"""Gera os mesmos embeddings de generate_embeddings.py, em lotes e retomável.

Mesmos textos (EmbeddingGenerator._compor_texto*), mesmo modelo
(settings.EMBEDDING_MODEL), mesma normalização e os mesmos filtros de linha;
muda só a gravação: em vez de um UPDATE por linha numa transação única (30 min
para 34 mil emendas neste notebook), grava em lotes com UPDATE ... FROM (VALUES)
e confirma a cada bloco. Se cair, roda de novo: só processa o que está sem vetor.

Uso (DATABASE_URL apontando para o Postgres):
    py scripts/generate_embeddings_lote.py
    py scripts/generate_embeddings_lote.py --bloco 4096 --batch 64 --threads 4
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import execute_values  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import SessionLocal, engine  # noqa: E402
from app.services.indexing.embedding_generator import EmbeddingGenerator  # noqa: E402
from app.services.indexing.index_manager import IndexManager  # noqa: E402

# (tabela, SELECT com o mesmo filtro de generate_embeddings.py, compositor de texto)
FONTES = [
    ("emendas", """
        SELECT e.id, e.nome_autor, e.funcao_nome, e.subfuncao_nome, e.descricao, e.uf,
               e.ano, e.localidade, e.tipo_emenda, p.partido
        FROM emendas e LEFT JOIN parlamentares p ON e.cod_autor = p.cod_autor
        WHERE e.embedding IS NULL AND e.id > :ultimo ORDER BY e.id LIMIT :n
    """, lambda g, r: g._compor_texto(r)),
    ("classificacao_orcamentaria", """
        SELECT id, funcao_nome, subfuncao_nome, programa_nome, descricao
        FROM classificacao_orcamentaria
        WHERE embedding IS NULL AND id > :ultimo ORDER BY id LIMIT :n
    """, lambda g, r: f"{r['funcao_nome']} {r['subfuncao_nome']} {r['programa_nome']} "
                      f"{r['descricao']}"),
    ("documentos_emenda", """
        SELECT id, observacao, favorecido_nome, funcao, subfuncao, elemento_despesa
        FROM documentos_emenda
        WHERE embedding IS NULL AND observacao IS NOT NULL AND observacao != ''
          AND id > :ultimo ORDER BY id LIMIT :n
    """, lambda g, r: g._compor_texto_documento(r)),
    ("favorecidos", """
        SELECT id, nome, tipo_pessoa, uf FROM favorecidos
        WHERE embedding IS NULL AND nome IS NOT NULL AND nome != ''
          AND nome != 'Sem informação' AND id > :ultimo ORDER BY id LIMIT :n
    """, lambda g, r: g._compor_texto_favorecido(r)),
]


def _vetor(v) -> str:
    # repr do float64 de cada float32: ida e volta exata; o pgvector guarda float32.
    return "[" + ",".join(map(repr, v.tolist())) + "]"


def processar(gerador, tabela, sql, compor, bloco, batch) -> int:
    restantes = None
    with engine.connect() as c:
        restantes = c.execute(text(
            f"SELECT COUNT(*) FROM ({sql.replace('LIMIT :n', '')}) s"),
            {"ultimo": 0}).scalar()
    print(f"[{tabela}] sem embedding: {restantes:,}", flush=True)
    if not restantes:
        return 0

    feitos, ultimo, t0 = 0, 0, time.monotonic()
    while True:
        with engine.connect() as c:
            linhas = [dict(r._mapping) for r in c.execute(
                text(sql), {"ultimo": ultimo, "n": bloco}).fetchall()]
        if not linhas:
            break
        textos = [compor(gerador, r) for r in linhas]
        vetores = gerador.model.encode(textos, batch_size=batch, show_progress_bar=False,
                                       normalize_embeddings=True)
        raw = engine.raw_connection()
        try:
            with raw.cursor() as cur:
                execute_values(
                    cur,
                    f"UPDATE {tabela} AS t SET embedding = v.emb::vector "
                    "FROM (VALUES %s) AS v(id, emb) WHERE t.id = v.id",
                    [(r["id"], _vetor(v)) for r, v in zip(linhas, vetores)],
                    page_size=500)
            raw.commit()
        finally:
            raw.close()
        feitos += len(linhas)
        ultimo = linhas[-1]["id"]
        taxa = feitos / (time.monotonic() - t0)
        eta = (restantes - feitos) / taxa / 60 if taxa else 0
        print(f"[{tabela}] {feitos:,}/{restantes:,} ({taxa:.0f}/s, ETA {eta:.0f} min)",
              flush=True)
    return feitos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bloco", type=int, default=4096, help="Linhas por commit")
    parser.add_argument("--batch", type=int, default=settings.EMBEDDING_BATCH_SIZE,
                        help="Batch do encode (padrão: EMBEDDING_BATCH_SIZE)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Threads do torch (0 = padrão do torch)")
    parser.add_argument("--sem-indices", action="store_true",
                        help="Não cria os índices HNSW de emendas/classificações no fim")
    args = parser.parse_args()

    if str(settings.DATABASE_URL).startswith("sqlite"):
        sys.exit("Este script grava com UPDATE ... ::vector: use um DATABASE_URL Postgres.")

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"modelo={settings.EMBEDDING_MODEL} batch={args.batch} bloco={args.bloco} "
          f"threads={torch.get_num_threads()}", flush=True)

    gerador = EmbeddingGenerator()
    for tabela, sql, compor in FONTES:
        processar(gerador, tabela, sql, compor, args.bloco, args.batch)

    if not args.sem_indices:  # igual ao fim de generate_embeddings.py
        db = SessionLocal()
        try:
            IndexManager.criar_indices_hnsw(db, m=settings.HNSW_M,
                                            ef_construction=settings.HNSW_EF_CONSTRUCTION)
        finally:
            db.close()
    print("embeddings concluídos", flush=True)


if __name__ == "__main__":
    main()
