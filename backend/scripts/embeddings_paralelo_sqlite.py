"""Gera embeddings no SQLite de ingestão enquanto a coleta ainda roda.

O encode na CPU deste notebook (~15 textos/s) é o gargalo depois da coleta:
730 mil documentos levariam ~14 h só no fim. Este processo gera os vetores em
paralelo com a coleta, direto em data/transparencia_local.db, usando o mesmo
modelo, os mesmos textos (EmbeddingGenerator._compor_texto*) e a mesma
normalização de generate_embeddings.py. sync_to_local_pg.py copia os vetores
prontos para o Postgres e generate_embeddings_lote.py só completa o que faltar.

Só pega linhas cujo texto já está final:
  * emendas: todas (o texto não muda depois da fase de emendas);
  * documentos: detalhes_coletados = 1 e observação preenchida;
  * favorecidos: com nome (o nome só é gravado uma vez).
O vetor vai como JSON em embedding, o formato que o próprio projeto usa no SQLite.

Uso (de backend/):
    py scripts/embeddings_paralelo_sqlite.py              # até a coleta acabar
    py scripts/embeddings_paralelo_sqlite.py --uma-passada
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.indexing.embedding_generator import EmbeddingGenerator  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent
SQLITE = BACKEND / "data" / "transparencia_local.db"
PARAR = BACKEND / "data" / "logs" / "embeddings_paralelo.parar"
# Chaves da fase detalhes no checkpoint: as 27 UFs. As emendas sem UF (SEM_UF) ficam
# fora desde 14/10 (decisão do usuário; ver INCLUIR_SEM_UF em popular_base_nacional.py),
# e esperar por elas deixaria o worker rodando para sempre depois da coleta.
CHAVES = ["RS", "PR", "SC", "SP", "RJ", "MG", "ES", "GO", "MS", "MT", "DF", "AC", "AL",
          "AM", "AP", "BA", "CE", "MA", "PA", "PB", "PE", "PI", "RN", "RO", "RR", "SE",
          "TO"]
# Escrita bloqueada por outro processo (a reconciliação de valores no fim de cada UF
# segura o SQLite por minutos): espera e tenta de novo em vez de morrer.
ESPERA_TRAVA_S = 30
MAX_TENTATIVAS_TRAVA = 40  # ~20 min

FONTES = [
    ("emendas", """
        SELECT e.id, e.nome_autor, e.funcao_nome, e.subfuncao_nome, e.descricao, e.uf,
               e.ano, e.localidade, e.tipo_emenda, p.partido
        FROM emendas e LEFT JOIN parlamentares p ON e.cod_autor = p.cod_autor
        WHERE e.embedding IS NULL AND e.id > ? ORDER BY e.id LIMIT ?
    """, lambda g, r: g._compor_texto(r)),
    ("documentos_emenda", """
        SELECT id, observacao, favorecido_nome, funcao, subfuncao, elemento_despesa
        FROM documentos_emenda
        WHERE embedding IS NULL AND detalhes_coletados = 1
          AND observacao IS NOT NULL AND observacao != '' AND id > ?
        ORDER BY id LIMIT ?
    """, lambda g, r: g._compor_texto_documento(r)),
    ("favorecidos", """
        SELECT id, nome, tipo_pessoa, uf FROM favorecidos
        WHERE embedding IS NULL AND nome IS NOT NULL AND nome != ''
          AND nome != 'Sem informação' AND id > ? ORDER BY id LIMIT ?
    """, lambda g, r: g._compor_texto_favorecido(r)),
]


def _conectar() -> sqlite3.Connection:
    c = sqlite3.connect(SQLITE, timeout=120)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout = 120000")
    return c


def _gravar_vetores(conectar, tabela: str, pares: list, espera_s: float = ESPERA_TRAVA_S,
                    tentativas: int = MAX_TENTATIVAS_TRAVA) -> None:
    """UPDATE em transação curta; se o SQLite seguir travado, espera e repete."""
    for tentativa in range(1, tentativas + 1):
        c = conectar()
        try:
            c.executemany(f"UPDATE {tabela} SET embedding = ? WHERE id = ?", pares)
            c.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or tentativa == tentativas:
                raise
            print(f"[{time.strftime('%H:%M:%S')}] {tabela}: banco travado por outro "
                  f"processo (tentativa {tentativa}); nova tentativa em {espera_s:.0f} s",
                  flush=True)
        finally:
            c.close()
        time.sleep(espera_s)


def passada(gerador, bloco: int, batch: int) -> int:
    feitos = 0
    for tabela, sql, compor in FONTES:
        ultimo = 0
        while not PARAR.exists():
            c = _conectar()
            try:
                linhas = [dict(r) for r in c.execute(sql, (ultimo, bloco)).fetchall()]
            finally:
                c.close()
            if not linhas:
                break
            t0 = time.monotonic()
            vetores = gerador.model.encode([compor(gerador, r) for r in linhas],
                                           batch_size=batch, normalize_embeddings=True,
                                           show_progress_bar=False)
            # transação curta: a coleta continua gravando no mesmo arquivo
            _gravar_vetores(_conectar, tabela,
                            [(json.dumps(v.tolist()), r["id"])
                             for r, v in zip(linhas, vetores)])
            feitos += len(linhas)
            ultimo = linhas[-1]["id"]
            print(f"[{time.strftime('%H:%M:%S')}] {tabela}: +{len(linhas)} "
                  f"({len(linhas) / (time.monotonic() - t0):.0f}/s)", flush=True)
    return feitos


def coleta_terminou() -> bool:
    cp = BACKEND / "data" / "checkpoint_brasil.json"
    try:
        dados = json.loads(cp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    detalhes = dados.get("detalhes", {})
    return all(detalhes.get(chave, {}).get("status") == "completed" for chave in CHAVES)


def _fixar_tmp() -> None:
    """Mesmo motivo de popular_base_nacional._fixar_tmp: um TMP herdado pode sumir."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    tmp = Path(base) / "Temp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(tmp)


def main():
    _fixar_tmp()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bloco", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--uma-passada", action="store_true")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    gerador = EmbeddingGenerator()
    print(f"embeddings paralelos: threads={args.threads} batch={args.batch} "
          f"bloco={args.bloco}", flush=True)
    while not PARAR.exists():
        feitos = passada(gerador, args.bloco, args.batch)
        if args.uma_passada or (feitos == 0 and coleta_terminou()):
            break
        if feitos == 0:
            time.sleep(120)  # nada pronto ainda: espera a coleta avançar
    print("embeddings paralelos encerrados", flush=True)


if __name__ == "__main__":
    main()
