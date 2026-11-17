"""Copia a base de ingestão (SQLite) para um PostgreSQL LOCAL.

O pipeline do projeto coleta tudo em data/transparencia_local.db (SQLite) com
run_ingestion_brasil.py e depois sincroniza com o Postgres via
sync_to_production.py. Aquele script tem o destino fixo no Supabase de produção;
este reaproveita as mesmas funções de sincronização, mas:

  * o destino vem de PG_DESTINO_URL e precisa ser localhost/127.0.0.1;
  * copia também a tabela parlamentares, que o sync de produção não copia;
  * alarga as colunas VARCHAR do destino quando o SQLite (que não aplica o
    limite de tamanho) tem valores maiores que o modelo;
  * desliga as checagens de FK durante a carga (o SQLite não as aplica, então
    um lookup pode apontar para um código ausente) e conta os órfãos no fim.

As tabelas precisam existir no destino: rode antes init_db() com
DATABASE_URL apontando para o mesmo banco. Embeddings não são copiados; gere-os
depois com scripts/generate_embeddings.py apontando para o Postgres.

Uso:
    $env:PG_DESTINO_URL = "postgresql://postgres:<senha>@localhost:5433/transparencia_nacional"
    py scripts/sync_to_local_pg.py
    py scripts/sync_to_local_pg.py --dry-run
    py scripts/sync_to_local_pg.py --sqlite data/outro.db
"""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
import sync_to_production as base  # noqa: E402

HOSTS_LOCAIS = {"localhost", "127.0.0.1", "::1"}

# Tabelas copiadas por base.run_sync (mais parlamentares, copiada aqui).
TABELAS_COPIADAS = [
    "parlamentares", "funcoes", "subfuncoes", "programas", "acoes", "orgaos",
    "unidades_gestoras", "categorias_despesa", "grupos_despesa",
    "elementos_despesa", "modalidades_aplicacao", "emendas",
    "documentos_emenda", "favorecidos", "beneficiarios", "sancoes_ceis", "sancoes_cnep",
    "sancoes_cepim", "sancoes_ceaf", "convenios",
]

ORFAOS = {
    "emendas com cod_autor sem parlamentar": """
        SELECT COUNT(*) FROM emendas e
        WHERE e.cod_autor IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM parlamentares p WHERE p.cod_autor = e.cod_autor)""",
    "documentos sem emenda": """
        SELECT COUNT(*) FROM documentos_emenda d
        WHERE NOT EXISTS (SELECT 1 FROM emendas e WHERE e.id = d.emenda_id)""",
    "vínculos documento_favorecido quebrados": """
        SELECT COUNT(*) FROM documento_favorecido df
        WHERE NOT EXISTS (SELECT 1 FROM documentos_emenda d WHERE d.id = df.documento_id)
           OR NOT EXISTS (SELECT 1 FROM favorecidos f WHERE f.id = df.favorecido_id)""",
    "subfuncoes com funcao_codigo ausente": """
        SELECT COUNT(*) FROM subfuncoes s
        WHERE s.funcao_codigo IS NOT NULL AND s.funcao_codigo <> ''
          AND NOT EXISTS (SELECT 1 FROM funcoes f WHERE f.codigo = s.funcao_codigo)""",
    "unidades_gestoras com orgao ausente": """
        SELECT COUNT(*) FROM unidades_gestoras u
        WHERE u.orgao_codigo IS NOT NULL AND u.orgao_codigo <> ''
          AND NOT EXISTS (SELECT 1 FROM orgaos o WHERE o.codigo = u.orgao_codigo)""",
}


def destino_url() -> str:
    url = os.environ.get("PG_DESTINO_URL", "")
    if not url:
        sys.exit("Defina PG_DESTINO_URL, ex.: "
                 "postgresql://postgres:<senha>@localhost:5433/transparencia_nacional")
    host = urlparse(url).hostname or ""
    if host not in HOSTS_LOCAIS:
        sys.exit(f"Destino recusado: o host '{host}' não é local. "
                 "Este script só grava em Postgres local.")
    return url


def conectar_destino():
    """Conexão com FKs desligadas na sessão (carga em lote a partir do SQLite)."""
    conn = psycopg2.connect(destino_url())
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role = replica")
    conn.commit()  # SET de sessão persiste após o commit
    return conn


def descrever_destino() -> str:
    u = urlparse(destino_url())
    return f"{u.hostname}:{u.port or 5432}{u.path}"


def alargar_colunas(lite, pg_cur, dry_run: bool):
    """Alarga VARCHAR(n) do destino quando o SQLite tem valores mais longos."""
    tabelas_sqlite = {r[0] for r in lite.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    alteradas = 0
    for tabela in TABELAS_COPIADAS:
        if tabela not in tabelas_sqlite:
            continue
        colunas_sqlite = {r[1] for r in lite.execute(f"PRAGMA table_info({tabela})")}
        pg_cur.execute("""
            SELECT column_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
              AND data_type = 'character varying'
              AND character_maximum_length IS NOT NULL
        """, (tabela,))
        for coluna, limite in pg_cur.fetchall():
            if coluna not in colunas_sqlite:
                continue
            maior = lite.execute(
                f"SELECT MAX(LENGTH({coluna})) FROM {tabela}").fetchone()[0] or 0
            if maior <= limite:
                continue
            sql = f"ALTER TABLE {tabela} ALTER COLUMN {coluna} TYPE VARCHAR({maior})"
            if dry_run:
                base.log(f"  [DRY-RUN] {sql}  (limite atual {limite})")
            else:
                pg_cur.execute(sql)
                base.log(f"  ~ {tabela}.{coluna}: VARCHAR({limite}) -> VARCHAR({maior})")
            alteradas += 1
    base.log(f"  colunas alargadas: {alteradas}")


def sync_parlamentares(lite, pg_cur, dry_run: bool):
    rows = lite.execute("""
        SELECT cod_autor, nome, nome_civil, partido, uf, legislaturas, url_foto
        FROM parlamentares
    """).fetchall()
    dados = []
    for r in rows:
        legislaturas = r["legislaturas"]
        if isinstance(legislaturas, str):
            legislaturas = json.loads(legislaturas) if legislaturas else None
        dados.append((r["cod_autor"], r["nome"], r["nome_civil"], r["partido"],
                      r["uf"], legislaturas, r["url_foto"]))

    sql = """
        INSERT INTO parlamentares
            (cod_autor, nome, nome_civil, partido, uf, legislaturas, url_foto)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (cod_autor) DO UPDATE SET
            nome = EXCLUDED.nome, nome_civil = EXCLUDED.nome_civil,
            partido = EXCLUDED.partido, uf = EXCLUDED.uf,
            legislaturas = EXCLUDED.legislaturas, url_foto = EXCLUDED.url_foto
    """
    if not dry_run:
        psycopg2.extras.execute_batch(pg_cur, sql, dados, page_size=1000)
    base.log(f"  parlamentares: {len(dados)} copiados")


def sync_beneficiarios(lite, pg_cur, dry_run: bool):
    """Recopia beneficiarios (favorecidos finais), mapeando documento_id pelo código.

    O sync de produção não copia esta tabela. Como o destino é local e é nosso,
    a cópia é completa: apaga e insere de novo.
    """
    local_doc = {r["id"]: r["codigo_documento"] for r in lite.execute(
        "SELECT id, codigo_documento FROM documentos_emenda").fetchall()}
    pg_cur.execute("SELECT codigo_documento, id FROM documentos_emenda")
    destino_doc = dict(pg_cur.fetchall())

    linhas, sem_documento = [], 0
    for r in lite.execute("""
        SELECT documento_id, codigo_pessoa, nome, tipo_pessoa, cpf_cnpj, valor_recebido
        FROM beneficiarios ORDER BY id
    """).fetchall():
        doc_id = destino_doc.get(local_doc.get(r["documento_id"]))
        if doc_id is None:
            sem_documento += 1
            continue
        linhas.append((doc_id, r["codigo_pessoa"], r["nome"], r["tipo_pessoa"],
                       r["cpf_cnpj"], r["valor_recebido"]))

    if not dry_run:
        pg_cur.execute("DELETE FROM beneficiarios")
        psycopg2.extras.execute_batch(pg_cur, """
            INSERT INTO beneficiarios
                (documento_id, codigo_pessoa, nome, tipo_pessoa, cpf_cnpj, valor_recebido)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, linhas, page_size=1000)
    base.log(f"  beneficiarios: {len(linhas)} copiados (sem documento no destino: {sem_documento})")


def copiar_embeddings(lite, pg_cur, dry_run: bool):
    """Leva para o Postgres os vetores já gerados no SQLite (JSON -> vector).

    Casa pelas chaves naturais (codigo_emenda, codigo_documento, cpf_cnpj), porque
    os ids mudam entre os dois bancos. Só preenche onde o destino ainda está sem
    vetor; o que faltar, generate_embeddings_lote.py completa.
    """
    alvos = [("emendas", "codigo_emenda"), ("documentos_emenda", "codigo_documento"),
             ("favorecidos", "cpf_cnpj")]
    for tabela, chave in alvos:
        cursor = lite.execute(
            f"SELECT {chave}, embedding FROM {tabela} WHERE embedding IS NOT NULL")
        total = 0
        while True:
            lote = cursor.fetchmany(2000)
            if not lote:
                break
            total += len(lote)
            if not dry_run:
                psycopg2.extras.execute_values(
                    pg_cur,
                    f"UPDATE {tabela} AS t SET embedding = v.emb::vector "
                    f"FROM (VALUES %s) AS v(chave, emb) "
                    f"WHERE t.{chave} = v.chave AND t.embedding IS NULL",
                    [(r[0], r[1]) for r in lote], page_size=500)
        base.log(f"  {tabela}: {total:,} vetores do SQLite")


def contar_orfaos(pg_cur):
    base.log("Conferência de integridade (órfãos):")
    for descricao, sql in ORFAOS.items():
        pg_cur.execute(sql)
        base.log(f"  {descricao}: {pg_cur.fetchone()[0]:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Copia o SQLite de ingestão para um Postgres local")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que faria, sem gravar no destino")
    parser.add_argument("--sqlite", type=Path, default=base.SQLITE_PATH,
                        help="SQLite de origem (padrão: data/transparencia_local.db)")
    args = parser.parse_args()

    if not args.sqlite.exists():
        sys.exit(f"SQLite de origem não encontrado: {args.sqlite}")

    base.SQLITE_PATH = args.sqlite
    base.pg_connect = conectar_destino  # run_sync passa a usar o destino local
    base.log(f"Origem: {args.sqlite}")
    base.log(f"Destino (local): {descrever_destino()}")

    lite = base.sqlite_connect()
    pg = conectar_destino()
    try:
        with pg.cursor() as cur:
            base.log("=== COLUNAS ===")
            alargar_colunas(lite, cur, args.dry_run)
            base.log("=== PARLAMENTARES ===")
            sync_parlamentares(lite, cur, args.dry_run)
        if not args.dry_run:
            pg.commit()
    except Exception:
        pg.rollback()
        raise
    finally:
        pg.close()
        lite.close()

    base.run_sync(None, args.dry_run)

    lite = base.sqlite_connect()
    pg = conectar_destino()
    try:
        with pg.cursor() as cur:
            base.log("=== BENEFICIARIOS ===")
            sync_beneficiarios(lite, cur, args.dry_run)
            base.log("=== EMBEDDINGS JÁ GERADOS NO SQLITE ===")
            copiar_embeddings(lite, cur, args.dry_run)
        if not args.dry_run:
            pg.commit()
        with pg.cursor() as cur:
            contar_orfaos(cur)
    except Exception:
        pg.rollback()
        raise
    finally:
        pg.close()
        lite.close()


if __name__ == "__main__":
    main()
