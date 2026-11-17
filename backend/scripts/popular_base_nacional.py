"""Monta a base nacional do artigo e roda a avaliação COM RAG, de ponta a ponta.

Etapas, em ordem. Cada uma é retomável, e o estado fica em
data/logs/pipeline_estado.json:

  emendas, documentos, detalhes   run_ingestion_brasil.py -> SQLite data/transparencia_local.db
  repescagem                      detalhes ainda pendentes por UF + dead letter queue
  sancoes                         CEIS, CNEP, CEPIM, CEAF (o RAG cruza favorecidos com sanções)
  beneficiarios_ac                favorecidos finais do Acre (tabela beneficiarios do artigo)
  sync_pg                         sync_to_local_pg.py -> Postgres local, banco transparencia_nacional
  embeddings                      generate_embeddings.py no Postgres
  indices                         índices e consultas_log das migrations 001/002 (alembic upgrade head)
  conferencia                     contagens mínimas antes de gastar tokens do LLM
  com_rag                         evaluate_120_com_sem_rag.py --fase com_rag (espera a VPN/LiteLLM)
  pontuar                         --fase pontuar: BERTScore de COM e SEM RAG, JSONs finais

O banco seed (transparencia_fiscal) não é tocado. Nada é apagado de tests/test_queries.

Uso (de backend/):
    py scripts/popular_base_nacional.py               # roda ou retoma tudo
    py scripts/popular_base_nacional.py --status      # estado, contagens e fim dos logs
    py scripts/popular_base_nacional.py --ate sync_pg # para depois dessa etapa
"""
import argparse
import ctypes
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

BACKEND = Path(__file__).resolve().parent.parent
DATA = BACKEND / "data"
LOGS = DATA / "logs"
ESTADO = LOGS / "pipeline_estado.json"
TRAVA = LOGS / "pipeline.lock"
LOG_PIPELINE = LOGS / "00_pipeline.log"
SQLITE = DATA / "transparencia_local.db"
SQLITE_URL = "sqlite:///data/transparencia_local.db"  # relativo a backend/
CP_BRASIL = DATA / "checkpoint_brasil.json"
CP_SANCOES = DATA / "checkpoint_sancoes.json"
CP_AC = DATA / "checkpoint_ac.json"
DLQ_BRASIL = DATA / "dead_letter_brasil.json"
BANCO_DESTINO = "transparencia_nacional"
MODELO = "gemini-2.5-flash"
DIR_PARCIAL = BACKEND / "tests" / "test_queries" / f"_parcial_{MODELO}"
SAIDA = BACKEND / "tests" / "test_queries"

ANOS = ["2020", "2021", "2022", "2023", "2024"]
UFS = ["RS", "PR", "SC", "SP", "RJ", "MG", "ES", "GO", "MS", "MT", "DF", "AC", "AL",
       "AM", "AP", "BA", "CE", "MA", "PA", "PB", "PE", "PI", "RN", "RO", "RR", "SE", "TO"]
SEM_UF = "SEM_UF"  # mesma chave de run_ingestion_brasil.py
CHAVES_UF = UFS + [SEM_UF]
# 14/10/2026: o usuário decidiu pular as emendas sem UF para ficar o mais perto da base
# do artigo. O checkpoint do André (commit 728d835) só tem documentos das 26 UFs e do AC.
# Os documentos SEM_UF já coletados ficam no SQLite e saem da base de avaliação no
# sync_pg; as emendas sem UF continuam na base, como na dele.
INCLUIR_SEM_UF = False


def _chaves_ativas() -> list[str]:
    return CHAVES_UF if INCLUIR_SEM_UF else UFS
SANCOES = ["ceis", "cnep", "cepim", "ceaf"]

# Limite por chave. O código original usava 400/min o dia todo. O limite oficial
# diurno não foi confirmado (a página da CGU recusa acesso automatizado); às
# 03h20 de Brasília uma rajada de 1.200 requisições em 34 s passou sem 429. Se a
# API devolver 429, o coletor espera e repete, e a repescagem cobre o que sobrar.
RPM_DIA = 380
# 14/10 22h-23h30: com 650/min o ritmo ficou nos mesmos ~320/min do dia (o WAF da CGU
# limita), os 403 triplicaram e 20 documentos do RJ esgotaram as tentativas. Igual ao
# dia até haver evidência de que a madrugada aceita mais.
RPM_NOITE = 380

MAX_ERROS_FASE = 20  # erros de etapa seguidos sem progresso (~10 min de retentativa) -> aborta
# O orquestrador roda DETACHED (sem console). Sem esta flag, cada filho python ganha
# uma janela de console visível, e fechar essa janela mata a coleta sem deixar nada
# no log: foi o que parou as emendas sem UF em 14/10 às 12:01.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
# Um filho que sai com chaves pendentes foi encerrado de fora (ele mesmo já repete
# os erros de etapa). Relança com --resume até este número de vezes antes de falhar.
MAX_RELANCES_FASE = 3
PAUSA_RELANCE_S = 60
# run_ingestion_brasil/ac registram erro_fase; run_ingestion_sancoes registra só "erro".
MARCAS_ERRO = (b"erro_fase", b"[error    ] erro ")
MARCAS_PROGRESSO = (b"progresso_", b"emendas_batch", b"_completos", b"_completas",
                    b"_completo", b"dlq_resolved", b"[COM RAG", b"[PONTUAR", b"batch_commit",
                    b"_progresso", b"[documentos_emenda]", b"[emendas]", b"[favorecidos]")
SONDA_LITELLM_S = 60

# Conferência antes da avaliação (a base do artigo: 32.787 / 915 / 730.571 / 371).
MINIMOS = {"emendas": 30_000, "parlamentares": 900, "documentos_emenda": 300_000,
           "favorecidos": 1_000}
MAX_PENDENTES_DETALHES = 0.02

INDICES = [  # migrations 001 e 002, na ordem
    "CREATE INDEX IF NOT EXISTS idx_parlamentares_partido ON parlamentares (partido)",
    "CREATE INDEX IF NOT EXISTS idx_parlamentares_uf ON parlamentares (uf)",
    "CREATE INDEX IF NOT EXISTS idx_parlamentares_nome ON parlamentares "
    "USING gin (to_tsvector('portuguese', nome))",
    "CREATE INDEX IF NOT EXISTS idx_classif_funcao ON classificacao_orcamentaria (funcao)",
    "CREATE INDEX IF NOT EXISTS idx_classif_subfuncao ON classificacao_orcamentaria (subfuncao)",
    "CREATE INDEX IF NOT EXISTS idx_classif_embedding ON classificacao_orcamentaria "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_ano ON emendas (ano)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_uf ON emendas (uf)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_cod_autor ON emendas (cod_autor)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_funcao ON emendas (funcao)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_subfuncao ON emendas (subfuncao)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_tipo ON emendas (tipo_emenda)",
    "CREATE INDEX IF NOT EXISTS idx_emendas_tsv ON emendas USING gin (to_tsvector('portuguese', "
    "coalesce(nome_autor, '') || ' ' || coalesce(funcao_nome, '') || ' ' || "
    "coalesce(subfuncao_nome, '') || ' ' || coalesce(descricao, '')))",
    "CREATE INDEX IF NOT EXISTS idx_emendas_embedding ON emendas "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
    "CREATE INDEX IF NOT EXISTS idx_execucao_emenda ON execucao (emenda_id)",
    "CREATE INDEX IF NOT EXISTS idx_execucao_fase ON execucao (fase)",
    "CREATE INDEX IF NOT EXISTS idx_execucao_data ON execucao (data_movimento)",
    """CREATE TABLE IF NOT EXISTS consultas_log (
        id SERIAL PRIMARY KEY, consulta_nl TEXT NOT NULL, entidades_json JSON,
        sql_gerada TEXT, modo_busca VARCHAR(20), num_resultados INTEGER,
        latencia_ms INTEGER, sucesso BOOLEAN DEFAULT true, erro TEXT,
        created_at TIMESTAMP DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS idx_log_created ON consultas_log (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_documentos_emenda_embedding ON documentos_emenda "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
    "CREATE INDEX IF NOT EXISTS idx_favorecidos_embedding ON favorecidos "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
]


class EtapaFalhou(Exception):
    pass


# ---------------------------------------------------------------------------
# Infraestrutura: log, estado, trava, subprocessos
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOG_PIPELINE, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def _ler_json(path: Path, padrao=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if padrao is None else padrao


def _gravar_json(path: Path, dados) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dados, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _estado() -> dict:
    return _ler_json(ESTADO) or {"etapas": {}}


def _marcar(etapa: str, status: str, detalhe: str = "") -> None:
    est = _estado()
    reg = est["etapas"].setdefault(etapa, {})
    agora = datetime.now().isoformat(timespec="seconds")
    if status == "rodando":
        reg["inicio"] = agora
        reg["tentativas"] = reg.get("tentativas", 0) + 1
    else:
        reg["fim"] = agora
    reg["status"] = status
    reg["detalhe"] = detalhe
    est["etapa_atual"] = etapa
    est["atualizado_em"] = agora
    est["pid"] = os.getpid()
    _gravar_json(ESTADO, est)


def _pid_vivo(pid: int) -> bool:
    """Sem os.kill(pid, 0): no Windows isso MATA o processo."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    codigo = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(codigo))
    k32.CloseHandle(h)
    return bool(ok) and codigo.value == 259  # STILL_ACTIVE


def _adquirir_trava() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if TRAVA.exists():
        try:
            pid = int(TRAVA.read_text().strip())
        except ValueError:
            pid = 0
        if pid and pid != os.getpid() and _pid_vivo(pid):
            sys.exit(f"Já existe um pipeline rodando (pid {pid}).")
    TRAVA.write_text(str(os.getpid()))


def _url_destino() -> str:
    url = (dotenv_values(BACKEND.parent / ".env").get("DATABASE_URL") or "").strip()
    u = urlparse(url)
    if u.hostname not in {"localhost", "127.0.0.1"}:
        sys.exit("DATABASE_URL do .env não aponta para um Postgres local.")
    return u._replace(path=f"/{BANCO_DESTINO}").geturl()


def _fixar_tmp() -> None:
    """Aponta TMP/TEMP para o temp do perfil do usuário, que não some.

    O pipeline roda por dias, desligado de quem o lançou. Se herdar o TMP de uma
    pasta com prazo (a pasta de rascunho de uma sessão do agente, por exemplo) e
    ela for apagada, o SQLite não consegue criar o arquivo temporário de
    ordenação e as consultas grandes falham com "unable to open database file".
    Os filhos herdam este valor por _env().
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    tmp = Path(base) / "Temp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(tmp)


def _env(banco: str) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NO_COLOR="1",
               CGU_RATE_LIMIT_RPM=str(RPM_DIA), CGU_RATE_LIMIT_RPM_NOITE=str(RPM_NOITE))
    if banco == "sqlite":
        env["DATABASE_URL"] = SQLITE_URL
    else:
        env["DATABASE_URL"] = env["PG_DESTINO_URL"] = _url_destino()
    return env


def _erros_sem_progresso(caminho: Path, inicio_log: int) -> int:
    """Erros de etapa desde a última linha de progresso (lê só o fim do log)."""
    tamanho = caminho.stat().st_size
    with open(caminho, "rb") as f:
        f.seek(max(inicio_log, tamanho - 1_000_000))
        trecho = f.read()
    ultimo = max(trecho.rfind(m) for m in MARCAS_PROGRESSO)
    depois = trecho[ultimo + 1:] if ultimo >= 0 else trecho
    return sum(depois.count(m) for m in MARCAS_ERRO)


def _rodar(nome_log: str, args: list[str], banco: str) -> int:
    """Roda um script do projeto, anexando a saída em data/logs/<nome_log>.log.

    Se o script entrar em laço de erro_fase sem progresso (mesma falha a cada
    30 s), mata o processo e falha a etapa: isso pede correção, não mais espera.
    """
    caminho = LOGS / f"{nome_log}.log"
    inicio_log = caminho.stat().st_size if caminho.exists() else 0
    cmd = [sys.executable, "-u", *args]
    log(f"  $ {' '.join(args)}  (banco={banco}, log={caminho.name})")
    with open(caminho, "ab") as saida:
        proc = subprocess.Popen(cmd, cwd=BACKEND, env=_env(banco), stdout=saida,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=CREATE_NO_WINDOW)
        while True:
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = None
            if rc is not None:
                if rc != 0:
                    # 0xC000013A = console fechado / Ctrl+C; 1 = exceção Python.
                    log(f"  saiu com código {rc} (0x{rc & 0xFFFFFFFF:08X})")
                return rc
            erros = _erros_sem_progresso(caminho, inicio_log)
            if erros >= MAX_ERROS_FASE:
                proc.kill()
                proc.wait()
                raise EtapaFalhou(f"{nome_log}: {erros} erro_fase seguidos sem progresso; "
                                  f"veja {caminho.name}")


def _sqlite(sql: str, params=()):
    conn = sqlite3.connect(SQLITE, timeout=60)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _pg():
    import psycopg2
    return psycopg2.connect(_url_destino())


def _pg_um(sql: str):
    conn = _pg()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()[0]
    finally:
        conn.close()


def _status_cp(path: Path, fonte: str, chave: str) -> str:
    return _ler_json(path).get(fonte, {}).get(chave, {}).get("status", "pendente")


def _todas_completas(path: Path, fonte: str, chaves) -> bool:
    return all(_status_cp(path, fonte, k) == "completed" for k in chaves)


def _pendentes_detalhes_por_uf() -> dict:
    # uf vazio vira SEM_UF: é a chave que --uf e o checkpoint usam para essas emendas.
    pend = dict(_sqlite(f"""
        SELECT CASE WHEN COALESCE(e.uf, '') = '' THEN '{SEM_UF}' ELSE e.uf END, COUNT(*)
        FROM documentos_emenda d JOIN emendas e ON e.id = d.emenda_id
        WHERE COALESCE(d.detalhes_coletados, 0) = 0 GROUP BY 1
    """))
    if not INCLUIR_SEM_UF:
        pend.pop(SEM_UF, None)  # fora da base de avaliação: a repescagem não os busca
    return pend


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------
def etapa_emendas() -> str:
    if not _todas_completas(CP_BRASIL, "emendas", ANOS):
        _rodar("01_emendas", ["scripts/run_ingestion_brasil.py", "--fase", "emendas",
                              "--resume"], "sqlite")
    if not _todas_completas(CP_BRASIL, "emendas", ANOS):
        raise EtapaFalhou("checkpoint de emendas incompleto")
    return f"{_sqlite('SELECT COUNT(*) FROM emendas')[0][0]:,} emendas"


def _rodar_ate_completar(fase: str, nome_log: str, args: list[str], chaves) -> None:
    """Roda a passada; se o filho sair com chaves pendentes, retoma do checkpoint.

    Retomar é seguro: o checkpoint para antes da primeira emenda inacabada e as
    gravações usam INSERT OR IGNORE. Se as relances acabarem, quem chama falha a etapa.
    """
    for relance in range(MAX_RELANCES_FASE + 1):
        pendentes = [k for k in chaves if _status_cp(CP_BRASIL, fase, k) != "completed"]
        if not pendentes:
            return
        if relance:
            log(f"  {fase}: filho saiu com {pendentes} pendentes; relance "
                f"{relance}/{MAX_RELANCES_FASE} em {PAUSA_RELANCE_S} s")
            time.sleep(PAUSA_RELANCE_S)
        _rodar(nome_log, args, "sqlite")


def _fase_por_uf(fase: str, nome_log: str) -> None:
    # A fase nacional pula o AC (o André coletava o Acre à parte); --uf AC cobre.
    # SEM_UF: emendas de localidade MÚLTIPLO/Nacional/regional, que ficam com uf vazio;
    # entram na passada nacional só se INCLUIR_SEM_UF.
    chaves = _chaves_ativas()
    nacional = ["scripts/run_ingestion_brasil.py", "--fase", fase, "--resume"]
    if not INCLUIR_SEM_UF:
        nacional.append("--pular-sem-uf")
    _rodar_ate_completar(fase, nome_log, nacional, [u for u in chaves if u != "AC"])
    _rodar_ate_completar(fase, nome_log,
                         ["scripts/run_ingestion_brasil.py", "--fase", fase, "--uf", "AC",
                          "--resume"], ["AC"])
    faltando = [u for u in chaves if _status_cp(CP_BRASIL, fase, u) != "completed"]
    if faltando:
        raise EtapaFalhou(f"{fase}: UFs incompletas {faltando}")


def etapa_documentos() -> str:
    _fase_por_uf("documentos", "02_documentos")
    return f"{_sqlite('SELECT COUNT(*) FROM documentos_emenda')[0][0]:,} documentos"


def etapa_detalhes() -> str:
    _fase_por_uf("detalhes", "03_detalhes")
    pend = sum(_pendentes_detalhes_por_uf().values())
    return f"{_sqlite('SELECT COUNT(*) FROM favorecidos')[0][0]:,} favorecidos; {pend:,} pendentes"


def _podar_dlq(fases=("documentos", "detalhes")) -> int:
    """Tira da DLQ o que a repescagem já resolveu; devolve o que sobrou."""
    itens = _ler_json(DLQ_BRASIL, padrao=[])
    if not itens:
        return 0
    restantes = []
    for item in itens:
        if item["fase"] not in fases:
            restantes.append(item)
            continue
        if item["fase"] == "detalhes":
            row = _sqlite("SELECT COALESCE(detalhes_coletados, 0) FROM documentos_emenda "
                          "WHERE codigo_documento = ?", (item["identificador"],))
            resolvido = bool(row and row[0][0])
        else:
            row = _sqlite("SELECT COUNT(*) FROM documentos_emenda d JOIN emendas e "
                          "ON e.id = d.emenda_id WHERE e.codigo_emenda = ?",
                          (item["identificador"],))
            resolvido = bool(row and row[0][0])
        if not resolvido:
            restantes.append(item)
    _gravar_json(DLQ_BRASIL, restantes)
    return len(restantes)


def _passadas_detalhes() -> None:
    for passada in (1, 2):
        pend = _pendentes_detalhes_por_uf()
        if not pend:
            break
        log(f"  passada {passada}: pendentes por UF {pend}")
        cp = _ler_json(CP_BRASIL)
        for uf in pend:
            cp.get("detalhes", {}).pop(uf, None)
        _gravar_json(CP_BRASIL, cp)
        for uf in pend:
            _rodar("04_repescagem", ["scripts/run_ingestion_brasil.py", "--fase", "detalhes",
                                     "--uf", uf, "--resume"], "sqlite")


def etapa_repescagem() -> str:
    _passadas_detalhes()
    # Só os detalhes são podados antes: uma emenda com documentos parciais (403 do WAF
    # no meio da paginação) conta como "resolvida" e perderia a rebusca completa.
    if _podar_dlq(fases=("detalhes",)):
        _rodar("04_repescagem", ["scripts/run_ingestion_brasil.py", "--retry-dlq"], "sqlite")
        _passadas_detalhes()  # documentos trazidos pela DLQ ainda não têm detalhes
    restam_dlq = _podar_dlq()
    pend = sum(_pendentes_detalhes_por_uf().values())
    return f"{pend:,} documentos ainda sem detalhes; DLQ restante {restam_dlq}"


def etapa_sancoes() -> str:
    for tipo in SANCOES:
        if _status_cp(CP_SANCOES, "sancoes", tipo) != "completed":
            _rodar("05_sancoes", ["scripts/run_ingestion_sancoes.py", "--tipo", tipo,
                                  "--resume"], "sqlite")
    faltando = [t for t in SANCOES if _status_cp(CP_SANCOES, "sancoes", t) != "completed"]
    if faltando:
        raise EtapaFalhou(f"sanções incompletas: {faltando}")
    return ", ".join(f"{t}={_sqlite(f'SELECT COUNT(*) FROM sancoes_{t}')[0][0]:,}"
                     for t in SANCOES)


def etapa_beneficiarios_ac() -> str:
    if _status_cp(CP_AC, "beneficiarios", "all") != "completed":
        _rodar("06_beneficiarios_ac", ["scripts/run_ingestion_ac.py", "--etapa",
                                       "beneficiarios", "--resume"], "sqlite")
    if _status_cp(CP_AC, "beneficiarios", "all") != "completed":
        raise EtapaFalhou("beneficiários do AC incompletos")
    return f"{_sqlite('SELECT COUNT(*) FROM beneficiarios')[0][0]:,} beneficiários"


_FILTRO_DOC_SEM_UF = """d.emenda_id = e.id AND COALESCE(e.uf, '') = ''
    AND NOT EXISTS (SELECT 1 FROM beneficiarios b WHERE b.documento_id = d.id)"""
SQL_APAGAR_VINCULOS_SEM_UF = f"""DELETE FROM documento_favorecido df
    USING documentos_emenda d, emendas e
    WHERE df.documento_id = d.id AND {_FILTRO_DOC_SEM_UF}"""
SQL_APAGAR_DOCS_SEM_UF = f"""DELETE FROM documentos_emenda d USING emendas e
    WHERE {_FILTRO_DOC_SEM_UF}"""


def _tirar_sem_uf_do_pg() -> tuple[int, int]:
    """Apaga do Postgres os documentos das emendas sem UF e seus vínculos.

    Só a base de avaliação perde essas linhas; o SQLite de coleta fica intacto, então
    um novo sync com INCLUIR_SEM_UF = True as traz de volta. beneficiarios aponta para
    documentos_emenda com ON DELETE CASCADE: documento citado por um beneficiário fica.
    """
    conn = _pg()
    try:
        with conn.cursor() as cur:
            # Índices de apoio ANTES do delete: sem eles o ON DELETE CASCADE de
            # documento_favorecido/beneficiarios apaga linha a linha (seq scan por
            # documento_id) e o DELETE de ~637 mil docs leva 20+ min. A etapa
            # 'indices' só roda DEPOIS do sync, então estes precisam existir aqui.
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documentos_emenda_emenda_id "
                        "ON documentos_emenda (emenda_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_docfav_documento_id "
                        "ON documento_favorecido (documento_id)")
            conn.commit()
            cur.execute(SQL_APAGAR_VINCULOS_SEM_UF)
            vinculos = cur.rowcount
            cur.execute(SQL_APAGAR_DOCS_SEM_UF)
            docs = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return docs, vinculos


# Contagem esperada no Postgres quando os documentos SEM_UF ficam de fora.
CONTAGEM_SQLITE_SEM_UF_FORA = {
    "documentos_emenda": "SELECT COUNT(*) FROM documentos_emenda d JOIN emendas e "
                         "ON e.id = d.emenda_id WHERE COALESCE(e.uf, '') <> ''",
    "documento_favorecido": "SELECT COUNT(*) FROM documento_favorecido df "
                            "JOIN documentos_emenda d ON d.id = df.documento_id "
                            "JOIN emendas e ON e.id = d.emenda_id "
                            "WHERE COALESCE(e.uf, '') <> ''",
}


def etapa_sync_pg() -> str:
    _rodar("07_sync_pg", ["-c", "import app.models; from app.database import init_db; "
                                "init_db(); print('schema ok')"], "pg")
    rc = _rodar("07_sync_pg", ["scripts/sync_to_local_pg.py"], "pg")
    if rc != 0:
        raise EtapaFalhou(f"sync_to_local_pg saiu com {rc}")
    if not INCLUIR_SEM_UF:
        docs, vinculos = _tirar_sem_uf_do_pg()
        log(f"  SEM_UF fora da base de avaliação: {docs:,} documentos e {vinculos:,} "
            "vínculos apagados do Postgres (o SQLite fica intacto)")
    diferencas = {}
    for tabela in ("emendas", "documentos_emenda", "favorecidos", "documento_favorecido",
                   "beneficiarios", "parlamentares"):
        if not INCLUIR_SEM_UF and tabela in CONTAGEM_SQLITE_SEM_UF_FORA:
            local = _sqlite(CONTAGEM_SQLITE_SEM_UF_FORA[tabela])[0][0]
        else:
            local = _sqlite(f"SELECT COUNT(*) FROM {tabela}")[0][0]
        destino = _pg_um(f"SELECT COUNT(*) FROM {tabela}")
        if destino < local:
            diferencas[tabela] = (local, destino)
    if diferencas:
        raise EtapaFalhou(f"Postgres com menos linhas que o SQLite: {diferencas}")
    return f"{_pg_um('SELECT COUNT(*) FROM documentos_emenda'):,} documentos no Postgres"


def etapa_embeddings() -> str:
    # Mesmos vetores de generate_embeddings.py, gravados em lote e retomáveis.
    rc = _rodar("08_embeddings", ["scripts/generate_embeddings_lote.py"], "pg")
    if rc != 0:
        raise EtapaFalhou(f"generate_embeddings_lote saiu com {rc}")
    faltam = _pg_um("SELECT COUNT(*) FROM emendas WHERE embedding IS NULL")
    faltam_docs = _pg_um("SELECT COUNT(*) FROM documentos_emenda WHERE embedding IS NULL "
                         "AND observacao IS NOT NULL AND observacao <> ''")
    if faltam or faltam_docs:
        raise EtapaFalhou(f"sem embedding: {faltam} emendas, {faltam_docs} documentos")
    docs = _pg_um("SELECT COUNT(*) FROM documentos_emenda WHERE embedding IS NOT NULL")
    return f"embeddings ok ({docs:,} documentos)"


def etapa_indices() -> str:
    conn = _pg()
    try:
        with conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem = '3GB'")
            # Build paralelo usa /dev/shm, que tem 64 MB no container: fica serial.
            cur.execute("SET max_parallel_maintenance_workers = 0")
            for sql in INDICES:
                t0 = time.monotonic()
                cur.execute(sql)
                conn.commit()
                log(f"  {' '.join(sql.split())[:90]}... {time.monotonic() - t0:.0f}s")
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("ANALYZE")
    finally:
        conn.close()
    return f"{len(INDICES)} comandos de schema/índice aplicados"


def etapa_conferencia() -> str:
    cont = {t: _pg_um(f"SELECT COUNT(*) FROM {t}")
            for t in ("emendas", "parlamentares", "documentos_emenda", "favorecidos",
                      "beneficiarios", "documento_favorecido")}
    pend = _pg_um("SELECT COUNT(*) FROM documentos_emenda WHERE COALESCE(detalhes_coletados, 0) = 0")
    baixo = {t: (cont[t], m) for t, m in MINIMOS.items() if cont[t] < m}
    if baixo or pend > MAX_PENDENTES_DETALHES * max(cont["documentos_emenda"], 1):
        raise EtapaFalhou(f"abaixo do mínimo {baixo}; detalhes pendentes {pend:,}")
    return json.dumps({**cont, "detalhes_pendentes": pend}, ensure_ascii=False)


def _aguardar_litellm() -> None:
    import httpx
    cfg = dotenv_values(BACKEND.parent / ".env")
    url = (cfg.get("LITELLM_BASE_URL") or "").rstrip("/") + "/v1/models"
    cab = {"Authorization": f"Bearer {cfg.get('LITELLM_API_KEY', '')}"}
    avisou = False
    while True:
        try:
            if httpx.get(url, headers=cab, timeout=10).status_code < 500:
                if avisou:
                    log("  LiteLLM respondeu de novo")
                return
        except httpx.HTTPError:
            pass
        if not avisou:
            log("  LiteLLM inacessível: aguardando a VPN da TechBiz (sondagem a cada 60 s)")
            _marcar("com_rag", "aguardando_vpn", "LiteLLM inacessível; ligue a VPN da TechBiz")
            avisou = True
        time.sleep(SONDA_LITELLM_S)


def _com_rag_ok() -> int:
    ckpt = DIR_PARCIAL / "com_rag.jsonl"
    if not ckpt.exists():
        return 0
    ok = {}
    for linha in ckpt.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            reg = json.loads(linha)
            ok[reg["id"]] = ok.get(reg["id"], False) or bool(reg.get("sucesso"))
    return sum(ok.values())


def etapa_com_rag() -> str:
    for tentativa in (1, 2, 3):
        if _com_rag_ok() >= 120:
            break
        _aguardar_litellm()
        _marcar("com_rag", "rodando", f"tentativa {tentativa}")
        _rodar("11_com_rag", ["scripts/evaluate_120_com_sem_rag.py", "--modelo", MODELO,
                              "--fase", "com_rag"], "pg")
    n = _com_rag_ok()
    if n < 120:
        raise EtapaFalhou(f"COM RAG com {n}/120 consultas bem-sucedidas")
    return "120/120 consultas COM RAG"


def etapa_pontuar() -> str:
    antes = set(SAIDA.glob(f"*_{MODELO}_resultados_120_com_rag_hibrido.json"))
    rc = _rodar("12_pontuar", ["scripts/evaluate_120_com_sem_rag.py", "--modelo", MODELO,
                               "--fase", "pontuar", "--manter-parciais"], "pg")
    novos = set(SAIDA.glob(f"*_{MODELO}_resultados_120_com_rag_hibrido.json")) - antes
    if rc != 0 or not novos:
        raise EtapaFalhou(f"pontuar saiu com {rc}; arquivos novos: {len(novos)}")
    return f"resultados em {sorted(novos)[-1].name}"


ETAPAS = [
    ("emendas", etapa_emendas),
    ("documentos", etapa_documentos),
    ("detalhes", etapa_detalhes),
    ("repescagem", etapa_repescagem),
    ("sancoes", etapa_sancoes),
    ("beneficiarios_ac", etapa_beneficiarios_ac),
    ("sync_pg", etapa_sync_pg),
    ("embeddings", etapa_embeddings),
    ("indices", etapa_indices),
    ("conferencia", etapa_conferencia),
    ("com_rag", etapa_com_rag),
    ("pontuar", etapa_pontuar),
]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def mostrar_status() -> None:
    est = _estado()
    pid = est.get("pid")
    vivo = bool(pid) and _pid_vivo(pid)
    print(f"pipeline: pid={pid} {'VIVO' if vivo else 'parado'} | etapa atual: "
          f"{est.get('etapa_atual')} | atualizado {est.get('atualizado_em')}")
    for nome, _ in ETAPAS:
        reg = est.get("etapas", {}).get(nome, {})
        print(f"  {nome:<17} {reg.get('status', 'pendente'):<15} "
              f"{reg.get('detalhe', '')[:110]}")
    if SQLITE.exists():
        tabelas = ("emendas", "documentos_emenda", "favorecidos", "documento_favorecido",
                   "beneficiarios", "parlamentares")
        print("SQLite:", {t: _sqlite(f"SELECT COUNT(*) FROM {t}")[0][0] for t in tabelas})
        print("detalhes pendentes por UF:", _pendentes_detalhes_por_uf())
    cp = _ler_json(CP_BRASIL)
    chaves = _chaves_ativas()
    rotulo = f"27 UFs + {SEM_UF}" if INCLUIR_SEM_UF else f"27 UFs; {SEM_UF} pulado"
    for fase in ("documentos", "detalhes"):
        reg_fase = cp.get(fase, {})
        feitas = [u for u in chaves if reg_fase.get(u, {}).get("status") == "completed"]
        em_curso = {u: {k: v.get(k) for k in ("last_emenda_id", "last_index", "coletados",
                                               "total_docs", "erros") if k in v}
                    for u, v in reg_fase.items()
                    if u in chaves and v.get("status") != "completed"}
        print(f"{fase}: {len(feitas)}/{len(chaves)} completas ({rotulo}); "
              f"em curso: {em_curso}")
    print("DLQ:", len(_ler_json(DLQ_BRASIL, padrao=[])), "itens")
    for nome in sorted(LOGS.glob("*.log")):
        linhas = nome.read_text(encoding="utf-8", errors="replace").splitlines()[-2:]
        for linha in linhas:
            print(f"  [{nome.name}] {linha[:160]}")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="Mostra o estado e sai")
    parser.add_argument("--ate", choices=[n for n, _ in ETAPAS],
                        help="Para depois desta etapa")
    args = parser.parse_args()
    _fixar_tmp()
    if args.status:
        mostrar_status()
        return

    _adquirir_trava()
    log(f"=== pipeline iniciado (pid {os.getpid()}) ===")
    try:
        for nome, func in ETAPAS:
            if _estado()["etapas"].get(nome, {}).get("status") == "ok":
                continue
            _marcar(nome, "rodando")
            log(f"--- {nome} ---")
            try:
                detalhe = func()
            except EtapaFalhou as e:
                _marcar(nome, "falhou", str(e))
                log(f"FALHOU {nome}: {e}")
                sys.exit(2)
            except Exception as e:  # erro inesperado do orquestrador
                _marcar(nome, "falhou", f"{type(e).__name__}: {e}")
                log(f"FALHOU {nome}: {type(e).__name__}: {e}")
                raise
            _marcar(nome, "ok", detalhe)
            log(f"ok {nome}: {detalhe}")
            if nome == args.ate:
                break
        log("=== pipeline concluído ===")
    finally:
        if TRAVA.exists() and TRAVA.read_text().strip() == str(os.getpid()):
            TRAVA.unlink()


if __name__ == "__main__":
    main()
