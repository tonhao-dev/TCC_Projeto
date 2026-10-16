"""Ingestão nacional de emendas parlamentares — Brasil completo (2020-2024).

Fases sequenciais:
  1. Emendas (todas as UFs, todas de 2020-2024)
  2. Documentos (coleta documentos de cada emenda)
  3. Detalhes (coleta detalhes+favorecidos de cada documento)

Uso:
    python scripts/run_ingestion_brasil.py
    python scripts/run_ingestion_brasil.py --resume
    python scripts/run_ingestion_brasil.py --fase emendas
    python scripts/run_ingestion_brasil.py --fase documentos [--uf SP]
    python scripts/run_ingestion_brasil.py --fase detalhes [--uf SP]
    python scripts/run_ingestion_brasil.py --retry-dlq
    python scripts/run_ingestion_brasil.py --stats
"""
import asyncio
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database import SessionLocal, init_db, engine
import app.models  # noqa: F401
from app.services.ingestion.cgu_collector import CGUCollector
from app.services.ingestion.normalizer import DataNormalizer
from app.models.documento import DocumentoEmenda
from app.utils.checkpoint import CheckpointManager
from app.utils.circuit_breaker import CircuitBreaker, DeadLetterQueue
import structlog

logger = structlog.get_logger()

ANOS = [2020, 2021, 2022, 2023, 2024]
TODAS_UFS = [
    # Prioridade: Sul, Sudeste, Centro-Oeste
    "RS", "PR", "SC",           # Sul
    "SP", "RJ", "MG", "ES",    # Sudeste
    "GO", "MS", "MT", "DF",    # Centro-Oeste
    # Depois: Norte e Nordeste
    "AC", "AL", "AM", "AP", "BA", "CE", "MA",
    "PA", "PB", "PE", "PI", "RN", "RO", "RR", "SE", "TO",
]
CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "checkpoint_brasil.json"
DLQ_PATH = Path(__file__).parent.parent / "data" / "dead_letter_brasil.json"
MAX_RETRIES = 3
ESPERA_DISJUNTOR_S = 5  # sondagem enquanto o disjuntor está aberto
CONCORRENCIA_DOCUMENTOS = 8  # emendas buscadas em paralelo na fase de documentos
# Emendas de localidade MÚLTIPLO, Nacional, Exterior ou regional ficam com uf vazio
# (46% da base). Sem esta chave as fases por UF nunca coletavam os documentos delas.
SEM_UF = "SEM_UF"


def _filtro_uf(uf: str) -> tuple[str, dict]:
    """Filtro SQL da UF e seus parâmetros; SEM_UF pega uf vazio ou nulo."""
    if uf == SEM_UF:
        return "COALESCE(e.uf, '') = ''", {}
    return "e.uf = :uf", {"uf": uf}


def _ufs_da_passada(uf_filter: str | None, pular_sem_uf: bool = False) -> list[str]:
    """Chaves de uma passada: só a UF pedida, ou a passada nacional.

    A nacional pula o AC (coletado à parte) e termina nas emendas sem UF, a menos
    que --pular-sem-uf as deixe de fora (a base do artigo não tem documentos delas).
    """
    if uf_filter:
        return [uf_filter]
    return [uf for uf in TODAS_UFS if uf != "AC"] + ([] if pular_sem_uf else [SEM_UF])


def _parse_date(date_str):
    if not date_str:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _extrair_codigo(valor_str: str) -> str:
    """Extrai código numérico de strings como '10 - Saúde' ou '301 - Atenção Básica'."""
    if not valor_str:
        return ""
    valor_str = valor_str.strip()
    if " - " in valor_str:
        return valor_str.split(" - ", 1)[0].strip()
    return valor_str


def _extrair_nome(valor_str: str) -> str:
    """Extrai nome de strings como '10 - Saúde'."""
    if not valor_str:
        return ""
    if " - " in valor_str:
        return valor_str.split(" - ", 1)[1].strip()
    return valor_str.strip()


def stats():
    """Mostra contagens atuais do banco."""
    with engine.connect() as c:
        for t in ["parlamentares", "emendas", "documentos_emenda",
                   "favorecidos", "documento_favorecido",
                   "funcoes", "subfuncoes", "orgaos",
                   "sancoes_ceis", "sancoes_cnep", "sancoes_cepim", "sancoes_ceaf",
                   "convenios"]:
            try:
                count = c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                logger.info(f"  {t}: {count:,}")
            except Exception:
                logger.info(f"  {t}: (tabela não existe)")

    # Contagem por UF
    with engine.connect() as c:
        try:
            result = c.execute(text(
                "SELECT uf, COUNT(*) as cnt FROM emendas GROUP BY uf ORDER BY cnt DESC"
            ))
            logger.info("=== EMENDAS POR UF ===")
            for row in result.fetchall():
                logger.info(f"  {row[0]}: {row[1]:,}")
        except Exception:
            pass


def show_stats_only():
    """Exibe estatísticas sem executar coleta."""
    init_db()
    stats()

    cp = CheckpointManager(path=CHECKPOINT_PATH)
    logger.info("=== CHECKPOINT ===")
    logger.info(f"  resumo: {cp.summary()}")

    dlq = DeadLetterQueue(DLQ_PATH)
    if dlq.count:
        logger.info(f"=== DLQ: {dlq.count} itens ===")
        logger.info(f"  por fase: {dlq.summary()}")


# ---------------------------------------------------------------------------
# Fase 1: Emendas (todas as UFs, 2020-2024)
# ---------------------------------------------------------------------------
async def fase_emendas(cp: CheckpointManager, dlq: DeadLetterQueue):
    collector = CGUCollector()

    for ano in ANOS:
        key = str(ano)
        if cp.get_status("emendas", key) == "completed":
            logger.info("emendas_skip", ano=ano)
            continue

        # Retoma NA página registrada, não na seguinte: o checkpoint é gravado no
        # meio da página e o resto dela se perderia. As repetidas são puladas abaixo.
        start_page = max(1, cp.get_last_page("emendas", key))
        cp.mark_started("emendas", key)
        logger.info("coletando_emendas_brasil", ano=ano, start_page=start_page)

        init_db()
        db = SessionLocal()
        normalizer = DataNormalizer(db)
        count = 0
        skip_existing = 0

        try:
            async for emenda_data in collector.coletar_emendas(ano, start_page=start_page):
                # Verificar se já existe (AC ou duplicata)
                codigo = emenda_data.get("codigo_emenda")
                if codigo:
                    existe = db.execute(
                        text("SELECT id FROM emendas WHERE codigo_emenda = :cod"),
                        {"cod": codigo}
                    ).fetchone()
                    if existe:
                        skip_existing += 1
                        continue

                # Popular lookup tables
                _popular_lookups_emenda(db, emenda_data)

                # Vincular autor e inserir
                emenda_data = normalizer.vincular_autor(emenda_data)
                normalizer.inserir_emenda(emenda_data)
                count += 1

                if count % 100 == 0:
                    normalizer.commit()
                    cp.update_progress("emendas", key,
                                       last_page=collector.current_page,
                                       total=count, skip=skip_existing)
                    logger.info("emendas_batch", ano=ano, novas=count,
                                skip=skip_existing, page=collector.current_page)

            normalizer.commit()
        finally:
            db.close()

        cp.mark_completed("emendas", key, total=count)
        logger.info("emendas_completas", ano=ano, novas=count, skip=skip_existing)


def _popular_lookups_emenda(db, emenda_data: dict):
    """Popula tabelas lookup a partir dos dados de uma emenda."""
    funcao_cod = emenda_data.get("funcao", "")
    funcao_nome = emenda_data.get("funcao_nome", "")
    subfuncao_cod = emenda_data.get("subfuncao", "")
    subfuncao_nome = emenda_data.get("subfuncao_nome", "")

    if funcao_cod and funcao_nome:
        db.execute(text(
            "INSERT OR IGNORE INTO funcoes (codigo, nome) VALUES (:cod, :nome)"
        ), {"cod": funcao_cod, "nome": funcao_nome})

    if subfuncao_cod and subfuncao_nome:
        db.execute(text(
            "INSERT OR IGNORE INTO subfuncoes (codigo, nome, funcao_codigo) "
            "VALUES (:cod, :nome, :func)"
        ), {"cod": subfuncao_cod, "nome": subfuncao_nome, "func": funcao_cod})


# ---------------------------------------------------------------------------
# Fase 2: Documentos (todas as UFs)
# ---------------------------------------------------------------------------
async def fase_documentos(cp: CheckpointManager, dlq: DeadLetterQueue,
                          uf_filter: str | None = None, pular_sem_uf: bool = False):
    cb = CircuitBreaker(failure_threshold=10, recovery_timeout=300)
    collector = CGUCollector()

    ufs = _ufs_da_passada(uf_filter, pular_sem_uf)

    for uf in ufs:
        key = uf
        if cp.get_status("documentos", key) == "completed":
            logger.info("documentos_skip", uf=uf)
            continue

        init_db()
        db = SessionLocal()
        # Retoma pelo id da última emenda processada. A lista abaixo é recalculada
        # a cada execução e encolhe, então um índice posicional pularia emendas.
        last_emenda_id = cp.get_value("documentos", key, "last_emenda_id", 0)
        cp.mark_started("documentos", key)

        try:
            filtro, params_uf = _filtro_uf(uf)
            result = db.execute(text(f"""
                SELECT e.id, e.codigo_emenda
                FROM emendas e
                LEFT JOIN documentos_emenda d ON d.emenda_id = e.id
                WHERE d.id IS NULL
                  AND e.codigo_emenda IS NOT NULL
                  AND {filtro}
                  AND e.id > :last_id
                ORDER BY e.id
            """), {**params_uf, "last_id": last_emenda_id})
            emendas = [(r.id, r.codigo_emenda) for r in result.fetchall()]

            total_emendas = len(emendas)
            logger.info("documentos_inicio", uf=uf, total_emendas=total_emendas,
                        resumindo_apos_emenda_id=last_emenda_id,
                        concorrencia=CONCORRENCIA_DOCUMENTOS)

            total_docs = 0
            erros = 0
            t0 = time.monotonic()
            # Várias emendas em voo ao mesmo tempo (o limitador por chave segura o
            # ritmo); a gravação no SQLite fica serializada. O checkpoint só avança
            # até a menor emenda ainda não concluída, para a retomada não pular nada.
            fila = asyncio.Queue()
            for item in emendas:
                fila.put_nowait(item)
            gravacao = asyncio.Lock()
            concluidas: set[int] = set()
            ordem = [eid for eid, _ in emendas]
            proxima = 0  # índice em `ordem` da menor emenda não concluída
            processadas = 0

            def _gravar_documentos(emenda_id: int, docs: list[dict]) -> int:
                novos = 0
                for doc_data in docs:
                    existe = db.execute(
                        text("SELECT id FROM documentos_emenda WHERE codigo_documento = :cod"),
                        {"cod": doc_data["codigo_documento"]}
                    ).fetchone()
                    if not existe:
                        db.add(DocumentoEmenda(
                            emenda_id=emenda_id,
                            codigo_documento=doc_data["codigo_documento"],
                            fase=doc_data.get("fase"),
                            valor=doc_data.get("valor", 0),
                            data_documento=_parse_date(doc_data.get("data_documento")),
                            tipo_documento=doc_data.get("tipo_documento"),
                            observacao=doc_data.get("observacao"),
                        ))
                        db.flush()
                        novos += 1
                db.commit()
                return novos

            async def trabalhador():
                nonlocal total_docs, erros, proxima, processadas
                while True:
                    try:
                        emenda_id, codigo_emenda = fila.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    # Disjuntor aberto: espera a API voltar em vez de mandar a emenda
                    # para a DLQ (numa coleta longa isso esvaziaria a UF inteira).
                    while not cb.can_proceed():
                        await asyncio.sleep(ESPERA_DISJUNTOR_S)

                    for tentativa in range(1, MAX_RETRIES + 1):
                        try:
                            docs = await collector.coletar_documentos_emenda(codigo_emenda)
                            cb.record_success()
                            async with gravacao:
                                try:
                                    total_docs += _gravar_documentos(emenda_id, docs)
                                except Exception:
                                    db.rollback()
                                    raise
                            break
                        except Exception as e:
                            cb.record_failure()
                            if tentativa < MAX_RETRIES:
                                wait = tentativa * 10
                                logger.warning("retry_doc", uf=uf, codigo=codigo_emenda,
                                               tentativa=tentativa, wait=wait,
                                               erro=str(e)[:80])
                                await asyncio.sleep(wait)
                            else:
                                erros += 1
                                dlq.add("documentos", codigo_emenda, str(e)[:200])
                                logger.error("falha_doc", uf=uf, codigo=codigo_emenda,
                                             erro=str(e)[:120])

                    concluidas.add(emenda_id)
                    processadas += 1
                    while proxima < len(ordem) and ordem[proxima] in concluidas:
                        proxima += 1
                    if processadas % 10 == 0 or processadas == total_emendas:
                        elapsed = time.monotonic() - t0
                        rate = processadas / elapsed if elapsed > 0 else 0
                        eta_min = (total_emendas - processadas) / rate / 60 if rate > 0 else 0
                        pct = processadas / total_emendas * 100 if total_emendas else 0
                        logger.info("progresso_documentos", uf=uf,
                                    pos=processadas, total=total_emendas,
                                    pct=f"{pct:.1f}%", docs=total_docs, erros=erros,
                                    eta_min=f"{eta_min:.1f}")
                        marca = ordem[proxima - 1] if proxima else last_emenda_id
                        cp.update_progress("documentos", key, last_emenda_id=marca,
                                           total_docs=total_docs, erros=erros)

            await asyncio.gather(*(trabalhador() for _ in
                                   range(min(CONCORRENCIA_DOCUMENTOS, max(total_emendas, 1)))))

            cp.mark_completed("documentos", key, total=total_docs)
            logger.info("documentos_completos", uf=uf, docs=total_docs, erros=erros)

        finally:
            db.close()


# ---------------------------------------------------------------------------
# Fase 3: Detalhes dos documentos + favorecidos normalizados
# ---------------------------------------------------------------------------
async def _fetch_detalhe(collector, codigo_documento, cb, semaphore):
    """Fetch a single document detail with concurrency control."""
    async with semaphore:
        # Disjuntor aberto: espera em vez de devolver "circuit_breaker_open", que
        # mandava para a DLQ, em segundos, todos os documentos pendentes da UF.
        while not cb.can_proceed():
            await asyncio.sleep(ESPERA_DISJUNTOR_S)
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                detalhe = await collector.coletar_detalhes_documento(codigo_documento)
                cb.record_success()
                return codigo_documento, detalhe, None
            except Exception as e:
                cb.record_failure()
                if tentativa < MAX_RETRIES:
                    await asyncio.sleep(tentativa * 5)
                else:
                    return codigo_documento, None, str(e)[:200]


def _get_concurrency():
    """Dynamic concurrency: higher at night (less WAF risk), lower during day."""
    hour = datetime.now().hour
    if 0 <= hour < 6:
        return 20  # Madrugada: mais agressivo
    elif 6 <= hour < 8 or 22 <= hour < 24:
        return 15  # Transição
    else:
        return 10  # Dia: conservador


async def fase_detalhes(cp: CheckpointManager, dlq: DeadLetterQueue,
                        uf_filter: str | None = None, pular_sem_uf: bool = False):
    cb = CircuitBreaker(failure_threshold=10, recovery_timeout=300)
    collector = CGUCollector()
    n_concurrent = _get_concurrency()
    logger.info("detalhes_pipeline", concurrency=n_concurrent,
                hour=datetime.now().hour)

    ufs = _ufs_da_passada(uf_filter, pular_sem_uf)

    for uf in ufs:
        key = uf
        if cp.get_status("detalhes", key) == "completed":
            logger.info("detalhes_skip", uf=uf)
            continue

        init_db()
        db = SessionLocal()
        # A consulta abaixo já exclui os documentos com detalhes coletados, então a
        # lista recomeça do início a cada retomada. O last_index do checkpoint é só
        # progresso: pular por ele descartaria documentos ainda pendentes.
        last_index = 0
        cp.mark_started("detalhes", key)

        try:
            filtro, params_uf = _filtro_uf(uf)
            result = db.execute(text(f"""
                SELECT d.id, d.codigo_documento
                FROM documentos_emenda d
                JOIN emendas e ON e.id = d.emenda_id
                WHERE {filtro}
                  AND (d.detalhes_coletados IS NULL OR d.detalhes_coletados = 0)
                ORDER BY d.id
            """), params_uf)
            documentos = [(r.id, r.codigo_documento) for r in result.fetchall()]

            total = len(documentos)
            n_concurrent = _get_concurrency()  # Recalculate per UF
            collector.update_rpm_for_current_hour()  # Recalculate RPM cap per UF
            logger.info("detalhes_inicio", uf=uf, total=total,
                        resumindo_de=last_index, concurrency=n_concurrent)

            coletados = 0
            erros = 0
            t0 = time.monotonic()
            semaphore = asyncio.Semaphore(n_concurrent)

            # Pipeline: use asyncio.Queue so fetching never stops while DB writes
            queue = asyncio.Queue(maxsize=n_concurrent * 4)
            writer_done = asyncio.Event()
            pending = [(i, doc_id, codigo) for i, (doc_id, codigo) in enumerate(documentos)
                       if i >= last_index]

            async def producer():
                """Continuously fetch details and put results in queue."""
                tasks = set()
                idx = 0
                while idx < len(pending) or tasks:
                    # Launch new tasks up to concurrency limit
                    while idx < len(pending) and len(tasks) < n_concurrent:
                        orig_i, doc_id, codigo = pending[idx]
                        task = asyncio.create_task(
                            _fetch_detalhe(collector, codigo, cb, semaphore)
                        )
                        task._doc_info = (orig_i, doc_id, codigo)
                        tasks.add(task)
                        idx += 1

                    if not tasks:
                        break

                    # Wait for ANY task to complete (not all)
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        orig_i, doc_id, codigo = task._doc_info
                        result = task.result()
                        await queue.put((orig_i, doc_id, codigo, result))

                await queue.put(None)  # Signal end

            async def consumer():
                """Write results to DB as they arrive."""
                nonlocal coletados, erros
                commit_count = 0
                while True:
                    item = await queue.get()
                    if item is None:
                        if commit_count > 0:
                            db.commit()
                        break

                    orig_i, doc_id, codigo, (_, detalhe, error) = item

                    if error:
                        if error != "circuit_breaker_open":
                            erros += 1
                            logger.error("falha_detalhe", uf=uf,
                                         codigo=codigo, erro=error[:120])
                        dlq.add("detalhes", codigo, error)
                    elif detalhe:
                        _atualizar_documento_detalhes(db, doc_id, detalhe)
                        _popular_lookups_detalhe(db, detalhe)
                        _popular_favorecido(db, doc_id, detalhe)
                        coletados += 1
                    else:
                        db.execute(text(
                            "UPDATE documentos_emenda SET detalhes_coletados = 1 "
                            "WHERE id = :id"
                        ), {"id": doc_id})

                    commit_count += 1
                    # Batch commit every 50 items
                    if commit_count >= 50:
                        db.commit()
                        commit_count = 0

                    # Progress reporting
                    pos = orig_i + 1
                    if pos % 200 == 0 or pos >= total:
                        elapsed = time.monotonic() - t0
                        start_pos = pending[0][0] if pending else 0
                        processed = pos - start_pos
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = total - pos
                        eta_min = remaining / rate / 60 if rate > 0 else 0
                        pct = pos / total * 100 if total else 0
                        logger.info("progresso_detalhes", uf=uf,
                                    pos=pos, total=total,
                                    pct=f"{pct:.1f}%", coletados=coletados,
                                    erros=erros, eta_min=f"{eta_min:.1f}")
                        cp.update_progress("detalhes", key,
                                           last_index=pos, coletados=coletados,
                                           erros=erros)

            # Run producer and consumer concurrently
            await asyncio.gather(producer(), consumer())

            # Reconciliar valores das emendas desta UF a partir dos documentos
            normalizer = DataNormalizer(db)
            is_sqlite = "sqlite" in str(db.bind.url)
            if is_sqlite:
                normalizer.recalcular_valores_de_documentos_sqlite()
            else:
                normalizer.recalcular_valores_de_documentos()

            cp.mark_completed("detalhes", key, total=coletados)
            logger.info("detalhes_completos", uf=uf, coletados=coletados, erros=erros)

        finally:
            db.close()

    await collector.close_shared_client()


def _parse_valor_detalhe(valor_str) -> float:
    """Converte valor do endpoint de detalhes (formato BR) para float."""
    if not valor_str:
        return 0.0
    if isinstance(valor_str, (int, float)):
        return float(valor_str)
    cleaned = str(valor_str).strip()
    if cleaned in ("-", "", "N/A", "null"):
        return 0.0
    try:
        return float(cleaned.replace(".", "").replace(",", "."))
    except ValueError:
        return 0.0


def _atualizar_documento_detalhes(db, doc_id: int, detalhe: dict):
    """Atualiza documento com informações detalhadas."""
    db.execute(text("""
        UPDATE documentos_emenda SET
            funcao = :funcao,
            subfuncao = :subfuncao,
            programa = :programa,
            acao = :acao,
            localizador_gasto = :localizador,
            especie = :especie,
            categoria = :categoria,
            grupo_despesa = :grupo,
            elemento_despesa = :elemento,
            modalidade = :modalidade,
            orgao = :orgao,
            orgao_superior = :orgao_sup,
            unidade_gestora = :ug,
            unidade_orcamentaria = :uo,
            favorecido_nome = :fav_nome,
            favorecido_cpf_cnpj = :fav_cpf,
            favorecido_uf = :fav_uf,
            numero_processo = :processo,
            plano_orcamentario = :plano,
            valor = :valor,
            observacao = CASE WHEN :observacao != '' THEN :observacao
                             ELSE COALESCE(observacao, '') END,
            detalhes_coletados = 1
        WHERE id = :id
    """), {
        "funcao": detalhe.get("funcao", ""),
        "subfuncao": detalhe.get("subfuncao", ""),
        "programa": detalhe.get("programa", ""),
        "acao": detalhe.get("acao", ""),
        "localizador": detalhe.get("localizadorGasto", ""),
        "especie": detalhe.get("especie", ""),
        "categoria": detalhe.get("categoria", ""),
        "grupo": detalhe.get("grupo", ""),
        "elemento": detalhe.get("elemento", ""),
        "modalidade": detalhe.get("modalidade", ""),
        "orgao": detalhe.get("orgao", ""),
        "orgao_sup": detalhe.get("orgaoSuperior", ""),
        "ug": detalhe.get("ug", ""),
        "uo": detalhe.get("uo", ""),
        "fav_nome": detalhe.get("nomeFavorecido", ""),
        "fav_cpf": detalhe.get("codigoFavorecido", ""),
        "fav_uf": detalhe.get("ufFavorecido", ""),
        "processo": detalhe.get("numeroProcesso", ""),
        "plano": detalhe.get("planoOrcamentario", ""),
        "valor": _parse_valor_detalhe(detalhe.get("valor", "0")),
        "observacao": detalhe.get("observacao", ""),
        "id": doc_id,
    })


def _popular_lookups_detalhe(db, detalhe: dict):
    """Popula tabelas lookup a partir dos detalhes de um documento."""
    # Funções e subfunções (do documento — podem ter código+nome)
    funcao = detalhe.get("funcao", "")
    subfuncao = detalhe.get("subfuncao", "")
    programa = detalhe.get("programa", "")
    acao = detalhe.get("acao", "")

    cod_f = _extrair_codigo(funcao)
    nome_f = _extrair_nome(funcao)
    if cod_f and nome_f:
        db.execute(text(
            "INSERT OR IGNORE INTO funcoes (codigo, nome) VALUES (:c, :n)"
        ), {"c": cod_f, "n": nome_f})

    cod_sf = _extrair_codigo(subfuncao)
    nome_sf = _extrair_nome(subfuncao)
    if cod_sf and nome_sf:
        db.execute(text(
            "INSERT OR IGNORE INTO subfuncoes (codigo, nome, funcao_codigo) "
            "VALUES (:c, :n, :f)"
        ), {"c": cod_sf, "n": nome_sf, "f": cod_f})

    cod_p = _extrair_codigo(programa)
    nome_p = _extrair_nome(programa)
    if cod_p:
        db.execute(text(
            "INSERT OR IGNORE INTO programas (codigo, nome) VALUES (:c, :n)"
        ), {"c": cod_p, "n": nome_p or cod_p})

    cod_a = _extrair_codigo(acao)
    nome_a = _extrair_nome(acao)
    if cod_a:
        db.execute(text(
            "INSERT OR IGNORE INTO acoes (codigo, nome) VALUES (:c, :n)"
        ), {"c": cod_a, "n": nome_a or cod_a})

    # Órgãos
    orgao = detalhe.get("orgao", "")
    orgao_sup = detalhe.get("orgaoSuperior", "")
    cod_org = _extrair_codigo(orgao)
    nome_org = _extrair_nome(orgao)
    cod_org_sup = _extrair_codigo(orgao_sup)
    nome_org_sup = _extrair_nome(orgao_sup)

    if cod_org_sup and nome_org_sup:
        db.execute(text(
            "INSERT OR IGNORE INTO orgaos (codigo, nome) VALUES (:c, :n)"
        ), {"c": cod_org_sup, "n": nome_org_sup})

    if cod_org and nome_org:
        db.execute(text(
            "INSERT OR IGNORE INTO orgaos (codigo, nome, orgao_superior_codigo) "
            "VALUES (:c, :n, :s)"
        ), {"c": cod_org, "n": nome_org, "s": cod_org_sup})

    # Unidades gestoras
    ug = detalhe.get("ug", "")
    cod_ug = _extrair_codigo(ug)
    nome_ug = _extrair_nome(ug)
    if cod_ug:
        db.execute(text(
            "INSERT OR IGNORE INTO unidades_gestoras (codigo, nome, orgao_codigo) "
            "VALUES (:c, :n, :o)"
        ), {"c": cod_ug, "n": nome_ug or cod_ug, "o": cod_org})

    # Classificação da despesa
    categoria = detalhe.get("categoria", "")
    grupo = detalhe.get("grupo", "")
    elemento = detalhe.get("elemento", "")
    modalidade = detalhe.get("modalidade", "")

    for valor, tabela in [(categoria, "categorias_despesa"),
                          (grupo, "grupos_despesa"),
                          (elemento, "elementos_despesa"),
                          (modalidade, "modalidades_aplicacao")]:
        cod = _extrair_codigo(valor)
        nome = _extrair_nome(valor)
        if cod:
            db.execute(text(
                f"INSERT OR IGNORE INTO {tabela} (codigo, nome) VALUES (:c, :n)"
            ), {"c": cod, "n": nome or cod})


def _popular_favorecido(db, doc_id: int, detalhe: dict):
    """Popula tabela favorecidos normalizada a partir dos detalhes do documento."""
    cpf_cnpj = (detalhe.get("codigoFavorecido") or "").strip()
    if not cpf_cnpj:
        return

    nome = (detalhe.get("nomeFavorecido") or "").strip()
    uf = (detalhe.get("ufFavorecido") or "").strip()
    limpo = cpf_cnpj.replace(".", "").replace("-", "").replace("/", "")
    tipo = "PJ" if len(limpo) > 11 else "PF"

    # Upsert favorecido
    existe = db.execute(
        text("SELECT id FROM favorecidos WHERE cpf_cnpj = :cpf"),
        {"cpf": cpf_cnpj}
    ).fetchone()

    if existe:
        fav_id = existe[0]
        # Atualizar nome se estava vazio
        if nome:
            db.execute(text(
                "UPDATE favorecidos SET nome = :nome, tipo_pessoa = :tipo, uf = :uf "
                "WHERE id = :id AND (nome IS NULL OR nome = '')"
            ), {"nome": nome, "tipo": tipo, "uf": uf, "id": fav_id})
    else:
        db.execute(text(
            "INSERT INTO favorecidos (cpf_cnpj, nome, tipo_pessoa, uf) "
            "VALUES (:cpf, :nome, :tipo, :uf)"
        ), {"cpf": cpf_cnpj, "nome": nome, "tipo": tipo, "uf": uf})
        fav_id = db.execute(
            text("SELECT id FROM favorecidos WHERE cpf_cnpj = :cpf"),
            {"cpf": cpf_cnpj}
        ).fetchone()[0]

    # Vincular documento ao favorecido
    ja_vinculado = db.execute(
        text("SELECT id FROM documento_favorecido "
             "WHERE documento_id = :did AND favorecido_id = :fid"),
        {"did": doc_id, "fid": fav_id}
    ).fetchone()

    if not ja_vinculado:
        db.execute(text(
            "INSERT INTO documento_favorecido (documento_id, favorecido_id) "
            "VALUES (:did, :fid)"
        ), {"did": doc_id, "fid": fav_id})


# ---------------------------------------------------------------------------
# Retry DLQ
# ---------------------------------------------------------------------------
async def retry_dead_letter_queue():
    """Reprocessa itens da Dead Letter Queue."""
    dlq = DeadLetterQueue(DLQ_PATH)
    if not dlq.count:
        logger.info("dlq_vazia")
        return

    logger.info("dlq_retry", total=dlq.count, por_fase=dlq.summary())
    collector = CGUCollector()
    init_db()
    db = SessionLocal()

    # Retry documentos
    for item in dlq.get_pending("documentos"):
        codigo = item["identificador"]
        try:
            docs = await collector.coletar_documentos_emenda(codigo)
            # Buscar emenda_id
            row = db.execute(
                text("SELECT id FROM emendas WHERE codigo_emenda = :cod"),
                {"cod": codigo}
            ).fetchone()
            if row:
                for doc_data in docs:
                    existe = db.execute(
                        text("SELECT id FROM documentos_emenda WHERE codigo_documento = :cod"),
                        {"cod": doc_data["codigo_documento"]}
                    ).fetchone()
                    if not existe:
                        doc = DocumentoEmenda(
                            emenda_id=row[0],
                            codigo_documento=doc_data["codigo_documento"],
                            fase=doc_data.get("fase"),
                            valor=doc_data.get("valor", 0),
                            data_documento=_parse_date(doc_data.get("data_documento")),
                            tipo_documento=doc_data.get("tipo_documento"),
                            observacao=doc_data.get("observacao"),
                        )
                        db.add(doc)
                db.commit()
            dlq.mark_resolved("documentos", codigo)
            logger.info("dlq_resolved", fase="documentos", codigo=codigo)
        except Exception as e:
            db.rollback()
            dlq.add("documentos", codigo, str(e)[:200])
            logger.warning("dlq_retry_falha", fase="documentos", codigo=codigo,
                           erro=str(e)[:80])

    # Retry detalhes
    for item in dlq.get_pending("detalhes"):
        codigo = item["identificador"]
        try:
            detalhe = await collector.coletar_detalhes_documento(codigo)
            row = db.execute(
                text("SELECT id FROM documentos_emenda WHERE codigo_documento = :cod"),
                {"cod": codigo}
            ).fetchone()
            if row and detalhe:
                _atualizar_documento_detalhes(db, row[0], detalhe)
                _popular_lookups_detalhe(db, detalhe)
                _popular_favorecido(db, row[0], detalhe)
            elif row:
                db.execute(text(
                    "UPDATE documentos_emenda SET detalhes_coletados = 1 WHERE id = :id"
                ), {"id": row[0]})
            db.commit()
            dlq.mark_resolved("detalhes", codigo)
            logger.info("dlq_resolved", fase="detalhes", codigo=codigo)
        except Exception as e:
            db.rollback()
            dlq.add("detalhes", codigo, str(e)[:200])
            logger.warning("dlq_retry_falha", fase="detalhes", codigo=codigo,
                           erro=str(e)[:80])

    db.close()
    logger.info("dlq_retry_completo", restante=dlq.count)


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------
async def main(resume: bool = False, etapa: str | None = None,
               uf_filter: str | None = None, retry_dlq: bool = False,
               pular_sem_uf: bool = False):
    cp = CheckpointManager(path=CHECKPOINT_PATH)
    dlq = DeadLetterQueue(DLQ_PATH)

    if not resume and not retry_dlq:
        cp.reset()

    if retry_dlq:
        await retry_dead_letter_queue()
        return

    logger.info("=" * 60)
    logger.info("INGESTÃO BRASIL — INÍCIO", resume=resume,
                etapa_unica=etapa or "todas", uf_filter=uf_filter or "todas")
    logger.info("=" * 60)

    init_db()

    etapas = {
        "emendas": lambda: fase_emendas(cp, dlq),
        "documentos": lambda: fase_documentos(cp, dlq, uf_filter, pular_sem_uf),
        "detalhes": lambda: fase_detalhes(cp, dlq, uf_filter, pular_sem_uf),
    }

    if etapa:
        etapas_exec = {etapa: etapas[etapa]}
    else:
        etapas_exec = etapas

    for nome, func in etapas_exec.items():
        while True:
            try:
                logger.info(f"=== FASE: {nome.upper()} ===")
                await func()
                break
            except KeyboardInterrupt:
                logger.warning("interrompido_pelo_usuario",
                               checkpoint=str(CHECKPOINT_PATH))
                raise
            except Exception as e:
                logger.error("erro_fase", fase=nome, erro=str(e)[:200])
                logger.info("aguardando_30s_para_retry")
                await asyncio.sleep(30)
                logger.info("retomando_fase", fase=nome)

    logger.info("=" * 60)
    logger.info("INGESTÃO BRASIL — COMPLETA")
    stats()
    if dlq.count:
        logger.info("dlq_pendente", total=dlq.count, por_fase=dlq.summary())
    logger.info("checkpoint", path=str(CHECKPOINT_PATH))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão nacional — Brasil")
    parser.add_argument("--resume", action="store_true",
                        help="Retomar do checkpoint")
    parser.add_argument("--fase",
                        choices=["emendas", "documentos", "detalhes"],
                        help="Executar apenas uma fase específica")
    parser.add_argument("--uf", type=str, default=None,
                        help="Filtrar por UF específica (ex: SP)")
    parser.add_argument("--retry-dlq", action="store_true",
                        help="Reprocessar itens da Dead Letter Queue")
    parser.add_argument("--pular-sem-uf", action="store_true",
                        help="Passada nacional sem as emendas sem UF (MÚLTIPLO/Nacional); "
                             "a base do artigo não tem documentos delas")
    parser.add_argument("--stats", action="store_true",
                        help="Exibir estatísticas sem executar coleta")
    args = parser.parse_args()

    if args.stats:
        show_stats_only()
    else:
        asyncio.run(main(
            resume=args.resume,
            etapa=args.fase,
            uf_filter=args.uf.upper() if args.uf else None,
            retry_dlq=args.retry_dlq,
            pular_sem_uf=args.pular_sem_uf,
        ))
