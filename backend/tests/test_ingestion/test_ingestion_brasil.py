"""Testes das correções na ingestão nacional (run_ingestion_brasil.py / cgu_collector.py).

Cobrem o que uma coleta de vários dias precisa: valores com sinal separado,
limite por minuto conforme o horário de Brasília, retomada sem pular emendas
nem documentos pendentes e espera (em vez de DLQ) com o disjuntor aberto.

Rode com DATABASE_URL=sqlite:// (tipos compatíveis com SQLite, como o resto da suíte).
"""
import asyncio
import importlib.util
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401
import app.models.favorecido  # noqa: F401  (DocumentoFavorecido)
import app.models.lookup  # noqa: F401
from app.database import Base, is_sqlite
from app.services.ingestion import cgu_collector
from app.services.ingestion.cgu_collector import CGUCollector
from app.utils.checkpoint import CheckpointManager
from app.utils.circuit_breaker import CircuitBreaker, DeadLetterQueue

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_ingestion_brasil.py"
_spec = importlib.util.spec_from_file_location("run_ingestion_brasil", SCRIPT)
rib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rib)

requer_sqlite = pytest.mark.skipif(
    not is_sqlite(), reason="rode com DATABASE_URL=sqlite:// (tipos do modelo para SQLite)")


class ColetorFalso:
    def __init__(self):
        self.documentos_pedidos = []
        self.detalhes_pedidos = []

    def update_rpm_for_current_hour(self):
        pass

    async def close_shared_client(self):
        pass

    async def coletar_documentos_emenda(self, codigo):
        self.documentos_pedidos.append(codigo)
        return [{"codigo_documento": f"DOC-{codigo}", "fase": "Empenho", "valor": 0,
                 "data_documento": "01/02/2024", "tipo_documento": "Empenho",
                 "observacao": ""}]

    async def coletar_detalhes_documento(self, codigo):
        self.detalhes_pedidos.append(codigo)
        return {"valor": "1.000,00", "observacao": "Aquisição de ambulância",
                "nomeFavorecido": "FUNDO MUNICIPAL DE SAUDE",
                "codigoFavorecido": "11.111.111/0001-11", "ufFavorecido": "SP",
                "funcao": "10 - Saúde", "subfuncao": "301 - Atenção Básica"}


@pytest.fixture
def banco(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(rib, "SessionLocal",
                        sessionmaker(bind=engine, autocommit=False, autoflush=False))
    monkeypatch.setattr(rib, "init_db", lambda: None)
    monkeypatch.setattr(rib, "ESPERA_DISJUNTOR_S", 0.01)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def _inserir_emendas(engine, n, uf="SP"):
    with engine.begin() as c:
        for i in range(1, n + 1):
            c.execute(text("""
                INSERT INTO emendas (id, codigo_emenda, ano, uf,
                                     valor_empenhado, valor_liquidado, valor_pago)
                VALUES (:i, :cod, 2024, :uf, 0, 0, 0)
            """), {"i": i, "cod": f"E{i}", "uf": uf})


# --- valores -----------------------------------------------------------------

@pytest.mark.parametrize("bruto, esperado", [
    ("1.234,56", 1234.56),
    ("- 4.504,92", -4504.92),   # formato que derrubou a fase de emendas
    ("-4.504,92", -4504.92),
    ("0,00", 0.0),
    ("-", 0.0),
    ("", 0.0),
    (None, 0.0),
    (12.5, 12.5),
])
def test_parse_valor(bruto, esperado):
    assert CGUCollector._parse_valor(bruto) == pytest.approx(esperado)


# --- limite por horário de Brasília -------------------------------------------

def test_rpm_vigente_usa_cota_noturna_de_brasilia(monkeypatch):
    monkeypatch.setattr(cgu_collector.settings, "CGU_RATE_LIMIT_RPM", 380)
    monkeypatch.setattr(cgu_collector.settings, "CGU_RATE_LIMIT_RPM_NOITE", 650)
    utc = timezone.utc
    assert cgu_collector.rpm_vigente(datetime(2026, 10, 14, 2, 59, tzinfo=utc)) == 380  # 23:59
    assert cgu_collector.rpm_vigente(datetime(2026, 10, 14, 3, 0, tzinfo=utc)) == 650   # 00:00
    assert cgu_collector.rpm_vigente(datetime(2026, 10, 14, 8, 59, tzinfo=utc)) == 650  # 05:59
    assert cgu_collector.rpm_vigente(datetime(2026, 10, 14, 9, 0, tzinfo=utc)) == 380   # 06:00


def test_rpm_vigente_sem_cota_noturna(monkeypatch):
    monkeypatch.setattr(cgu_collector.settings, "CGU_RATE_LIMIT_RPM", 400)
    monkeypatch.setattr(cgu_collector.settings, "CGU_RATE_LIMIT_RPM_NOITE", 0)
    assert cgu_collector.rpm_vigente(datetime(2026, 10, 14, 5, 0, tzinfo=timezone.utc)) == 400


@pytest.mark.asyncio
async def test_coletor_troca_o_limite_na_virada_do_horario(monkeypatch):
    monkeypatch.setattr(cgu_collector.settings, "CGU_API_KEY", "chave-teste")
    monkeypatch.setattr(cgu_collector.settings, "CGU_API_KEYS", "")
    atual = {"rpm": 380}
    monkeypatch.setattr(cgu_collector, "rpm_vigente", lambda agora=None: atual["rpm"])
    coletor = CGUCollector()
    assert coletor._rate_limiters["chave-teste"].rpm == 380

    atual["rpm"] = 650
    await coletor._acquire_and_get_headers()
    limiter = coletor._rate_limiters["chave-teste"]
    assert limiter.rpm == 650
    assert limiter.interval == pytest.approx(60 / 650)


# --- disjuntor ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_detalhe_espera_o_disjuntor_em_vez_de_desistir(monkeypatch):
    monkeypatch.setattr(rib, "ESPERA_DISJUNTOR_S", 0.01)
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.2)
    cb.record_failure()
    assert cb.state == "OPEN"
    coletor = ColetorFalso()

    inicio = time.monotonic()
    codigo, detalhe, erro = await rib._fetch_detalhe(coletor, "D9", cb, asyncio.Semaphore(1))

    assert (codigo, erro) == ("D9", None)
    assert detalhe["nomeFavorecido"] == "FUNDO MUNICIPAL DE SAUDE"
    assert time.monotonic() - inicio >= 0.2
    assert cb.state == "CLOSED"


# --- retomada -----------------------------------------------------------------

@requer_sqlite
@pytest.mark.asyncio
async def test_emendas_retoma_na_pagina_registrada(banco, tmp_path, monkeypatch):
    paginas = {}

    class ColetorEmendas:
        current_page = 1

        async def coletar_emendas(self, ano, start_page=1):
            paginas[ano] = start_page
            return
            yield  # noqa: B901 — gerador assíncrono vazio

    monkeypatch.setattr(rib, "CGUCollector", ColetorEmendas)
    cp = CheckpointManager(path=tmp_path / "cp.json")
    cp.update_progress("emendas", "2023", last_page=61)

    await rib.fase_emendas(cp, DeadLetterQueue(tmp_path / "dlq.json"))

    assert paginas[2023] == 61  # antes: 62, perdendo o resto da página 61
    assert paginas[2020] == 1


@requer_sqlite
@pytest.mark.asyncio
async def test_documentos_retoma_apos_a_ultima_emenda_processada(banco, tmp_path, monkeypatch):
    _inserir_emendas(banco, 5)
    with banco.begin() as c:  # E1 e E2 já têm documentos; E3 foi processada sem nenhum
        c.execute(text("""
            INSERT INTO documentos_emenda (emenda_id, codigo_documento, fase, valor)
            VALUES (1, 'X1', 'Empenho', 0), (2, 'X2', 'Empenho', 0)
        """))
    coletor = ColetorFalso()
    monkeypatch.setattr(rib, "CGUCollector", lambda: coletor)
    cp = CheckpointManager(path=tmp_path / "cp.json")
    cp.update_progress("documentos", "SP", last_emenda_id=3)

    await rib.fase_documentos(cp, DeadLetterQueue(tmp_path / "dlq.json"), uf_filter="SP")

    # Antes, um índice posicional na lista recalculada [E3, E4, E5] pulava todas.
    assert coletor.documentos_pedidos == ["E4", "E5"]
    assert cp.get_status("documentos", "SP") == "completed"


@requer_sqlite
@pytest.mark.asyncio
async def test_documentos_em_paralelo_processa_todas_e_checkpoint_seguro(banco, tmp_path,
                                                                         monkeypatch):
    _inserir_emendas(banco, 25)
    lentas = {"E3"}  # termina por último: o checkpoint não pode passar de E2 antes dela

    class ColetorLento(ColetorFalso):
        async def coletar_documentos_emenda(self, codigo):
            await asyncio.sleep(0.2 if codigo in lentas else 0.001)
            return await super().coletar_documentos_emenda(codigo)

    coletor = ColetorLento()
    monkeypatch.setattr(rib, "CGUCollector", lambda: coletor)
    monkeypatch.setattr(rib, "CONCORRENCIA_DOCUMENTOS", 4)
    marcas = []
    cp = CheckpointManager(path=tmp_path / "cp.json")
    original = cp.update_progress

    def registrar(fonte, chave, **kw):
        if fonte == "documentos":
            marcas.append(kw["last_emenda_id"])
        original(fonte, chave, **kw)

    monkeypatch.setattr(cp, "update_progress", registrar)
    await rib.fase_documentos(cp, DeadLetterQueue(tmp_path / "dlq.json"), uf_filter="SP")

    assert sorted(coletor.documentos_pedidos) == sorted(f"E{i}" for i in range(1, 26))
    with banco.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM documentos_emenda")).scalar() == 25
    # Enquanto E3 estava em voo, nenhuma marca podia passar de 2.
    assert marcas and marcas[0] <= 2
    assert marcas == sorted(marcas)
    assert cp.get_status("documentos", "SP") == "completed"


@requer_sqlite
@pytest.mark.asyncio
async def test_detalhes_processa_pendentes_mesmo_com_last_index_antigo(banco, tmp_path,
                                                                       monkeypatch):
    _inserir_emendas(banco, 2)
    with banco.begin() as c:
        c.execute(text("""
            INSERT INTO documentos_emenda
                (id, emenda_id, codigo_documento, fase, valor, detalhes_coletados)
            VALUES (1, 1, 'D1', 'Empenho', 0, 1),
                   (2, 1, 'D2', 'Empenho', 0, 0),
                   (3, 2, 'D3', 'Pagamento', 0, 0)
        """))
    coletor = ColetorFalso()
    monkeypatch.setattr(rib, "CGUCollector", lambda: coletor)
    cp = CheckpointManager(path=tmp_path / "cp.json")
    cp.update_progress("detalhes", "SP", last_index=2)  # posição na lista antiga

    await rib.fase_detalhes(cp, DeadLetterQueue(tmp_path / "dlq.json"), uf_filter="SP")

    assert sorted(coletor.detalhes_pedidos) == ["D2", "D3"]
    with banco.connect() as c:
        pendentes = c.execute(text(
            "SELECT COUNT(*) FROM documentos_emenda WHERE detalhes_coletados = 0")).scalar()
        favorecidos = c.execute(text("SELECT COUNT(*) FROM favorecidos")).scalar()
        valor = c.execute(text(
            "SELECT valor FROM documentos_emenda WHERE codigo_documento = 'D2'")).scalar()
    assert pendentes == 0
    assert favorecidos == 1
    assert float(valor) == pytest.approx(1000.0)



# --- emendas sem UF (MÚLTIPLO, Nacional, regionais) ----------------------------

def _inserir_emendas_com_uf(engine, ufs):
    with engine.begin() as c:
        for i, uf in enumerate(ufs, start=1):
            c.execute(text("""
                INSERT INTO emendas (id, codigo_emenda, ano, uf,
                                     valor_empenhado, valor_liquidado, valor_pago)
                VALUES (:i, :cod, 2024, :uf, 0, 0, 0)
            """), {"i": i, "cod": f"E{i}", "uf": uf})


def test_filtro_uf():
    assert rib._filtro_uf("SP") == ("e.uf = :uf", {"uf": "SP"})
    assert rib._filtro_uf(rib.SEM_UF) == ("COALESCE(e.uf, '') = ''", {})


def test_ufs_da_passada_nacional_pula_ac_e_opcionalmente_sem_uf():
    assert rib._ufs_da_passada("SP") == ["SP"]
    assert rib._ufs_da_passada(rib.SEM_UF, pular_sem_uf=True) == [rib.SEM_UF]  # pedido explícito
    nacional = rib._ufs_da_passada(None)
    assert nacional[-1] == rib.SEM_UF and "AC" not in nacional and len(nacional) == 27
    sem = rib._ufs_da_passada(None, pular_sem_uf=True)
    assert sem == nacional[:-1] and len(sem) == 26


@requer_sqlite
@pytest.mark.asyncio
async def test_documentos_sem_uf_pega_uf_vazio_e_nulo(banco, tmp_path, monkeypatch):
    _inserir_emendas_com_uf(banco, ["SP", "", None, "RJ", ""])
    coletor = ColetorFalso()
    monkeypatch.setattr(rib, "CGUCollector", lambda: coletor)
    cp = CheckpointManager(path=tmp_path / "cp.json")

    await rib.fase_documentos(cp, DeadLetterQueue(tmp_path / "dlq.json"), uf_filter=rib.SEM_UF)

    assert sorted(coletor.documentos_pedidos) == ["E2", "E3", "E5"]
    assert cp.get_status("documentos", rib.SEM_UF) == "completed"


@requer_sqlite
@pytest.mark.asyncio
async def test_fase_nacional_termina_nas_emendas_sem_uf(banco, tmp_path, monkeypatch):
    _inserir_emendas_com_uf(banco, ["SP", "", "AC", None])
    coletor = ColetorFalso()
    monkeypatch.setattr(rib, "CGUCollector", lambda: coletor)
    cp = CheckpointManager(path=tmp_path / "cp.json")

    await rib.fase_documentos(cp, DeadLetterQueue(tmp_path / "dlq.json"))

    # AC continua fora da fase nacional (o orquestrador roda --uf AC à parte).
    assert sorted(coletor.documentos_pedidos) == ["E1", "E2", "E4"]
    assert cp.get_status("documentos", rib.SEM_UF) == "completed"


@requer_sqlite
@pytest.mark.asyncio
async def test_detalhes_sem_uf(banco, tmp_path, monkeypatch):
    _inserir_emendas_com_uf(banco, ["", "SP"])
    with banco.begin() as c:
        c.execute(text("""
            INSERT INTO documentos_emenda
                (id, emenda_id, codigo_documento, fase, valor, detalhes_coletados)
            VALUES (1, 1, 'D1', 'Empenho', 0, 0), (2, 2, 'D2', 'Empenho', 0, 0)
        """))
    coletor = ColetorFalso()
    monkeypatch.setattr(rib, "CGUCollector", lambda: coletor)
    cp = CheckpointManager(path=tmp_path / "cp.json")

    await rib.fase_detalhes(cp, DeadLetterQueue(tmp_path / "dlq.json"), uf_filter=rib.SEM_UF)

    assert coletor.detalhes_pedidos == ["D1"]
    assert cp.get_status("detalhes", rib.SEM_UF) == "completed"



# --- checkpoint no Windows -------------------------------------------------------

def test_checkpoint_tenta_de_novo_quando_o_arquivo_esta_preso(tmp_path, monkeypatch):
    from app.utils import checkpoint as modulo
    reais = modulo.os.replace
    falhas = {"restam": 3}

    def replace_preso(origem, destino):
        if falhas["restam"]:
            falhas["restam"] -= 1
            raise PermissionError(5, "Access is denied")  # antivírus segurando o arquivo
        reais(origem, destino)

    monkeypatch.setattr(modulo.os, "replace", replace_preso)
    monkeypatch.setattr(modulo.time, "sleep", lambda s: None)
    cp = CheckpointManager(path=tmp_path / "cp.json")
    cp.mark_completed("documentos", "SP", total=1)

    assert falhas["restam"] == 0
    assert CheckpointManager(path=tmp_path / "cp.json").get_status("documentos", "SP") == "completed"


def test_checkpoint_desiste_depois_das_tentativas(tmp_path, monkeypatch):
    from app.utils import checkpoint as modulo

    def sempre_preso(origem, destino):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(modulo.os, "replace", sempre_preso)
    monkeypatch.setattr(modulo.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        modulo.substituir_atomico(tmp_path / "a.tmp", tmp_path / "a.json", tentativas=3)



# --- bloqueios do WAF (403) e falhas de rede ------------------------------------

def _coletor_com_respostas(monkeypatch, respostas):
    """CGUCollector cujas requisições recebem, em ordem, as respostas da lista."""
    import httpx
    monkeypatch.setattr(cgu_collector.settings, "CGU_API_KEY", "chave-teste")
    monkeypatch.setattr(cgu_collector.settings, "CGU_API_KEYS", "")
    monkeypatch.setattr(cgu_collector, "rpm_vigente", lambda agora=None: 60_000)
    monkeypatch.setattr(CGUCollector, "ESPERA_BLOQUEIO_DOCS_S", 0)
    pedidos = []

    def responder(request):
        pedidos.append(request.url.params.get("pagina"))
        resposta = respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        status, corpo = resposta
        return httpx.Response(status, json=corpo)

    cliente_real = httpx.AsyncClient
    monkeypatch.setattr(cgu_collector.httpx, "AsyncClient",
                        lambda **kw: cliente_real(transport=httpx.MockTransport(responder), **kw))
    return CGUCollector(), pedidos


DOC = {"codigoDocumento": "D1", "fase": "Empenho", "data": "10/03/2026", "observacao": "x"}


@pytest.mark.asyncio
async def test_documentos_esperam_o_bloqueio_do_waf(monkeypatch):
    coletor, pedidos = _coletor_com_respostas(monkeypatch, [(403, {}), (403, {}), (200, [DOC]),
                                                            (200, [])])
    docs = await coletor.coletar_documentos_emenda("E1")
    assert [d["codigo_documento"] for d in docs] == ["D1"]
    assert pedidos == ["1", "1", "1", "2"]  # a página 1 é pedida de novo, nada é pulado


@pytest.mark.asyncio
async def test_documentos_com_bloqueio_persistente_viram_erro(monkeypatch):
    import httpx
    coletor, _ = _coletor_com_respostas(monkeypatch, [(200, [DOC])] + [(403, {})] * 10)
    # Antes: devolvia [D1] e a emenda era gravada como completa, sem as outras páginas.
    with pytest.raises(httpx.HTTPStatusError):
        await coletor.coletar_documentos_emenda("E1")


@pytest.mark.asyncio
async def test_documentos_sem_rede_viram_erro(monkeypatch):
    import httpx
    falha = httpx.ConnectError("sem rede")
    coletor, _ = _coletor_com_respostas(monkeypatch, [falha])
    with pytest.raises(httpx.ConnectError):
        await coletor.coletar_documentos_emenda("E1", max_network_retries=1)


@pytest.mark.asyncio
async def test_detalhe_sem_rede_vira_erro_e_nao_documento_vazio(monkeypatch):
    import httpx
    monkeypatch.setattr(cgu_collector.settings, "CGU_API_KEY", "chave-teste")
    monkeypatch.setattr(cgu_collector.settings, "CGU_API_KEYS", "")
    monkeypatch.setattr(cgu_collector, "rpm_vigente", lambda agora=None: 60_000)

    def sem_rede(request):
        raise httpx.ConnectError("sem rede", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(sem_rede)) as cliente:
        # None faria o writer marcar detalhes_coletados = 1 sem detalhe nenhum.
        with pytest.raises(httpx.ConnectError):
            await CGUCollector().coletar_detalhes_documento("D1", max_network_retries=1,
                                                            client=cliente)
