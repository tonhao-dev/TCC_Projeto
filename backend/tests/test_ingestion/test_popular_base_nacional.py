"""Testes do detector de laço de erro do orquestrador (popular_base_nacional.py)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "popular_base_nacional.py"
_spec = importlib.util.spec_from_file_location("popular_base_nacional", SCRIPT)
pbn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pbn)


@pytest.fixture(autouse=True)
def _log_isolado(tmp_path, monkeypatch):
    # log() anexa em data/logs/00_pipeline.log; os testes não podem sujar o log real.
    monkeypatch.setattr(pbn, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(pbn, "LOG_PIPELINE", tmp_path / "logs" / "00_pipeline.log")
    monkeypatch.setattr(pbn, "PAUSA_RELANCE_S", 0)


ERRO_BRASIL = "2026-10-14 02:21:18 [error    ] erro_fase                      erro=x fase=emendas\n"
ERRO_SANCOES = "2026-10-14 02:21:18 [error    ] erro                           erro=x tipo=ceis\n"
PROGRESSO = "2026-10-14 02:22:00 [info     ] progresso_detalhes             pos=200 uf=SP\n"


def _log(tmp_path, conteudo: str) -> Path:
    caminho = tmp_path / "etapa.log"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


def test_conta_erros_seguidos_sem_progresso(tmp_path):
    caminho = _log(tmp_path, ERRO_BRASIL * 3)
    assert pbn._erros_sem_progresso(caminho, 0) == 3


def test_progresso_zera_a_contagem(tmp_path):
    caminho = _log(tmp_path, ERRO_BRASIL * 5 + PROGRESSO + ERRO_BRASIL)
    assert pbn._erros_sem_progresso(caminho, 0) == 1


def test_reconhece_o_formato_do_script_de_sancoes(tmp_path):
    caminho = _log(tmp_path, ERRO_SANCOES * 4)
    assert pbn._erros_sem_progresso(caminho, 0) == 4


def test_ignora_o_que_veio_antes_do_inicio_da_etapa(tmp_path):
    antigo = ERRO_BRASIL * 30
    caminho = _log(tmp_path, antigo + ERRO_BRASIL)
    assert pbn._erros_sem_progresso(caminho, len(antigo.encode("utf-8"))) == 1



def test_fixar_tmp_troca_um_tmp_que_sumiu(tmp_path, monkeypatch):
    sumiu = tmp_path / "scratch-da-sessao-reclamado"
    monkeypatch.setenv("TMP", str(sumiu))
    monkeypatch.setenv("TEMP", str(sumiu))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    pbn._fixar_tmp()
    esperado = tmp_path / "Local" / "Temp"
    assert esperado.is_dir()
    assert Path(pbn.os.environ["TMP"]) == esperado
    assert Path(pbn.os.environ["TEMP"]) == esperado


def test_filhos_herdam_o_tmp_fixado(tmp_path, monkeypatch):
    monkeypatch.setenv("TMP", str(tmp_path / "sumiu"))
    monkeypatch.setenv("TEMP", str(tmp_path / "sumiu"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    pbn._fixar_tmp()
    env = pbn._env("sqlite")
    assert env["TMP"] == env["TEMP"] == str(tmp_path / "Local" / "Temp")



def test_fase_por_uf_exige_as_emendas_sem_uf(tmp_path, monkeypatch):
    cp = tmp_path / "checkpoint_brasil.json"
    cp.write_text(pbn.json.dumps({"documentos": {
        uf: {"status": "completed"} for uf in pbn.UFS}}), encoding="utf-8")
    monkeypatch.setattr(pbn, "CP_BRASIL", cp)
    monkeypatch.setattr(pbn, "INCLUIR_SEM_UF", True)
    chamadas = []
    monkeypatch.setattr(pbn, "_rodar", lambda nome, args, banco: chamadas.append(args) or 0)

    try:
        pbn._fase_por_uf("documentos", "02_documentos")
        assert False, "devia falhar com SEM_UF pendente"
    except pbn.EtapaFalhou as e:
        assert pbn.SEM_UF in str(e)
    # As 27 UFs completas não bastam: a fase nacional roda de novo para o SEM_UF,
    # e é relançada até o limite antes de a etapa falhar.
    nacional = ["scripts/run_ingestion_brasil.py", "--fase", "documentos", "--resume"]
    assert chamadas == [nacional] * (1 + pbn.MAX_RELANCES_FASE)


def test_fase_por_uf_relanca_filho_encerrado_de_fora(tmp_path, monkeypatch):
    # 14/10 12:01: o filho morreu no meio do SEM_UF sem log; a etapa deve retomar.
    cp = tmp_path / "checkpoint_brasil.json"
    estado = {"documentos": {uf: {"status": "completed"} for uf in pbn.UFS}}
    cp.write_text(pbn.json.dumps(estado), encoding="utf-8")
    monkeypatch.setattr(pbn, "CP_BRASIL", cp)
    monkeypatch.setattr(pbn, "INCLUIR_SEM_UF", True)
    chamadas = []

    def rodar(nome, args, banco):
        chamadas.append(args)
        if len(chamadas) == 2:  # a relance termina o SEM_UF
            estado["documentos"][pbn.SEM_UF] = {"status": "completed"}
            cp.write_text(pbn.json.dumps(estado), encoding="utf-8")
        return 0xC000013A if len(chamadas) == 1 else 0

    monkeypatch.setattr(pbn, "_rodar", rodar)
    pbn._fase_por_uf("documentos", "02_documentos")
    nacional = ["scripts/run_ingestion_brasil.py", "--fase", "documentos", "--resume"]
    assert chamadas == [nacional, nacional]  # o AC já estava completo


def test_fase_por_uf_pula_sem_uf_por_padrao(tmp_path, monkeypatch):
    # Decisão de 14/10: a base do artigo não tem documentos das emendas sem UF.
    assert pbn.INCLUIR_SEM_UF is False
    cp = tmp_path / "checkpoint_brasil.json"
    estado = {"detalhes": {uf: {"status": "completed"} for uf in pbn.UFS if uf != "TO"}}
    estado["detalhes"][pbn.SEM_UF] = {"status": "in_progress", "last_emenda_id": 8348}
    cp.write_text(pbn.json.dumps(estado), encoding="utf-8")
    monkeypatch.setattr(pbn, "CP_BRASIL", cp)
    chamadas = []

    def rodar(nome, args, banco):
        chamadas.append(args)
        estado["detalhes"]["TO"] = {"status": "completed"}
        cp.write_text(pbn.json.dumps(estado), encoding="utf-8")
        return 0

    monkeypatch.setattr(pbn, "_rodar", rodar)
    pbn._fase_por_uf("detalhes", "03_detalhes")  # SEM_UF pendente não falha a etapa
    assert chamadas == [["scripts/run_ingestion_brasil.py", "--fase", "detalhes",
                         "--resume", "--pular-sem-uf"]]


class _ProcFalso:
    def __init__(self, rc):
        self.rc = rc

    def wait(self, timeout=None):
        return self.rc


def test_rodar_nao_abre_janela_e_registra_codigo_de_saida(tmp_path, monkeypatch, capsys):
    recebido = {}

    def popen(cmd, **kwargs):
        recebido.update(kwargs)
        return _ProcFalso(0xC000013A)

    monkeypatch.setattr(pbn.subprocess, "Popen", popen)
    assert pbn._rodar("02_documentos", ["scripts/x.py"], "sqlite") == 0xC000013A
    assert recebido["creationflags"] == pbn.CREATE_NO_WINDOW
    if pbn.os.name == "nt":
        assert pbn.CREATE_NO_WINDOW == 0x08000000
    assert "0xC000013A" in capsys.readouterr().out


def test_pendentes_por_uf_chama_uf_vazio_de_sem_uf(tmp_path, monkeypatch):
    banco = tmp_path / "local.db"
    conn = pbn.sqlite3.connect(banco)
    conn.executescript("""
        CREATE TABLE emendas (id INTEGER PRIMARY KEY, uf TEXT);
        CREATE TABLE documentos_emenda (id INTEGER PRIMARY KEY, emenda_id INTEGER,
                                        detalhes_coletados INTEGER);
        INSERT INTO emendas VALUES (1, 'SP'), (2, ''), (3, NULL);
        INSERT INTO documentos_emenda VALUES (1, 1, 0), (2, 2, 0), (3, 3, NULL), (4, 2, 1);
    """)
    conn.commit()
    conn.close()
    monkeypatch.setattr(pbn, "SQLITE", banco)
    monkeypatch.setattr(pbn, "INCLUIR_SEM_UF", True)
    assert pbn._pendentes_detalhes_por_uf() == {"SP": 1, pbn.SEM_UF: 2}
    # Pulando o SEM_UF, a repescagem não deve buscar detalhes desses documentos.
    monkeypatch.setattr(pbn, "INCLUIR_SEM_UF", False)
    assert pbn._pendentes_detalhes_por_uf() == {"SP": 1}



def test_repescagem_rebusca_emenda_parcial_e_depois_coleta_detalhes(tmp_path, monkeypatch):
    banco = tmp_path / "local.db"
    conn = pbn.sqlite3.connect(banco)
    conn.executescript("""
        CREATE TABLE emendas (id INTEGER PRIMARY KEY, uf TEXT, codigo_emenda TEXT);
        CREATE TABLE documentos_emenda (id INTEGER PRIMARY KEY, emenda_id INTEGER,
                                        codigo_documento TEXT, detalhes_coletados INTEGER);
        INSERT INTO emendas VALUES (1, 'SP', 'E1');
        INSERT INTO documentos_emenda VALUES (1, 1, 'D1', 1);  -- só a 1ª página veio
    """)
    conn.commit()
    conn.close()
    dlq = tmp_path / "dlq.json"
    dlq.write_text(pbn.json.dumps([{"fase": "documentos", "identificador": "E1",
                                    "erro": "403", "retry_count": 1}]), encoding="utf-8")
    cp = tmp_path / "cp.json"
    cp.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pbn, "SQLITE", banco)
    monkeypatch.setattr(pbn, "DLQ_BRASIL", dlq)
    monkeypatch.setattr(pbn, "CP_BRASIL", cp)
    monkeypatch.setattr(pbn, "log", lambda msg: None)
    chamadas = []

    def rodar(nome, args, banco_alvo):
        chamadas.append(args[1:])
        if "--retry-dlq" in args:  # a rebusca traz a página que faltava, sem detalhes
            c = pbn.sqlite3.connect(banco)
            c.execute("INSERT INTO documentos_emenda VALUES (2, 1, 'D2', 0)")
            c.commit()
            c.close()
            dlq.write_text("[]", encoding="utf-8")
        else:
            c = pbn.sqlite3.connect(banco)
            c.execute("UPDATE documentos_emenda SET detalhes_coletados = 1")
            c.commit()
            c.close()
        return 0

    monkeypatch.setattr(pbn, "_rodar", rodar)
    pbn.etapa_repescagem()

    # Antes a emenda com 1 documento era podada como resolvida e a DLQ nem rodava.
    assert chamadas == [["--retry-dlq"], ["--fase", "detalhes", "--uf", "SP", "--resume"]]
