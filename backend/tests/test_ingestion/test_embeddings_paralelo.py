import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "embeddings_paralelo_sqlite.py"
_spec = importlib.util.spec_from_file_location("embeddings_paralelo_sqlite", SCRIPT)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


class _ConexaoFalsa:
    def __init__(self, erro=None):
        self.erro, self.fechada, self.commits = erro, False, 0

    def executemany(self, sql, pares):
        if self.erro:
            raise self.erro

    def commit(self):
        self.commits += 1

    def close(self):
        self.fechada = True


def test_gravar_vetores_espera_o_banco_destravar():
    # 14/10 18:45: a reconciliação de valores no fim do RS segurou o SQLite por
    # mais de 120 s e o worker morreu com "database is locked".
    conexoes = [_ConexaoFalsa(sqlite3.OperationalError("database is locked")),
                _ConexaoFalsa(sqlite3.OperationalError("database is locked")),
                _ConexaoFalsa()]
    fila = iter(conexoes)
    ep._gravar_vetores(lambda: next(fila), "documentos_emenda", [("[0.1]", 1)], espera_s=0)
    assert [c.commits for c in conexoes] == [0, 0, 1]
    assert all(c.fechada for c in conexoes)


def test_gravar_vetores_desiste_depois_do_limite():
    trava = sqlite3.OperationalError("database is locked")
    with pytest.raises(sqlite3.OperationalError):
        ep._gravar_vetores(lambda: _ConexaoFalsa(trava), "emendas", [("[0.1]", 1)],
                           espera_s=0, tentativas=3)


def test_gravar_vetores_nao_mascara_outros_erros():
    chamadas = []

    def conectar():
        chamadas.append(1)
        return _ConexaoFalsa(sqlite3.OperationalError("no such table: emendas"))

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        ep._gravar_vetores(conectar, "emendas", [("[0.1]", 1)], espera_s=0)
    assert len(chamadas) == 1


def test_gravar_vetores_grava_no_sqlite(tmp_path):
    banco = tmp_path / "local.db"
    conn = sqlite3.connect(banco)
    conn.executescript("CREATE TABLE emendas (id INTEGER PRIMARY KEY, embedding TEXT);"
                       "INSERT INTO emendas VALUES (1, NULL), (2, NULL);")
    conn.commit()
    conn.close()
    ep._gravar_vetores(lambda: sqlite3.connect(banco), "emendas",
                       [("[0.5]", 1), ("[0.25]", 2)], espera_s=0)
    conn = sqlite3.connect(banco)
    assert conn.execute("SELECT embedding FROM emendas ORDER BY id").fetchall() == [
        ("[0.5]",), ("[0.25]",)]
    conn.close()


def test_coleta_termina_sem_esperar_o_sem_uf(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    cp = tmp_path / "data" / "checkpoint_brasil.json"
    monkeypatch.setattr(ep, "BACKEND", tmp_path)
    detalhes = {uf: {"status": "completed"} for uf in ep.CHAVES}
    detalhes["SEM_UF"] = {"status": "in_progress"}  # pulado: nunca completa
    cp.write_text(json.dumps({"detalhes": detalhes}), encoding="utf-8")
    assert "SEM_UF" not in ep.CHAVES and len(ep.CHAVES) == 27
    assert ep.coleta_terminou()
    detalhes["TO"] = {"status": "in_progress"}
    cp.write_text(json.dumps({"detalhes": detalhes}), encoding="utf-8")
    assert not ep.coleta_terminou()
