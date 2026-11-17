"""Normalização das sanções (CEIS, CNEP, CEPIM, CEAF) do Portal da Transparência.

Os itens abaixo seguem o formato oficial da API (CeisDTO, CnepDTO, CepimDTO e
CeafDTO em https://api.portaldatransparencia.gov.br/v3/api-docs). Quem foi
punido vem dentro de objetos aninhados (`sancionado`, `pessoa`,
`pessoaJuridica`, `punicao`), não em campos soltos no topo do item.
"""
import asyncio

import pytest

from app.services.ingestion.cgu_collector import CGUCollector

ORGAO_SANCIONADOR = {
    "nome": "Tribunal de Justiça do Estado de Roraima",
    "siglaUf": "RR",
    "poder": "Judiciário",
    "esfera": "Estadual",
}

CEIS_PJ = {
    "id": 1,
    "dataInicioSancao": "23/05/2022",
    "dataFimSancao": "23/05/2032",
    "tipoSancao": {
        "descricaoResumida": "Impedimento/proibição de contratar com prazo determinado",
        "descricaoPortal": "Impedimento de contratar",
    },
    "fundamentacao": [
        {"codigo": "12846", "descricao": "Lei 12.846/2013, art. 6º"},
        {"codigo": "8666", "descricao": "Lei 8.666/1993, art. 87"},
    ],
    "orgaoSancionador": ORGAO_SANCIONADOR,
    "sancionado": {"nome": "CONSTRUTORA EXEMPLO LTDA", "codigoFormatado": "00.360.305/0001-04"},
    "pessoa": {
        "id": 10,
        "cpfFormatado": "",
        "cnpjFormatado": "00.360.305/0001-04",
        "nome": "CONSTRUTORA EXEMPLO LTDA",
        "tipo": "Pessoa Jurídica",
    },
}

CNEP_PF = {
    "id": 2,
    "dataInicioSancao": "27/01/2016",
    "dataFimSancao": None,
    "tipoSancao": {"descricaoResumida": "Multa", "descricaoPortal": "Multa"},
    "fundamentacao": [{"codigo": "12846", "descricao": "Lei 12.846/2013, art. 6º, I"}],
    "orgaoSancionador": ORGAO_SANCIONADOR,
    "sancionado": {"nome": "FULANO DE TAL", "codigoFormatado": "***.456.789-**"},
    "pessoa": {"cpfFormatado": "***.456.789-**", "cnpjFormatado": "", "nome": "FULANO DE TAL"},
    "valorMulta": "1.234,56",
}

CEPIM = {
    "id": 3,
    "motivo": "Omissão no dever de prestar contas",
    "orgaoSuperior": {
        "nome": "Ministério da Saúde",
        "codigoSIAFI": "36000",
        "sigla": "MS",
        "descricaoPoder": "Executivo",
        "orgaoMaximo": {"codigo": "36000", "sigla": "MS", "nome": "Ministério da Saúde"},
    },
    "pessoaJuridica": {
        "cnpjFormatado": "11.222.333/0001-44",
        "nome": "ASSOCIACAO EXEMPLO",
        "tipo": "Pessoa Jurídica",
    },
    "convenio": {"codigo": "1", "objeto": "Obra", "numero": "123"},
}

CEAF = {
    "id": 4,
    "dataPublicacao": "21/07/2021",
    "punicao": {
        "cpfPunidoFormatado": "***.111.222-**",
        "nomePunido": "BELTRANO DA SILVA",
        "portaria": "Portaria 1",
    },
    "tipoPunicao": {"descricao": "Demissão"},
    "pessoa": {"cpfFormatado": "***.111.222-**", "nome": "BELTRANO DA SILVA"},
    "orgaoLotacao": {
        "siglaDaPasta": "MJSP",
        "sigla": "MJSP",
        "nome": "MINISTÉRIO DA JUSTIÇA E SEGURANÇA PÚBLICA",
        "nomeSemAcento": "MINISTERIO DA JUSTICA E SEGURANCA PUBLICA",
    },
    "fundamentacao": [{"codigo": "132", "descricao": "Lei 8.112/1990, art. 132"}],
}


@pytest.fixture
def collector():
    return CGUCollector.__new__(CGUCollector)  # sem rate limiter nem chaves de API


def _coletar(collector, metodo, item, monkeypatch):
    async def fake_paginado(self, endpoint, label, start_page=1, params_extra=None):
        yield item

    monkeypatch.setattr(CGUCollector, "_coletar_paginado", fake_paginado)

    async def run():
        return [x async for x in getattr(collector, metodo)()]

    return asyncio.run(run())


def test_ceis_le_cnpj_e_nome_do_sancionado(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_ceis", CEIS_PJ, monkeypatch)
    assert r["cpf_cnpj"] == "00.360.305/0001-04"
    assert r["nome"] == "CONSTRUTORA EXEMPLO LTDA"


def test_ceis_fundamentacao_vira_texto_legivel(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_ceis", CEIS_PJ, monkeypatch)
    assert r["fundamentacao"] == "Lei 12.846/2013, art. 6º; Lei 8.666/1993, art. 87"


def test_ceis_mantem_campos_que_ja_funcionavam(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_ceis", CEIS_PJ, monkeypatch)
    assert r["tipo_sancao"] == "Impedimento/proibição de contratar com prazo determinado"
    assert r["orgao_sancionador"] == "Tribunal de Justiça do Estado de Roraima"
    assert r["data_inicio"] == "23/05/2022"
    assert r["data_fim"] == "23/05/2032"


def test_ceis_usa_pessoa_quando_sancionado_falta(collector, monkeypatch):
    item = {**CEIS_PJ, "sancionado": None}
    [r] = _coletar(collector, "coletar_sancoes_ceis", item, monkeypatch)
    assert r["cpf_cnpj"] == "00.360.305/0001-04"
    assert r["nome"] == "CONSTRUTORA EXEMPLO LTDA"


def test_cnep_le_cpf_nome_e_multa(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_cnep", CNEP_PF, monkeypatch)
    assert r["cpf_cnpj"] == "***.456.789-**"
    assert r["nome"] == "FULANO DE TAL"
    assert r["tipo_sancao"] == "Multa"
    assert float(r["valor_multa"]) == pytest.approx(1234.56)


def test_cepim_le_todos_os_campos(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_cepim", CEPIM, monkeypatch)
    assert r == {
        "cnpj": "11.222.333/0001-44",
        "nome": "ASSOCIACAO EXEMPLO",
        "motivo": "Omissão no dever de prestar contas",
        "orgao_maximo": "Ministério da Saúde",
    }


def test_ceaf_le_cpf_nome_e_punicao(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_ceaf", CEAF, monkeypatch)
    assert r["cpf"] == "***.111.222-**"
    assert r["nome"] == "BELTRANO DA SILVA"
    assert r["tipo_punicao"] == "Demissão"
    assert r["orgao_lotacao"] == "MINISTÉRIO DA JUSTIÇA E SEGURANÇA PÚBLICA"
    assert r["data_inicio"] == "21/07/2021"
    assert r["data_fim"] is None


def test_item_sem_objetos_aninhados_nao_quebra(collector, monkeypatch):
    [r] = _coletar(collector, "coletar_sancoes_ceis", {"id": 9}, monkeypatch)
    assert r["cpf_cnpj"] == ""
    assert r["nome"] == ""
    assert r["fundamentacao"] == ""
