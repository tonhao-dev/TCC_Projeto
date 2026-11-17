"""Verifica a integridade do gabarito gerado por scripts/gerar_gabarito_consultas.py.

Testes offline: leem apenas o JSON versionado, sem tocar o banco. O que se
garante aqui é que o gabarito continua alinhado ao conjunto de consultas e que
cada resposta de referência é utilizável como alvo de BERTScore.
"""
import json
import re
from pathlib import Path

import pytest

from tests.test_queries.test_evaluation import (
    CONSULTAS_TIPO_A, CONSULTAS_TIPO_B, CONSULTAS_TIPO_C, CONSULTAS_TIPO_D,
)

GABARITO = Path(__file__).parent / "gabarito_120_consultas.json"

TOPICOS = {"factual_direct_lookup", "semantic_cross_source",
           "temporal_comparison", "beneficiary_lookup"}


@pytest.fixture(scope="module")
def doc() -> dict:
    if not GABARITO.exists():
        pytest.skip(f"gabarito ausente: rode "
                    f"scripts/gerar_gabarito_consultas.py ({GABARITO.name})")
    return json.loads(GABARITO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def consultas() -> dict:
    todas = (CONSULTAS_TIPO_A + CONSULTAS_TIPO_B
             + CONSULTAS_TIPO_C + CONSULTAS_TIPO_D)
    return {c["id"]: c for c in todas}


def test_tem_120_consultas(doc):
    assert len(doc["queries"]) == 120


def test_ids_batem_com_conjunto_de_avaliacao(doc, consultas):
    ids_gab = [q["id"] for q in doc["queries"]]
    assert len(ids_gab) == len(set(ids_gab)), "IDs duplicados no gabarito"
    assert set(ids_gab) == set(consultas), \
        "gabarito e test_evaluation divergem nos IDs"


def test_enunciado_e_filtros_identicos_a_origem(doc, consultas):
    """O gabarito não pode reescrever a pergunta nem os filtros esperados."""
    for q in doc["queries"]:
        origem = consultas[q["id"]]
        assert q["query"] == origem["consulta"], f"{q['id']}: enunciado alterado"
        assert q["waited"] == origem["esperado"], f"{q['id']}: filtros alterados"


def test_topico_valido_e_coerente_com_o_id(doc):
    esperado = {"A": "factual_direct_lookup", "B": "semantic_cross_source",
                "C": "temporal_comparison", "D": "beneficiary_lookup"}
    for q in doc["queries"]:
        assert q["topic"] in TOPICOS, f"{q['id']}: topic inválido"
        assert q["topic"] == esperado[q["id"][0]], f"{q['id']}: topic incoerente"


def test_respostas_sao_utilizaveis_como_referencia(doc):
    """Resposta não vazia, em uma linha, terminada em ponto."""
    for q in doc["queries"]:
        a = q["answer"]
        assert a and a.strip(), f"{q['id']}: resposta vazia"
        assert len(a) >= 60, f"{q['id']}: resposta curta demais ({len(a)} car)"
        assert "\n" not in a, f"{q['id']}: resposta com quebra de linha"
        assert "  " not in a, f"{q['id']}: resposta com espaço duplo"
        assert a.rstrip().endswith("."), f"{q['id']}: resposta sem ponto final"


def test_respostas_sem_marcadores_de_template_nao_resolvidos(doc):
    """Pega placeholders e defeitos de formatação que escapariam à revisão."""
    proibidos = [
        (r"\bNone\b", "None vazado"),
        (r"\bnan\b", "nan vazado"),
        (r"\{", "chave de f-string não resolvida"),
        (r"\bemendas emendas\b", "palavra repetida"),
        (r"\b1 emendas\b", "concordância de plural"),
        (r"\bde a \b", "contração ausente"),
        (r"\ba o \b", "contração ausente"),
        (r"R\$ \.", "valor monetário malformado"),
    ]
    for q in doc["queries"]:
        for padrao, descricao in proibidos:
            assert not re.search(padrao, q["answer"]), \
                f"{q['id']}: {descricao} — {q['answer'][:120]}"


def test_toda_resposta_tem_evidencia_estruturada(doc):
    for q in doc["queries"]:
        assert "evidencia" in q, f"{q['id']}: sem campo evidencia"
        assert isinstance(q["evidencia"], dict) and q["evidencia"], \
            f"{q['id']}: evidencia vazia"


def test_consultas_sem_dados_declaram_ausencia_na_resposta(doc):
    """Quem não tem dado na base precisa dizer isso, não inventar número."""
    vazias = doc["metadata"]["consultas_sem_dados"]
    por_id = {q["id"]: q for q in doc["queries"]}
    for cid in vazias:
        a = por_id[cid]["answer"].lower()
        assert ("não há" in a or "não registra" in a), \
            f"{cid}: marcada sem dados mas a resposta não declara a ausência"


def test_metadata_descreve_a_base_e_os_criterios(doc):
    md = doc["metadata"]
    for campo in ("description", "created", "source", "base_de_dados",
                  "criterios", "consultas_origem", "gerado_por",
                  "total_consultas", "consultas_sem_dados"):
        assert campo in md, f"metadata sem campo {campo}"
    assert md["total_consultas"] == len(doc["queries"])
    # A base nacional, não a de seed: o gabarito perde o sentido se for gerado
    # contra as 8 emendas de desenvolvimento.
    assert md["base_de_dados"]["emendas"] > 10_000
    assert md["base_de_dados"]["documento_favorecido"] > 100_000


def test_base_do_gabarito_e_a_mesma_das_baterias(doc):
    """O gabarito tem de descrever a MESMA base usada nas rodadas de avaliação.

    Se as contagens divergirem, as referências e as respostas avaliadas vêm de
    universos diferentes e o BERTScore compara coisas incomparáveis.
    """
    pasta = Path(__file__).parent
    baterias = sorted(pasta.glob("*resultados_120_*.json"))
    if not baterias:
        pytest.skip("nenhuma bateria de 120 consultas para comparar")
    gab = doc["metadata"]["base_de_dados"]
    for arquivo in baterias:
        base = json.loads(arquivo.read_text(encoding="utf-8")).get(
            "base_de_dados")
        if not base:
            continue
        for chave in ("emendas", "documentos_emenda", "favorecidos"):
            assert gab.get(chave) == base.get(chave), (
                f"{arquivo.name}: {chave} divergente — gabarito "
                f"{gab.get(chave)} vs bateria {base.get(chave)}")
