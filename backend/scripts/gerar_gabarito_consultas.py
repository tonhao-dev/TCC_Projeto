"""Gera o gabarito (ground truth) das 120 consultas de avaliação.

Para cada consulta de ``tests/test_queries/test_evaluation.py`` este script roda
SQL determinístico sobre a base nacional e redige a resposta de referência em
português. O JSON resultante é a referência usada para calcular BERTScore e
para auditar a veracidade das respostas geradas pelo pipeline RAG.

Critérios adotados (documentados em ``metadata.criterios``):

* **Área temática** → ``emendas.funcao`` igual ao código da função orçamentária
  resolvido pelo dicionário (``data/dictionary.json``). É a classificação
  canônica da emenda, independente de heurísticas de subfunção do pipeline.
* **Valores da emenda** → ``valor_empenhado`` / ``valor_liquidado`` /
  ``valor_pago`` da tabela ``emendas``. A métrica destacada na resposta segue o
  verbo da pergunta (pago, liquidado ou, por omissão, empenhado).
* **Beneficiários** → ``favorecidos`` ligados por ``documento_favorecido``.
  ``documento_favorecido.valor_recebido`` é 100% NULL nesta base, então o valor
  vem de ``documentos_emenda.valor``, separado por fase: ``Pagamento`` (recebido)
  e ``Empenho`` (empenhado). Há exatamente 1 favorecido por documento, portanto
  a soma não duplica.
* **Busca em documentos** → palavra-chave em ``documentos_emenda.observacao``,
  com os filtros de UF/ano da pergunta mas SEM restringir a função, porque a
  pergunta é sobre o objeto efetivamente custeado.
* **Ausência de dados é resposta válida**: quando o SQL não retorna linhas, a
  referência afirma explicitamente que não há registro na base.

Uso:
    py scripts/gerar_gabarito_consultas.py
    py scripts/gerar_gabarito_consultas.py --saida tests/test_queries/gabarito.json
    $env:GABARITO_PG_URL = "postgresql://user:senha@localhost:5433/transparencia_nacional"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg2

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "tests" / "test_queries"))

from test_evaluation import (  # noqa: E402
    CONSULTAS_TIPO_A, CONSULTAS_TIPO_B, CONSULTAS_TIPO_C, CONSULTAS_TIPO_D,
)

HOSTS_LOCAIS = {"localhost", "127.0.0.1", "::1"}
DB_PADRAO = os.environ.get("GABARITO_PG_DB", "transparencia_nacional")
MIN_EMENDAS = 10_000  # trava contra rodar sobre a base de seed (8 linhas)

UF_NOME = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso",
    "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí",
    "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo",
    "TO": "Tocantins",
}

# Preposição locativa correta para cada UF ("no Pará", "em Pernambuco",
# "na Bahia"). Estados cujo nome não leva artigo usam "em".
UF_PREP = {
    "AC": "no", "AL": "em", "AM": "no", "AP": "no", "BA": "na", "CE": "no",
    "DF": "no", "ES": "no", "GO": "em", "MA": "no", "MG": "em", "MS": "no",
    "MT": "no", "PA": "no", "PB": "na", "PE": "em", "PI": "no", "PR": "no",
    "RJ": "no", "RN": "no", "RO": "em", "RR": "em", "RS": "no", "SC": "em",
    "SE": "em", "SP": "em", "TO": "no",
}
UF_ARTIGO = {
    "AC": "o ", "AL": "", "AM": "o ", "AP": "o ", "BA": "a ", "CE": "o ",
    "DF": "o ", "ES": "o ", "GO": "", "MA": "o ", "MG": "", "MS": "o ",
    "MT": "o ", "PA": "o ", "PB": "a ", "PE": "", "PI": "o ", "PR": "o ",
    "RJ": "o ", "RN": "o ", "RO": "", "RR": "", "RS": "o ", "SC": "",
    "SE": "", "SP": "", "TO": "o ",
}

# tipo_emenda na base: 'Emenda Individual - Transferências com Finalidade
# Definida', 'Emenda Individual - Transferências Especiais', 'Emenda de
# Bancada', 'Emenda de Relator', 'Emenda de Comissão'. O padrão ILIKE evita
# o problema de acento em "Comissão".
TIPO_LIKE = {
    "individual": "%Individual%",
    "bancada": "%Bancada%",
    "comissao": "%Comiss%",
    "relator": "%Relator%",
}
TIPO_ROTULO = {
    "individual": "individuais", "bancada": "de bancada",
    "comissao": "de comissão", "relator": "de relator",
}
TIPO_ROTULO_SING = {
    "individual": "individual", "bancada": "de bancada",
    "comissao": "de comissão", "relator": "de relator",
}

TOPICO = {
    "A": "factual_direct_lookup",
    "B": "semantic_cross_source",
    "C": "temporal_comparison",
    "D": "beneficiary_lookup",
}

_DICT = json.loads((_BACKEND / "data" / "dictionary.json").read_text("utf-8"))
MAP_FUNCAO = _DICT["mapeamentos_funcao"]

# Preenchido em main() com emendas.funcao -> emendas.funcao_nome, para que o
# nome da área na resposta seja exatamente o rótulo que a base usa.
FUNCAO_NOME: dict[str, str] = {}


# --------------------------------------------------------------------------
# formatação
# --------------------------------------------------------------------------
def brl(v) -> str:
    """Formata valor em reais no padrão brasileiro."""
    v = float(v or 0)
    base = f"{v:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"R$ {base}"


def num(n) -> str:
    return f"{int(n or 0):,}".replace(",", ".")


def pct(v) -> str:
    return f"{float(v):.1f}".replace(".", ",") + "%"


def vezes(v) -> str:
    return f"{float(v):.1f}".replace(".", ",")


def emendas(n) -> str:
    """'1 emenda' / '25 emendas', com concordância."""
    return f"{num(n)} emenda" if int(n or 0) == 1 else f"{num(n)} emendas"


def cap(s: str) -> str:
    """Maiúscula só na primeira letra, preservando o resto da frase."""
    return s[0].upper() + s[1:] if s else s


def contrair_a(rotulo: str) -> str:
    """'o Fundo' -> 'ao Fundo'; 'a Prefeitura' -> 'à Prefeitura'."""
    for art, contr in (("o ", "ao "), ("a ", "à "), ("os ", "aos "),
                       ("as ", "às ")):
        if rotulo.startswith(art):
            return contr + rotulo[len(art):]
    return f"a {rotulo}"


_CONECTIVOS = {"de", "da", "do", "das", "dos", "e", "em", "no", "na"}
# Siglas que devem permanecer em caixa alta mesmo tendo vogal.
_SIGLAS = {"SA", "S/A", "ME", "EPP", "CIA", "ONG", "APAE", "SUS", "UBS",
           "UPA", "UTI", "CRAS", "CREAS", "EIRELI", "UFRN", "UFF", "IFES"}


def nome_proprio(s: str) -> str:
    """Title case respeitando conectivos, siglas e tokens com dígito.

    A base guarda nomes em caixa alta sem acento ('BANCADA DO PARA'), que em
    prosa fica melhor como 'Bancada do Para'. Siglas conhecidas, siglas curtas
    sem vogal (CM), abreviações pontuadas curtas (S.A.) e tokens com dígito
    (2A) são mantidos em caixa alta.
    """
    saida = []
    for i, tok in enumerate(s.split()):
        baixo = tok.lower()
        if i and baixo in _CONECTIVOS:
            saida.append(baixo)
        elif tok.upper().strip(".,") in _SIGLAS:
            saida.append(tok.upper())
        elif any(ch.isdigit() for ch in tok):
            saida.append(tok.upper())
        elif "." in tok and len(tok) <= 4:
            saida.append(tok.upper())
        elif len(tok) <= 3 and not any(v in baixo for v in "aeiou"):
            saida.append(tok.upper())
        else:
            saida.append(tok.capitalize())
    return " ".join(saida)


def sem_acento(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def enumerar(itens: list[str], conj: str = "e") -> str:
    itens = [i for i in itens if i]
    if not itens:
        return ""
    if len(itens) == 1:
        return itens[0]
    return ", ".join(itens[:-1]) + f" {conj} " + itens[-1]


def local(f: dict) -> str:
    """Trecho locativo já com a preposição correta: 'no Pará (PA)'."""
    uf = f.get("uf")
    if not uf:
        return ""
    return f"{UF_PREP[uf]} {UF_NOME[uf]} ({uf})"


def local_para(f: dict) -> str:
    """Trecho de destino: 'para o Pará (PA)'."""
    uf = f.get("uf")
    if not uf:
        return ""
    return f"para {UF_ARTIGO[uf]}{UF_NOME[uf]} ({uf})".replace("para  ", "para ")


def periodo(f: dict) -> str:
    if f.get("ano"):
        return f"em {f['ano']}"
    if f.get("ano_inicio") and f.get("ano_fim"):
        return f"entre {f['ano_inicio']} e {f['ano_fim']}"
    return "entre 2020 e 2024"


def rotulo_tipo(f: dict, n: int | None = None) -> str:
    t = f.get("tipo_emenda")
    if not t:
        return ""
    tabela = TIPO_ROTULO_SING if n == 1 else TIPO_ROTULO
    return f" {tabela[t]}"


def rotulo_area(f: dict) -> str:
    """Nome da função orçamentária conforme a própria base (funcao_nome)."""
    cod = f.get("funcao")
    if not cod:
        return ""
    nome = FUNCAO_NOME.get(cod) or f.get("area_nome") or cod
    return f" de {nome[0].lower() + nome[1:]}"


def area_nome(f: dict) -> str:
    cod = f.get("funcao")
    if not cod:
        return ""
    nome = FUNCAO_NOME.get(cod) or f.get("area_nome") or cod
    return nome[0].lower() + nome[1:]


# --------------------------------------------------------------------------
# construção de WHERE
# --------------------------------------------------------------------------
def resolver_filtros(esperado: dict) -> dict:
    """Converte o dict 'esperado' da consulta em filtros SQL resolvidos."""
    f = {}
    for k in ("uf", "ano", "ano_inicio", "ano_fim", "partido", "tipo_emenda"):
        if esperado.get(k) is not None:
            f[k] = esperado[k]
    if esperado.get("autor"):
        f["autor"] = esperado["autor"]
    if esperado.get("area"):
        codigo = MAP_FUNCAO.get(esperado["area"].lower().strip())
        if codigo is None:
            raise SystemExit(f"área não mapeada no dicionário: {esperado['area']!r}")
        f["funcao"] = codigo
        f["area_nome"] = esperado["area"]
    if esperado.get("beneficiario"):
        f["beneficiario"] = esperado["beneficiario"]
    return f


def where(f: dict, extra: list[str] | None = None) -> tuple[str, dict]:
    cl, p = [], {}
    if f.get("ano"):
        cl.append("e.ano = %(ano)s")
        p["ano"] = f["ano"]
    elif f.get("ano_inicio") and f.get("ano_fim"):
        cl.append("e.ano BETWEEN %(ano_inicio)s AND %(ano_fim)s")
        p["ano_inicio"], p["ano_fim"] = f["ano_inicio"], f["ano_fim"]
    if f.get("uf"):
        cl.append("e.uf = %(uf)s")
        p["uf"] = f["uf"]
    if f.get("autor"):
        cl.append("UPPER(e.nome_autor) LIKE %(autor)s")
        p["autor"] = f"%{sem_acento(f['autor'])}%"
    if f.get("partido"):
        cl.append("p.partido = %(partido)s")
        p["partido"] = f["partido"]
    if f.get("tipo_emenda"):
        cl.append("e.tipo_emenda ILIKE %(tipo_emenda)s")
        p["tipo_emenda"] = TIPO_LIKE[f["tipo_emenda"]]
    if f.get("funcao"):
        cl.append("e.funcao = %(funcao)s")
        p["funcao"] = f["funcao"]
    cl.extend(extra or [])
    return (" AND ".join(cl) or "1=1"), p


FROM_EMENDAS = ("FROM emendas e "
                "LEFT JOIN parlamentares p ON p.cod_autor = e.cod_autor")
FROM_BENEF = FROM_EMENDAS + """
    JOIN documentos_emenda d ON d.emenda_id = e.id
    JOIN documento_favorecido df ON df.documento_id = d.id
    JOIN favorecidos fa ON fa.id = df.favorecido_id"""

METRICA_COL = {"empenhado": "valor_empenhado", "liquidado": "valor_liquidado",
               "pago": "valor_pago"}
METRICA_VERBO = {"empenhado": "empenhado", "liquidado": "liquidado",
                 "pago": "pago"}


def metrica_da_pergunta(texto: str) -> str:
    t = texto.lower()
    if "liquidad" in t:
        return "liquidado"
    if "pago" in t or "pagas" in t or "pagos" in t or "paga " in t:
        return "pago"
    return "empenhado"


class Ctx:
    """Executor de SQL que registra todas as consultas usadas."""

    def __init__(self, cur):
        self.cur = cur
        self.sqls: list[str] = []

    def run(self, sql: str, params: dict | None = None) -> list[tuple]:
        self.cur.execute(sql, params or {})
        self.sqls.append(" ".join(
            (self.cur.mogrify(sql, params or {}).decode("utf-8")).split()))
        return self.cur.fetchall()

    def one(self, sql: str, params: dict | None = None) -> tuple:
        r = self.run(sql, params)
        return r[0] if r else ()


def agg(ctx: Ctx, f: dict, extra=None) -> dict:
    w, p = where(f, extra)
    r = ctx.one(f"""
        SELECT COUNT(*),
               COALESCE(SUM(e.valor_empenhado),0),
               COALESCE(SUM(e.valor_liquidado),0),
               COALESCE(SUM(e.valor_pago),0),
               COALESCE(AVG(e.valor_empenhado),0),
               COUNT(DISTINCT e.nome_autor)
        {FROM_EMENDAS} WHERE {w}""", p)
    return {"qtd": int(r[0]), "empenhado": float(r[1]),
            "liquidado": float(r[2]), "pago": float(r[3]),
            "media_empenhado": float(r[4]), "autores": int(r[5])}


def serie_anual(ctx: Ctx, f: dict, col="valor_empenhado") -> list[dict]:
    w, p = where(f)
    rows = ctx.run(f"""
        SELECT e.ano, COUNT(*), COALESCE(SUM(e.{col}),0)
        {FROM_EMENDAS} WHERE {w} GROUP BY e.ano ORDER BY e.ano""", p)
    return [{"ano": int(a), "qtd": int(n), "valor": float(v)} for a, n, v in rows]


def ranking(ctx: Ctx, f: dict, dim: str, n: int = 5,
            col="valor_empenhado") -> list[dict]:
    """dim: autor | partido | funcao | tipo | uf."""
    expr = {"autor": "e.nome_autor", "partido": "p.partido",
            "funcao": "COALESCE(NULLIF(e.funcao_nome,''),'Não informada')",
            "tipo": "e.tipo_emenda", "uf": "e.uf"}[dim]
    w, p = where(f)
    p["lim"] = n
    rows = ctx.run(f"""
        SELECT {expr} AS k, COUNT(*), COALESCE(SUM(e.{col}),0) v
        {FROM_EMENDAS} WHERE {w} AND {expr} IS NOT NULL
        GROUP BY k ORDER BY v DESC LIMIT %(lim)s""", p)
    return [{"chave": k, "qtd": int(q), "valor": float(v)} for k, q, v in rows]


def benef(ctx: Ctx, f: dict, like: str | None = None, n: int = 10,
          tipo_pessoa: str | None = None, minimo: float | None = None,
          ) -> tuple[list[dict], dict]:
    """Favorecidos agregados por nome, com valores de pagamento e empenho."""
    extra, w_extra = [], {}
    if like:
        ors = []
        for i, pat in enumerate(like.split("|")):
            ors.append(f"UPPER(fa.nome) LIKE %(like_{i})s")
            w_extra[f"like_{i}"] = f"%{sem_acento(pat)}%"
        extra.append("(" + " OR ".join(ors) + ")")
    if tipo_pessoa:
        extra.append("fa.tipo_pessoa = %(tp)s")
        w_extra["tp"] = tipo_pessoa
    extra.append("fa.nome NOT ILIKE '%%sem informa%%'")
    w, p = where(f, extra)
    p.update(w_extra)
    p["lim"] = n

    having = ""
    if minimo is not None:
        having = ("HAVING SUM(CASE WHEN d.fase='Pagamento' "
                  "THEN COALESCE(d.valor,0) ELSE 0 END) >= %(minimo)s")
        p["minimo"] = minimo

    rows = ctx.run(f"""
        SELECT fa.nome, fa.tipo_pessoa, COUNT(DISTINCT e.id),
               SUM(CASE WHEN d.fase='Pagamento' THEN COALESCE(d.valor,0)
                        ELSE 0 END) pago,
               SUM(CASE WHEN d.fase='Empenho' THEN COALESCE(d.valor,0)
                        ELSE 0 END) empenhado
        {FROM_BENEF} WHERE {w}
        GROUP BY fa.nome, fa.tipo_pessoa {having}
        ORDER BY pago DESC, empenhado DESC LIMIT %(lim)s""", p)
    lista = [{"nome": nm, "tipo_pessoa": tp, "emendas": int(q),
              "pago": float(pg), "empenhado": float(em)}
             for nm, tp, q, pg, em in rows]

    # O total tem de respeitar o mesmo HAVING, senão conta favorecidos que
    # ficaram fora do recorte (ex.: abaixo do limiar de R$ 1 milhão). A CTE
    # isola os nomes qualificados e o COUNT(DISTINCT e.id) evita contar a
    # mesma emenda uma vez por favorecido.
    p_tot = {k: v for k, v in p.items() if k != "lim"}
    tot = ctx.one(f"""
        WITH qualificados AS (
            SELECT fa.nome {FROM_BENEF} WHERE {w}
            GROUP BY fa.nome {having}
        )
        SELECT COUNT(DISTINCT fa.nome), COUNT(DISTINCT e.id),
               SUM(CASE WHEN d.fase='Pagamento' THEN COALESCE(d.valor,0)
                        ELSE 0 END),
               SUM(CASE WHEN d.fase='Empenho' THEN COALESCE(d.valor,0)
                        ELSE 0 END)
        {FROM_BENEF}
        JOIN qualificados q ON q.nome = fa.nome
        WHERE {w}""", p_tot)
    total = {"favorecidos": int(tot[0] or 0), "emendas": int(tot[1] or 0),
             "pago": float(tot[2] or 0), "empenhado": float(tot[3] or 0)}
    return lista, total


def docs_palavra_chave(ctx: Ctx, f: dict, kws: list[str], n: int = 8,
                       ) -> tuple[list[dict], dict]:
    """Emendas cujos documentos de despesa mencionam as palavras-chave."""
    f_sem_funcao = {k: v for k, v in f.items() if k not in ("funcao",)}
    ors, pk = [], {}
    for i, kw in enumerate(kws):
        ors.append(f"LOWER(d.observacao) LIKE %(kw_{i})s")
        pk[f"kw_{i}"] = f"%{kw.lower()}%"
    w, p = where(f_sem_funcao, ["(" + " OR ".join(ors) + ")"])
    p.update(pk)
    p["lim"] = n

    base_from = FROM_EMENDAS + " JOIN documentos_emenda d ON d.emenda_id = e.id"
    rows = ctx.run(f"""
        SELECT e.codigo_emenda, e.nome_autor, p.partido, e.ano,
               COALESCE(NULLIF(e.funcao_nome,''),'Não informada'),
               e.localidade, e.valor_empenhado, e.valor_pago,
               COUNT(d.id)
        {base_from} WHERE {w}
        GROUP BY e.codigo_emenda, e.nome_autor, p.partido, e.ano,
                 e.funcao_nome, e.localidade, e.valor_empenhado, e.valor_pago
        ORDER BY e.valor_empenhado DESC LIMIT %(lim)s""", p)
    lista = [{"codigo": c, "autor": a, "partido": pt, "ano": int(an),
              "funcao": fn, "localidade": lo, "empenhado": float(ve),
              "pago": float(vp), "documentos": int(nd)}
             for c, a, pt, an, fn, lo, ve, vp, nd in rows]

    tot = ctx.one(f"""
        SELECT COUNT(DISTINCT e.id), COUNT(d.id),
               COALESCE(SUM(DISTINCT e.valor_empenhado),0)
        {base_from} WHERE {w}""",
                  {k: v for k, v in p.items() if k != "lim"})
    return lista, {"emendas": int(tot[0] or 0), "documentos": int(tot[1] or 0)}


# --------------------------------------------------------------------------
# diagnóstico de resultado vazio
# --------------------------------------------------------------------------
def diagnostico_vazio(ctx: Ctx, f: dict) -> tuple[str, dict]:
    """Explica POR QUE o filtro não retornou nada.

    Remove um filtro de cada vez e reporta apenas aqueles cuja remoção faz o
    recorte deixar de ser vazio — isto é, os filtros que de fato bloquearam o
    resultado. Distingue "o fato não existe" de "a base não cobre esse
    recorte", o que é essencial para julgar se uma resposta do RAG alucinou ou
    apenas refletiu uma lacuna de dados.
    """
    ev: dict = {}
    base = {k: v for k, v in f.items()
            if k in ("uf", "ano", "ano_inicio", "ano_fim")}
    a_base = agg(ctx, base)
    ev["emendas_no_recorte_uf_ano"] = a_base["qtd"]
    if a_base["qtd"] == 0:
        return (f"a base não registra nenhuma emenda {local(f)} "
                f"{periodo(f)}"), ev

    culpados = []
    for chave in ("partido", "funcao", "tipo_emenda", "autor"):
        if not f.get(chave):
            continue
        relaxado = {k: v for k, v in f.items() if k != chave}
        if agg(ctx, relaxado)["qtd"] > 0:
            culpados.append(chave)

    frases = []
    if "partido" in culpados:
        rk = ranking(ctx, {k: v for k, v in f.items() if k != "partido"},
                     "partido", 6)
        ev["partidos_presentes"] = [{"partido": r["chave"], "qtd": r["qtd"]}
                                    for r in rk]
        presentes = [r["chave"] for r in rk if r["chave"]]
        if presentes:
            frases.append(f"no recorte há emendas de "
                          f"{enumerar(presentes)}, mas nenhuma do "
                          f"{f['partido']}")
        else:
            frases.append(f"nenhuma emenda do recorte tem partido "
                          f"identificado, então não há como atribuí-las ao "
                          f"{f['partido']}")
        sem_autor = agg(ctx, {k: v for k, v in f.items() if k != "partido"},
                        ["e.cod_autor IS NULL"])
        ev["emendas_sem_autor_identificado"] = sem_autor["qtd"]
        if sem_autor["qtd"]:
            frases.append(f"outras {emendas(sem_autor['qtd'])} do recorte não "
                          f"têm parlamentar autor identificado na base e "
                          f"portanto não podem ser atribuídas a partido algum")
    if "funcao" in culpados:
        rk = ranking(ctx, {k: v for k, v in f.items() if k != "funcao"},
                     "funcao", 5)
        ev["areas_presentes"] = [{"area": r["chave"], "qtd": r["qtd"]}
                                 for r in rk]
        if rk:
            frases.append(f"as áreas presentes no recorte são "
                          f"{enumerar([r['chave'] for r in rk])}, sem "
                          f"{area_nome(f)}")
    if "tipo_emenda" in culpados:
        rk = ranking(ctx, {k: v for k, v in f.items() if k != "tipo_emenda"},
                     "tipo", 6)
        ev["tipos_presentes"] = [{"tipo": r["chave"], "qtd": r["qtd"]}
                                 for r in rk]
        if rk:
            frases.append(f"os tipos presentes no recorte são "
                          f"{enumerar([r['chave'] for r in rk])}")
    if "autor" in culpados:
        frases.append(f"{nome_proprio(f['autor'])} não tem emendas nesse "
                      f"recorte")
    if not frases and a_base["qtd"]:
        frases.append(f"o recorte de local e período tem "
                      f"{emendas(a_base['qtd'])}, mas a combinação de filtros "
                      f"da pergunta não é satisfeita por nenhuma delas")
    ev["filtros_bloqueantes"] = culpados
    return "; ".join(frases), ev


def diagnostico_benef(ctx: Ctx, f: dict,
                      alvo: str = "o beneficiário consultado",
                      ) -> tuple[str, dict]:
    """Explica a ausência de beneficiários: filtro vazio ou elo não coletado."""
    a = agg(ctx, f)
    w, p = where(f)
    com_elo = ctx.one(f"""
        SELECT COUNT(DISTINCT e.id) {FROM_BENEF} WHERE {w}""", p)
    n_elo = int(com_elo[0] or 0) if com_elo else 0
    ev = {"emendas_no_filtro": a["qtd"], "emendas_com_favorecido": n_elo}
    if a["qtd"] == 0:
        frase, ev_d = diagnostico_vazio(ctx, f)
        ev.update(ev_d)
        return frase, ev
    if n_elo == 0:
        return (f"o recorte tem {emendas(a['qtd'])}, mas nenhuma delas possui "
                f"beneficiário final identificado na base — os documentos de "
                f"despesa com favorecido não foram coletados para essas "
                f"emendas"), ev
    quantas = ("todas com beneficiário identificado" if n_elo == a["qtd"]
               else f"{num(n_elo)} com beneficiário identificado")
    return (f"o recorte tem {emendas(a['qtd'])}, {quantas}, mas nenhum deles "
            f"corresponde {contrair_a(alvo)}"), ev


# --------------------------------------------------------------------------
# handlers: cada um devolve (texto_da_resposta, evidencia)
# --------------------------------------------------------------------------
def _sujeito(f: dict, n: int | None = None) -> str:
    """'emendas individuais de saúde do PT de X no Ceará (CE)'.

    Com ``n=1`` concorda no singular ('1 emenda individual'). O resultado já
    começa com a palavra 'emenda(s)', então pode ser prefixado por um número.
    """
    palavra = "emenda" if n == 1 else "emendas"
    alvo = f"{palavra}{rotulo_tipo(f, n)}{rotulo_area(f)}"
    if f.get("partido"):
        alvo += f" do {f['partido']}"
    if f.get("autor"):
        alvo += f" de autoria de {nome_proprio(f['autor'])}"
    if f.get("uf"):
        alvo += f" {local(f)}"
    return alvo


def _nada(ctx, f, cauda: str) -> str:
    motivo, ev_d = diagnostico_vazio(ctx, f)
    txt = f"Não há registro de {_sujeito(f)} {periodo(f)} na base consultada"
    if motivo:
        txt += f": {motivo}"
    return txt + f". {cauda}".rstrip(), ev_d


def h_contagem(ctx, f, c, exige=None, **_):
    a = agg(ctx, f)
    ev = {"quantidade": a["qtd"], "empenhado": a["empenhado"],
          "liquidado": a["liquidado"], "pago": a["pago"]}
    if a["qtd"] == 0:
        txt, ev_d = _nada(ctx, f, "A contagem correta é 0.")
        ev.update(ev_d)
        return txt, ev
    verbo = "Foi registrada" if a["qtd"] == 1 else "Foram registradas"
    txt = (f"{verbo} {num(a['qtd'])} {_sujeito(f, a['qtd'])} {periodo(f)}, "
           f"somando {brl(a['empenhado'])} em valor empenhado e "
           f"{brl(a['pago'])} em valor pago.")
    if exige:
        extra = {"empenhado": ["e.valor_empenhado > 0"],
                 "pago": ["e.valor_pago > 0"]}[exige]
        sub = agg(ctx, f, extra)
        ev["com_movimento"] = sub["qtd"]
        mov = "empenho" if exige == "empenhado" else "pagamento"
        if sub["qtd"] == a["qtd"]:
            txt += (f" Todas elas têm {mov} efetivamente registrado, com "
                    f"valor maior que zero.")
        else:
            txt += (f" Dessas, {emendas(sub['qtd'])} têm {mov} efetivamente "
                    f"registrado, com valor maior que zero.")
    return txt, ev


def h_contagem_distinta(ctx, f, c, **_):
    a = agg(ctx, f)
    ev = {"parlamentares_distintos": a["autores"], "quantidade": a["qtd"],
          "empenhado": a["empenhado"]}
    if a["autores"] == 0:
        txt, ev_d = _nada(ctx, f, "A contagem correta é 0 parlamentares.")
        ev.update(ev_d)
        return txt, ev
    return (f"{num(a['autores'])} parlamentares distintos tiveram "
            f"{_sujeito(f)} {periodo(f)}, em um total de {emendas(a['qtd'])} "
            f"e {brl(a['empenhado'])} empenhados."), ev


def h_soma(ctx, f, c, **_):
    m = metrica_da_pergunta(c["consulta"])
    a = agg(ctx, f)
    ev = {"quantidade": a["qtd"], "empenhado": a["empenhado"],
          "liquidado": a["liquidado"], "pago": a["pago"], "metrica": m}
    if a["qtd"] == 0:
        txt, ev_d = _nada(ctx, f, f"O valor {METRICA_VERBO[m]} correto é "
                                  f"R$ 0,00.")
        ev.update(ev_d)
        return txt, ev
    return (f"O valor {METRICA_VERBO[m]} em {_sujeito(f)} {periodo(f)} é "
            f"{brl(a[m])}, distribuído em {emendas(a['qtd'])}. Para "
            f"comparação, o total empenhado no mesmo recorte é "
            f"{brl(a['empenhado'])} e o total pago é {brl(a['pago'])}."), ev


def h_media(ctx, f, c, **_):
    a = agg(ctx, f)
    ev = {"quantidade": a["qtd"], "media_empenhado": a["media_empenhado"],
          "empenhado": a["empenhado"]}
    if a["qtd"] == 0:
        txt, ev_d = _nada(ctx, f, "Não há valor médio a calcular.")
        ev.update(ev_d)
        return txt, ev
    return (f"O valor médio das {_sujeito(f)} {periodo(f)} é "
            f"{brl(a['media_empenhado'])} por emenda, resultado de "
            f"{brl(a['empenhado'])} empenhados em {emendas(a['qtd'])}."), ev


def h_lista_emendas(ctx, f, c, n=5, **_):
    a = agg(ctx, f)
    top = ranking(ctx, f, "autor", n)
    areas = ranking(ctx, f, "funcao", 3)
    ev = {"quantidade": a["qtd"], "empenhado": a["empenhado"],
          "pago": a["pago"], "top_autores": top, "top_areas": areas}
    if a["qtd"] == 0:
        txt, ev_d = _nada(ctx, f, "")
        ev.update(ev_d)
        return txt, ev
    frase_autor = enumerar([f"{nome_proprio(t['chave'])} ({brl(t['valor'])})"
                            for t in top[:3]])
    txt = (f"A base registra {num(a['qtd'])} {_sujeito(f, a['qtd'])} "
           f"{periodo(f)}, com {brl(a['empenhado'])} empenhados "
           f"e {brl(a['pago'])} pagos.")
    if not f.get("funcao") and areas:
        frase_area = enumerar([f"{x['chave']} ({brl(x['valor'])})"
                               for x in areas])
        txt += f" As áreas que mais receberam recursos foram {frase_area}."
    txt += f" Os maiores autores por valor empenhado são {frase_autor}."
    return txt, ev


def h_lista_autores(ctx, f, c, n=10, **_):
    a = agg(ctx, f)
    top = ranking(ctx, f, "autor", n)
    ev = {"parlamentares_distintos": a["autores"], "quantidade": a["qtd"],
          "empenhado": a["empenhado"], "parlamentares": top}
    if a["qtd"] == 0:
        txt, ev_d = _nada(ctx, f, "")
        ev.update(ev_d)
        return txt, ev
    nomes = enumerar([f"{nome_proprio(t['chave'])} ({brl(t['valor'])})"
                      for t in top])
    abre = ("São eles" if a["autores"] <= len(top)
            else f"Os {len(top)} maiores por valor empenhado são")
    return (f"{num(a['autores'])} parlamentares tiveram {_sujeito(f)} "
            f"{periodo(f)}, somando {brl(a['empenhado'])} empenhados em "
            f"{emendas(a['qtd'])}. {abre}: {nomes}."), ev


def h_ranking(ctx, f, c, dim="autor", n=5, **_):
    m = metrica_da_pergunta(c["consulta"])
    top = ranking(ctx, f, dim, n, METRICA_COL[m])
    a = agg(ctx, f)
    rot_dim, artigo = {
        "autor": ("parlamentar", "O"), "partido": ("partido", "O"),
        "funcao": ("área temática", "A"), "tipo": ("tipo de emenda", "O"),
        "uf": ("estado", "O"),
    }[dim]
    ev = {"dimensao": dim, "metrica": m, "ranking": top,
          "quantidade": a["qtd"], "empenhado": a["empenhado"]}
    if not top:
        # Duas causas distintas: o recorte é vazio, ou o recorte tem emendas
        # mas nenhuma delas tem a dimensão do ranking preenchida (o caso
        # comum é emenda de bancada/relator sem cod_autor, logo sem partido).
        if a["qtd"] > 0:
            sem_dim = {"autor": "parlamentar autor identificado",
                       "partido": "partido identificado",
                       "funcao": "função orçamentária informada",
                       "tipo": "tipo informado",
                       "uf": "UF informada"}[dim]
            ev["sem_dimensao"] = a["qtd"]
            return (f"O recorte tem {num(a['qtd'])} "
                    f"{_sujeito(f, a['qtd'])} {periodo(f)}, com "
                    f"{brl(a[m])} {METRICA_VERBO[m]}s, mas nenhuma delas tem "
                    f"{sem_dim} na base — portanto não é possível apontar o "
                    f"{rot_dim} que mais destinou recursos."), ev
        txt, ev_d = _nada(ctx, f, "Não há ranking a apresentar.")
        ev.update(ev_d)
        return txt, ev

    def rotular(chave):
        return nome_proprio(chave) if dim == "autor" else chave

    corpo = enumerar([f"{rotular(t['chave'])} ({brl(t['valor'])} em "
                      f"{emendas(t['qtd'])})" for t in top])
    pri = top[0]
    if len(top) == 1:
        return (f"{artigo} único {rot_dim} com {_sujeito(f)} {periodo(f)} é "
                f"{rotular(pri['chave'])}, com {brl(pri['valor'])} "
                f"{METRICA_VERBO[m]}s em {emendas(pri['qtd'])}."), ev
    verbo = "recebeu" if dim in ("funcao", "uf") else "destinou"
    return (f"{artigo} {rot_dim} que mais {verbo} {_sujeito(f)} {periodo(f)} "
            f"é {rotular(pri['chave'])}, com {brl(pri['valor'])} "
            f"{METRICA_VERBO[m]}s em {emendas(pri['qtd'])}. "
            f"O ranking dos {len(top)} maiores é: {corpo}."), ev


def h_tendencia(ctx, f, c, **_):
    m = metrica_da_pergunta(c["consulta"])
    serie = serie_anual(ctx, f, METRICA_COL[m])
    ev = {"metrica": m, "serie": serie}
    if not serie:
        txt, ev_d = _nada(ctx, f, "Não há série temporal a analisar.")
        ev.update(ev_d)
        return txt, ev
    detalhe = "; ".join(f"{s['ano']}: {brl(s['valor'])} em "
                        f"{emendas(s['qtd'])}" for s in serie)
    pri, ult = serie[0], serie[-1]
    if pri["valor"] > 0:
        var = (ult["valor"] - pri["valor"]) / pri["valor"] * 100
        ev["variacao_pct"] = var
        if var > 5:
            dir_ = f"houve aumento de {pct(var)}"
        elif var < -5:
            dir_ = f"houve redução de {pct(abs(var))}"
        else:
            dir_ = f"o valor ficou praticamente estável, variação de {pct(var)}"
    else:
        dir_ = ("não havia valor registrado no ano inicial, então não há base "
                "de comparação percentual")
    maior = max(serie, key=lambda s: s["valor"])
    ev["ano_pico"] = maior["ano"]
    anos_faltando = [a for a in range(pri["ano"], ult["ano"] + 1)
                     if a not in {s["ano"] for s in serie}]
    ev["anos_sem_registro"] = anos_faltando
    cauda = (f" Os anos {enumerar([str(a) for a in anos_faltando])} não têm "
             f"registro nesse recorte." if anos_faltando else "")
    return (f"Comparando {pri['ano']} ({brl(pri['valor'])}) com {ult['ano']} "
            f"({brl(ult['valor'])}) em {_sujeito(f)}, {dir_}. O pico ocorreu "
            f"em {maior['ano']}, com {brl(maior['valor'])}. Série anual por "
            f"valor {METRICA_VERBO[m]} — {detalhe}.{cauda}"), ev


def h_comparar_areas(ctx, f, c, areas=(), **_):
    m = metrica_da_pergunta(c["consulta"])
    blocos, ev_itens = [], []
    for termo in areas:
        cod = MAP_FUNCAO[termo.lower()]
        fa = dict(f, funcao=cod, area_nome=termo)
        a = agg(ctx, fa)
        rot = area_nome(fa)
        blocos.append((rot, a))
        ev_itens.append({"area": rot, "funcao": cod, "quantidade": a["qtd"],
                         "valor": a[m]})
    ev = {"metrica": m, "comparacao": ev_itens}
    rotulos = [b[0] for b in blocos]
    if all(b[1]["qtd"] == 0 for b in blocos):
        return (f"Não há emendas de {enumerar(rotulos)} {local(f)} "
                f"{periodo(f)} na base consultada: o valor {METRICA_VERBO[m]} "
                f"é R$ 0,00 nas duas áreas."), ev
    partes = [f"{rot} com {brl(a[m])} em {emendas(a['qtd'])}"
              if a["qtd"] else f"{rot} sem nenhum registro"
              for rot, a in blocos]
    ordenado = sorted(blocos, key=lambda b: b[1][m], reverse=True)
    lider, ultimo = ordenado[0], ordenado[-1]
    if ultimo[1][m] > 0:
        comp = (f"{cap(lider[0])} recebeu "
                f"{vezes(lider[1][m] / ultimo[1][m])} vezes mais que "
                f"{ultimo[0]}")
    else:
        comp = (f"{cap(lider[0])} concentra todo o valor, pois "
                f"{ultimo[0]} não registra valor {METRICA_VERBO[m]} no recorte")
    return (f"{cap(local(f))} {periodo(f)}, a comparação por valor "
            f"{METRICA_VERBO[m]} é: {enumerar(partes)}. {comp}."), ev


def h_comparar_partidos(ctx, f, c, partidos=(), **_):
    m = metrica_da_pergunta(c["consulta"])
    blocos, ev_itens = [], []
    for sigla in partidos:
        a = agg(ctx, dict(f, partido=sigla))
        blocos.append((sigla, a))
        ev_itens.append({"partido": sigla, "quantidade": a["qtd"],
                         "valor": a[m]})
    ev = {"metrica": m, "comparacao": ev_itens}
    if all(b[1]["qtd"] == 0 for b in blocos):
        motivo, ev_d = diagnostico_vazio(ctx, dict(f, partido=partidos[0]))
        ev.update(ev_d)
        return (f"Nem {enumerar(list(partidos))} registram emendas"
                f"{rotulo_area(f)} {local(f)} {periodo(f)} na base "
                f"consultada" + (f": {motivo}." if motivo else ".")), ev
    partes = [f"{s} com {brl(a[m])} em {emendas(a['qtd'])}" for s, a in blocos]
    ordenado = sorted(blocos, key=lambda b: b[1][m], reverse=True)
    lider = ordenado[0]
    return (f"Em emendas{rotulo_area(f)} {local(f)} {periodo(f)}, o "
            f"{lider[0]} lidera com {brl(lider[1][m])} {METRICA_VERBO[m]}s. "
            f"Comparativo — {enumerar(partes)}."), ev


def h_comparar_tipos(ctx, f, c, tipos=(), **_):
    m = metrica_da_pergunta(c["consulta"])
    blocos, ev_itens = [], []
    for t in tipos:
        a = agg(ctx, dict(f, tipo_emenda=t))
        blocos.append((TIPO_ROTULO[t], a))
        ev_itens.append({"tipo": t, "quantidade": a["qtd"], "valor": a[m]})
    ev = {"metrica": m, "comparacao": ev_itens}
    if all(b[1]["qtd"] == 0 for b in blocos):
        return (f"Não há emendas dos tipos consultados {local(f)} "
                f"{periodo(f)} na base."), ev
    partes = [f"emendas {rot} com {brl(a[m])} em {emendas(a['qtd'])}"
              if a["qtd"] else f"emendas {rot} sem nenhum registro"
              for rot, a in blocos]
    ordenado = sorted(blocos, key=lambda b: b[1][m], reverse=True)
    return (f"{cap(local(f))} {periodo(f)}, as emendas "
            f"{ordenado[0][0]} predominam, com {brl(ordenado[0][1][m])} "
            f"{METRICA_VERBO[m]}s. Comparativo — {enumerar(partes)}."), ev


def h_comparar_ufs(ctx, f, c, ufs=(), **_):
    m = metrica_da_pergunta(c["consulta"])
    blocos, ev_itens = [], []
    for uf in ufs:
        a = agg(ctx, dict(f, uf=uf))
        blocos.append((uf, a))
        ev_itens.append({"uf": uf, "quantidade": a["qtd"], "valor": a[m],
                         "media_empenhado": a["media_empenhado"]})
    ev = {"metrica": m, "comparacao": ev_itens}
    partes = [f"{UF_NOME[uf]} ({uf}) com {brl(a[m])} em {emendas(a['qtd'])}"
              for uf, a in blocos]
    ordenado = sorted(blocos, key=lambda b: b[1][m], reverse=True)
    lider, ultimo = ordenado[0], ordenado[-1]
    dif = lider[1][m] - ultimo[1][m]
    return (f"{cap(periodo(f))}, {enumerar(partes)}. "
            f"{UF_NOME[lider[0]]} lidera em valor {METRICA_VERBO[m]}, com "
            f"{brl(dif)} mais que {UF_NOME[ultimo[0]]}."), ev


def h_execucao(ctx, f, c, **_):
    a = agg(ctx, f)
    taxa = (a["pago"] / a["empenhado"] * 100) if a["empenhado"] else 0.0
    ev = {"empenhado": a["empenhado"], "liquidado": a["liquidado"],
          "pago": a["pago"], "taxa_pago_sobre_empenhado": taxa,
          "quantidade": a["qtd"]}
    if a["qtd"] == 0:
        txt, ev_d = _nada(ctx, f, "Não há proporção a calcular.")
        ev.update(ev_d)
        return txt, ev
    return (f"{cap(local(f))} {periodo(f)}, foram empenhados "
            f"{brl(a['empenhado'])} e pagos {brl(a['pago'])} em "
            f"{emendas(a['qtd'])}, o que corresponde a uma taxa de execução "
            f"(pago sobre empenhado) de {pct(taxa)}. O valor liquidado "
            f"registrado é {brl(a['liquidado'])}."), ev


def h_tendencia_execucao(ctx, f, c, **_):
    emp = {s["ano"]: s["valor"] for s in serie_anual(ctx, f, "valor_empenhado")}
    pg = {s["ano"]: s["valor"] for s in serie_anual(ctx, f, "valor_pago")}
    serie = [{"ano": a, "empenhado": emp[a], "pago": pg.get(a, 0.0),
              "taxa": (pg.get(a, 0.0) / emp[a] * 100) if emp[a] else 0.0}
             for a in sorted(emp)]
    ev = {"serie": serie}
    if not serie:
        txt, ev_d = _nada(ctx, f, "")
        ev.update(ev_d)
        return txt, ev
    detalhe = "; ".join(f"{s['ano']}: {pct(s['taxa'])} ({brl(s['pago'])} "
                        f"pagos de {brl(s['empenhado'])} empenhados)"
                        for s in serie)
    pri, ult = serie[0], serie[-1]
    delta = ult["taxa"] - pri["taxa"]
    ev["delta_pontos_percentuais"] = delta
    if delta > 2:
        dir_ = f"a taxa melhorou {vezes(delta)} pontos percentuais"
    elif delta < -2:
        dir_ = f"a taxa piorou {vezes(abs(delta))} pontos percentuais"
    else:
        dir_ = "a taxa ficou praticamente estável"
    return (f"A taxa de execução (pago sobre empenhado) {local(f)} passou de "
            f"{pct(pri['taxa'])} em {pri['ano']} para {pct(ult['taxa'])} em "
            f"{ult['ano']}: {dir_}. Detalhamento anual — {detalhe}."), ev


def h_tendencia_media(ctx, f, c, **_):
    w, p = where(f)
    rows = ctx.run(f"""
        SELECT e.ano, COUNT(*), COALESCE(AVG(e.valor_empenhado),0),
               COALESCE(SUM(e.valor_empenhado),0)
        {FROM_EMENDAS} WHERE {w} GROUP BY e.ano ORDER BY e.ano""", p)
    serie = [{"ano": int(a), "qtd": int(n), "media": float(md),
              "total": float(t)} for a, n, md, t in rows]
    ev = {"serie": serie}
    if not serie:
        txt, ev_d = _nada(ctx, f, "")
        ev.update(ev_d)
        return txt, ev
    detalhe = "; ".join(f"{s['ano']}: {brl(s['media'])} em "
                        f"{emendas(s['qtd'])}" for s in serie)
    pri, ult = serie[0], serie[-1]
    var = ((ult["media"] - pri["media"]) / pri["media"] * 100) \
        if pri["media"] else 0.0
    ev["variacao_pct"] = var
    dir_ = ("cresceu" if var > 5 else "caiu" if var < -5
            else "ficou praticamente estável")
    return (f"O valor médio por emenda {local(f)} {dir_} de "
            f"{brl(pri['media'])} em {pri['ano']} para {brl(ult['media'])} em "
            f"{ult['ano']}, variação de {pct(var)}. Série anual — "
            f"{detalhe}."), ev


def h_tendencia_dim(ctx, f, c, dim="funcao", n=5, **_):
    """Crescimento por dimensão entre o primeiro e o último ano do intervalo."""
    m = metrica_da_pergunta(c["consulta"])
    col = METRICA_COL[m]
    a0, a1 = f["ano_inicio"], f["ano_fim"]
    expr = {"funcao": "COALESCE(NULLIF(e.funcao_nome,''),'Não informada')",
            "autor": "e.nome_autor", "partido": "p.partido"}[dim]
    w, p = where(f)
    p.update({"a0": a0, "a1": a1, "lim": n})
    rows = ctx.run(f"""
        SELECT {expr} AS k,
               COALESCE(SUM(CASE WHEN e.ano=%(a0)s THEN e.{col} END),0) ini,
               COALESCE(SUM(CASE WHEN e.ano=%(a1)s THEN e.{col} END),0) fim
        {FROM_EMENDAS} WHERE {w} AND {expr} IS NOT NULL
        GROUP BY k
        ORDER BY (COALESCE(SUM(CASE WHEN e.ano=%(a1)s THEN e.{col} END),0)
                  - COALESCE(SUM(CASE WHEN e.ano=%(a0)s THEN e.{col} END),0))
                 DESC LIMIT %(lim)s""", p)
    itens = [{"chave": k, f"valor_{a0}": float(i), f"valor_{a1}": float(fi),
              "crescimento": float(fi) - float(i)} for k, i, fi in rows]
    ev = {"metrica": m, "dimensao": dim, "crescimento": itens}
    rot = {"funcao": "áreas temáticas", "autor": "parlamentares",
           "partido": "partidos"}[dim]
    if not itens:
        txt, ev_d = _nada(ctx, f, "")
        ev.update(ev_d)
        return txt, ev

    def rotular(k):
        return nome_proprio(k) if dim == "autor" else k

    partes = [f"{rotular(x['chave'])} (de {brl(x[f'valor_{a0}'])} para "
              f"{brl(x[f'valor_{a1}'])}, variação de {brl(x['crescimento'])})"
              for x in itens]
    pri = itens[0]
    escopo = f"em emendas{rotulo_area(f)}" if f.get("funcao") else "em emendas"
    return (f"Entre {a0} e {a1}, {local(f)}, o maior crescimento em valor "
            f"{METRICA_VERBO[m]} {escopo} entre {rot} foi de "
            f"{rotular(pri['chave'])}, com alta de "
            f"{brl(pri['crescimento'])}. Os {len(itens)} maiores crescimentos "
            f"são: {enumerar(partes)}."), ev


def h_serie_tipos(ctx, f, c, tipos=(), **_):
    m = metrica_da_pergunta(c["consulta"])
    series, ev_itens, partes = [], [], []
    for t in tipos:
        s = serie_anual(ctx, dict(f, tipo_emenda=t), METRICA_COL[m])
        series.append((TIPO_ROTULO[t], s))
        ev_itens.append({"tipo": t, "serie": s})
    ev = {"metrica": m, "series": ev_itens}
    for rot, s in series:
        if not s:
            partes.append(f"emendas {rot} não aparecem no recorte")
            continue
        var = (((s[-1]["valor"] - s[0]["valor"]) / s[0]["valor"] * 100)
               if s[0]["valor"] else 0.0)
        partes.append(f"emendas {rot} foram de {brl(s[0]['valor'])} em "
                      f"{s[0]['ano']} para {brl(s[-1]['valor'])} em "
                      f"{s[-1]['ano']}, variação de {pct(var)}")
    return (f"{cap(local(f))}, {periodo(f)}: "
            f"{enumerar(partes)}."), ev


def h_distribuicao_tipos(ctx, f, c, anos=(), **_):
    m = metrica_da_pergunta(c["consulta"])
    ev_itens, blocos = [], []
    for ano in anos:
        fa = {k: v for k, v in f.items()
              if k not in ("ano_inicio", "ano_fim")}
        fa["ano"] = ano
        top = ranking(ctx, fa, "tipo", 10, METRICA_COL[m])
        total = sum(t["valor"] for t in top) or 1.0
        dist = [{"tipo": t["chave"], "valor": t["valor"], "qtd": t["qtd"],
                 "share_pct": t["valor"] / total * 100} for t in top]
        ev_itens.append({"ano": ano, "distribuicao": dist})
        blocos.append((ano, dist))
    ev = {"metrica": m, "anos": ev_itens}
    partes = []
    for ano, dist in blocos:
        if not dist:
            partes.append(f"em {ano} não há emendas registradas")
            continue
        corpo = ", ".join(f"{d['tipo']} com {pct(d['share_pct'])}"
                          for d in dist[:4])
        partes.append(f"em {ano} a composição por valor {METRICA_VERBO[m]} é "
                      f"{corpo}")
    return (f"{cap(local(f))}, a distribuição por tipo de emenda "
            f"mudou entre os dois anos: {'; '.join(partes)}."), ev


def h_share_partido(ctx, f, c, partido=None, **_):
    """Participação de um partido no total da área/UF ao longo dos anos.

    O denominador tem de ignorar o filtro de partido — se não, a razão é
    sempre 100% e a resposta fica trivialmente errada.
    """
    m = metrica_da_pergunta(c["consulta"])
    col = METRICA_COL[m]
    partido = partido or f.get("partido")
    f_total = {k: v for k, v in f.items() if k != "partido"}
    total = {s["ano"]: s["valor"] for s in serie_anual(ctx, f_total, col)}
    do_part = {s["ano"]: s["valor"] for s in
               serie_anual(ctx, dict(f_total, partido=partido), col)}
    serie = [{"ano": a, "total": total[a], "partido": do_part.get(a, 0.0),
              "share_pct": (do_part.get(a, 0.0) / total[a] * 100)
              if total[a] else 0.0} for a in sorted(total)]
    ev = {"metrica": m, "partido": partido, "serie": serie,
          "denominador": "total da área/UF, sem filtro de partido"}
    if not serie:
        txt, ev_d = _nada(ctx, f_total, "")
        ev.update(ev_d)
        return txt, ev
    detalhe = "; ".join(f"{s['ano']}: {pct(s['share_pct'])} "
                        f"({brl(s['partido'])} de {brl(s['total'])})"
                        for s in serie)
    pri, ult = serie[0], serie[-1]
    if len(serie) == 1:
        return (f"Só o ano {pri['ano']} tem emendas{rotulo_area(f_total)} "
                f"{local(f_total)} no intervalo consultado, então não há "
                f"evolução a medir: nesse ano a participação do {partido} foi "
                f"de {pct(pri['share_pct'])} do valor {METRICA_VERBO[m]} "
                f"({brl(pri['partido'])} de {brl(pri['total'])})."), ev
    delta = ult["share_pct"] - pri["share_pct"]
    ev["delta_pontos_percentuais"] = delta
    dir_ = ("cresceu" if delta > 1 else "caiu" if delta < -1
            else "ficou praticamente estável")
    return (f"A participação do {partido} em emendas{rotulo_area(f_total)} "
            f"{local(f_total)} {dir_}: passou de {pct(pri['share_pct'])} do "
            f"valor {METRICA_VERBO[m]} em {pri['ano']} para "
            f"{pct(ult['share_pct'])} em {ult['ano']}. Detalhamento anual — "
            f"{detalhe}."), ev


def h_docs(ctx, f, c, kws=(), rotulo="o objeto consultado", **_):
    lista, tot = docs_palavra_chave(ctx, f, list(kws))
    ev = {"palavras_chave": list(kws), "emendas": tot["emendas"],
          "documentos": tot["documentos"], "exemplos": lista}
    termos = enumerar([f"'{k}'" for k in kws], "ou")
    if tot["emendas"] == 0:
        sem_kw = agg(ctx, {k: v for k, v in f.items() if k != "funcao"})
        ev["emendas_no_recorte"] = sem_kw["qtd"]
        return (f"Não há, na base consultada, emendas {local(f)} {periodo(f)} "
                f"cujos documentos de despesa mencionem {rotulo}. O recorte "
                f"tem {emendas(sem_kw['qtd'])}, mas nenhum de seus documentos "
                f"de despesa contém os termos buscados ({termos}). Vale notar "
                f"que o campo de observação dos documentos descreve o "
                f"pagamento (nota fiscal, retenção), não necessariamente o "
                f"objeto da obra."), ev
    exemplos = enumerar([
        f"{x['codigo']} ({nome_proprio(x['autor'])}"
        f"{'/' + x['partido'] if x['partido'] else ''}, área {x['funcao']}, "
        f"{brl(x['empenhado'])} empenhados)" for x in lista[:4]])
    verbo = ("Foi identificada 1 emenda" if tot["emendas"] == 1
             else f"Foram identificadas {num(tot['emendas'])} emendas")
    return (f"{verbo} {local(f)} {periodo(f)} com "
            f"{num(tot['documentos'])} documentos de despesa que mencionam "
            f"{rotulo} (termos buscados: {termos}). "
            f"{'Trata-se de' if tot['emendas'] == 1 else 'Entre as de maior valor empenhado estão'} "
            f"{exemplos}."), ev


def _recorte_benef(f: dict) -> str:
    """'emendas individuais de saúde do PL na Bahia (BA)'."""
    alvo = f"emendas{rotulo_tipo(f)}{rotulo_area(f)}"
    if f.get("partido"):
        alvo += f" do {f['partido']}"
    if f.get("autor"):
        alvo += f" de {nome_proprio(f['autor'])}"
    if f.get("uf"):
        alvo += f" {local(f)}"
    return alvo


def h_benef(ctx, f, c, like=None, n=10, tipo_pessoa=None, minimo=None,
            rotulo="beneficiários", **_):
    like = like or f.get("beneficiario")
    lista, tot = benef(ctx, f, like, n, tipo_pessoa, minimo)
    ev = {"padrao_nome": like, "favorecidos_distintos": tot["favorecidos"],
          "emendas": tot["emendas"], "pago": tot["pago"],
          "empenhado": tot["empenhado"], "beneficiarios": lista}
    if tot["favorecidos"] == 0:
        motivo, ev_d = diagnostico_benef(ctx, f, rotulo)
        ev.update(ev_d)
        return (f"A base não registra {rotulo} recebendo recursos de "
                f"{_recorte_benef(f)} {periodo(f)}"
                + (f": {motivo}." if motivo else ".")), ev
    itens = enumerar([
        f"{nome_proprio(x['nome'])} ({brl(x['pago'])} pagos"
        + ("" if x["empenhado"] == 0
           else f", {brl(x['empenhado'])} empenhados")
        + f", em {emendas(x['emendas'])})" for x in lista[:5]])
    plural_rot = rotulo if tot["favorecidos"] != 1 else rotulo.rstrip("s")
    return (f"A base registra {num(tot['favorecidos'])} {plural_rot} "
            f"recebendo recursos de {_recorte_benef(f)} {periodo(f)}, "
            f"em {emendas(tot['emendas'])}, totalizando "
            f"{brl(tot['pago'])} em valores pagos e {brl(tot['empenhado'])} "
            f"em valores empenhados. "
            f"{'O maior por valor pago é' if tot['favorecidos'] == 1 else 'Os maiores por valor pago são'} "
            f"{itens}."), ev


def h_benef_total(ctx, f, c, like=None, rotulo=None, sim_nao=False, **_):
    like = like or f.get("beneficiario")
    rotulo = rotulo or f.get("beneficiario") or "o beneficiário consultado"
    lista, tot = benef(ctx, f, like, 5)
    ev = {"padrao_nome": like, "favorecidos_distintos": tot["favorecidos"],
          "emendas": tot["emendas"], "pago": tot["pago"],
          "empenhado": tot["empenhado"], "maiores": lista}
    if tot["favorecidos"] == 0:
        motivo, ev_d = diagnostico_benef(ctx, f, rotulo)
        ev.update(ev_d)
        abre = "Não. " if sim_nao else ""
        return (f"{abre}A base não registra recursos de "
                f"{_recorte_benef(f)} {periodo(f)} para {rotulo}"
                + (f": {motivo}." if motivo else ".")), ev
    abre = "Sim. " if sim_nao else ""
    corpo = (f"{rotulo[0].upper() + rotulo[1:]} recebeu recursos de "
             f"{_recorte_benef(f)} {periodo(f)}: "
             f"{num(tot['favorecidos'])} "
             f"{'entidade' if tot['favorecidos'] == 1 else 'entidades'} com "
             f"esse nome "
             f"{'aparece' if tot['favorecidos'] == 1 else 'aparecem'} "
             f"vinculada{'' if tot['favorecidos'] == 1 else 's'} a "
             f"{emendas(tot['emendas'])}, totalizando {brl(tot['pago'])} em "
             f"valores pagos e {brl(tot['empenhado'])} em valores empenhados.")
    if lista:
        corpo += (" A maior delas é "
                  f"{nome_proprio(lista[0]['nome'])}, com "
                  f"{brl(lista[0]['pago'])} pagos.")
    return abre + corpo, ev


def h_benef_autores(ctx, f, c, like=None, n=8, rotulo=None, **_):
    like = like or f.get("beneficiario")
    rotulo = rotulo or f.get("beneficiario") or "o beneficiário consultado"
    ors, w_extra = [], {}
    for i, pat in enumerate((like or "").split("|")):
        ors.append(f"UPPER(fa.nome) LIKE %(like_{i})s")
        w_extra[f"like_{i}"] = f"%{sem_acento(pat)}%"
    w, p = where(f, ["(" + " OR ".join(ors) + ")"])
    p.update(w_extra)
    p["lim"] = n
    rows = ctx.run(f"""
        SELECT e.nome_autor, p.partido, COUNT(DISTINCT e.id),
               SUM(CASE WHEN d.fase='Pagamento' THEN COALESCE(d.valor,0)
                        ELSE 0 END) pago,
               SUM(CASE WHEN d.fase='Empenho' THEN COALESCE(d.valor,0)
                        ELSE 0 END) emp
        {FROM_BENEF} WHERE {w}
        GROUP BY e.nome_autor, p.partido
        ORDER BY pago DESC, emp DESC LIMIT %(lim)s""", p)
    itens = [{"autor": a, "partido": pt, "emendas": int(q), "pago": float(pg),
              "empenhado": float(em)} for a, pt, q, pg, em in rows]
    ev = {"padrao_nome": like, "parlamentares": itens}
    if not itens:
        motivo, ev_d = diagnostico_benef(ctx, f, rotulo)
        ev.update(ev_d)
        return (f"A base não registra parlamentares com emendas destinadas "
                f"para {rotulo} {local(f)} {periodo(f)}"
                + (f": {motivo}." if motivo else ".")), ev
    corpo = enumerar([
        f"{nome_proprio(x['autor'])}"
        f"{'/' + x['partido'] if x['partido'] else ''} "
        f"({brl(x['pago'])} pagos em {emendas(x['emendas'])})"
        for x in itens])
    return (f"{rotulo[0].upper() + rotulo[1:]} recebeu emendas de "
            f"{num(len(itens))} "
            f"{'parlamentar' if len(itens) == 1 else 'parlamentares'} "
            f"{local(f)} {periodo(f)}: {corpo}."), ev


def h_benef_rank_autor(ctx, f, c, like=None, n=5, **_):
    extra, w_extra, ors = [], {}, []
    for i, pat in enumerate((like or "").split("|")):
        ors.append(f"UPPER(fa.nome) LIKE %(like_{i})s")
        w_extra[f"like_{i}"] = f"%{sem_acento(pat)}%"
    if like:
        extra.append("(" + " OR ".join(ors) + ")")
    extra.append("fa.nome NOT ILIKE '%%sem informa%%'")
    w, p = where(f, extra)
    p.update(w_extra)
    p["lim"] = n
    rows = ctx.run(f"""
        SELECT e.nome_autor, p.partido, COUNT(DISTINCT fa.id),
               SUM(CASE WHEN d.fase='Pagamento' THEN COALESCE(d.valor,0)
                        ELSE 0 END) pago,
               SUM(CASE WHEN d.fase='Empenho' THEN COALESCE(d.valor,0)
                        ELSE 0 END) emp
        {FROM_BENEF} WHERE {w}
        GROUP BY e.nome_autor, p.partido
        ORDER BY pago DESC, emp DESC LIMIT %(lim)s""", p)
    itens = [{"autor": a, "partido": pt, "beneficiarios": int(nb),
              "pago": float(pg), "empenhado": float(em)}
             for a, pt, nb, pg, em in rows]
    ev = {"padrao_nome": like, "ranking": itens}
    if not itens:
        motivo, ev_d = diagnostico_benef(ctx, f)
        ev.update(ev_d)
        return (f"A base não registra parlamentares destinando recursos a "
                f"esse tipo de beneficiário {local(f)} {periodo(f)}"
                + (f": {motivo}." if motivo else ".")), ev
    pri = itens[0]
    corpo = enumerar([
        f"{nome_proprio(x['autor'])}"
        f"{'/' + x['partido'] if x['partido'] else ''} "
        f"({brl(x['pago'])} pagos a {num(x['beneficiarios'])} "
        f"{'beneficiário' if x['beneficiarios'] == 1 else 'beneficiários'})"
        for x in itens])
    return (f"O parlamentar que mais destinou recursos foi "
            f"{nome_proprio(pri['autor'])}"
            f"{'/' + pri['partido'] if pri['partido'] else ''}, com "
            f"{brl(pri['pago'])} pagos a {num(pri['beneficiarios'])} "
            f"{'beneficiário' if pri['beneficiarios'] == 1 else 'beneficiários'} "
            f"{local(f)} {periodo(f)}. Ranking — {corpo}."), ev


HANDLERS = {
    "contagem": h_contagem, "contagem_distinta": h_contagem_distinta,
    "soma": h_soma, "media": h_media, "lista_emendas": h_lista_emendas,
    "lista_autores": h_lista_autores, "ranking": h_ranking,
    "tendencia": h_tendencia, "comparar_areas": h_comparar_areas,
    "comparar_partidos": h_comparar_partidos,
    "comparar_tipos": h_comparar_tipos, "comparar_ufs": h_comparar_ufs,
    "execucao": h_execucao, "tendencia_execucao": h_tendencia_execucao,
    "tendencia_media": h_tendencia_media, "tendencia_dim": h_tendencia_dim,
    "serie_tipos": h_serie_tipos, "distribuicao_tipos": h_distribuicao_tipos,
    "share_partido": h_share_partido, "docs": h_docs, "benef": h_benef,
    "benef_total": h_benef_total, "benef_autores": h_benef_autores,
    "benef_rank_autor": h_benef_rank_autor,
}

# --------------------------------------------------------------------------
# especificação por consulta
# --------------------------------------------------------------------------
FMS = "FUNDO MUNICIPAL DE SAUDE"
FES = "FUNDO ESTADUAL DE SAUDE"
FMAS = "FUNDO MUNICIPAL DE ASSISTENCIA SOCIAL"
HOSP = "HOSPITAL|SANTA CASA|HOSPITALAR"
PREF = "PREFEITURA|MUNICIPIO"
ENSINO = "UNIVERSIDADE|INSTITUTO FEDERAL|ESCOLA|FACULDADE|FUNDACAO UNIVERSIT|CENTRO EDUCACIONAL|COLEGIO"
ASSIST = "ASSISTENCIA SOCIAL|APAE|LAR |ABRIGO|CASA DE ACOLHIMENTO|ENTIDADE ASSISTENCIAL"

SPEC: dict[str, dict] = {
    # ---------------- TIPO A ----------------
    "A1": {"h": "contagem"},
    "A2": {"h": "contagem"},
    "A3": {"h": "ranking", "dim": "autor", "n": 5},
    "A4": {"h": "soma"},
    "A5": {"h": "contagem", "exige": "empenhado"},
    "A6": {"h": "soma"},
    "A7": {"h": "lista_autores", "n": 10},
    "A8": {"h": "media"},
    "A9": {"h": "contagem_distinta"},
    "A10": {"h": "soma"},
    "A11": {"h": "contagem"},
    "A12": {"h": "contagem"},
    "A13": {"h": "lista_emendas"},
    "A14": {"h": "soma"},
    "A15": {"h": "lista_autores", "n": 10},
    "A16": {"h": "soma"},
    "A17": {"h": "contagem", "exige": "pago"},
    "A18": {"h": "ranking", "dim": "autor", "n": 5},
    "A19": {"h": "soma"},
    "A20": {"h": "contagem"},
    "A21": {"h": "soma"},
    "A22": {"h": "soma"},
    "A23": {"h": "lista_autores", "n": 10},
    "A24": {"h": "soma"},
    "A25": {"h": "contagem"},
    "A26": {"h": "ranking", "dim": "autor", "n": 5},
    "A27": {"h": "contagem"},
    "A28": {"h": "contagem"},
    "A29": {"h": "media"},
    "A30": {"h": "soma"},

    # ---------------- TIPO B ----------------
    "B1": {"h": "lista_emendas"},
    "B2": {"h": "docs", "kws": ["ponte"], "rotulo": "pontes"},
    "B3": {"h": "docs", "kws": ["escola", "educandario", "colegio"],
           "rotulo": "escolas"},
    "B4": {"h": "lista_emendas"},
    "B5": {"h": "docs",
           "kws": ["equipamento hospitalar", "equipamentos hospitalares",
                   "material hospitalar", "equipamento medico",
                   "equipamentos medicos"],
           "rotulo": "equipamentos hospitalares"},
    "B6": {"h": "lista_emendas"},
    "B7": {"h": "docs", "kws": ["pavimenta"], "rotulo": "pavimentação"},
    "B8": {"h": "lista_autores", "n": 10},
    "B9": {"h": "docs", "kws": ["ubs", "unidade basica de saude",
                                "unidade de saude"],
           "rotulo": "unidades básicas de saúde (UBS)"},
    "B10": {"h": "lista_emendas"},
    "B11": {"h": "docs", "kws": ["saneamento", "esgoto"],
            "rotulo": "saneamento ou esgoto"},
    "B12": {"h": "lista_emendas"},
    "B13": {"h": "docs", "kws": ["ambulanc"], "rotulo": "ambulâncias"},
    "B14": {"h": "lista_emendas"},
    "B15": {"h": "docs", "kws": ["creche"], "rotulo": "creches"},
    "B16": {"h": "lista_emendas"},
    "B17": {"h": "docs", "kws": ["trator", "escavadeira", "retroescavadeira"],
            "rotulo": "tratores ou escavadeiras"},
    "B18": {"h": "lista_emendas"},
    "B19": {"h": "docs", "kws": ["pavimenta", "obra", "construcao",
                                 "drenagem", "calcamento"],
            "rotulo": "obras de infraestrutura"},
    "B20": {"h": "lista_emendas"},
    "B21": {"h": "soma"},
    "B22": {"h": "lista_emendas"},
    "B23": {"h": "ranking", "dim": "autor", "n": 5},
    "B24": {"h": "lista_emendas"},
    "B25": {"h": "soma"},
    "B26": {"h": "soma"},
    "B27": {"h": "lista_emendas"},
    "B28": {"h": "ranking", "dim": "autor", "n": 5},
    "B29": {"h": "lista_emendas"},
    "B30": {"h": "soma"},

    # ---------------- TIPO C ----------------
    "C1": {"h": "tendencia"},
    "C2": {"h": "comparar_partidos", "partidos": ["PT", "PL"]},
    "C3": {"h": "tendencia"},
    "C4": {"h": "ranking", "dim": "funcao", "n": 5},
    "C5": {"h": "serie_tipos", "tipos": ["bancada", "individual"]},
    "C6": {"h": "ranking", "dim": "autor", "n": 3},
    "C7": {"h": "tendencia"},
    "C8": {"h": "comparar_areas", "areas": ["transporte", "saude"]},
    "C9": {"h": "tendencia_dim", "dim": "funcao", "n": 5},
    "C10": {"h": "execucao"},
    "C11": {"h": "tendencia"},
    "C12": {"h": "comparar_areas", "areas": ["educacao", "saude"]},
    "C13": {"h": "tendencia"},
    "C14": {"h": "ranking", "dim": "autor", "n": 5},
    "C15": {"h": "tendencia_execucao"},
    "C16": {"h": "comparar_areas", "areas": ["agricultura", "urbanismo"]},
    "C17": {"h": "tendencia_dim", "dim": "autor", "n": 5},
    "C18": {"h": "tendencia"},
    "C19": {"h": "comparar_areas", "areas": ["seguranca", "educacao"]},
    "C20": {"h": "tendencia_dim", "dim": "partido", "n": 5},
    "C21": {"h": "ranking", "dim": "partido", "n": 3},
    "C22": {"h": "tendencia"},
    "C23": {"h": "comparar_tipos", "tipos": ["individual", "comissao"]},
    "C24": {"h": "ranking", "dim": "autor", "n": 5},
    "C25": {"h": "share_partido", "partido": "PT"},
    "C26": {"h": "comparar_ufs", "ufs": ["SP", "RJ"]},
    "C27": {"h": "ranking", "dim": "funcao", "n": 5},
    "C28": {"h": "distribuicao_tipos", "anos": [2021, 2024]},
    "C29": {"h": "comparar_areas", "areas": ["habitacao", "saneamento"]},
    "C30": {"h": "tendencia_media"},

    # ---------------- TIPO D ----------------
    # 'sim_nao' marca as perguntas de resposta binária ("X recebeu...?"),
    # que ganham um "Sim."/"Não." explícito no início da referência.
    "D1": {"h": "benef_total", "like": FMS,
           "rotulo": "o Fundo Municipal de Saúde"},
    "D2": {"h": "benef_total", "like": "BANCO DO BRASIL",
           "rotulo": "o Banco do Brasil", "sim_nao": True},
    "D3": {"h": "benef_total", "like": FES,
           "rotulo": "o Fundo Estadual de Saúde"},
    "D4": {"h": "benef_total", "like": FMAS,
           "rotulo": "o Fundo Municipal de Assistência Social",
           "sim_nao": True},
    "D5": {"h": "benef", "like": HOSP,
           "rotulo": "hospitais e entidades hospitalares", "n": 10},
    "D6": {"h": "benef_total", "like": FMS,
           "rotulo": "o Fundo Municipal de Saúde", "sim_nao": True},
    "D7": {"h": "benef_total", "like": FMS,
           "rotulo": "o Fundo Municipal de Saúde"},
    "D8": {"h": "benef_total", "like": FES,
           "rotulo": "o Fundo Estadual de Saúde", "sim_nao": True},
    "D9": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D10": {"h": "benef_total", "like": FMS,
            "rotulo": "o Fundo Municipal de Saúde", "sim_nao": True},
    "D11": {"h": "benef", "tipo_pessoa": "PJ",
            "rotulo": "pessoas jurídicas (empresas e entidades)", "n": 10},
    "D12": {"h": "benef", "like": PREF,
            "rotulo": "prefeituras e municípios", "n": 10},
    "D13": {"h": "benef", "like": ENSINO,
            "rotulo": "instituições de ensino", "n": 10},
    "D14": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D15": {"h": "benef", "like": "FUNDO", "rotulo": "fundos", "n": 10},
    "D16": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D17": {"h": "benef", "like": PREF, "rotulo": "municípios", "n": 10},
    "D18": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D19": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D20": {"h": "benef", "like": HOSP,
            "rotulo": "hospitais e entidades hospitalares", "n": 10},
    "D21": {"h": "benef_rank_autor", "like": ENSINO, "n": 5},
    "D22": {"h": "benef", "rotulo": "beneficiários", "n": 10},
    "D23": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D24": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D25": {"h": "benef", "rotulo": "beneficiários", "n": 5},
    "D26": {"h": "benef", "like": PREF,
            "rotulo": "prefeituras e municípios", "n": 10},
    "D27": {"h": "benef_autores", "like": FMS, "n": 8,
            "rotulo": "o Fundo Municipal de Saúde"},
    "D28": {"h": "benef", "rotulo": "beneficiários com mais de R$ 1 milhão "
            "em pagamentos", "minimo": 1_000_000, "n": 10},
    "D29": {"h": "benef", "rotulo": "entidades beneficiárias", "n": 10},
    "D30": {"h": "benef_total", "like": FMS,
            "rotulo": "o Fundo Municipal de Saúde"},
}

# D23 pergunta por entidades de saúde: a consulta original não traz 'area',
# mas o enunciado restringe a saúde. Aplicado explicitamente.
AREA_EXTRA = {"D23": "saude"}


# --------------------------------------------------------------------------
# execução
# --------------------------------------------------------------------------
def resolver_url() -> str:
    url = os.environ.get("GABARITO_PG_URL")
    if not url:
        from app.config import settings
        partes = urlparse(settings.DATABASE_URL)
        url = urlunparse(partes._replace(path=f"/{DB_PADRAO}"))
    host = urlparse(url).hostname or ""
    if host not in HOSTS_LOCAIS:
        sys.exit(f"Destino recusado: host '{host}' não é local. O gabarito "
                 f"só é gerado a partir de um Postgres local.")
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saida", default="tests/test_queries/"
                    "gabarito_120_consultas.json")
    ap.add_argument("--incluir-sql", action="store_true",
                    help="inclui o SQL executado em cada entrada")
    args = ap.parse_args()

    url = resolver_url()
    cx = psycopg2.connect(url)
    cx.set_session(readonly=True)
    cur = cx.cursor()

    cur.execute("SELECT COUNT(*) FROM emendas")
    n_emendas = cur.fetchone()[0]
    if n_emendas < MIN_EMENDAS:
        sys.exit(f"A base '{urlparse(url).path.lstrip('/')}' tem apenas "
                 f"{n_emendas} emendas — parece ser a base de seed, não a "
                 f"base nacional. Aponte GABARITO_PG_URL para a base correta.")

    contagens = {}
    for t in ("emendas", "parlamentares", "documentos_emenda",
              "documento_favorecido", "favorecidos"):
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        contagens[t] = cur.fetchone()[0]
    print(f"base: {urlparse(url).path.lstrip('/')} @ "
          f"{urlparse(url).hostname}:{urlparse(url).port}")
    print("contagens:", contagens)

    # Rótulo oficial de cada função orçamentária, direto da base, para que o
    # nome da área na resposta seja o mesmo que o dado usa.
    cur.execute("SELECT funcao, MAX(funcao_nome) FROM emendas "
                "WHERE funcao IS NOT NULL AND funcao <> '' "
                "AND funcao_nome IS NOT NULL AND funcao_nome <> '' "
                "GROUP BY funcao")
    FUNCAO_NOME.update({c: n for c, n in cur.fetchall()})

    todas = ([("A", c) for c in CONSULTAS_TIPO_A]
             + [("B", c) for c in CONSULTAS_TIPO_B]
             + [("C", c) for c in CONSULTAS_TIPO_C]
             + [("D", c) for c in CONSULTAS_TIPO_D])

    entradas, vazias = [], []
    for tipo, c in todas:
        cid = c["id"]
        spec = dict(SPEC[cid])
        handler = HANDLERS[spec.pop("h")]
        esperado = dict(c["esperado"])
        if cid in AREA_EXTRA:
            esperado["area"] = AREA_EXTRA[cid]
        f = resolver_filtros(esperado)
        ctx = Ctx(cur)
        try:
            texto, ev = handler(ctx, f, c, **spec)
        except Exception as exc:  # pragma: no cover - diagnóstico
            raise SystemExit(f"falha em {cid}: {type(exc).__name__}: {exc}")
        texto = " ".join(texto.split())
        entrada = {
            "id": cid,
            "topic": TOPICO[tipo],
            "query": c["consulta"],
            "answer": texto,
            "waited": c["esperado"],
            "evidencia": ev,
        }
        if args.incluir_sql:
            entrada["sql"] = ctx.sqls
        entradas.append(entrada)
        marcador = ""
        if ev.get("quantidade") == 0 or ev.get("emendas") == 0 \
                or ev.get("favorecidos_distintos") == 0:
            vazias.append(cid)
            marcador = "  [sem dados]"
        print(f"  {cid:4s} {len(texto):>4d} car{marcador}")

    doc = {
        "metadata": {
            "description": "Gabarito (ground truth) das 120 consultas de "
                           "avaliação do FiscalIA: respostas de referência "
                           "extraídas por SQL determinístico da base nacional "
                           "de emendas parlamentares, para cálculo de "
                           "BERTScore e auditoria de veracidade das respostas "
                           "do pipeline RAG.",
            "created": date.today().isoformat(),
            "source": "Base local transparencia_nacional (Portal da "
                      "Transparência/CGU + Câmara dos Deputados), emendas "
                      "parlamentares de 2020 a 2024",
            "base_de_dados": contagens,
            "consultas_origem": "backend/tests/test_queries/test_evaluation.py",
            "gerado_por": "backend/scripts/gerar_gabarito_consultas.py",
            "total_consultas": len(entradas),
            "consultas_sem_dados": vazias,
            "criterios": {
                "area_tematica": "emendas.funcao = código da função "
                                 "orçamentária resolvido por "
                                 "data/dictionary.json",
                "valores_da_emenda": "emendas.valor_empenhado / "
                                     "valor_liquidado / valor_pago; a métrica "
                                     "destacada segue o verbo da pergunta",
                "tipo_emenda": "emendas.tipo_emenda ILIKE padrão "
                               "(individual, bancada, comissão, relator)",
                "partido": "parlamentares.partido via cod_autor",
                "autor": "UPPER(emendas.nome_autor) LIKE padrão",
                "beneficiarios": "favorecidos via documento_favorecido; "
                                 "valor_recebido é 100% NULL nesta base, "
                                 "então o valor vem de documentos_emenda."
                                 "valor separado por fase (Pagamento = "
                                 "recebido, Empenho = empenhado); 1 "
                                 "favorecido por documento, sem duplicação",
                "busca_em_documentos": "palavra-chave em documentos_emenda."
                                       "observacao com filtros de UF/ano, sem "
                                       "restringir a função",
                "ausencia_de_dados": "quando o SQL não retorna linhas, a "
                                     "resposta de referência afirma "
                                     "explicitamente que não há registro",
            },
        },
        "queries": entradas,
    }

    saida = Path(args.saida)
    if not saida.is_absolute():
        saida = _BACKEND / saida
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    cx.close()

    print(f"\n{len(entradas)} consultas escritas em {saida}")
    print(f"sem dados na base ({len(vazias)}): {', '.join(vazias) or '-'}")


if __name__ == "__main__":
    main()
