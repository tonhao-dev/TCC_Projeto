#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recalcula o BERTScore das baterias contra o GABARITO independente.

Motivação
---------
As baterias originais mediram o BERTScore contra uma referência *sintética*
construída pelo próprio pipeline a partir dos dados recuperados do banco
(``evaluate.construir_referencia``). Essa referência tem dois problemas como
gabarito:

1. **Circularidade.** Quando a recuperação falha, a referência também fica
   vazia ("Foram encontrados 0 registros no total. 1. N/A (N/A/N/A) ..."), e o
   modelo é *premiado* por repetir fielmente a falha de recuperação em vez de
   ser penalizado por não responder o que a base realmente contém.
2. **Estilo telegráfico.** A referência é uma lista de campos (``N/A``,
   ``R$ 0.00``), não prosa em português, o que deprime o BERTScore de qualquer
   resposta bem escrita.

Este script substitui a referência pelo gabarito independente
(``gabarito_120_consultas.json``), extraído por SQL determinístico da base
nacional, e recalcula P/R/F1 por consulta e os agregados por tipo (A-D) e
Geral. **As respostas dos modelos NÃO são geradas de novo** -- apenas o campo
``resposta`` já gravado é recomparado.

Garantias de comparabilidade
----------------------------
* Mesmo scorer da execução original: ``BERTScorer(lang="pt",
  rescale_with_baseline=True, use_fast_tokenizer=True)`` -- bert-base-
  multilingual-cased, camada 9, com rescale de baseline.
* Mesma regra de agregação: média simples dos scores por consulta
  (verificada contra os agregados armazenados nos JSONs originais).
* As métricas de recuperação (Entity Accuracy, Precision@5, Recall@5,
  F1 Retrieval, nDCG@5, MRR) são **copiadas sem alteração**: não dependem do
  texto de referência.

Saídas
------
* ``<prefixo>_bertscore_gabarito.json`` por bateria: cópia do JSON original
  com ``metricas.bertscore_*`` recalculado, ``referencia`` trocada pelo
  gabarito, os valores antigos preservados em ``bertscore_referencia_sintetica``
  e os agregados refeitos.
* Um CSV comparativo no layout do estudo exploratório (bloco por tipo de
  consulta + Geral; linha COM RAG, linha SEM RAG e linha de diferença).

Uso
---
    python recalcular_bertscore_gabarito.py
    python recalcular_bertscore_gabarito.py --apenas gemini-2.5-flash
    python recalcular_bertscore_gabarito.py --saida-csv ../tests/test_queries/x.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

AQUI = Path(__file__).resolve().parent
DIR_RESULTADOS = AQUI.parent / "tests" / "test_queries"
GABARITO_PADRAO = DIR_RESULTADOS / "gabarito_120_consultas.json"

# Baterias a recalcular: (rótulo no CSV, modelo, arquivo COM RAG, arquivo SEM RAG)
BATERIAS = [
    (
        "gemini 2.5 flash",
        "gemini-2.5-flash",
        "2026-10-16_16-21-39-800_gemini-2.5-flash_resultados_120_com_rag_hibrido.json",
        "2026-10-16_16-21-39-800_gemini-2.5-flash_resultados_120_sem_rag_direto.json",
    ),
    (
        "llama3.1",
        "llama3.1",
        "2026-10-18_02-56-56-577_llama3.1_resultados_120_com_rag_hibrido.json",
        "2026-10-18_02-56-56-577_llama3.1_resultados_120_sem_rag_direto.json",
    ),
    (
        "gemma3:1b",
        "gemma3-1b",
        "2026-10-20_05-26-59-658_gemma3-1b_resultados_120_com_rag_hibrido.json",
        "2026-10-20_05-26-59-658_gemma3-1b_resultados_120_sem_rag_direto.json",
    ),
]

# (chave no agregado, rótulo da coluna, é_métrica_de_recuperação)
METRICAS = [
    ("entity_accuracy", "Entity Accuracy", True),
    ("precision_at_5", "Precision@5", True),
    ("recall_at_5", "Recall@5", True),
    ("f1_score", "F1 Retrieval", True),
    ("ndcg_at_5", "nDCG@5", True),
    ("mrr", "MRR", True),
    ("bertscore_precision", "BERTScore P", False),
    ("bertscore_recall", "BERTScore R", False),
    ("bertscore_f1", "BERTScore F1", False),
]

BLOCOS = [
    ("A", "Tipo A",
     "Tipo A - Factual Direto - Consultas com filtros simples sobre campos estruturados"),
    ("B", "Tipo B",
     "Tipo B - Cruzamento Semântico - Exigem mapear uma área temática em "
     "linguagem natural para a classificação orçamentária"),
    ("C", "Tipo C",
     "Tipo C - Temporal - Envolvem raciocínio temporal, comparando ou "
     "acompanhando a evolução de valores ao longo de anos"),
    ("D", "Tipo D",
     "Tipo D - Beneficiários - Sobre os destinatários finais dos recursos - "
     "quem efetivamente recebeu"),
    ("G", "Geral", "GERAL"),
]

CHAVES_BS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")


# ---------------------------------------------------------------- scorer

_SCORER = None


def get_scorer():
    """Carrega (uma vez) o BERTScorer na MESMA configuração da execução original."""
    global _SCORER
    if _SCORER is None:
        from bert_score import BERTScorer
        t0 = time.time()
        _SCORER = BERTScorer(
            lang="pt",
            rescale_with_baseline=True,
            use_fast_tokenizer=True,
        )
        print("  scorer: %s camada=%s rescale=%s (%.0fs)"
              % (_SCORER._model_type, _SCORER._num_layers,
                 _SCORER.rescale_with_baseline, time.time() - t0),
              flush=True)
    return _SCORER


# ---------------------------------------------------------------- gabarito


def carregar_gabarito(caminho: Path) -> dict[str, str]:
    doc = json.loads(caminho.read_text(encoding="utf-8"))
    gab: dict[str, str] = {}
    for q in doc["queries"]:
        resposta = (q.get("answer") or "").strip()
        if not resposta:
            raise SystemExit(f"Gabarito sem 'answer' para a consulta {q.get('id')}.")
        gab[q["id"]] = resposta
    if len(gab) != 120:
        raise SystemExit(f"Gabarito deveria ter 120 consultas, tem {len(gab)}.")
    return gab


# ---------------------------------------------------------------- recálculo


def _agregar(registros: list[dict]) -> dict[str, Optional[float]]:
    """Média simples dos BERTScores por consulta (regra da execução original)."""
    saida: dict[str, Optional[float]] = {}
    for chave in CHAVES_BS:
        vals = [(r.get("metricas") or {}).get(chave) for r in registros]
        vals = [v for v in vals if v is not None]
        saida[chave] = mean(vals) if vals else None
    saida["n_bertscore"] = len([
        r for r in registros
        if (r.get("metricas") or {}).get("bertscore_f1") is not None
    ])
    return saida


def recalcular(caminho: Path, gabarito: dict[str, str],
               batch_size: int) -> tuple[dict, Path]:
    """Recalcula o BERTScore de uma bateria contra o gabarito."""
    doc = json.loads(caminho.read_text(encoding="utf-8"))
    resultados = doc["resultados"]

    faltando = sorted({r["id"] for r in resultados} - set(gabarito))
    if faltando:
        raise SystemExit(f"{caminho.name}: sem gabarito para {faltando}.")

    candidatos: list[str] = []
    referencias: list[str] = []
    indices: list[int] = []
    for i, r in enumerate(resultados):
        cand = (r.get("resposta") or "").strip()
        if not cand:
            continue
        candidatos.append(cand)
        referencias.append(gabarito[r["id"]])
        indices.append(i)

    print("  %s: %d/%d respostas pontuáveis" % (caminho.name, len(indices), len(resultados)), flush=True)

    scorer = get_scorer()
    t0 = time.time()
    P, R, F1 = scorer.score(candidatos, referencias, batch_size=batch_size)
    print("     pontuado em %.0fs" % (time.time() - t0), flush=True)

    # Zera os scores de quem não foi pontuado e grava os novos
    for r in resultados:
        m = r.setdefault("metricas", {})
        anterior = {k: m.get(k) for k in CHAVES_BS}
        if any(v is not None for v in anterior.values()):
            r["bertscore_referencia_sintetica"] = anterior
        for k in CHAVES_BS:
            m[k] = None

    for pos, idx in enumerate(indices):
        r = resultados[idx]
        r["metricas"]["bertscore_precision"] = float(P[pos])
        r["metricas"]["bertscore_recall"] = float(R[pos])
        r["metricas"]["bertscore_f1"] = float(F1[pos])
        r["referencia_sintetica"] = r.get("referencia")
        r["referencia"] = gabarito[r["id"]]

    # Agregados: BERTScore refeito, métricas de recuperação intactas
    doc["geral"].update(_agregar(resultados))
    for tipo in ("A", "B", "C", "D"):
        bloco = [r for r in resultados if r["tipo"] == tipo]
        if bloco:
            doc["por_tipo"][tipo].update(_agregar(bloco))

    doc["referencias_de"] = "gabarito_120_consultas.json (ground truth independente, SQL determinístico)"
    doc["bertscore_recalculado"] = {
        "em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gabarito": GABARITO_PADRAO.name,
        "scorer": "bert-base-multilingual-cased lang=pt rescale_with_baseline=True",
        "respostas_regeradas": False,
        "observacao": (
            "Somente o BERTScore foi recalculado, contra o gabarito independente. "
            "As respostas dos modelos e as métricas de recuperação são as da "
            "execução original. Os BERTScores antigos (contra a referência "
            "sintética derivada da recuperação) ficam preservados em "
            "'bertscore_referencia_sintetica' por consulta."
        ),
    }

    destino = caminho.with_name(caminho.stem + "_bertscore_gabarito.json")
    destino.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return doc, destino


# ---------------------------------------------------------------- CSV


def _bloco_agg(doc: dict, bloco: str) -> dict:
    if bloco == "G":
        return doc.get("geral", {})
    return (doc.get("por_tipo") or {}).get(bloco, {})


def _fmt(valor) -> str:
    if valor is None:
        return ""
    return f"{round(valor, 4):.4f}"


def gerar_csv(linhas_modelos: list[tuple[str, dict, dict]], saida: Path) -> Path:
    n_col = 1 + len(BLOCOS) * len(METRICAS)

    # Linha 1: cabeçalho de grupo (uma célula por bloco, resto vazio)
    grupo = ["Modelo"]
    for _, _, titulo in BLOCOS:
        grupo.append(titulo)
        grupo.extend([""] * (len(METRICAS) - 1))

    # Linha 2: rótulos das métricas
    rotulos = [""]
    for _, curto, _ in BLOCOS:
        for _, rotulo, _ in METRICAS:
            rotulos.append(f"{rotulo} ({curto})")

    def linha(nome: str, doc: dict, so_bertscore: bool) -> list[str]:
        row = [nome]
        for cod, _, _ in BLOCOS:
            agg = _bloco_agg(doc, cod)
            for chave, _, e_recuperacao in METRICAS:
                if so_bertscore and e_recuperacao:
                    row.append("")
                else:
                    row.append(_fmt(agg.get(chave)))
        return row

    def linha_diferenca(rotulo_modelo: str, com: dict, sem: dict) -> list[str]:
        row = [f"Diferença {rotulo_modelo} (COM vs SEM RAG)"]
        for cod, _, _ in BLOCOS:
            a = _bloco_agg(com, cod)
            b = _bloco_agg(sem, cod)
            for chave, _, e_recuperacao in METRICAS:
                if e_recuperacao:
                    row.append("-")
                    continue
                va, vb = a.get(chave), b.get(chave)
                row.append(_fmt(va - vb) if va is not None and vb is not None else "")
        return row

    with saida.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(grupo)
        w.writerow(rotulos)
        for rotulo, com, sem in linhas_modelos:
            w.writerow(linha(f"{rotulo} COM RAG", com, so_bertscore=False))
            w.writerow(linha(f"{rotulo} SEM RAG", sem, so_bertscore=True))
            w.writerow(linha_diferenca(rotulo, com, sem))
            w.writerow([""] * n_col)
        w.writerow(["Link para as consultas"] + [""] * (n_col - 1))

    return saida


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gabarito", type=Path, default=GABARITO_PADRAO,
                    help="JSON do gabarito (ground truth).")
    ap.add_argument("--saida-csv", type=Path,
                    default=DIR_RESULTADOS / "Estudo_Exploratorio_Comparativo_COM_vs_SEM_RAG_bertscore_gabarito.csv",
                    help="CSV comparativo de saída.")
    ap.add_argument("--apenas", action="append", default=None,
                    help="Recalcular só este modelo (repetível).")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch do BERTScorer (default 16).")
    args = ap.parse_args()

    if not args.gabarito.exists():
        raise SystemExit(f"Gabarito não encontrado: {args.gabarito}")

    gabarito = carregar_gabarito(args.gabarito)
    print(f"Gabarito: {args.gabarito.name} ({len(gabarito)} consultas)")

    baterias = BATERIAS
    if args.apenas:
        alvos = set(args.apenas)
        baterias = [b for b in BATERIAS if b[0] in alvos or b[1] in alvos]
        if not baterias:
            raise SystemExit(f"Nenhuma bateria casa com {sorted(alvos)}.")

    linhas: list[tuple[str, dict, dict]] = []
    for rotulo, modelo, arq_com, arq_sem in baterias:
        print(f"\n{rotulo} ({modelo})", flush=True)
        docs = []
        for arq in (arq_com, arq_sem):
            caminho = DIR_RESULTADOS / arq
            if not caminho.exists():
                raise SystemExit(f"Bateria não encontrada: {caminho}")
            doc, destino = recalcular(caminho, gabarito, args.batch_size)
            print("     -> %s" % destino.name, flush=True)
            docs.append(doc)
        com, sem = docs
        g_com, g_sem = com["geral"], sem["geral"]
        print("     Geral F1: COM=%.4f  SEM=%.4f  Δ=%+.4f"
              % (g_com["bertscore_f1"], g_sem["bertscore_f1"],
                 g_com["bertscore_f1"] - g_sem["bertscore_f1"]), flush=True)
        linhas.append((rotulo, com, sem))

    saida = gerar_csv(linhas, args.saida_csv)
    print(f"\nCSV gerado: {saida}")
    print(f"  modelos={len(linhas)}  colunas={1 + len(BLOCOS) * len(METRICAS)}")


if __name__ == "__main__":
    sys.exit(main())
