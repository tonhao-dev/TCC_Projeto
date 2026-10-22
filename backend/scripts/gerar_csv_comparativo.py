#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gera o CSV comparativo COM RAG vs SEM RAG (importável no Google Sheets).

Formato idêntico ao gerado para o gemini-2.5-flash:
  - Linhas: COM RAG, SEM RAG
  - Colunas: 6 métricas de retrieval + 3 de BERTScore, estratificadas por
    tipo de consulta (A, B, C, D) e, ao final, o bloco Geral (agregado das
    120 consultas).

As 6 métricas de retrieval (Entity Accuracy, Precision@5, Recall@5,
F1 Retrieval, nDCG@5, MRR) só existem no cenário COM RAG; na linha SEM RAG
essas células ficam vazias (o baseline não recupera do banco). As 3 colunas
de BERTScore (P, R, F1) existem nos dois cenários e são as comparáveis.

Uso:
    python gerar_csv_comparativo.py --com <com_rag.json> --sem <sem_rag.json> \
        [--saida <arquivo.csv>]

Se --com/--sem forem omitidos, o script localiza automaticamente o par mais
recente de ``*_resultados_120_com_rag_hibrido.json`` e
``*_resultados_120_sem_rag_direto.json`` em tests/test_queries com o mesmo
prefixo de timestamp+modelo.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DIR_RESULTADOS = AQUI.parent / "tests" / "test_queries"

# (chave no JSON agregado, rótulo da coluna, é_metrica_de_retrieval)
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
BLOCOS = [("A", "Tipo A"), ("B", "Tipo B"), ("C", "Tipo C"),
          ("D", "Tipo D"), ("G", "Geral")]


def _carregar(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _agg_do_bloco(data: dict, bloco: str) -> dict:
    if bloco == "G":
        return data.get("geral", {})
    return (data.get("por_tipo") or {}).get(bloco, {})


def _fmt(valor) -> str:
    if valor is None:
        return ""
    return f"{round(valor, 4)}"


def _localizar_par() -> tuple[Path, Path]:
    coms = sorted(DIR_RESULTADOS.glob("*_resultados_120_com_rag_hibrido.json"))
    if not coms:
        sys.exit("Nenhum arquivo *_resultados_120_com_rag_hibrido.json encontrado.")
    com = coms[-1]
    prefixo = com.name.replace("_resultados_120_com_rag_hibrido.json", "")
    sem = DIR_RESULTADOS / f"{prefixo}_resultados_120_sem_rag_direto.json"
    if not sem.exists():
        sys.exit(f"Par SEM RAG não encontrado para o prefixo '{prefixo}'.")
    return com, sem


def _modelo_do_nome(nome_com: str) -> str:
    # <timestamp>_<modelo>_resultados_120_com_rag_hibrido.json
    m = re.match(r"^[0-9_\-]+_(.+?)_resultados_120_com_rag_hibrido\.json$", nome_com)
    return m.group(1) if m else "modelo"


def gerar(com_path: Path, sem_path: Path, saida: Path | None) -> Path:
    com = _carregar(com_path)
    sem = _carregar(sem_path)
    modelo = _modelo_do_nome(com_path.name)

    if saida is None:
        saida = DIR_RESULTADOS / f"comparativo_{modelo}_com_vs_sem_rag.csv"

    cabecalho = ["Modelo"]
    for _, rotulo_bloco in BLOCOS:
        for _, rotulo_metrica, _ in METRICAS:
            cabecalho.append(f"{rotulo_metrica} ({rotulo_bloco})")

    def linha(nome_cenario: str, data: dict) -> list[str]:
        row = [nome_cenario]
        for cod_bloco, _ in BLOCOS:
            agg = _agg_do_bloco(data, cod_bloco)
            for chave, _, _ in METRICAS:
                row.append(_fmt(agg.get(chave)))
        return row

    # UTF-8 com BOM para o Google Sheets abrir acentos corretamente.
    with saida.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cabecalho)
        w.writerow(linha("COM RAG", com))
        w.writerow(linha("SEM RAG", sem))

    print(f"CSV gerado: {saida}")
    print(f"  modelo={modelo}  linhas=2  colunas={len(cabecalho)}")
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--com", type=Path, help="JSON final COM RAG (hibrido).")
    ap.add_argument("--sem", type=Path, help="JSON final SEM RAG (direto).")
    ap.add_argument("--saida", type=Path, help="Caminho do CSV de saída.")
    args = ap.parse_args()

    if args.com and args.sem:
        com_path, sem_path = args.com, args.sem
    elif args.com or args.sem:
        sys.exit("Passe --com E --sem juntos, ou nenhum (auto-localiza o par).")
    else:
        com_path, sem_path = _localizar_par()
        print(f"Auto-localizado:\n  COM: {com_path.name}\n  SEM: {sem_path.name}")

    gerar(com_path, sem_path, args.saida)


if __name__ == "__main__":
    main()
