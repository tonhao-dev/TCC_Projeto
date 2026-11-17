"""Gera gráficos comparativos BERTScore: COM RAG vs. SEM RAG (llama3.1).

Lê os valores EXATOS dos JSONs de resultado já armazenados -- não recalcula
nenhuma métrica. Apenas plota o que está em disco. Os rótulos usam o nome do
modelo gravado no JSON de resultado (ex.: llama3.1).

Saída: PNGs em backend/tests/test_queries/graficos/
Uso:   py -u backend/scripts/gerar_graficos_bertscore.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem display
import matplotlib.pyplot as plt
import numpy as np

from output_naming import nome_com_prefixo

# ---------------------------------------------------------------- caminhos
BASE = Path(__file__).resolve().parents[1] / "tests" / "test_queries"
HIBRIDO_JSON = BASE / "resultados_bertscore_seed_hibrido.json"
DIRETO_JSON = BASE / "resultados_baseline_direto_llama31.json"
OUT = BASE / "graficos"
OUT.mkdir(exist_ok=True)

COR_HIBRIDO = "#2563eb"  # azul
COR_DIRETO = "#dc2626"   # vermelho
COR_DELTA = "#16a34a"    # verde


def carregar(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _mais_recente(sufixo_base: str, caminho_legado: Path) -> Path:
    """Localiza o JSON de resultado tolerando o novo prefixo de nome.

    Procura por arquivos ``*<sufixo_base>`` (ex.: ``*resultados_bertscore_seed_hibrido.json``)
    e devolve o mais recente por mtime. Se nenhum prefixado existir, usa o nome
    legado (sem prefixo), para continuar funcionando com dados antigos.
    """
    candidatos = sorted(
        BASE.glob(f"*{sufixo_base}"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if candidatos:
        return candidatos[0]
    return caminho_legado


def por_id(dados: dict) -> dict[str, dict]:
    return {r["id"]: r for r in dados["resultados"]}


def main() -> None:
    hibrido_path = _mais_recente("resultados_bertscore_seed_hibrido.json", HIBRIDO_JSON)
    direto_path = _mais_recente("resultados_baseline_direto_llama31.json", DIRETO_JSON)
    hibrido = carregar(hibrido_path)
    direto = carregar(direto_path)

    # Modelo e instante de escrita para o prefixo dos PNGs (usa o do híbrido).
    modelo = hibrido.get("modelo") or direto.get("modelo") or "modelo"
    agora = datetime.now()

    # Rótulos de legenda com o nome do modelo correspondente.
    LBL_COM_RAG = f"COM RAG ({modelo})"
    LBL_SEM_RAG = f"SEM RAG ({modelo})"

    def salvar(fig, nome_base: str) -> None:
        fig.savefig(OUT / nome_com_prefixo(nome_base, modelo, agora), dpi=150)

    h = por_id(hibrido)
    d = por_id(direto)

    # ordem estável S1..S8 (interseção, preservando a ordem do cenário COM RAG)
    ids = [r["id"] for r in hibrido["resultados"] if r["id"] in d]

    f1_h = [h[i]["metricas"]["bertscore_f1"] for i in ids]
    f1_d = [d[i]["metricas"]["bertscore_f1"] for i in ids]
    lat_h = [h[i]["latencia_ms"] / 1000.0 for i in ids]
    lat_d = [d[i]["latencia_ms"] / 1000.0 for i in ids]

    x = np.arange(len(ids))
    w = 0.38

    # ---------------------------------------------- 1) BERTScore F1 por consulta
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, f1_h, w, label=LBL_COM_RAG, color=COR_HIBRIDO)
    b2 = ax.bar(x + w / 2, f1_d, w, label=LBL_SEM_RAG, color=COR_DIRETO)
    ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--",
               label="baseline aleatória pt (0)")
    ax.set_title(f"BERTScore F1 por consulta — {modelo} (rescale_with_baseline)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Consulta")
    ax.set_ylabel("BERTScore F1")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    fig.tight_layout()
    salvar(fig, "bertscore_f1_por_consulta.png")
    plt.close(fig)

    # ------------------------------------- 2) P/R/F1 agregado (barras agrupadas)
    metr = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    rotulos = ["Precision", "Recall", "F1"]
    agg_h = [hibrido["geral"][m] for m in metr]
    agg_d = [direto["geral"][m] for m in metr]
    xa = np.arange(len(rotulos))
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(xa - w / 2, agg_h, w, label=LBL_COM_RAG, color=COR_HIBRIDO)
    b2 = ax.bar(xa + w / 2, agg_d, w, label=LBL_SEM_RAG, color=COR_DIRETO)
    ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--")
    ax.set_title(f"BERTScore agregado (8 consultas) — COM RAG vs. SEM RAG ({modelo})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor médio")
    ax.set_xticks(xa)
    ax.set_xticklabels(rotulos)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
    fig.tight_layout()
    salvar(fig, "bertscore_prf_agregado.png")
    plt.close(fig)

    # ------------------------------------------------ 3) Latência por consulta
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, lat_h, w, label=LBL_COM_RAG, color=COR_HIBRIDO)
    b2 = ax.bar(x + w / 2, lat_d, w, label=LBL_SEM_RAG, color=COR_DIRETO)
    ax.set_title("Latência por consulta — COM RAG (3 chamadas) vs. SEM RAG (1 chamada)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Consulta")
    ax.set_ylabel("Latência (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0fs", fontsize=7, padding=2)
    fig.tight_layout()
    salvar(fig, "latencia_por_consulta.png")
    plt.close(fig)

    # ------------------------------------- 4) Ganho (Δ F1 = COM RAG - SEM RAG)
    delta = [fh - fd for fh, fd in zip(f1_h, f1_d)]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, delta, color=COR_DELTA)
    ax.set_title("Ganho do RAG em BERTScore F1 por consulta (COM RAG − SEM RAG)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Consulta")
    ax.set_ylabel("Δ BERTScore F1")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="+%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    salvar(fig, "bertscore_f1_delta.png")
    plt.close(fig)

    print("OK — 4 gráficos gerados em:", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
