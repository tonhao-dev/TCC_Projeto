"""Insere o bloco gemma3:1b (COM RAG / SEM RAG / Diferenca) no CSV comparativo
seguindo o formato do arquivo modelo. As linhas do gemma ocupam as posicoes
11, 12 e 13 (1-based), empurrando 'Link para as consultas' para o fim.

- Metricas de retrieval (Entity Accuracy, Precision@5, Recall@5, F1 Retrieval,
  nDCG@5, MRR) so existem COM RAG -> na linha de Diferenca ficam '-', igual aos
  outros modelos.
- Metricas BERTScore (P, R, F1) existem COM e SEM RAG -> Diferenca = COM - SEM.
"""
import csv
import sys

MODELO = sys.argv[1]       # CSV modelo anexado
GEMMA = sys.argv[2]        # comparativo_gemma3-1b_com_vs_sem_rag.csv
OUT = sys.argv[3]          # novo CSV de saida

with open(MODELO, encoding="utf-8-sig", newline="") as f:
    modelo_rows = list(csv.reader(f))

with open(GEMMA, encoding="utf-8-sig", newline="") as f:
    gemma_rows = list(csv.reader(f))

header = modelo_rows[0]           # linha de grupos (Tipo A/B/C/D/GERAL)
colmap = modelo_rows[1]           # linha de nomes de metrica por coluna

gemma_com = gemma_rows[1][:]      # COM RAG
gemma_sem = gemma_rows[2][:]      # SEM RAG

# Monta linha de Diferenca a partir dos nomes de coluna
diff = ["Diferença gemma3:1b (COM vs SEM RAG)"]
for i in range(1, len(colmap)):
    nome = colmap[i]
    com = gemma_com[i].strip()
    sem = gemma_sem[i].strip()
    if nome.startswith("BERTScore") and com != "" and sem != "":
        d = float(com) - float(sem)
        diff.append(f"{d:.4f}")
    else:
        diff.append("-")

# Renomeia rotulos do bloco gemma para o padrao dos demais modelos
gemma_com[0] = "gemma3:1b COM RAG"
gemma_sem[0] = "gemma3:1b SEM RAG"

# Reconstroi: linhas 1..10 do modelo (ate o separador em branco apos llama3.1),
# depois o bloco gemma (3 linhas), um separador em branco, e por fim o
# 'Link para as consultas'.
blank = [""] * len(header)
link_row = modelo_rows[10]        # 'Link para as consultas' (0-based idx 10)

out_rows = []
out_rows.extend(modelo_rows[0:10])   # grupos, metricas, gemini(3), blank, llama(3), blank
out_rows.append(gemma_com)           # linha 11
out_rows.append(gemma_sem)           # linha 12
out_rows.append(diff)                # linha 13
out_rows.append(blank)               # separador
out_rows.append(link_row)            # link para consultas

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerows(out_rows)

print("OK ->", OUT)
print("Total de linhas:", len(out_rows))
for i, r in enumerate(out_rows, 1):
    print(i, "|", r[0])
