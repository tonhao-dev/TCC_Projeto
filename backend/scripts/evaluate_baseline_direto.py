"""Baseline SEM RAG — pergunta enviada DIRETO ao modelo (llama3.1).

Objetivo: medir como o llama3.1 responde às mesmas 8 consultas do seed SEM
nenhum mecanismo de recuperação (sem interpreter, sem query_planner, sem busca
SQL/vetorial, sem synthesizer com contexto). A pergunta vai pura ao modelo, que
responde "de cabeça". Isso isola a contribuição do RAG: comparando este
baseline contra o 1º resultado (modo híbrido), vemos quanto o pipeline de
recuperação melhora a fidelidade factual das respostas.

Metodologia da comparação (justa):
- Mesmas 8 consultas (CONSULTAS_SEED, importadas do runner híbrido).
- MESMA referência factual por consulta — reutilizada do resultado híbrido
  (resultados_bertscore_seed_hibrido.json), construída a partir dos dados reais
  do banco. Assim ambos os candidatos (resposta-RAG vs. resposta-direta) são
  medidos contra o MESMO gabarito factual, com o MESMO BERTScorer multilíngue.
- As métricas de RETRIEVAL (P@5, R@5, F1, NDCG@5, MRR) NÃO se aplicam ao
  baseline direto: não há documentos recuperados. São registradas como null.
  A comparação de interesse é BERTScore (fidelidade da geração) + latência.

Uso:
    py scripts/evaluate_baseline_direto.py
"""
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.services.rag.llm_factory import build_chat_llm
from scripts.evaluate import calcular_bertscore
from scripts.evaluate_bertscore_seed import CONSULTAS_SEED
from scripts.output_naming import caminho_com_prefixo, timestamp_conteudo
import structlog

logger = structlog.get_logger()

# Prompt de sistema mínimo: define a tarefa sem injetar NENHUM dado recuperado.
# Equivalente ao papel do synthesizer, porém SEM contexto — o modelo responde
# apenas com o que "sabe", que é exatamente o baseline que queremos medir.
SYSTEM_PROMPT = (
    "Você é um assistente especializado em emendas parlamentares e gastos "
    "públicos do Brasil (orçamento federal, execução de emendas, valores "
    "empenhados e pagos por autor, partido, UF, ano e área temática). "
    "Responda à pergunta do usuário de forma factual, objetiva e em português, "
    "informando valores e registros quando souber. Se não tiver certeza dos "
    "números, deixe isso claro."
)

# Arquivo do 1º resultado (híbrido) de onde reutilizamos as referências.
HIBRIDO_JSON = (
    Path(__file__).parent.parent / "tests" / "test_queries"
    / "resultados_bertscore_seed_hibrido.json"
)


def carregar_referencias_hibrido() -> dict:
    """Carrega as referências factuais por id do resultado híbrido (1º resultado).

    Tolera o novo prefixo de nome: procura ``*resultados_bertscore_seed_hibrido.json``
    e usa o mais recente; cai no nome legado se nenhum prefixado existir.
    """
    base = HIBRIDO_JSON.parent
    candidatos = sorted(
        base.glob("*resultados_bertscore_seed_hibrido.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    caminho = candidatos[0] if candidatos else HIBRIDO_JSON
    with open(caminho, encoding="utf-8") as f:
        d = json.load(f)
    refs = {}
    for r in d.get("resultados", []):
        if r.get("referencia"):
            refs[r["id"]] = r["referencia"]
    return refs


async def executar():
    referencias = carregar_referencias_hibrido()
    print(f"SEM RAG (direto ao modelo) — {len(CONSULTAS_SEED)} consultas, "
          f"modelo={settings.OLLAMA_MODEL}", flush=True)
    print(f"Referências carregadas do híbrido: {len(referencias)}", flush=True)

    # Mesmo modelo do pipeline. temperature=0 para determinismo (igual aos
    # serviços RAG). Sem num_predict explícito -> default do Ollama, para não
    # limitar artificialmente a resposta direta.
    llm = build_chat_llm(temperature=0.0)

    resultados = []
    for i, consulta_def in enumerate(CONSULTAS_SEED, 1):
        cid = consulta_def["id"]
        consulta = consulta_def["consulta"]
        print(f"[{i}/{len(CONSULTAS_SEED)}] {cid}: {consulta[:55]}...", flush=True)

        inicio = time.time()
        try:
            mensagens = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=consulta),
            ]
            resp = await llm.ainvoke(mensagens)
            resposta = resp.content if hasattr(resp, "content") else str(resp)
            latencia = int((time.time() - inicio) * 1000)

            referencia = referencias.get(cid, "")
            bert = calcular_bertscore(resposta, referencia)

            metricas = {
                # Retrieval não se aplica ao baseline sem recuperação.
                "precision_at_5": None,
                "recall_at_5": None,
                "f1_score": None,
                "ndcg_at_5": None,
                "mrr": None,
                "entity_accuracy": None,
            }
            if bert is not None:
                metricas["bertscore_precision"] = bert["precision"]
                metricas["bertscore_recall"] = bert["recall"]
                metricas["bertscore_f1"] = bert["f1"]

            resultados.append({
                "id": cid,
                "tipo": cid[0],
                "consulta": consulta,
                "latencia_ms": latencia,
                "resposta": resposta,
                "referencia": referencia,
                "metricas": metricas,
                "sucesso": True,
            })

            b_f1 = metricas.get("bertscore_f1")
            b_str = f"{b_f1:.3f}" if b_f1 is not None else "N/A"
            print(f"    OK lat={latencia/1000:.0f}s len={len(resposta)} "
                  f"BERT-F1={b_str}", flush=True)

        except Exception as e:
            latencia = int((time.time() - inicio) * 1000)
            resultados.append({
                "id": cid, "tipo": cid[0], "consulta": consulta,
                "latencia_ms": latencia, "sucesso": False, "erro": str(e),
            })
            print(f"    ERRO: {str(e)[:120]}", flush=True)

    # Agregação do BERTScore (único grupo de métricas aplicável aqui).
    sucessos = [r for r in resultados if r.get("sucesso")]

    def media(key):
        vals = [r["metricas"][key] for r in sucessos
                if r["metricas"].get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    geral = {
        "total": len(resultados),
        "sucessos": len(sucessos),
        "latencia_media_ms": (sum(r["latencia_ms"] for r in sucessos) / len(sucessos))
                              if sucessos else 0,
        "bertscore_precision": media("bertscore_precision"),
        "bertscore_recall": media("bertscore_recall"),
        "bertscore_f1": media("bertscore_f1"),
        # Retrieval não aplicável (sem recuperação).
        "precision_at_5": None,
        "recall_at_5": None,
        "f1_score": None,
        "ndcg_at_5": None,
        "mrr": None,
        "entity_accuracy": None,
    }

    agora = datetime.now()
    relatorio = {
        "modo": "baseline_direto",
        "conjunto": "seed",
        "descricao": "Pergunta enviada direto ao modelo, sem RAG (sem "
                     "interpreter/planner/retrieval/synthesizer). BERTScore "
                     "medido contra a mesma referência factual do resultado híbrido.",
        "modelo": settings.OLLAMA_MODEL,
        "timestamp": timestamp_conteudo(agora),
        "geral": geral,
        "resultados": resultados,
    }

    output_path = caminho_com_prefixo(
        Path(__file__).parent.parent / "tests" / "test_queries",
        "resultados_baseline_direto_llama31.json",
        settings.OLLAMA_MODEL,
        agora,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    # Relatório
    print(f"\n{'='*72}", flush=True)
    print(f"  SEM RAG (pergunta direto ao modelo) — modelo {settings.OLLAMA_MODEL}", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  Consultas: {geral['total']} | Sucessos: {geral['sucessos']}", flush=True)
    print(f"  Latência média: {geral['latencia_media_ms']/1000:.0f}s", flush=True)
    print(f"  Retrieval: N/A (baseline sem recuperação)", flush=True)
    print(f"  BERTScore — P={geral['bertscore_precision']:.2%}  "
          f"R={geral['bertscore_recall']:.2%}  "
          f"F1={geral['bertscore_f1']:.2%}", flush=True)
    print(f"{'-'*72}", flush=True)
    print(f"  {'ID':<5} {'lat(s)':>7} {'len':>5} {'BERT-P':>8} {'BERT-R':>8} {'BERT-F1':>8}  consulta", flush=True)
    for r in resultados:
        if r.get("sucesso"):
            m = r["metricas"]
            print(f"  {r['id']:<5} {r['latencia_ms']/1000:>7.0f} "
                  f"{len(r['resposta']):>5} "
                  f"{(m.get('bertscore_precision') or 0):>8.2%} "
                  f"{(m.get('bertscore_recall') or 0):>8.2%} "
                  f"{(m.get('bertscore_f1') or 0):>8.2%}  "
                  f"{r['consulta'][:38]}", flush=True)
        else:
            print(f"  {r['id']:<5} {'ERRO':>7}  {r.get('erro','')[:60]}", flush=True)
    print(f"{'='*72}\n", flush=True)
    print(f"Resultados salvos em: {output_path}", flush=True)

    return relatorio


if __name__ == "__main__":
    asyncio.run(executar())
