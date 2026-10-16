"""Avaliação focada com BERTScore sobre o conjunto de dados semeado (seed).

O conjunto oficial de 120 consultas (test_evaluation.py) foi projetado para a
base nacional completa (~32.787 emendas). Em um ambiente de desenvolvimento com
apenas o seed de ~8 emendas, a maioria dessas consultas retornaria vazio — o que
não exercita a geração do LLM nem permite medir BERTScore de forma significativa.

Este runner usa um subconjunto de consultas cujos filtros CASAM com os dados
semeados (PA/saúde, SP/educação, MG/transporte, MG/saúde, BA/educação), de modo
que o pipeline recupera registros reais, o llama3.1 sintetiza uma resposta, e o
BERTScore mede a fidelidade dessa resposta contra uma referência factual
determinística construída a partir dos próprios dados.

Métricas por consulta: retrieval (P@5, R@5, F1, NDCG@5, MRR, entity_accuracy)
+ BERTScore (precision, recall, f1).

Uso:
    py scripts/evaluate_bertscore_seed.py --modo hibrido
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from app.config import settings
from app.database import SessionLocal
from app.services.rag.pipeline import RAGPipeline
from app.services.cache import query_cache
from scripts.evaluate import (
    avaliar_resultado, calcular_metricas_agregadas, construir_referencia,
)
from scripts.output_naming import caminho_com_prefixo, timestamp_conteudo
import structlog

logger = structlog.get_logger()


# Consultas que CASAM com os dados semeados (ver seed_dev_data.py).
# Mistura dos 4 tipos para exercitar diferentes caminhos do pipeline.
CONSULTAS_SEED = [
    # Tipo A — Factual direta
    {"id": "S1", "consulta": "Qual o total gasto em saúde no estado do Pará em 2023?",
     "esperado": {"uf": "PA", "ano": 2023, "area": "saude", "operacao": "soma"}},
    {"id": "S2", "consulta": "Quantas emendas foram destinadas ao Pará em 2022?",
     "esperado": {"uf": "PA", "ano": 2022, "operacao": "contagem"}},
    {"id": "S3", "consulta": "Qual o total empenhado em emendas para São Paulo em 2024?",
     "esperado": {"uf": "SP", "ano": 2024, "operacao": "soma"}},
    # Tipo B — Cruzamento semântico / área temática
    {"id": "S4", "consulta": "Quais emendas de educação foram destinadas a São Paulo em 2023?",
     "esperado": {"area": "educacao", "uf": "SP", "ano": 2023}},
    {"id": "S5", "consulta": "Quais emendas para transporte foram pagas em Minas Gerais em 2023?",
     "esperado": {"area": "transporte", "uf": "MG", "ano": 2023}},
    {"id": "S6", "consulta": "Quais emendas de educação existem na Bahia em 2023?",
     "esperado": {"area": "educacao", "uf": "BA", "ano": 2023}},
    # Tipo A — Saúde em MG (2024)
    {"id": "S7", "consulta": "Qual o total empenhado em saúde em Minas Gerais em 2024?",
     "esperado": {"area": "saude", "uf": "MG", "ano": 2024, "operacao": "soma"}},
    # Tipo A — Pará saúde agregado multi-ano (2022+2023 existem)
    {"id": "S8", "consulta": "Quais emendas de saúde foram destinadas ao Pará?",
     "esperado": {"area": "saude", "uf": "PA"}},
]


async def executar(modo: str = "hibrido"):
    logger.info("iniciando_avaliacao_bertscore_seed", total=len(CONSULTAS_SEED), modo=modo)
    print(f"Avaliação BERTScore (seed) — {len(CONSULTAS_SEED)} consultas, modo={modo}", flush=True)

    db = SessionLocal()
    pipeline = RAGPipeline()
    resultados = []

    try:
        for i, consulta_def in enumerate(CONSULTAS_SEED, 1):
            cid = consulta_def["id"]
            consulta = consulta_def["consulta"]
            print(f"[{i}/{len(CONSULTAS_SEED)}] {cid}: {consulta[:55]}...", flush=True)

            # Garantir execução limpa (sem cache) para medir latência real
            cache_key = f"{modo}:{consulta}"
            query_cache._cache.pop(cache_key, None)

            inicio = time.time()
            try:
                resultado = await pipeline.processar(consulta, db, modo=modo)
                latencia = int((time.time() - inicio) * 1000)

                metricas = avaliar_resultado(resultado, consulta_def, calcular_bert=True)
                referencia = construir_referencia(resultado, consulta_def)

                resultados.append({
                    "id": cid,
                    "tipo": cid[0],
                    "consulta": consulta,
                    "latencia_ms": latencia,
                    "num_resultados": resultado["metadata"]["num_resultados"],
                    "total_no_banco": resultado["metadata"].get("total_no_banco"),
                    "modo": resultado["metadata"]["modo"],
                    "resposta": resultado.get("resposta", ""),
                    "referencia": referencia,
                    "metricas": metricas,
                    "sucesso": True,
                })

                b_f1 = metricas.get("bertscore_f1")
                b_str = f"{b_f1:.3f}" if b_f1 is not None else "N/A"
                print(f"    OK lat={latencia/1000:.0f}s num={resultado['metadata']['num_resultados']} "
                      f"F1={metricas['f1_score']:.2f} BERT-F1={b_str}", flush=True)

            except Exception as e:
                latencia = int((time.time() - inicio) * 1000)
                resultados.append({
                    "id": cid, "tipo": cid[0], "consulta": consulta,
                    "latencia_ms": latencia, "sucesso": False, "erro": str(e),
                })
                print(f"    ERRO: {str(e)[:100]}", flush=True)

        # Agregação (reusa a mesma função do evaluate principal)
        geral = calcular_metricas_agregadas(resultados)

        agora = datetime.now()
        relatorio = {
            "modo": modo,
            "conjunto": "seed",
            "modelo": settings.OLLAMA_MODEL,
            "timestamp": timestamp_conteudo(agora),
            "geral": geral,
            "resultados": resultados,
        }

        output_path = caminho_com_prefixo(
            Path(__file__).parent.parent / "tests" / "test_queries",
            f"resultados_bertscore_seed_{modo}.json",
            settings.OLLAMA_MODEL,
            agora,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)

        # Relatório
        print(f"\n{'='*72}", flush=True)
        print(f"  RELATÓRIO BERTScore (seed) — COM RAG · Modo: {modo.upper()}", flush=True)
        print(f"{'='*72}", flush=True)
        print(f"  Consultas: {geral.get('total', 0)} | Sucessos: {geral.get('sucessos', 0)}", flush=True)
        print(f"  Latência média: {geral.get('latencia_media_ms', 0)/1000:.0f}s", flush=True)
        print(f"{'-'*72}", flush=True)
        print(f"  Retrieval  — P@5={geral.get('precision_at_5',0):.2%}  "
              f"R@5={geral.get('recall_at_5',0):.2%}  "
              f"F1={geral.get('f1_score',0):.2%}  "
              f"Entity={geral.get('entity_accuracy',0):.2%}", flush=True)
        print(f"  BERTScore  — P={geral.get('bertscore_precision',0):.2%}  "
              f"R={geral.get('bertscore_recall',0):.2%}  "
              f"F1={geral.get('bertscore_f1',0):.2%}", flush=True)
        print(f"{'-'*72}", flush=True)
        print(f"  {'ID':<5} {'num':>4} {'lat(s)':>7} {'ret-F1':>8} {'BERT-F1':>8}  consulta", flush=True)
        for r in resultados:
            if r.get("sucesso"):
                m = r["metricas"]
                bf1 = m.get("bertscore_f1")
                print(f"  {r['id']:<5} {r.get('num_resultados',0):>4} "
                      f"{r['latencia_ms']/1000:>7.0f} "
                      f"{m['f1_score']:>8.2%} "
                      f"{(bf1 if bf1 is not None else 0):>8.2%}  "
                      f"{r['consulta'][:40]}", flush=True)
            else:
                print(f"  {r['id']:<5} {'ERRO':>4}  {r.get('erro','')[:50]}", flush=True)
        print(f"{'='*72}\n", flush=True)
        print(f"Resultados salvos em: {output_path}", flush=True)

        return relatorio

    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", default="hibrido",
                        choices=["hibrido", "sql_puro", "vetorial_puro"])
    args = parser.parse_args()
    asyncio.run(executar(args.modo))
