"""Avaliação das 120 consultas oficiais — COM RAG vs. SEM RAG (mesmo modelo).

COM RAG = pipeline completo do projeto (RAGPipeline, modo "hibrido"):
          interpreter -> planner + decomposer -> busca SQL/vetorial/RRF ->
          synthesizer com os dados recuperados do banco.
SEM RAG = a pergunta vai pura ao modelo, com o MESMO prompt de sistema do
          baseline ``evaluate_baseline_direto.py`` e nenhum dado recuperado.

Metodologia (a mesma do comparativo llama3.1 no seed):
- Mesmas 120 consultas (tests/test_queries/test_evaluation.py, tipos A/B/C/D).
- Métricas de retrieval (P@5, R@5, F1, NDCG@5, MRR, entity accuracy) só existem
  no COM RAG; no SEM RAG não há recuperação e elas ficam null.
- BERTScore (P/R/F1) nos dois cenários contra a MESMA referência factual por
  consulta, construída de forma determinística a partir dos dados recuperados no
  COM RAG (``construir_referencia``), com o mesmo BERTScorer multilíngue.

Fases (a geração é retomável por checkpoint JSONL; COM RAG e SEM RAG podem rodar
em paralelo, em processos separados, pois o SEM RAG não usa o banco):
  com_rag  gera as respostas COM RAG + métricas de retrieval + referência.
  sem_rag  gera as respostas SEM RAG.
  pontuar  calcula o BERTScore dos dois cenários e grava os JSONs finais
           ``<timestamp>_<modelo>_resultados_120_*.json``. Nunca sobrescreve
           arquivos existentes.
  todas    com_rag -> sem_rag -> pontuar, em sequência.
  paralelo com_rag e sem_rag em 2 processos simultâneos e, ao final, pontuar.

Latência: o SDK do provedor roda com 0 retentativas; 429, quedas de conexão
(VPN) e fallbacks do pipeline descartam a tentativa e repetem a consulta fora
do cronômetro. A latência registrada é sempre a de uma tentativa limpa.

Uso:
    py scripts/evaluate_120_com_sem_rag.py --modelo gemini-2.5-flash --fase paralelo
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import subprocess
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import httpx
import structlog
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from app.config import settings
from app.services.cache import query_cache
from app.services.rag.llm_factory import build_chat_llm
from scripts.evaluate import (
    TODAS_CONSULTAS,
    avaliar_resultado,
    calcular_bertscore,
    construir_referencia,
)
from scripts.evaluate_baseline_direto import SYSTEM_PROMPT
from scripts.output_naming import caminho_com_prefixo, timestamp_conteudo

SAIDA_PADRAO = BACKEND / "tests" / "test_queries"

# Política de retentativas. O SDK do provedor roda com 0 retentativas (ver
# main): toda espera acontece aqui, FORA da latência medida, e cada tentativa
# reinicia o cronômetro — a latência registrada é sempre de uma chamada limpa.
MAX_TENTATIVAS = 4                 # erros "outros" (não 429, não conexão)
BACKOFF_S = [5, 15, 45]
ESPERA_RATE_LIMIT_S = 60           # o limite do proxy é de tokens por minuto
MAX_RATE_LIMIT = 15                # 429 seguidos antes de desistir da consulta
MAX_ERROS_CONEXAO = 10             # falhas de conexão com o proxy respondendo
ESPERA_CONEXAO_S = 30              # intervalo entre sondagens do proxy
MAX_ESPERA_CONEXAO_S = 4 * 3600    # aguarda a VPN voltar por até 4 h
TIMEOUT_LLM_S = 300.0              # teto por chamada ao modelo

POLITICA_RETENTATIVAS = {
    "retentativas_do_sdk": 0,
    "timeout_por_chamada_s": TIMEOUT_LLM_S,
    "latencia": "medida só na tentativa aceita; tentativas descartadas (429, "
                "conexão, fallback do pipeline) e as esperas entre elas ficam fora",
    "rate_limit_429": f"espera {ESPERA_RATE_LIMIT_S}s, até {MAX_RATE_LIMIT} vezes",
    "conexao": "aguarda o proxy voltar (VPN) e repete a mesma consulta",
    "outros_erros": f"até {MAX_TENTATIVAS} tentativas, backoff {BACKOFF_S}s",
}

NOMES_TIPO = {"A": "Factual", "B": "Semântico", "C": "Complexo", "D": "Benefic."}

METRICAS_RETRIEVAL = [
    "entity_accuracy", "precision_at_5", "recall_at_5",
    "f1_score", "ndcg_at_5", "mrr",
]
METRICAS_BERT = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]

# ---------------------------------------------------------------------------
# Captura de fallbacks silenciosos do pipeline
# ---------------------------------------------------------------------------
# O planner e a resposta "sem resultados" do synthesizer engolem exceções do LLM
# e caem em heurística/texto fixo. Sem esta captura, uma falha transitória do
# provedor viraria uma resposta "válida" e contaminaria as métricas. Os eventos
# abaixo disparam nova tentativa; se persistirem, ficam registrados no JSON.
EVENTOS_FALLBACK = {"planner_fallback", "erro_resposta_vazia", "interpretacao_falha"}
_EVENTOS_CAPTURADOS: list[dict] = []


def _capturar_fallbacks(_logger, metodo, event_dict):
    nome = event_dict.get("event")
    if nome in EVENTOS_FALLBACK:
        _EVENTOS_CAPTURADOS.append({
            "evento": nome,
            "nivel": metodo,
            "erro": str(event_dict.get("erro", ""))[:300],
        })
    return event_dict


def _configurar_logs() -> None:
    """Mantém só WARNING+ no console e intercepta os eventos de fallback."""
    processadores = list(structlog.get_config()["processors"])
    structlog.configure(
        processors=[_capturar_fallbacks, *processadores],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _modelo_ativo() -> str:
    provedor = (settings.LLM_PROVIDER or "ollama").lower()
    if provedor == "litellm":
        return settings.LITELLM_MODEL
    if provedor == "anthropic":
        return settings.ANTHROPIC_MODEL
    return settings.OLLAMA_MODEL


def _tag_segura(valor: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", valor.replace(":", "-")).strip("-")


def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


def _carregar_jsonl(path: Path) -> dict[str, dict]:
    """Lê o checkpoint; a última ocorrência de cada id vence."""
    registros: dict[str, dict] = {}
    if path.exists():
        for linha in path.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                reg = json.loads(linha)
                registros[reg["id"]] = reg
    return registros


def _anexar_jsonl(path: Path, registro: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(registro, ensure_ascii=False, default=_json_default) + "\n")


def _ler_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _gravar_json(path: Path, dados: dict) -> None:
    path.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _resumir_erro(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:400]}"


def _texto(conteudo) -> str:
    """Normaliza AIMessage.content (str ou lista de partes) para texto."""
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        partes = []
        for p in conteudo:
            if isinstance(p, str):
                partes.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                partes.append(p.get("text", ""))
        return "".join(partes)
    return str(conteudo or "")


def _registrar_meta(path: Path, modelo: str, extra: dict | None = None) -> dict:
    """Cria/atualiza o meta da fase, recusando misturar modelos no checkpoint."""
    meta = _ler_json(path)
    if meta and meta.get("modelo") != modelo:
        raise SystemExit(
            f"Checkpoint {path.parent} pertence ao modelo {meta.get('modelo')!r}, "
            f"não a {modelo!r}. Use outro --tag."
        )
    meta.setdefault("modelo", modelo)
    meta.setdefault("provedor", settings.LLM_PROVIDER)
    meta.setdefault("inicio", timestamp_conteudo())
    if extra:
        meta.update(extra)
    _gravar_json(path, meta)
    return meta


def _snapshot_base(db) -> dict:
    """Volume da base no momento da execução (contexto para ler as métricas)."""
    contagens = {}
    for tabela in ("emendas", "parlamentares", "documentos_emenda",
                   "beneficiarios", "favorecidos"):
        try:
            contagens[tabela] = db.execute(text(f"SELECT count(*) FROM {tabela}")).scalar()
        except Exception:
            db.rollback()
            contagens[tabela] = None
    return contagens


# ---------------------------------------------------------------------------
# Retentativas (fora da latência medida)
# ---------------------------------------------------------------------------
def _classificar_erro(erro: str) -> str:
    e = erro.lower()
    if "ratelimit" in e or "error code: 429" in e or "rate limit" in e:
        return "rate_limit"
    if any(s in e for s in ("connection error", "connecterror", "connecttimeout",
                            "apiconnectionerror", "connectionerror", "timed out",
                            "timeout", "remoteprotocolerror", "connection refused")):
        return "conexao"
    return "outro"


async def _aguardar_proxy(rotulo: str) -> None:
    """Bloqueia até o proxy LiteLLM responder (ex.: VPN caiu e voltou)."""
    if (settings.LLM_PROVIDER or "").lower() != "litellm":
        await asyncio.sleep(ESPERA_CONEXAO_S)
        return
    url = settings.LITELLM_BASE_URL.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {settings.LITELLM_API_KEY}"}
    inicio = time.monotonic()
    avisou = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as cli:
                resp = await cli.get(url, headers=headers)
            if resp.status_code < 500:
                if avisou:
                    print(f"[{rotulo}] proxy LiteLLM de volta após "
                          f"{int(time.monotonic() - inicio)}s", flush=True)
                return
        except httpx.HTTPError:
            pass
        if not avisou:
            print(f"[{rotulo}] proxy LiteLLM inacessível (VPN?) — aguardando "
                  f"reconexão (sondagem a cada {ESPERA_CONEXAO_S}s)...", flush=True)
            avisou = True
        if time.monotonic() - inicio > MAX_ESPERA_CONEXAO_S:
            raise SystemExit(f"[{rotulo}] proxy inacessível por mais de "
                             f"{MAX_ESPERA_CONEXAO_S // 3600}h — abortando "
                             "(checkpoint preservado; rode de novo para retomar).")
        await asyncio.sleep(ESPERA_CONEXAO_S)


async def _com_retentativas(tentativa, rotulo: str):
    """Executa ``tentativa()`` -> (payload, problema) até obter um resultado limpo.

    ``problema`` None = sucesso limpo. 429 espera a janela do limite de tokens;
    falha de conexão espera o proxy voltar; outros erros usam backoff curto.
    Retorna (payload, problema_final, falhas_descartadas). Se desistir, o
    payload pode ser None (exceção) ou um resultado aceito com ``problema``.
    """
    falhas: list[dict] = []
    contagem = {"rate_limit": 0, "conexao": 0, "outro": 0}
    limites = {"rate_limit": MAX_RATE_LIMIT, "conexao": MAX_ERROS_CONEXAO,
               "outro": MAX_TENTATIVAS}
    while True:
        try:
            payload, problema = await tentativa()
        except Exception as exc:
            payload, problema = None, _resumir_erro(exc)
        if problema is None:
            return payload, None, falhas

        tipo = _classificar_erro(problema)
        contagem[tipo] += 1
        if contagem[tipo] >= limites[tipo]:
            return payload, problema, falhas
        falhas.append({"tipo": tipo, "erro": problema[:400]})

        if tipo == "rate_limit":
            print(f"[{rotulo}] 429 (limite de tokens/min do proxy) — aguardando "
                  f"{ESPERA_RATE_LIMIT_S}s", flush=True)
            await asyncio.sleep(ESPERA_RATE_LIMIT_S)
        elif tipo == "conexao":
            await asyncio.sleep(5)
            await _aguardar_proxy(rotulo)
        else:
            await asyncio.sleep(BACKOFF_S[contagem[tipo] - 1])


# ---------------------------------------------------------------------------
# Fase COM RAG
# ---------------------------------------------------------------------------
async def _executar_com_rag(pipeline, db, consulta_def: dict) -> dict:
    cid, consulta = consulta_def["id"], consulta_def["consulta"]
    base = {"id": cid, "tipo": cid[0], "consulta": consulta,
            "esperado": consulta_def["esperado"]}

    async def tentativa():
        # Execução limpa (sem cache) para medir latência real.
        query_cache._cache.pop(f"hibrido:{consulta}", None)
        _EVENTOS_CAPTURADOS.clear()
        inicio = time.perf_counter()
        try:
            with get_usage_metadata_callback() as uso_cb:
                resultado = await pipeline.processar(consulta, db, modo="hibrido")
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
            raise
        latencia = int((time.perf_counter() - inicio) * 1000)
        fallbacks = list(_EVENTOS_CAPTURADOS)
        problema = "; ".join(f"{f['evento']}: {f['erro']}" for f in fallbacks) or None
        return (resultado, latencia, dict(uso_cb.usage_metadata), fallbacks), problema

    payload, problema, falhas = await _com_retentativas(tentativa, f"COM RAG {cid}")
    if payload is None:
        return {**base, "sucesso": False, "erro": problema,
                "tentativas_descartadas": falhas}

    resultado, latencia, uso, fallbacks = payload
    meta = resultado.get("metadata", {})
    return {
        **base,
        "sucesso": True,
        "latencia_ms": latencia,
        "resposta": resultado.get("resposta", ""),
        "resumo": resultado.get("resumo"),
        "sugestoes_followup": resultado.get("sugestoes_followup", []),
        "fontes": resultado.get("fontes", []),
        "disclaimer": resultado.get("disclaimer"),
        "entidades": meta.get("entidades"),
        "estrategia_hibrida": meta.get("estrategia_hibrida"),
        "num_resultados": meta.get("num_resultados"),
        "total_no_banco": meta.get("total_no_banco"),
        "dados_completos": meta.get("dados_completos"),
        "referencia": construir_referencia(resultado, consulta_def),
        "dados": resultado.get("dados", []),
        "metricas": avaliar_resultado(resultado, consulta_def, calcular_bert=False),
        "uso_tokens": uso,
        # Só preenchido se a tentativa ACEITA ainda caiu em fallback (após esgotar
        # as retentativas); vazio = todas as etapas usaram a resposta do modelo.
        "fallbacks_aceitos": fallbacks if problema else [],
        "tentativas_descartadas": falhas,
    }


async def fase_com_rag(consultas: list[dict], dir_parcial: Path, modelo: str) -> None:
    from app.database import SessionLocal
    from app.services.rag.pipeline import RAGPipeline

    ckpt = dir_parcial / "com_rag.jsonl"
    feitos = {k for k, v in _carregar_jsonl(ckpt).items() if v.get("sucesso")}
    pendentes = [c for c in consultas if c["id"] not in feitos]
    print(f"[COM RAG] modelo={modelo} pendentes={len(pendentes)}/{len(consultas)}",
          flush=True)
    if not pendentes:
        return

    db = SessionLocal()
    try:
        pipeline = RAGPipeline()
        # Carrega o modelo de embeddings ANTES de medir latência.
        pipeline.decomposer.embedder.encode("aquecimento")
        await _aguardar_proxy("COM RAG")
        _registrar_meta(dir_parcial / "meta_com_rag.json", modelo,
                        {"base_de_dados": _snapshot_base(db)})

        for i, consulta_def in enumerate(pendentes, 1):
            reg = await _executar_com_rag(pipeline, db, consulta_def)
            _anexar_jsonl(ckpt, reg)
            if reg["sucesso"]:
                m = reg["metricas"]
                aviso = " FALLBACK" if reg["fallbacks_aceitos"] else ""
                print(f"[COM RAG {i}/{len(pendentes)}] {reg['id']:<4} "
                      f"lat={reg['latencia_ms'] / 1000:5.1f}s num={reg['num_resultados']} "
                      f"banco={reg['total_no_banco']} ret-F1={m['f1_score']:.2f}{aviso}",
                      flush=True)
            else:
                print(f"[COM RAG {i}/{len(pendentes)}] {reg['id']:<4} ERRO {reg['erro']}",
                      flush=True)

        _registrar_meta(dir_parcial / "meta_com_rag.json", modelo,
                        {"fim": timestamp_conteudo()})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fase SEM RAG
# ---------------------------------------------------------------------------
async def _executar_sem_rag(llm, consulta_def: dict) -> dict:
    cid, consulta = consulta_def["id"], consulta_def["consulta"]
    base = {"id": cid, "tipo": cid[0], "consulta": consulta,
            "esperado": consulta_def["esperado"]}
    mensagens = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=consulta)]

    async def tentativa():
        inicio = time.perf_counter()
        resp = await llm.ainvoke(mensagens)
        latencia = int((time.perf_counter() - inicio) * 1000)
        resposta = _texto(resp.content)
        finish = (resp.response_metadata or {}).get("finish_reason")
        problema = None if resposta.strip() else f"resposta vazia (finish_reason={finish})"
        return (resp, resposta, finish, latencia), problema

    payload, problema, falhas = await _com_retentativas(tentativa, f"SEM RAG {cid}")
    if payload is None or problema is not None:
        return {**base, "sucesso": False, "erro": problema,
                "tentativas_descartadas": falhas}

    resp, resposta, finish, latencia = payload
    return {
        **base,
        "sucesso": True,
        "latencia_ms": latencia,
        "resposta": resposta,
        "finish_reason": finish,
        "uso_tokens": dict(resp.usage_metadata or {}),
        "tentativas_descartadas": falhas,
    }


async def fase_sem_rag(consultas: list[dict], dir_parcial: Path, modelo: str) -> None:
    ckpt = dir_parcial / "sem_rag.jsonl"
    feitos = {k for k, v in _carregar_jsonl(ckpt).items() if v.get("sucesso")}
    pendentes = [c for c in consultas if c["id"] not in feitos]
    print(f"[SEM RAG] modelo={modelo} pendentes={len(pendentes)}/{len(consultas)}",
          flush=True)
    if not pendentes:
        return

    # Mesmo modelo e temperatura 0 (determinismo), igual ao baseline existente.
    llm = build_chat_llm(temperature=0.0)
    await _aguardar_proxy("SEM RAG")
    _registrar_meta(dir_parcial / "meta_sem_rag.json", modelo)

    for i, consulta_def in enumerate(pendentes, 1):
        reg = await _executar_sem_rag(llm, consulta_def)
        _anexar_jsonl(ckpt, reg)
        if reg["sucesso"]:
            print(f"[SEM RAG {i}/{len(pendentes)}] {reg['id']:<4} "
                  f"lat={reg['latencia_ms'] / 1000:5.1f}s len={len(reg['resposta'])}",
                  flush=True)
        else:
            print(f"[SEM RAG {i}/{len(pendentes)}] {reg['id']:<4} ERRO {reg['erro']}",
                  flush=True)

    _registrar_meta(dir_parcial / "meta_sem_rag.json", modelo,
                    {"fim": timestamp_conteudo()})


# ---------------------------------------------------------------------------
# Fase de pontuação (BERTScore) + agregação + escrita final
# ---------------------------------------------------------------------------
def _aplicar_bert(metricas: dict, bert: dict | None) -> None:
    metricas["bertscore_precision"] = bert["precision"] if bert else None
    metricas["bertscore_recall"] = bert["recall"] if bert else None
    metricas["bertscore_f1"] = bert["f1"] if bert else None


def _media(valores: list[float]) -> float | None:
    return sum(valores) / len(valores) if valores else None


def _agregar(registros: list[dict]) -> dict:
    sucessos = [r for r in registros if r.get("sucesso")]
    out: dict = {"total": len(registros), "sucessos": len(sucessos)}
    for chave in METRICAS_RETRIEVAL + METRICAS_BERT:
        out[chave] = _media([r["metricas"][chave] for r in sucessos
                             if r.get("metricas", {}).get(chave) is not None])
    latencias = [r["latencia_ms"] for r in sucessos]
    out["latencia_media_ms"] = _media(latencias)
    out["latencia_mediana_ms"] = statistics.median(latencias) if latencias else None
    out["n_bertscore"] = sum(1 for r in sucessos
                             if r["metricas"].get("bertscore_f1") is not None)
    return out


def _agregar_cenario(registros: list[dict], tem_registros: dict[str, bool]) -> dict:
    por_tipo = {}
    for tipo in "ABCD":
        sub = [r for r in registros if r["tipo"] == tipo]
        if sub:
            por_tipo[tipo] = _agregar(sub)
    por_disp = {
        "com_registros_no_banco": _agregar(
            [r for r in registros if tem_registros.get(r["id"])]),
        "sem_registros_no_banco": _agregar(
            [r for r in registros if not tem_registros.get(r["id"])]),
    }
    return {"geral": _agregar(registros), "por_tipo": por_tipo,
            "por_disponibilidade_de_dados": por_disp}


def _fmt(valor, pct: bool = True) -> str:
    if valor is None:
        return "   N/A"
    return f"{valor:>7.2%}" if pct else f"{valor:>7.1f}"


def _imprimir_relatorio(modelo, base, agg_com, agg_sem, n_com_dados, total) -> None:
    linha = "=" * 96
    print(f"\n{linha}", flush=True)
    print(f"  120 CONSULTAS — COM RAG (híbrido) vs. SEM RAG (direto) · modelo {modelo}",
          flush=True)
    print(linha, flush=True)
    print(f"  Base: {base.get('emendas')} emendas | consultas com registros no banco: "
          f"{n_com_dados}/{total}", flush=True)
    print(f"  {'Tipo':<10} {'Cenário':<8} {'Suc.':>7} {'BERT-P':>7} {'BERT-R':>7} "
          f"{'BERT-F1':>7} {'ΔF1':>7} {'Lat(s)':>7} {'ret-F1':>7} {'NDCG@5':>7} "
          f"{'Entity':>7}", flush=True)
    print("  " + "-" * 92, flush=True)

    blocos = [(t, NOMES_TIPO[t], agg_com["por_tipo"].get(t), agg_sem["por_tipo"].get(t))
              for t in "ABCD"]
    blocos.append(("G", "GERAL", agg_com["geral"], agg_sem["geral"]))
    for _, nome, c, s in blocos:
        if not c or not s:
            continue
        delta = (c["bertscore_f1"] - s["bertscore_f1"]
                 if c["bertscore_f1"] is not None and s["bertscore_f1"] is not None
                 else None)
        for cen, a, d in (("COM RAG", c, None), ("SEM RAG", s, delta)):
            lat = a["latencia_media_ms"] / 1000 if a["latencia_media_ms"] else None
            print(f"  {nome if cen == 'COM RAG' else '':<10} {cen:<8} "
                  f"{a['sucessos']:>3}/{a['total']:<3} "
                  f"{_fmt(a['bertscore_precision'])} {_fmt(a['bertscore_recall'])} "
                  f"{_fmt(a['bertscore_f1'])} "
                  f"{(f'{d:>+7.3f}' if d is not None else ' ' * 7)} "
                  f"{_fmt(lat, pct=False)} {_fmt(a['f1_score'])} "
                  f"{_fmt(a['ndcg_at_5'])} {_fmt(a['entity_accuracy'])}", flush=True)
        print("  " + "-" * 92, flush=True)

    for chave, rotulo in (("com_registros_no_banco", "com registros no banco"),
                          ("sem_registros_no_banco", "sem registros no banco")):
        c = agg_com["por_disponibilidade_de_dados"][chave]
        s = agg_sem["por_disponibilidade_de_dados"][chave]
        print(f"  BERT-F1 {rotulo:<24} (n={c['total']:>3}): COM RAG {_fmt(c['bertscore_f1'])}"
              f" | SEM RAG {_fmt(s['bertscore_f1'])}", flush=True)
    print(linha, flush=True)


def fase_pontuar(consultas: list[dict], dir_parcial: Path, saida: Path,
                 modelo: str, manter_parciais: bool) -> None:
    com = _carregar_jsonl(dir_parcial / "com_rag.jsonl")
    sem = _carregar_jsonl(dir_parcial / "sem_rag.jsonl")
    faltando = ([f"COM RAG:{c['id']}" for c in consultas if c["id"] not in com]
                + [f"SEM RAG:{c['id']}" for c in consultas if c["id"] not in sem])
    if faltando:
        raise SystemExit(f"Geração incompleta ({len(faltando)} pendentes): "
                         f"{', '.join(faltando[:20])}... Rode as fases de geração de novo.")

    ordem = [c["id"] for c in consultas]
    res_com = [com[i] for i in ordem]
    res_sem = [sem[i] for i in ordem]

    print(f"[PONTUAR] BERTScore de {len(ordem)} x 2 respostas...", flush=True)
    for n, (rc, rs) in enumerate(zip(res_com, res_sem), 1):
        referencia = rc.get("referencia") if rc.get("sucesso") else None
        if rc.get("sucesso"):
            candidato = rc.get("resposta") or rc.get("resumo") or ""
            _aplicar_bert(rc["metricas"], calcular_bertscore(candidato, referencia))
        if rs.get("sucesso"):
            rs["referencia"] = referencia
            rs["metricas"] = {k: None for k in METRICAS_RETRIEVAL}
            _aplicar_bert(rs["metricas"],
                          calcular_bertscore(rs["resposta"], referencia)
                          if referencia else None)
        if n % 20 == 0:
            print(f"[PONTUAR] {n}/{len(ordem)}", flush=True)

    # Disponibilidade de dados por consulta (define a referência dos 2 cenários).
    tem_registros = {r["id"]: bool(r.get("total_no_banco")) for r in res_com}
    n_com_dados = sum(tem_registros.values())

    agg_com = _agregar_cenario(res_com, tem_registros)
    agg_sem = _agregar_cenario(res_sem, tem_registros)
    n_recuperados = sum(1 for r in res_com if r.get("num_resultados"))
    for agg, regs in ((agg_com, res_com), (agg_sem, res_sem)):
        agg["geral"]["consultas_com_registros_no_banco"] = n_com_dados
        agg["geral"]["consultas_com_dados_recuperados_com_rag"] = n_recuperados
        agg["geral"]["consultas_com_fallback_aceito"] = sum(
            1 for r in regs if r.get("fallbacks_aceitos"))
        agg["geral"]["tentativas_descartadas"] = sum(
            len(r.get("tentativas_descartadas") or []) for r in regs)

    meta_com = _ler_json(dir_parcial / "meta_com_rag.json")
    meta_sem = _ler_json(dir_parcial / "meta_sem_rag.json")
    base = meta_com.get("base_de_dados", {})
    bert_cfg = {
        "metrica": "BERTScore (Zhang et al., ICLR 2020)",
        "modelo": "bert-base-multilingual-cased (lang='pt')",
        "rescale_with_baseline": True,
        "candidato_com_rag": "campo 'resposta' do synthesizer (ou 'resumo' se vazio)",
        "referencia": "texto factual determinístico (construir_referencia) a partir "
                      "dos dados recuperados no COM RAG — a mesma nos dois cenários",
    }

    agora = datetime.now()
    caminho_com = caminho_com_prefixo(saida, "resultados_120_com_rag_hibrido.json",
                                      modelo, agora)
    caminho_sem = caminho_com_prefixo(saida, "resultados_120_sem_rag_direto.json",
                                      modelo, agora)
    for caminho in (caminho_com, caminho_sem):
        if caminho.exists():
            raise SystemExit(f"Recusado: {caminho} já existe (não sobrescrevo resultados).")

    relatorio_com = {
        "cenario": "com_rag",
        "modo": "hibrido",
        "conjunto": "oficial_120",
        "descricao": "Pipeline RAG completo do projeto (interpreter -> planner + "
                     "decomposer -> busca SQL/vetorial/RRF -> synthesizer com os "
                     "dados recuperados).",
        "modelo": modelo,
        "provedor": meta_com.get("provedor"),
        "timestamp": timestamp_conteudo(agora),
        "execucao": {"inicio": meta_com.get("inicio"), "fim": meta_com.get("fim")},
        "base_de_dados": base,
        "configuracao": {
            "temperaturas": {"interpreter": 0.0, "planner": 0.0, "synthesizer": 0.1},
            "limite_tokens": "não enviado (proxy LiteLLM rejeita max_tokens)",
            "retentativas": POLITICA_RETENTATIVAS,
            "bertscore": bert_cfg,
        },
        **agg_com,
        "resultados": res_com,
    }
    relatorio_sem = {
        "cenario": "sem_rag",
        "modo": "baseline_direto",
        "conjunto": "oficial_120",
        "descricao": "Pergunta enviada direto ao modelo, sem RAG (sem interpreter/"
                     "planner/retrieval/synthesizer). BERTScore medido contra a mesma "
                     "referência factual do cenário COM RAG.",
        "modelo": modelo,
        "provedor": meta_sem.get("provedor"),
        "timestamp": timestamp_conteudo(agora),
        "execucao": {"inicio": meta_sem.get("inicio"), "fim": meta_sem.get("fim")},
        "base_de_dados": base,
        "referencias_de": caminho_com.name,
        "configuracao": {
            "temperatura": 0.0,
            "system_prompt": SYSTEM_PROMPT,
            "limite_tokens": "não enviado (proxy LiteLLM rejeita max_tokens)",
            "retentativas": POLITICA_RETENTATIVAS,
            "bertscore": bert_cfg,
        },
        **agg_sem,
        "resultados": res_sem,
    }
    _gravar_json(caminho_com, relatorio_com)
    _gravar_json(caminho_sem, relatorio_sem)

    _imprimir_relatorio(modelo, base, agg_com, agg_sem, n_com_dados, len(ordem))
    print(f"COM RAG salvo em: {caminho_com}", flush=True)
    print(f"SEM RAG salvo em: {caminho_sem}", flush=True)

    falhas_finais = [f"{cen}:{r['id']}" for cen, regs in (("COM RAG", res_com),
                                                         ("SEM RAG", res_sem))
                     for r in regs if not r.get("sucesso")]
    if falhas_finais:
        print(f"ATENÇÃO: {len(falhas_finais)} consulta(s) sem sucesso: "
              f"{', '.join(falhas_finais)}. Checkpoints mantidos em {dir_parcial} — "
              "rode a geração de novo para repetir só essas.", flush=True)
    elif not manter_parciais:
        for nome in ("com_rag.jsonl", "sem_rag.jsonl",
                     "meta_com_rag.json", "meta_sem_rag.json"):
            (dir_parcial / nome).unlink(missing_ok=True)
        try:
            dir_parcial.rmdir()
        except OSError:
            pass


def _rodar_paralelo(args) -> None:
    """COM RAG e SEM RAG em 2 processos (event loops independentes), depois pontua.

    Processos separados evitam que o código síncrono do pipeline (SQL,
    embeddings) bloqueie o event loop e infle a latência do SEM RAG.
    """
    base = [sys.executable, "-u", str(Path(__file__).resolve())]
    for flag, valor in (("--modelo", args.modelo), ("--ids", args.ids),
                        ("--saida", str(args.saida) if args.saida else None),
                        ("--tag", args.tag)):
        if valor:
            base += [flag, valor]
    filhos = {fase: subprocess.Popen(base + ["--fase", fase])
              for fase in ("com_rag", "sem_rag")}
    codigos = {fase: proc.wait() for fase, proc in filhos.items()}
    print(f"[PARALELO] códigos de saída: {codigos}", flush=True)
    if any(codigos.values()):
        raise SystemExit("Geração não concluída; rode de novo para retomar do checkpoint.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", help="Modelo do proxy LiteLLM (força "
                        "LLM_PROVIDER=litellm). Sem ele, usa o provedor/modelo do .env.")
    parser.add_argument("--fase", default="todas",
                        choices=["com_rag", "sem_rag", "pontuar", "todas", "paralelo"])
    parser.add_argument("--ids", help="Subconjunto de ids separados por vírgula "
                        "(ex.: A1,B2) — útil para smoke test.")
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO,
                        help="Diretório dos JSONs finais (default: tests/test_queries).")
    parser.add_argument("--tag", help="Nome do checkpoint (default: nome do modelo).")
    parser.add_argument("--manter-parciais", action="store_true",
                        help="Não remove os checkpoints após gravar os JSONs finais.")
    args = parser.parse_args()

    if args.modelo:
        # settings é singleton compartilhado: vale para todo build_chat_llm().
        settings.LLM_PROVIDER = "litellm"
        settings.LITELLM_MODEL = args.modelo
    # Sem retentativas escondidas no SDK: quem repete é _com_retentativas, fora
    # do cronômetro (ver POLITICA_RETENTATIVAS).
    settings.LLM_MAX_RETRIES = 0
    settings.LLM_TIMEOUT_S = TIMEOUT_LLM_S
    modelo = _modelo_ativo()
    _configurar_logs()

    consultas = TODAS_CONSULTAS
    if args.ids:
        ids = {i.strip() for i in args.ids.split(",") if i.strip()}
        consultas = [c for c in TODAS_CONSULTAS if c["id"] in ids]
        desconhecidos = ids - {c["id"] for c in consultas}
        if desconhecidos:
            raise SystemExit(f"Ids desconhecidos: {sorted(desconhecidos)}")

    saida = args.saida.resolve()
    saida.mkdir(parents=True, exist_ok=True)
    dir_parcial = saida / f"_parcial_{_tag_segura(args.tag or modelo)}"
    dir_parcial.mkdir(exist_ok=True)

    if args.fase == "paralelo":
        _rodar_paralelo(args)
        fase_pontuar(consultas, dir_parcial, saida, modelo, args.manter_parciais)
        return

    async def _gerar() -> None:
        # Um único event loop: o langchain-openai reaproveita um httpx.AsyncClient
        # em cache, que não pode atravessar loops diferentes.
        if args.fase in ("com_rag", "todas"):
            await fase_com_rag(consultas, dir_parcial, modelo)
        if args.fase in ("sem_rag", "todas"):
            await fase_sem_rag(consultas, dir_parcial, modelo)

    if args.fase != "pontuar":
        asyncio.run(_gerar())
    if args.fase in ("pontuar", "todas"):
        fase_pontuar(consultas, dir_parcial, saida, modelo, args.manter_parciais)


if __name__ == "__main__":
    main()
