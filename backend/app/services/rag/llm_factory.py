"""Factory de LLM para o pipeline RAG.

Centraliza a criação do modelo de linguagem usado pelos componentes do RAG
(interpreter, query_planner, synthesizer), permitindo alternar entre um modelo
local via Ollama (ex: llama3.1), a API da Anthropic (Claude) e um proxy LiteLLM
(API compatível com OpenAI que roteia para claude/gemini/gpt) apenas por
configuração (settings.LLM_PROVIDER), sem alterar o código dos serviços.

Referência: LangChain Ollama integration — ChatOllama (langchain-ollama).
Ollama expõe um servidor local em http://localhost:11434 e o modelo precisa
ter sido baixado antes (ex: `ollama pull llama3.1`).
"""
from langchain_core.language_models.chat_models import BaseChatModel
from app.config import settings
import structlog

logger = structlog.get_logger()


def build_chat_llm(
    *,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Constrói o chat model de acordo com settings.LLM_PROVIDER.

    Args:
        max_tokens: limite de tokens de saída. Para Ollama vira ``num_predict``;
            para Anthropic e LiteLLM (OpenAI-compatível) vira ``max_tokens``.
            Se None, usa o default do provedor.
        temperature: temperatura de amostragem (0 = determinístico).

    Returns:
        Uma instância de BaseChatModel (ChatOllama, ChatAnthropic ou ChatOpenAI
        via LiteLLM) com a mesma interface LangChain (``invoke`` / ``ainvoke`` /
        pipe ``|``), de modo que os serviços RAG não precisam saber qual
        provedor está ativo.
    """
    provider = (settings.LLM_PROVIDER or "ollama").lower()

    if provider == "ollama":
        # Import tardio: só exige langchain-ollama quando o provedor é usado.
        from langchain_ollama import ChatOllama

        kwargs: dict = {
            "model": settings.OLLAMA_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "temperature": temperature,
        }
        # No Ollama o limite de tokens de saída é num_predict (não max_tokens).
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        logger.info(
            "llm_provider_selecionado",
            provider="ollama",
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        return ChatOllama(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": settings.ANTHROPIC_MODEL,
            "api_key": settings.ANTHROPIC_API_KEY,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if settings.LLM_MAX_RETRIES is not None:
            kwargs["max_retries"] = settings.LLM_MAX_RETRIES
        if settings.LLM_TIMEOUT_S is not None:
            kwargs["default_request_timeout"] = settings.LLM_TIMEOUT_S
        logger.info(
            "llm_provider_selecionado",
            provider="anthropic",
            model=settings.ANTHROPIC_MODEL,
        )
        return ChatAnthropic(**kwargs)

    if provider == "litellm":
        # LiteLLM expõe uma API compatível com OpenAI, então usamos ChatOpenAI
        # apontando base_url para o proxy. O modelo real (claude/gemini/gpt) é
        # escolhido por settings.LITELLM_MODEL e roteado pelo próprio LiteLLM.
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.LITELLM_MODEL,
            "api_key": settings.LITELLM_API_KEY,
            "base_url": settings.LITELLM_BASE_URL,
            "temperature": temperature,
        }
        if settings.LLM_MAX_RETRIES is not None:
            kwargs["max_retries"] = settings.LLM_MAX_RETRIES
        if settings.LLM_TIMEOUT_S is not None:
            kwargs["timeout"] = settings.LLM_TIMEOUT_S
        # NÃO enviamos limite de tokens para o LiteLLM: este proxy acaba
        # setando max_tokens E max_completion_tokens no backend e rejeita a
        # chamada com HTTP 400 ("Setting 'max_tokens' and 'max_completion_tokens'
        # at the same time is not supported"). Os prompts já pedem respostas
        # concisas e os modelos têm defaults de saída adequados, então o
        # max_tokens solicitado é intencionalmente ignorado aqui.
        if max_tokens is not None:
            logger.debug(
                "litellm_max_tokens_ignorado",
                motivo="proxy rejeita max_tokens+max_completion_tokens juntos",
                solicitado=max_tokens,
            )
        logger.info(
            "llm_provider_selecionado",
            provider="litellm",
            model=settings.LITELLM_MODEL,
            base_url=settings.LITELLM_BASE_URL,
        )
        return ChatOpenAI(**kwargs)

    raise ValueError(
        f"LLM_PROVIDER inválido: {settings.LLM_PROVIDER!r}. "
        "Use 'ollama' (local, llama3.1), 'anthropic' (Claude) ou "
        "'litellm' (proxy OpenAI-compatível)."
    )
