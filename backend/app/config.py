from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # TCC_Projeto/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_PROJECT_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Banco de Dados (Supabase)
    DATABASE_URL: str = "postgresql://postgres:secret@localhost:5432/transparencia_fiscal"

    # Provedor de LLM: "ollama" (local, llama3.1), "anthropic" (Claude) ou
    # "litellm" (proxy compatível com OpenAI — roteia para claude/gemini/gpt).
    LLM_PROVIDER: str = "ollama"

    # LLM local via Ollama (http://localhost:11434). Rode `ollama pull llama3.1`.
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # API LLM (Anthropic Claude) — usado apenas quando LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_MAX_TOKENS: int = 2048

    # Proxy LiteLLM (API compatível com OpenAI) — usado quando LLM_PROVIDER=litellm.
    # O LiteLLM roteia para o provedor real conforme LITELLM_MODEL (ex:
    # "claude-sonnet-4", "claude-haiku-3-5", "gemini-2.5-flash", "gpt-4o-mini").
    LITELLM_API_KEY: str = ""
    LITELLM_BASE_URL: str = "http://localhost:4000"
    LITELLM_MODEL: str = "gpt-4o-mini"

    # Retentativas/timeout do cliente HTTP dos provedores via API (litellm e
    # anthropic). None = default do SDK (2 retentativas, timeout longo). A
    # avaliação usa 0 retentativas para que nenhuma espera por 429 fique
    # escondida dentro da latência medida.
    LLM_MAX_RETRIES: int | None = None
    LLM_TIMEOUT_S: float | None = None

    # API Portal da Transparência (CGU)
    CGU_API_KEY: str = ""
    CGU_API_KEYS: str = ""  # Comma-separated extra keys for rotation
    CGU_API_BASE_URL: str = "https://api.portaldatransparencia.gov.br/api-de-dados"
    CGU_RATE_LIMIT_RPM: int = 400
    # Limite por chave das 00h às 06h (horário de Brasília), quando a API permite
    # mais requisições. 0 = usar CGU_RATE_LIMIT_RPM o dia todo.
    CGU_RATE_LIMIT_RPM_NOITE: int = 0

    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIMENSIONS: int = 384
    EMBEDDING_BATCH_SIZE: int = 256

    # HNSW Index
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200

    # Busca
    SIMILARITY_THRESHOLD: float = 0.45
    MAX_RESULTS: int = 20
    MAX_RESULTS_CAP: int = 50  # Limite máximo absoluto (planner pode sugerir até este valor)

    # RRF (Reciprocal Rank Fusion)
    RRF_K: int = 60  # Parâmetro k do RRF (Cormack et al., 2009)
    RRF_FETCH_MULTIPLIER: int = 3  # Buscar N*limit candidatos para fusão
    RRF_WEIGHT_SQL: float = 0.7  # Peso do ranking SQL na fusão RRF
    RRF_WEIGHT_VECTOR: float = 0.3  # Peso do ranking vetorial na fusão RRF
    RRF_SIMILARITY_GATE: float = 0.55  # Similaridade mínima para vetor entrar na fusão

    # Aplicação
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:5173"
    PRODUCTION_URL: str = ""
    LOG_LEVEL: str = "INFO"


settings = Settings()
