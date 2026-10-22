"""Lança o orquestrador ou o worker de embeddings desligado do terminal.

Uso (de backend/):
    py scripts/lancar_destacado.py orquestrador   # popular_base_nacional.py
    py scripts/lancar_destacado.py embeddings     # embeddings_paralelo_sqlite.py

Imprime o PID e sai na hora; a saída do processo vai para data/logs. O processo
sobrevive ao fechamento do terminal (ou da sessão que o lançou) e recebe um TMP
que não some: o temp do perfil do usuário. O orquestrador tem trava própria
(data/logs/pipeline.lock), então lançar duas vezes não duplica a coleta.
"""
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
LOGS = BACKEND / "data" / "logs"
TMP = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Temp"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _url_nacional() -> str:
    """DATABASE_URL do .env com o banco trocado para transparencia_nacional.

    O evaluate COM RAG lê os dados do Postgres nacional; o .env aponta para o
    seed (transparencia_fiscal), então reescrevemos só o nome do banco.
    """
    from urllib.parse import urlparse
    from dotenv import dotenv_values
    url = (dotenv_values(BACKEND.parent / ".env").get("DATABASE_URL") or "").strip()
    u = urlparse(url)
    return u._replace(path="/transparencia_nacional").geturl()

# alvo: (argumentos, log, variáveis extras, variáveis removidas, prioridade)
ALVOS = {
    # O orquestrador escolhe o banco de cada etapa; não pode herdar DATABASE_URL.
    "orquestrador": (["scripts/popular_base_nacional.py"], "00_pipeline.stdout.log",
                     {}, ("DATABASE_URL", "PG_DESTINO_URL"), 0),
    # Prioridade baixa: a coleta e o resto da máquina vêm primeiro.
    "embeddings": (["scripts/embeddings_paralelo_sqlite.py"], "09_embeddings_paralelo.log",
                   {"DATABASE_URL": "sqlite:///data/transparencia_local.db"},
                   ("PG_DESTINO_URL",), BELOW_NORMAL_PRIORITY_CLASS),
    # Avaliação 120 COM/SEM RAG no provedor do .env (ollama local). Lê do Postgres
    # nacional (não o seed transparencia_fiscal). A fase paralelo faz spawn dos
    # 2 subprocessos com_rag/sem_rag e depois pontua.
    "avaliar_ollama": (["scripts/evaluate_120_com_sem_rag.py", "--fase", "paralelo"],
                       "10_avaliar_ollama.log",
                       {"DATABASE_URL": _url_nacional()}, ("PG_DESTINO_URL",), 0),
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ALVOS:
        sys.exit(f"uso: py scripts/lancar_destacado.py {'|'.join(ALVOS)}")
    alvo = sys.argv[1]
    args, nome_log, extra, remover, prioridade = ALVOS[alvo]
    TMP.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NO_COLOR="1",
               TMP=str(TMP), TEMP=str(TMP), **extra)
    for chave in (*remover, "KIROCREW_SCRATCH"):
        env.pop(chave, None)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / nome_log, "ab") as saida:
        proc = subprocess.Popen(
            [sys.executable, "-u", *args], cwd=BACKEND, env=env,
            stdin=subprocess.DEVNULL, stdout=saida, stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                           | CREATE_BREAKAWAY_FROM_JOB | prioridade))
    print(f"{alvo} pid={proc.pid} TMP={TMP} log={LOGS / nome_log}")


if __name__ == "__main__":
    main()
