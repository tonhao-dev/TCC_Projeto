"""Abre 3 terminais visiveis e desacoplados (COM RAG, SEM RAG, PONTUAR) para o qwen2.5:7b.

Cada terminal roda scripts/_qwen_fase.ps1 <fase> em janela propria e sobrevive ao fim
da sessao que o lancou (CREATE_BREAKAWAY_FROM_JOB), mesmo padrao de _gemma_launch.py.
As fases com_rag/sem_rag retomam do checkpoint JSONL; pontuar espera as duas terminarem.
"""
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "_qwen_fase.ps1"
PWSH = r"C:\Program Files\PowerShell\7\pwsh.exe"

CREATE_NEW_CONSOLE = 0x00000010
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

for fase in ("com_rag", "sem_rag", "pontuar"):
    proc = subprocess.Popen(
        [PWSH, "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(SCRIPT), fase],
        stdin=subprocess.DEVNULL, close_fds=True,
        creationflags=CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB)
    print(f"{fase} pid={proc.pid}")
    time.sleep(2)
