"""Backup do trabalho do TCC para o GitHub (fork tonhao-dev/TCC_Projeto).

Faz, nesta ordem:
  1. Para o orquestrador, a coleta e o worker de embeddings (eles retomam do
     checkpoint depois, nada se perde).
  2. PRAGMA wal_checkpoint(TRUNCATE) + integrity_check no SQLite.
  3. Compacta data/transparencia_local.db em 7z (ou zip) e mede o tamanho.
  4. Cria a branch backup/base-nacional-<data> se ainda não existir.
  5. Commita código novo/alterado + JSONs de resultado + .gitattributes.
  6. Rastreia o .7z no Git LFS, add -f, commit.
  7. git push origin <branch> (o LFS sobe o binário).
  8. NÃO relança a coleta (quem chama decide; use lancar_destacado.py).

Uso (de backend/):
    py scripts/backup_base.py --preparar     # só para processos + checkpoint + compacta + mede
    py scripts/backup_base.py                # faz tudo, mas PARA antes do push (mostra o plano)
    py scripts/backup_base.py --push         # faz tudo e dá push de verdade
    py scripts/backup_base.py --relancar     # ao final, relança coleta + embeddings

O push exige confirmação explícita (--push) porque sobe ~centenas de MB via LFS
para a cota da conta tonhao-dev (plano grátis: 1 GB armazenamento + 1 GB/mês).
"""
import argparse
import ctypes
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
DATA = BACKEND / "data"
LOGS = DATA / "logs"
SQLITE = DATA / "transparencia_local.db"
BRANCH = f"backup/base-nacional-{date.today():%Y%m%d}"
# Caminho do zip DENTRO do repo (versionado via LFS). Fica em backend/data/, que o
# .gitignore ignora por *.db mas não por *.7z; mesmo assim entra por add -f.
ZIP_REL = f"backend/data/transparencia_local_{date.today():%Y%m%d}.7z"
ZIP_ABS = REPO / ZIP_REL
SETE_ZIP = r"C:\Program Files\7-Zip\7z.exe"

# Processos a parar: casam pelo nome do script na linha de comando.
ALVOS_CMD = ("popular_base_nacional.py", "run_ingestion_brasil.py",
             "run_ingestion_sancoes.py", "run_ingestion_ac.py",
             "embeddings_paralelo_sqlite.py", "generate_embeddings_lote.py")

# Resultados a versionar (os JSONs/relatórios/gráficos; NUNCA o .env).
PADROES_RESULTADO = ["backend/tests/test_queries/*.json",
                     "backend/tests/test_queries/*.md",
                     "backend/tests/test_queries/_parcial_*/",
                     "backend/tests/test_queries/_descartado_*/",
                     "backend/tests/test_queries/graficos/"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cp = subprocess.run(["git", *args], cwd=REPO, text=True,
                        capture_output=capture, encoding="utf-8")
    if check and cp.returncode != 0:
        saida = (cp.stderr or cp.stdout or "").strip()
        raise SystemExit(f"git {' '.join(args)} falhou ({cp.returncode}): {saida}")
    return cp


def _pid_vivo(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    codigo = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(codigo))
    k32.CloseHandle(h)
    return bool(ok) and codigo.value == 259


def _listar_pids_coleta() -> list[int]:
    """PIDs dos processos da coleta, via CIM (wmic foi removido do Windows recente).

    Lê ProcessId+CommandLine com PowerShell/Get-CimInstance e filtra pelos nomes
    de script em ALVOS_CMD. Saída em linhas 'PID\\tCOMMANDLINE'.
    """
    ps = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe' or name='py.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        text=True, capture_output=True, encoding="utf-8", errors="replace").stdout
    pids = []
    for linha in out.splitlines():
        if "\t" not in linha or not any(alvo in linha for alvo in ALVOS_CMD):
            continue
        pid_txt = linha.split("\t", 1)[0].strip()
        if pid_txt.isdigit():
            pids.append(int(pid_txt))
    return pids


def parar_processos() -> list[int]:
    """Encerra os processos da coleta. Retorna os pids encerrados."""
    mortos = []
    for pid in _listar_pids_coleta():
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True)
        mortos.append(pid)
        log(f"  encerrado pid {pid}")
    for _ in range(20):  # espera os handles do SQLite soltarem
        if not any(_pid_vivo(p) for p in mortos):
            break
        time.sleep(0.5)
    return mortos


def checkpoint_sqlite() -> None:
    """Passa o WAL para o .db principal e confere a integridade."""
    conn = sqlite3.connect(SQLITE, timeout=120)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        (res,) = conn.execute("PRAGMA integrity_check").fetchone()
        if res != "ok":
            raise SystemExit(f"integrity_check falhou: {res}")
        conn.execute("VACUUM")  # compacta páginas livres antes de zipar
    finally:
        conn.close()
    for sufixo in ("-wal", "-shm"):
        extra = SQLITE.with_name(SQLITE.name + sufixo)
        if extra.exists() and extra.stat().st_size == 0:
            extra.unlink(missing_ok=True)
    log(f"  integrity_check ok; .db = {SQLITE.stat().st_size / 1e6:.0f} MB")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def compactar() -> float:
    """Compacta o SQLite. Retorna o tamanho do .7z em MB."""
    if ZIP_ABS.exists():
        ZIP_ABS.unlink()
    if Path(SETE_ZIP).exists():
        subprocess.run([SETE_ZIP, "a", "-t7z", "-mx=5", str(ZIP_ABS), str(SQLITE)],
                       check=True, capture_output=True, text=True)
    else:  # zip da stdlib como alternativa
        import zipfile
        zpath = ZIP_ABS.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(SQLITE, SQLITE.name)
        return zpath.stat().st_size / 1e6
    mb = ZIP_ABS.stat().st_size / 1e6
    sha = _sha256(ZIP_ABS)
    (ZIP_ABS.with_suffix(".7z.sha256")).write_text(f"{sha}  {Path(ZIP_REL).name}\n")
    log(f"  {Path(ZIP_REL).name} = {mb:.0f} MB  sha256={sha[:16]}...")
    return mb


def preparar_branch() -> None:
    atual = _git("rev-parse", "--abbrev-ref", "HEAD", capture=True).stdout.strip()
    existe = _git("rev-parse", "--verify", BRANCH, check=False, capture=True).returncode == 0
    if atual != BRANCH:
        _git("checkout", "-B", BRANCH) if not existe else _git("checkout", BRANCH)
    log(f"  branch: {BRANCH}")


def _lfs_track() -> None:
    _git("lfs", "install", "--local")
    _git("lfs", "track", "backend/data/*.7z")
    _git("lfs", "track", "backend/data/*.zip")
    _git("add", ".gitattributes")


def commitar_codigo() -> None:
    """Commit 1: código + resultados (push normal, sem binário pesado)."""
    _git("add", "-A", "backend/app", "backend/scripts", "backend/tests",
         "backend/requirements.txt", "backend/pyproject.toml",
         "frontend/src", ".gitignore", ".env.example", "requirements.txt")
    for pad in PADROES_RESULTADO:
        _git("add", "-f", pad, check=False)
    tem = _git("diff", "--cached", "--quiet", check=False).returncode != 0
    if not tem:
        log("  (nada novo em código/resultados)")
        return
    _git("commit", "-m",
         "backup: base nacional TCC - scripts de ingestao/avaliacao e resultados")
    log("  commit de codigo + resultados feito")


def commitar_zip() -> None:
    """Commit 2: o SQLite compactado, via LFS."""
    _lfs_track()
    _git("add", "-f", ZIP_REL)
    sha = ZIP_ABS.with_suffix(".7z.sha256")
    if sha.exists():
        _git("add", "-f", str(sha.relative_to(REPO)).replace("\\", "/"))
    _git("commit", "-m",
         f"backup: SQLite da base nacional ({date.today():%Y-%m-%d}) via LFS")
    log("  commit do .7z (LFS) feito")


def fazer_push() -> None:
    log(f"  push para origin {BRANCH} (LFS sobe o binario)...")
    _git("push", "-u", "origin", BRANCH)
    log("  push concluido")


def relancar() -> None:
    for alvo in ("orquestrador", "embeddings"):
        subprocess.run([sys.executable, "scripts/lancar_destacado.py", alvo],
                       cwd=BACKEND, check=True)
    log("  coleta e embeddings relancados")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preparar", action="store_true",
                    help="So para processos + checkpoint + compacta + mede")
    ap.add_argument("--push", action="store_true", help="Da push de verdade")
    ap.add_argument("--relancar", action="store_true",
                    help="Relanca coleta + embeddings ao final")
    args = ap.parse_args()

    log("1/7 parando processos da coleta...")
    parar_processos()
    log("2/7 checkpoint + integrity_check do SQLite...")
    checkpoint_sqlite()
    log("3/7 compactando SQLite...")
    mb = compactar()
    if args.preparar:
        log(f"PREPARADO. .7z = {mb:.0f} MB em {ZIP_REL}")
        if args.relancar:
            relancar()
        return
    if mb > 1024:
        log(f"ATENCAO: .7z = {mb:.0f} MB passa de 1 GB (cota LFS do plano gratis).")
    log("4/7 preparando branch...")
    preparar_branch()
    log("5/7 commit de codigo + resultados...")
    commitar_codigo()
    log("6/7 commit do .7z via LFS...")
    commitar_zip()
    if args.push:
        log("7/7 push...")
        fazer_push()
    else:
        log("7/7 PRONTO para push. Rode de novo com --push para enviar ao GitHub.")
    if args.relancar:
        relancar()


if __name__ == "__main__":
    main()
