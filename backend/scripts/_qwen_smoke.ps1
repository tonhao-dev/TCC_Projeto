# Smoke-test do pipeline com qwen2.5:7b em UMA consulta (A1), COM e SEM RAG,
# para validar tool-calling/JSON/latencia antes da bateria completa.
$ErrorActionPreference = 'Continue'
$backend = 'C:\Users\luis.santiago\Second Brain\dev\tcc-andre\TCC_Projeto\backend'
Set-Location $backend
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$env:LLM_PROVIDER = 'ollama'
$env:OLLAMA_MODEL = 'qwen2.5:7b'
$env:PYTHONIOENCODING = 'utf-8'
$env:NO_COLOR = '1'
$env:DATABASE_URL = (py -c "from scripts.lancar_destacado import _url_nacional; print(_url_nacional())").Trim()

Write-Host "DB=$($env:DATABASE_URL -replace ':[^:@/]+@', ':****@')"
Write-Host "Modelo=$($env:OLLAMA_MODEL)"

py -u scripts\evaluate_120_com_sem_rag.py --fase todas --ids A1 --tag qwen2.5-7b-smoke --manter-parciais
