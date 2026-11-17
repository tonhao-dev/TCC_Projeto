# Executa UMA fase da avaliacao 120 COM/SEM RAG com gemma3:1b (Ollama), mostrando a
# saida ao vivo no terminal e gravando em data\logs\20_gemma3_1b_<fase>.log.
# Fases: com_rag | sem_rag | pontuar (pontuar espera as outras duas terminarem com exit 0).
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('com_rag', 'sem_rag', 'pontuar')]
    [string]$Fase
)

$ErrorActionPreference = 'Continue'
$backend = 'C:\Users\luis.santiago\Second Brain\dev\tcc-andre\TCC_Projeto\backend'
Set-Location $backend
$logDir = Join-Path $backend 'data\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "20_gemma3_1b_$Fase.log"
$marker = Join-Path $logDir "20_gemma3_1b_$Fase.exit"
$Host.UI.RawUI.WindowTitle = "gemma3:1b | $Fase"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Ambiente: modelo local e base NACIONAL (o .env aponta para o seed de 8 emendas).
$env:LLM_PROVIDER = 'ollama'
$env:OLLAMA_MODEL = 'gemma3:1b'
$env:PYTHONIOENCODING = 'utf-8'
$env:NO_COLOR = '1'
$env:DATABASE_URL = (py -c "from scripts.lancar_destacado import _url_nacional; print(_url_nacional())").Trim()

function Log([string]$msg) {
    $linha = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $linha
    Add-Content -Path $log -Value $linha -Encoding utf8
}

Remove-Item $marker -ErrorAction SilentlyContinue

if ($Fase -eq 'pontuar') {
    Log 'Aguardando COM RAG e SEM RAG terminarem (exit 0) para pontuar...'
    $m1 = Join-Path $logDir '20_gemma3_1b_com_rag.exit'
    $m2 = Join-Path $logDir '20_gemma3_1b_sem_rag.exit'
    while (-not ((Test-Path $m1) -and (Test-Path $m2))) { Start-Sleep -Seconds 20 }
    $c1 = (Get-Content $m1 -Raw).Trim()
    $c2 = (Get-Content $m2 -Raw).Trim()
    Log "COM RAG exit=$c1 | SEM RAG exit=$c2"
    if ($c1 -ne '0' -or $c2 -ne '0') {
        Log 'Alguma fase falhou; NAO vou pontuar. Rode de novo a fase que falhou (retoma do checkpoint).'
        Set-Content -Path $marker -Value '1'
        return
    }
    $args_py = @('-u', 'scripts\evaluate_120_com_sem_rag.py', '--fase', 'pontuar', '--manter-parciais')
} else {
    $args_py = @('-u', 'scripts\evaluate_120_com_sem_rag.py', '--fase', $Fase)
}

Log "Modelo=$($env:OLLAMA_MODEL) fase=$Fase db=nacional :: py $($args_py -join ' ')"
py @args_py 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
Set-Content -Path $marker -Value $code
Log "FIM fase=$Fase exit=$code"
