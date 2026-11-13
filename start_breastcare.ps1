<#
.SYNOPSIS
    Reliable Windows 11 setup, verification, and startup script for BreastCare AI.

.DESCRIPTION
    Starts or verifies the BreastCare AI research application (FastAPI backend on
    http://127.0.0.1:8000 and Next.js frontend on http://127.0.0.1:3000) with
    public classification fail-closed by default (BREASTCARE_CLASSIFICATION_ENABLED=false).

.PARAMETER Mode
    - "check": Verify Python/Node environment, dataset manifest, and locked artifacts.
    - "api"  : Run the FastAPI research backend in the foreground on port 8000.
    - "web"  : Run the Next.js frontend in the foreground on port 3000.
    - "both" : Start the FastAPI backend in a background process and run the Next.js
               frontend on port 3000 (default).
#>
param(
    [ValidateSet("both", "api", "web", "check")]
    [string]$Mode = "both",
    [int]$ApiPort = 8000,
    [int]$WebPort = 3000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment Python not found at $PythonExe. Run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
}

# Enforce fail-closed research classification by default
$env:BREASTCARE_CLASSIFICATION_ENABLED = "false"
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:$ApiPort"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " BreastCare AI Research Application (Windows 11)" -ForegroundColor Cyan
Write-Host " Mode: $Mode | BREASTCARE_CLASSIFICATION_ENABLED=$($env:BREASTCARE_CLASSIFICATION_ENABLED)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($Mode -eq "check") {
    Write-Host "[1/3] Checking Python & PyTorch environment..." -ForegroundColor Yellow
    & $PythonExe -c "import torch, fastapi, sklearn, PIL; print('Python OK | PyTorch:', torch.__version__, '| FastAPI:', fastapi.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "Python environment check failed." }

    Write-Host "[2/3] Verifying unified dataset manifest (2,381 acquired / 2,232 eligible)..." -ForegroundColor Yellow
    & $PythonExe -m ml.build_unified_manifest
    if ($LASTEXITCODE -ne 0) { throw "Unified dataset manifest verification failed." }

    Write-Host "[3/3] Verifying full-source baseline and fine-tuned experiment artifacts..." -ForegroundColor Yellow
    & $PythonExe -m ml.verify_full_source_baseline --intervals
    if ($LASTEXITCODE -ne 0) { throw "Full-source baseline verification failed." }
    if (Test-Path (Join-Path $RepoRoot "reports\full_source_finetune_v1\result.json")) {
        & $PythonExe -m ml.verify_full_source_finetune --intervals
        if ($LASTEXITCODE -ne 0) { throw "Full-source fine-tuning verification failed." }
    }
    if (Test-Path (Join-Path $RepoRoot "reports\gpu_encoder_finetune_v3\lock.json")) {
        & $PythonExe -m ml.gpu_encoder_finetune_v2 verify_package
        if ($LASTEXITCODE -ne 0) { throw "GPU encoder fine-tuning v3 package verification failed." }
    }
    Write-Host "All environment and artifact checks passed." -ForegroundColor Green
    exit 0
}

if ($Mode -eq "api") {
    Write-Host "Starting FastAPI backend at http://127.0.0.1:$ApiPort ..." -ForegroundColor Green
    & $PythonExe -m uvicorn backend.app.main:app --host 127.0.0.1 --port $ApiPort
    exit $LASTEXITCODE
}

if ($Mode -eq "web") {
    Set-Location (Join-Path $RepoRoot "web")
    Write-Host "Starting Next.js web interface at http://127.0.0.1:$WebPort ..." -ForegroundColor Green
    if (Test-Path (Join-Path $RepoRoot "web\.next\BUILD_ID")) {
        & node .\node_modules\next\dist\bin\next start -H 127.0.0.1 -p $WebPort
    } else {
        & node .\node_modules\next\dist\bin\next dev -H 127.0.0.1 -p $WebPort
    }
    exit $LASTEXITCODE
}

if ($Mode -eq "both") {
    Write-Host "Starting FastAPI backend in background on http://127.0.0.1:$ApiPort ..." -ForegroundColor Green
    $ApiProc = Start-Process -FilePath $PythonExe -ArgumentList "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" -WorkingDirectory $RepoRoot -PassThru -NoNewWindow
    try {
        $Ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                $Resp = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
                if ($Resp.status -eq "ok") {
                    $Ready = $true
                    break
                }
            } catch {}
        }
        if (-not $Ready) {
            throw "FastAPI backend did not become ready on http://127.0.0.1:$ApiPort/health"
        }
        Write-Host "Backend ready (/health status=ok, inference_enabled=false)." -ForegroundColor Green
        Write-Host "Starting Next.js frontend on http://127.0.0.1:$WebPort ..." -ForegroundColor Green
        Set-Location (Join-Path $RepoRoot "web")
        if (Test-Path (Join-Path $RepoRoot "web\.next\BUILD_ID")) {
            & node .\node_modules\next\dist\bin\next start -H 127.0.0.1 -p $WebPort
        } else {
            & node .\node_modules\next\dist\bin\next dev -H 127.0.0.1 -p $WebPort
        }
    } finally {
        if ($ApiProc -and -not $ApiProc.HasExited) {
            Write-Host "Stopping FastAPI backend process (PID $($ApiProc.Id))..." -ForegroundColor Yellow
            Stop-Process -Id $ApiProc.Id -Force -ErrorAction SilentlyContinue
        }
        Set-Location $RepoRoot
    }
}

