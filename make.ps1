<#
    make.ps1 - the Makefile targets for Windows machines without GNU make.

    Usage:  ./make.ps1 setup
            ./make.ps1 ingest-samples
            ./make.ps1 app
            ./make.ps1 eval
            ./make.ps1 test
            ./make.ps1 demo
            ./make.ps1 reset | clean | distclean
            ./make.ps1 setup ingest-samples app     (several targets in order)
#>

param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Targets = @("help"))

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Venv = Join-Path $Root ".venv"
$VPy = Join-Path $Venv "Scripts\python.exe"

function Invoke-Step([string]$Exe, [string[]]$Arguments) {
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "failed with exit code $LASTEXITCODE" }
}

function Assert-Venv {
    if (-not (Test-Path $VPy)) { throw "No virtualenv yet - run ./make.ps1 setup first." }
}

function Step-Setup {
    Invoke-Step "python" @("-m", "venv", ".venv")
    Invoke-Step $VPy @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    # CPU wheels keep the torch download small; ignore failure and let
    # requirements.txt install the default build instead.
    & $VPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    Invoke-Step $VPy @("-m", "pip", "install", "-r", "requirements.txt")
    Write-Host "`nDone. Copy .env.example to .env to configure (optional)." -ForegroundColor Green
}

foreach ($target in $Targets) {
    switch ($target) {
        "help" {
            Write-Host "targets: setup ingest-samples app eval test demo reset clean distclean"
        }
        "setup" { Step-Setup }
        "ingest-samples" { Assert-Venv; Invoke-Step $VPy @("scripts/ingest_samples.py") }
        "app" { Assert-Venv; Invoke-Step $VPy @("-m", "streamlit", "run", "app.py") }
        "eval" { Assert-Venv; Invoke-Step $VPy @("scripts/eval.py", "--ingest-if-empty") }
        "test" { Assert-Venv; Invoke-Step $VPy @("-m", "pytest") }
        "demo" { Assert-Venv; Invoke-Step $VPy @("scripts/demo.py", "--mock") }
        "reset" {
            Assert-Venv
            Invoke-Step $VPy @("-c", "import sys; sys.path.insert(0, 'src'); from docrag.store import DocStore; DocStore().reset(); print('index reset')")
        }
        "clean" { Assert-Venv; Invoke-Step $VPy @("scripts/clean.py") }
        "distclean" { Invoke-Step "python" @("scripts/clean.py", "--venv") }
        default { throw "Unknown target '$target'. Try ./make.ps1 help" }
    }
}
