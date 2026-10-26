$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
foreach ($Script in @('src\evaluate_offline.py', 'src\benchmark_latency.py')) {
    & $Python (Join-Path $ProjectRoot $Script)
    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }
}
