$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
foreach ($Script in @('src\extract_features.py', 'src\train_policy.py')) {
    & $Python (Join-Path $ProjectRoot $Script)
    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }
}
