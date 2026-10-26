param([switch]$CollectData, [ValidateSet('raw','pca16','pca64')][string]$Policy = 'raw')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if ($CollectData) {
    & (Join-Path $PSScriptRoot 'collect.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Dataset collection failed.' }
}
foreach ($Script in @('src\extract_features.py', 'src\train_policy.py', 'src\evaluate_offline.py', 'src\benchmark_latency.py')) {
    & $Python (Join-Path $ProjectRoot $Script)
    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }
}
& (Join-Path $PSScriptRoot 'demo.ps1') -Policy $Policy
if ($LASTEXITCODE -ne 0) { throw 'Closed-loop simulation/demo failed.' }
& $Python -m pytest -v (Join-Path $ProjectRoot 'tests')
if ($LASTEXITCODE -ne 0) { throw 'Project tests failed.' }
& $Python -m pip freeze | Set-Content (Join-Path $ProjectRoot 'environment.txt')
if ($LASTEXITCODE -ne 0) { throw 'Writing the environment manifest failed.' }
