param([ValidateSet('raw','pca16','pca64','moe16')][string]$Policy = 'raw')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
& (Join-Path $PSScriptRoot 'run_isaac.ps1') -EntryPoint 'src\run_policy.py' -ScriptArgs @('--episodes', '5', '--max-decisions', '120', '--steps-per-decision', '9', '--smoothing-window', '3', '--policy', $Policy, '--seed', '62024')
foreach ($Script in @('src\evaluate_closed_loop.py', 'src\make_figures.py')) {
    & $Python (Join-Path $ProjectRoot $Script)
    if ($LASTEXITCODE -ne 0) { throw "$Script failed with exit code $LASTEXITCODE" }
}
