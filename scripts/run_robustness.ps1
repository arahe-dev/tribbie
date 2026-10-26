param(
    [ValidateSet('raw','pca16','pca64','moe16')][string]$Policy = 'moe16',
    [int]$StartIndex = 0,
    [ValidateRange(1,25)][int]$Count = 10,
    [string]$RunTag = '',
    [string]$ModelCheckpoint = '',
    [ValidateSet('auto','cuda','cpu')][string]$MoeDevice = 'auto'
)
$ErrorActionPreference = 'Stop'
$ScriptArgs = @(
    '--case-manifest', 'data/robustness_challenge_v1.json',
    '--case-start-index', "$StartIndex",
    '--case-count', "$Count",
    '--no-record',
    '--policy', $Policy,
    '--moe-device', $MoeDevice
)
if ($RunTag) { $ScriptArgs += @('--run-tag', $RunTag) }
if ($ModelCheckpoint) { $ScriptArgs += @('--model-checkpoint', $ModelCheckpoint) }
& (Join-Path $PSScriptRoot 'run_isaac.ps1') -EntryPoint 'src\run_policy.py' -ScriptArgs $ScriptArgs
if ($LASTEXITCODE -ne 0) { throw "Robustness run failed with exit code $LASTEXITCODE" }
