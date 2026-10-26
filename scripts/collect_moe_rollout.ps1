param(
    [int]$Episodes = 30,
    [int]$RolloutsPerEpisode = 3,
    [int]$Seed = 9262026,
    [string]$OutputDir = 'data/moe_rollout_v2'
)
$ErrorActionPreference = 'Stop'
$ScriptArgs = @(
    '--episodes', "$Episodes",
    '--rollouts-per-episode', "$RolloutsPerEpisode",
    '--seed', "$Seed",
    '--output-dir', $OutputDir
)
& (Join-Path $PSScriptRoot 'run_isaac.ps1') -EntryPoint 'src\collect_moe_rollout_data.py' -ScriptArgs $ScriptArgs
if ($LASTEXITCODE -ne 0) { throw "collect_moe_rollout_data.py failed with exit code $LASTEXITCODE" }
