param(
    [int]$Episodes = 25,
    [int]$SamplesPerClass = 10,
    [int]$Seed = 7262026
)
$ErrorActionPreference = 'Stop'
$ScriptArgs = @('--episodes', "$Episodes", '--samples-per-class', "$SamplesPerClass", '--seed', "$Seed")
& (Join-Path $PSScriptRoot 'run_isaac.ps1') -EntryPoint 'src\collect_moe_domain_data.py' -ScriptArgs $ScriptArgs
if ($LASTEXITCODE -ne 0) { throw "collect_moe_domain_data.py failed with exit code $LASTEXITCODE" }
