$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_isaac.ps1') -EntryPoint 'src\collect_data.py' -ScriptArgs @('--episodes', '10', '--samples-per-class', '13', '--seed', '43017')
