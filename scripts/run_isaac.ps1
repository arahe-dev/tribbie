param(
    [Parameter(Mandatory = $true)][string]$EntryPoint,
    [string[]]$ScriptArgs = @()
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Launcher = 'C:\isaacsim\python.bat'
if (-not (Test-Path $Launcher)) { throw "Isaac Sim launcher not found at $Launcher" }
$ScriptPath = Join-Path $ProjectRoot $EntryPoint
& $Launcher $ScriptPath @ScriptArgs
if ($LASTEXITCODE -ne 0) { throw "Isaac Sim script failed with exit code $LASTEXITCODE" }
