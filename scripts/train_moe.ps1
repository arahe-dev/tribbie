param(
    [ValidateRange(1,8)][int]$Experts = 4,
    [string]$MetadataPath = 'features/metadata.csv',
    [string]$ModelName = 'moe_latent16',
    [double]$ClassWeightPower = 0.0,
    [ValidateRange(1,4)][int]$SpatialPool = 1
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
& $Python (Join-Path $ProjectRoot 'src\train_moe.py') --experts $Experts --metadata-path $MetadataPath --model-name $ModelName --class-weight-power $ClassWeightPower --spatial-pool $SpatialPool
if ($LASTEXITCODE -ne 0) { throw "train_moe.py failed with exit code $LASTEXITCODE" }
