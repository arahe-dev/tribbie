$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IsaacPython = 'C:\isaacsim\kit\python\python.exe'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $IsaacPython)) { throw "Isaac Sim Python not found at $IsaacPython" }
if (-not (Test-Path $Python)) {
    & $IsaacPython -m venv --system-site-packages (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating the project virtual environment failed.' }
}
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Upgrading pip failed.' }
& $Python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA-enabled PyTorch failed.' }
& $Python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing project requirements failed.' }
& $Python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch verification failed.' }
