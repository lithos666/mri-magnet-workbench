$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$environmentPython = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & python -m venv --system-site-packages (Join-Path $projectDirectory '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $environmentPython -m pip install -r (Join-Path $projectDirectory 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Environment ready. Run .\.venv\Scripts\python.exe .\py_main.py check'
