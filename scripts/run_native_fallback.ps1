param(
    [string]$Config = "configs/photonic_parallel_quick.json",
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $repo ".venv-photonic"
if (-not (Test-Path -LiteralPath $venv)) {
    py -3.12 -m venv $venv
}
$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r (Join-Path $repo "requirements-photonic.txt")
& $python -m pip install -e $repo --no-deps

$arguments = @(
    (Join-Path $repo "experiments\photonic_parallel_inference.py"),
    "--config", (Join-Path $repo $Config),
    "--output-dir", (Join-Path $repo "artifacts\photonic_parallel\$RunId")
)
if ($Resume) { $arguments += "--resume" }
& $python @arguments
