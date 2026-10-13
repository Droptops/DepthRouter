param(
    [ValidateSet("dense", "triton")]
    [string]$Method = "triton",
    [int]$Batch = 1,
    [double]$Fraction = 0.75,
    [string]$RunId = "manual-ncu",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
if ($Batch -lt 1) { throw "Batch must be positive" }
if ($Fraction -le 0 -or $Fraction -gt 1) { throw "Fraction must be in (0, 1]" }

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cache = Join-Path $env:LOCALAPPDATA "DepthRouter\huggingface-cache"
$output = Join-Path $repo "artifacts\fused_sparse_swiglu\$RunId\profiler"
New-Item -ItemType Directory -Force -Path $cache, $output | Out-Null

if (-not $SkipBuild) {
    docker build -f (Join-Path $repo "Dockerfile.ncu") -t depthrouter-photonic:ncu $repo
    if ($LASTEXITCODE -ne 0) { throw "Nsight image build failed" }
}

$stem = "ncu-$Method-b$Batch-f$($Fraction.ToString('0.##', [Globalization.CultureInfo]::InvariantCulture))"
$fractionText = $Fraction.ToString([Globalization.CultureInfo]::InvariantCulture)
$command = @"
set -o pipefail
ncu --profile-from-start off --set detailed --force-overwrite \
  --export /workspace/artifacts/fused_sparse_swiglu/$RunId/profiler/$stem \
  python experiments/profile_fused_swiglu.py --method $Method --batch $Batch --fraction $fractionText \
  2>&1 | tee /workspace/artifacts/fused_sparse_swiglu/$RunId/profiler/$stem.log
"@
$dockerArgs = @(
    "run", "--rm", "--gpus", "all",
    "--shm-size", "6g", "--memory", "40g",
    "--mount", "type=bind,source=$repo,target=/workspace",
    "--mount", "type=bind,source=$cache,target=/cache/huggingface",
    "-e", "HF_HOME=/cache/huggingface",
    "-e", "PYTHONPATH=/workspace/src",
    "-w", "/workspace",
    "depthrouter-photonic:ncu",
    "bash", "-lc", $command
)
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "Nsight capture exited with code $LASTEXITCODE" }
