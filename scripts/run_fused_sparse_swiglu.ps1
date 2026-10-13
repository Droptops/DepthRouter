param(
    [string]$Config = "configs/fused_sparse_swiglu_full.json",
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [ValidateSet("all", "kernels", "quality")]
    [string]$Stage = "all",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cache = Join-Path $env:LOCALAPPDATA "DepthRouter\huggingface-cache"
$artifacts = Join-Path $repo "artifacts\fused_sparse_swiglu"
New-Item -ItemType Directory -Force -Path $cache, $artifacts | Out-Null

if (-not $SkipBuild) {
    docker build -f (Join-Path $repo "Dockerfile.photonic") -t depthrouter-photonic:local $repo
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed" }
}

$containerName = "depthrouter-fused-$($RunId -replace '[^A-Za-z0-9_.-]', '-')"
$command = "python -m pip install -e . --no-deps && python experiments/fused_sparse_swiglu.py --config '$Config' --output-dir 'artifacts/fused_sparse_swiglu/$RunId' --stage '$Stage'"
$dockerArgs = @(
    "run", "--rm", "--gpus", "all",
    "--name", $containerName,
    "--shm-size", "6g",
    "--memory", "40g",
    "--mount", "type=bind,source=$repo,target=/workspace",
    "--mount", "type=bind,source=$cache,target=/cache/huggingface",
    "-e", "HF_HOME=/cache/huggingface",
    "-e", "PYTHONUNBUFFERED=1",
    "depthrouter-photonic:local",
    "bash", "-lc", $command
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class FusedSparseExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]0x00000001

try {
    [void][FusedSparseExecutionState]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment container exited with code $LASTEXITCODE" }
}
finally {
    [void][FusedSparseExecutionState]::SetThreadExecutionState($ES_CONTINUOUS)
}
