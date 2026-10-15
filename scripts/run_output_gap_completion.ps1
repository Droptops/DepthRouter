param(
    [string]$Config = "configs/output_gap_completion.json",
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$dockerExe = (Get-Command docker.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = (Resolve-Path (Join-Path $repo $Config)).Path
$repoPrefix = $repo.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $configPath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Config must resolve inside the experiment checkout"
}
$configRelative = $configPath.Substring($repoPrefix.Length).Replace('\', '/')
$cache = Join-Path $env:LOCALAPPDATA "DepthRouter\huggingface-cache"
$artifactRoot = Join-Path $repo "artifacts\output_gap_completion"
$runPath = Join-Path $artifactRoot $RunId
if (Test-Path -LiteralPath $runPath) {
    throw "Refusing to overwrite existing experiment run: $runPath"
}
New-Item -ItemType Directory -Force -Path $cache, $artifactRoot | Out-Null

if (-not $SkipBuild) {
    & $dockerExe build -f (Join-Path $repo "Dockerfile.photonic") -t depthrouter-photonic:local $repo
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed with exit code $LASTEXITCODE" }
}

# Positional shell arguments avoid interpreting config paths or run IDs as code.
$containerCommand = 'python -m pip install -e . --no-deps && python experiments/output_gap_completion.py --config "$1" --output-dir "$2" && python experiments/output_gap_report.py --run-dir "$2"'
$containerName = "depthrouter-output-gap-$RunId"
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
    "bash", "-lc", $containerCommand, "output-gap",
    $configRelative, "artifacts/output_gap_completion/$RunId"
)

# The Python entry point owns creation of the run directory and rejects reuse.
# Keep launcher output beside it so logging cannot pre-create that directory.
$launcherLog = Join-Path $artifactRoot "$RunId-launcher.log"
if (Test-Path -LiteralPath $launcherLog) {
    throw "Refusing to overwrite existing launcher log: $launcherLog"
}
if (-not ("OutputGapExecutionState" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class OutputGapExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
}
$executionContinuous = [uint32]2147483648
$executionRequired = [uint32]1

try {
    [void][OutputGapExecutionState]::SetThreadExecutionState($executionContinuous -bor $executionRequired)
    # Windows PowerShell 5.1 represents redirected native stderr as ErrorRecords,
    # including ordinary pip warnings. Preserve it and use the actual exit code.
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $dockerExe @dockerArgs 2>&1 | Tee-Object -FilePath $launcherLog
        $dockerExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($dockerExitCode -ne 0) {
        throw "Experiment container exited with code $dockerExitCode. Log: $launcherLog"
    }
}
finally {
    [void][OutputGapExecutionState]::SetThreadExecutionState($executionContinuous)
}
