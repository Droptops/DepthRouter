param(
    [string]$Config = "configs/expert_granularity.json",
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
$artifactRoot = Join-Path $repo "artifacts\expert_granularity"
$runPath = Join-Path $artifactRoot $RunId
$launcherLog = Join-Path $artifactRoot "$RunId-launcher.log"
if ((Test-Path -LiteralPath $runPath) -or (Test-Path -LiteralPath $launcherLog)) {
    throw "Refusing to overwrite an existing experiment run or launcher log: $RunId"
}
New-Item -ItemType Directory -Force -Path $cache, $artifactRoot | Out-Null
if (-not $SkipBuild) {
    & $dockerExe build -f (Join-Path $repo "Dockerfile.photonic") -t depthrouter-photonic:local $repo
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed with exit code $LASTEXITCODE" }
}

# The entry point owns registration, the four-hour ceiling, and output creation.
# Positional config/run arguments are never interpreted as executable shell text.
$containerCommand = 'python -m pip install -e . --no-deps && python experiments/expert_granularity.py --config "$1" --output-dir "$2"'
$dockerArgs = @(
    "run", "--rm", "--gpus", "all",
    "--name", "depthrouter-expert-granularity-$RunId",
    "--shm-size", "6g", "--memory", "40g",
    "--mount", "type=bind,source=$repo,target=/workspace",
    "--mount", "type=bind,source=$cache,target=/cache/huggingface",
    "-e", "HF_HOME=/cache/huggingface", "-e", "PYTHONPATH=src", "-e", "PYTHONUNBUFFERED=1",
    "depthrouter-photonic:local", "bash", "-lc", $containerCommand, "expert-granularity",
    $configRelative, "artifacts/expert_granularity/$RunId"
)
if (-not ("ExpertGranularityExecutionState" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class ExpertGranularityExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
}
$executionContinuous = [uint32]2147483648
$executionRequired = [uint32]1
try {
    [void][ExpertGranularityExecutionState]::SetThreadExecutionState($executionContinuous -bor $executionRequired)
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $dockerExe @dockerArgs 2>&1 | Tee-Object -FilePath $launcherLog
        $dockerExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousErrorPreference }
    if ($dockerExitCode -ne 0) {
        throw "Experiment container exited with code $dockerExitCode. Log: $launcherLog"
    }
}
finally {
    [void][ExpertGranularityExecutionState]::SetThreadExecutionState($executionContinuous)
}
