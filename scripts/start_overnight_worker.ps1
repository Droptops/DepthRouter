param(
    [string]$Config = "configs/photonic_parallel_full.json",
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [switch]$SkipBuild,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $env:LOCALAPPDATA "DepthRouter\worker-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "$RunId.stdout.log"
$stderr = Join-Path $logDir "$RunId.stderr.log"
$pidFile = Join-Path $logDir "$RunId.pid"
$runner = Join-Path $PSScriptRoot "run_photonic_docker.ps1"

$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner, "-Config", $Config, "-RunId", $RunId)
if ($SkipBuild) { $arguments += "-SkipBuild" }
if ($Resume) { $arguments += "-Resume" }

$process = Start-Process -FilePath "pwsh.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii
[pscustomobject]@{
    RunId = $RunId
    ProcessId = $process.Id
    Stdout = $stdout
    Stderr = $stderr
    PidFile = $pidFile
}
