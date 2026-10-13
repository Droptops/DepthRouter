# Docker Desktop startup repair — 2026-10-12

## Observed environment

- Windows build: 10.0.26300.9457 (64-bit)
- Docker Desktop: 4.82.0 (233772)
- Docker Engine after repair: 29.6.1
- WSL: 2.7.10.0, kernel 6.18.33.2-2, WSL2 backend
- NVIDIA Windows driver: 610.88
- GPU: NVIDIA GeForce RTX 4070, 12,282 MiB

## Failure evidence

The Docker backend log ended with:

```text
starting services: initializing Inference manager: listening on
unix://C:/Users/m4vil/AppData/Local/Docker/run/dockerInference: remove ...:
The file cannot be accessed by the system. (listener: The filename, directory
name, or volume label syntax is incorrect.)
```

All three transient endpoints were zero-byte NTFS reparse points and
`fsutil reparsepoint query` returned Windows error 1920:

- `%LOCALAPPDATA%\Docker\run\dockerInference`
- `%LOCALAPPDATA%\Docker\run\userAnalyticsOtlpHttp.sock`
- `%LOCALAPPDATA%\docker-secrets-engine\engine.sock`

The `dockerInference` and secrets sockets had timestamps from the preceding day.
Multiple backend/frontend/CLI processes remained after one graceful
`docker desktop stop` attempt, while the `docker-desktop` WSL distribution was
already stopped. This is consistent with the stale AF_UNIX endpoint lifecycle
failure described in docker/desktop-feedback issues
[#448](https://github.com/docker/desktop-feedback/issues/448),
[#527](https://github.com/docker/desktop-feedback/issues/527), and
[#625](https://github.com/docker/desktop-feedback/issues/625).

## Recovery performed

1. Copied Docker settings and host logs to
   `%LOCALAPPDATA%\DockerRepairBackup-20261012-221535`.
2. Attempted a graceful Docker Desktop stop once. It did not clear the confirmed
   Docker-owned orphan processes.
3. Force-stopped only processes whose executable path was under
   `C:\Program Files\Docker` and whose executable name matched Docker Desktop,
   backend, or CLI components.
4. Terminated only the `docker-desktop` WSL distribution. No distribution was
   unregistered.
5. Validated absolute paths under `%LOCALAPPDATA%`, then renamed the disposable
   runtime IPC directories to:
   - `%LOCALAPPDATA%\Docker\run.stale-20261012-221636`
   - `%LOCALAPPDATA%\docker-secrets-engine.stale-20261012-221636`
6. Restarted Docker Desktop once.

No factory reset, prune, VHDX change, persistent volume deletion, reinstall,
update, reboot, or WSL unregister was performed.

## Verification

- `docker info` succeeded; the pre-existing two stopped containers and four
  images remained present.
- Docker recreated fresh runtime endpoints with current timestamps.
- `docker-desktop` was running under WSL2.
- `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
  succeeded and reported the RTX 4070 with 12,282 MiB VRAM.

## Root cause conclusion

The directly observed cause was an inaccessible stale AF_UNIX reparse point at
`dockerInference`, compounded by orphaned Docker-owned processes after failed
starts. Quarantining only the transient runtime directories broke the crash loop.
This machine's OS/Docker combination falls in the affected version/build family;
the repair demonstrates recovery but does not establish that Docker's underlying
socket lifecycle bug is permanently fixed.
