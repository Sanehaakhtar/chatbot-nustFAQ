param(
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Write-Host "Port $Port is busy (PID $($listener.OwningProcess)). Stopping it..."
    Stop-Process -Id $listener.OwningProcess -Force
    Start-Sleep -Milliseconds 300
}

Write-Host "Starting app on http://127.0.0.1`:$Port"
Push-Location $RepoRoot
try {
    $args = @("-m", "uvicorn", "app:app", "--port", "$Port")
    & $PythonExe @args
}
finally {
    Pop-Location
}
