# ResearchX on localhost in one window: the API (http://127.0.0.1:8000) and the web app (http://localhost:4730),
# then the browser. Closing this window or pressing Ctrl+C stops both.
#   powershell -ExecutionPolicy Bypass -File scripts/dev/start_local.ps1          # uses the existing web build
#   powershell -ExecutionPolicy Bypass -File scripts/dev/start_local.ps1 -Build   # rebuilds the web app first
#   ... -NoLaya   # without the Laya decision service (http://127.0.0.1:8090; runs are traced "Laya unreachable")
param([switch]$Build, [switch]$NoLaya)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$Host.UI.RawUI.WindowTitle = "ResearchX local (API :8000, web :4730)"

foreach ($port in 8000, 4730) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "Port $port is already in use: stop that process first." }
}
if ($Build -or -not (Test-Path "$root\apps\web\.next\BUILD_ID")) {
    Push-Location "$root\apps\web"; npm run build; Pop-Location
}

function Up([string]$url) {
    try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $url).StatusCode -lt 500 } catch { $false }
}

$env:PYTHONPATH = $root
$laya = $null
if (-not $NoLaya -and (Test-Path "$root\.venv\Scripts\laya-serve.exe") -and -not (Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue)) {
    # Laya decision service in its own process: localhost only, English checkpoint built at start (~1 min on CPU)
    $env:LAYA_HOST, $env:LAYA_PORT, $env:LAYA_MODELS, $env:LAYA_PRELOAD = "127.0.0.1", "8090", "english", "1"
    $env:LAYA_DEFAULT_MODEL, $env:LAYA_MAX_LOADED, $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "english", "1", "1"
    $laya = Start-Process -FilePath "$root\.venv\Scripts\laya-serve.exe" -WorkingDirectory $root -NoNewWindow -PassThru
}
$api = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "services.api.app:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $root -NoNewWindow -PassThru
$web = Start-Process -FilePath "node" -ArgumentList "node_modules/next/dist/bin/next", "start", "-p", "4730" -WorkingDirectory "$root\apps\web" -NoNewWindow -PassThru
try {
    $deadline = (Get-Date).AddSeconds(90)
    while (-not ((Up "http://127.0.0.1:8000/api/health") -and (Up "http://127.0.0.1:4730/login"))) {
        if ((Get-Date) -gt $deadline -or $api.HasExited -or $web.HasExited) { throw "A server did not start: see the output above." }
        Start-Sleep -Milliseconds 500
    }
    Start-Process "http://localhost:4730"
    Write-Host "`nResearchX is running: web http://localhost:4730 | API http://127.0.0.1:8000/api/docs | close this window to stop`n" -ForegroundColor Green
    Wait-Process -Id $api.Id, $web.Id
}
finally {
    foreach ($p in $api, $web, $laya) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
}
