Write-Host "=== Griptape Video Generation Server ===" -ForegroundColor Cyan
Write-Host "Port: 8090" -ForegroundColor Gray
Write-Host "Log file: C:\Foundry\Griptape\video_gen_server\server.log" -ForegroundColor Gray
Write-Host ""
Write-Host "Waiting for server to start..." -ForegroundColor Yellow
Write-Host "(This window shows real-time server logs)" -ForegroundColor Gray
Write-Host ""

$log = "C:\Foundry\Griptape\video_gen_server\server.log"

# 로그 파일이 생길 때까지 대기
$waited = 0
while (-not (Test-Path $log) -and $waited -lt 10) {
    Start-Sleep 1
    $waited++
}

if (Test-Path $log) {
    Get-Content -Path $log -Wait -Tail 50
} else {
    Write-Host "[ERROR] Log file not found: $log" -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
