# Flash the Atom Echo over WiFi (no USB). Device must be running firmware that
# already has the /update endpoint (everything from the min_spiffs build onward).
#
#   .\flash_ota.ps1            # compile + OTA flash to default IP
#   .\flash_ota.ps1 192.168.1.42
param([string]$ip = "192.168.1.19")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli  = Join-Path $here "tools\arduino-cli.exe"
$sk   = Join-Path $here "atom_echo_v2v"
$fqbn = "esp32:esp32:m5stack_atom"
$opts = "PartitionScheme=min_spiffs"   # OTA-capable layout (two app slots)

Write-Host "Compiling..."
& $cli compile --fqbn $fqbn --board-options $opts --output-dir "$sk\build" $sk | Select-Object -Last 2

$bin = Join-Path $sk "build\atom_echo_v2v.ino.bin"
Write-Host "OTA flashing $bin -> http://$ip/update"
curl.exe -s -F "firmware=@$bin" "http://$ip/update"
Write-Host "`nWaiting for reboot..."
Start-Sleep -Seconds 12
try { Write-Host ("ALIVE:`n" + (Invoke-WebRequest "http://$ip/log" -TimeoutSec 6 -UseBasicParsing).Content) }
catch { Write-Host "no web yet: $($_.Exception.Message)" }
