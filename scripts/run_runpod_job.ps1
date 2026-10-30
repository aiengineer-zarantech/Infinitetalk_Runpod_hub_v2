<#
.SYNOPSIS
  Submits one payload to a RunPod endpoint (/run), polls /status until it finishes and saves the full response.

.DESCRIPTION
  The API key is read ONLY from $env:RUNPOD_API_KEY and sent as an Authorization header
  (never in the URL, never printed). This makes a live, billable GPU call.

.EXAMPLE
  $env:RUNPOD_API_KEY = "<your key>"   # set in your own shell; never commit it
  .\scripts\run_runpod_job.ps1 -EndpointId <test-endpoint-id> -Payload bench_compile_off.json -Out off.json
#>
param(
    [Parameter(Mandatory = $true)][string]$EndpointId,
    [Parameter(Mandatory = $true)][string]$Payload,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$PollSeconds = 10,
    [int]$MaxMinutes = 70
)

if (-not $env:RUNPOD_API_KEY) { Write-Host "ERROR: set `$env:RUNPOD_API_KEY first" -ForegroundColor Red; exit 2 }
$headers = @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" }
$base = "https://api.runpod.ai/v2/$EndpointId"

$body = Get-Content -Raw -Encoding UTF8 $Payload
$job = Invoke-RestMethod -Method Post -Uri "$base/run" -Headers $headers -ContentType "application/json" -Body $body
Write-Host "submitted job $($job.id)"
$start = Get-Date

while ($true) {
    Start-Sleep -Seconds $PollSeconds
    $status = Invoke-RestMethod -Method Get -Uri "$base/status/$($job.id)" -Headers $headers
    $elapsed = [int]((Get-Date) - $start).TotalSeconds
    Write-Host "$elapsed s: $($status.status)"
    if ($status.status -in @("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")) { break }
    if ($elapsed -gt $MaxMinutes * 60) { Write-Host "giving up after $MaxMinutes min (job keeps running on RunPod)"; break }
}

$status | ConvertTo-Json -Depth 10 -Compress | Set-Content -Encoding UTF8 $Out
Write-Host "saved $Out (status $($status.status), executionTime $($status.executionTime) ms)"
if ($status.status -ne "COMPLETED") { Write-Host "error: $($status.error)" -ForegroundColor Red; exit 1 }
