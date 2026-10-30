<#
.SYNOPSIS
  Builds Dockerfile.<Tag> from the worker repo and pushes digamberjhazaran/infinitetalk-runpod:<Tag>.

.DESCRIPTION
  Works from any folder (it always uses the worker repo as the build context and Dockerfile.<Tag>).
  Refuses to push a tag that already exists on Docker Hub, because Dockerfile.vNN files copy the
  CURRENT working tree: rebuilding an old tag would silently change what that tag contains.
  Create a new Dockerfile.vNN for every new version instead.

.EXAMPLE
  D:\Serverless2\Infinitetalk_Runpod_hub_v2\scripts\build_and_push.ps1 v20
.EXAMPLE
  D:\Serverless2\Infinitetalk_Runpod_hub_v2\scripts\build_and_push.ps1 v19 -NoPush
#>
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [switch]$NoPush,
    [switch]$Overwrite
)

$Image = "digamberjhazaran/infinitetalk-runpod"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Dockerfile = "Dockerfile.$Tag"

if (-not (Test-Path (Join-Path $RepoRoot $Dockerfile))) {
    $available = (Get-ChildItem -Path $RepoRoot -Filter "Dockerfile.v*" | ForEach-Object { $_.Name }) -join ", "
    Write-Host "ERROR: $Dockerfile not found in $RepoRoot. Available: $available" -ForegroundColor Red
    exit 1
}

if (-not $NoPush -and -not $Overwrite) {
    # cmd /c keeps Windows PowerShell 5.1 from turning docker's stderr into errors
    cmd /c "docker manifest inspect ${Image}:$Tag >nul 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "ERROR: ${Image}:$Tag already exists on Docker Hub. Create a new Dockerfile.vNN for a new version, or pass -Overwrite to replace it." -ForegroundColor Red
        exit 1
    }
}

Push-Location $RepoRoot
try {
    Write-Host "=== docker build -f $Dockerfile -t ${Image}:$Tag . (in $RepoRoot)"
    docker build -f $Dockerfile -t "${Image}:$Tag" .
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: docker build failed" -ForegroundColor Red; exit 1 }

    if ($NoPush) {
        Write-Host "Built ${Image}:$Tag (not pushed: -NoPush)"
        exit 0
    }

    Write-Host "=== docker push ${Image}:$Tag"
    docker push "${Image}:$Tag"
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: docker push failed (run 'docker login' first?)" -ForegroundColor Red; exit 1 }
    $digest = docker image inspect "${Image}:$Tag" --format "{{index .RepoDigests 0}}"
    Write-Host "Pushed: $digest" -ForegroundColor Green
}
finally {
    Pop-Location
}
