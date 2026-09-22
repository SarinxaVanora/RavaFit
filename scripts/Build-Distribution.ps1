param(
    [string]$GitHubRepository = "SarinxaVanora/RavaFit",
    [string]$Branch = "master",
    [string]$Configuration = "Release",
    [string]$RuntimeVersion = "1.1.2",
    [string]$BodiesVersion = "1.0.0",
    [string]$MinimumPluginVersion = "1.1.1",
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Push-Location $RepoRoot
try {
    Write-Host "[RavaFit] Building/hydrating hosted runtime and body assets..." -ForegroundColor Cyan
    & .\scripts\Build-GitHubAssets.ps1 `
        -GitHubRepository $GitHubRepository `
        -Branch $Branch `
        -RuntimeVersion $RuntimeVersion `
        -BodiesVersion $BodiesVersion `
        -MinimumPluginVersion $MinimumPluginVersion

    Write-Host ""
    Write-Host "[RavaFit] Building plugin distribution..." -ForegroundColor Cyan
    if ($RunTests) {
        & .\scripts\Build.ps1 -Configuration $Configuration -RunTests
    }
    else {
        & .\scripts\Build.ps1 -Configuration $Configuration
    }

    Write-Host ""
    Write-Host "[RavaFit] Complete distribution is ready under distribution\." -ForegroundColor Green
}
finally { Pop-Location }
