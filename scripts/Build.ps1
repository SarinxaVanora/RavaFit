param(
    [string]$Configuration = "Release",
    [switch]$RunTests,
    [switch]$OpenOutput
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Output = Join-Path $RepoRoot "artifacts\latest.zip"

Push-Location $RepoRoot
try {
    $assetManifestPath = Join-Path $RepoRoot "distribution\ravafit-assets.json"
    $assetManifest = Get-Content -LiteralPath $assetManifestPath -Raw | ConvertFrom-Json
    if ([string]$assetManifest.runtime.sha256 -notmatch '^[0-9a-fA-F]{64}$' -or [int64]$assetManifest.runtime.size -le 0) {
        throw "Hosted runtime manifest is not published yet. Run .\scripts\Build-GitHubAssets.ps1 first."
    }
    if ([string]$assetManifest.runtime.version -ne "1.1.8") {
        throw "Hosted runtime manifest version '$($assetManifest.runtime.version)' is stale; expected 1.1.8. Run .\scripts\Build-GitHubAssets.ps1 first."
    }

    dotnet restore .\RavaFit.sln -p:Platform=x64
    if ($LASTEXITCODE -ne 0) { throw "dotnet restore failed with exit code $LASTEXITCODE." }

    & .\scripts\Build-PluginRelease.ps1 -Configuration $Configuration -Output $Output
    & .\scripts\Build-PluginMaster.ps1

    if ([string]::Equals($Configuration, "Release", [StringComparison]::OrdinalIgnoreCase)) {
        & .\scripts\Test-ReleasePathPrivacy.ps1 -Configuration $Configuration
    }

    if ($RunTests) {
        & .\scripts\Test.ps1 -Configuration $Configuration
    }

    Write-Host ""
    Write-Host "RavaFit plugin package:" -ForegroundColor Green
    Write-Host "  distribution\latest.zip" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Runtime/Bodies are separate. Use .\scripts\Build-GitHubAssets.ps1 when publishing those assets." -ForegroundColor DarkGray
    if ($OpenOutput) { Start-Process explorer.exe (Join-Path $RepoRoot 'distribution') }
}
finally { Pop-Location }
