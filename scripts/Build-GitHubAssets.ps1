param(
    [string]$GitHubRepository = "SarinxaVanora/RavaFit",
    [string]$Branch = "master",
    [string]$RuntimeVersion = "1.1.8",
    [string]$BodiesVersion = "1.0.0",
    [string]$MinimumPluginVersion = "1.1.5",
    [string]$Runtime = "",
    [string]$Bodies = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($Runtime)) { $Runtime = Join-Path $RepoRoot "DevAssets\Runtime" }
if ([string]::IsNullOrWhiteSpace($Bodies)) { $Bodies = Join-Path $RepoRoot "DevAssets\Bodies\Bodies.rbody" }
$Runtime = [IO.Path]::GetFullPath($Runtime)
$Bodies = [IO.Path]::GetFullPath($Bodies)
$AssetDirectory = Join-Path $RepoRoot "distribution\assets"
$RuntimeFileName = "RavaFit.Runtime.win-x64.$RuntimeVersion.zip"
$RuntimeZip = Join-Path $AssetDirectory $RuntimeFileName
$BodiesOut = Join-Path $AssetDirectory "Bodies.rbody"
$ChecksumsOut = Join-Path $AssetDirectory "SHA256SUMS.txt"
$ManifestOut = Join-Path $RepoRoot "distribution\ravafit-assets.json"

function Test-GitLfsPointer([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -gt 4096) { return $false }
    try {
        $first = Get-Content -LiteralPath $Path -TotalCount 1 -ErrorAction Stop
        return [string]::Equals([string]$first, 'version https://git-lfs.github.com/spec/v1', [StringComparison]::Ordinal)
    }
    catch { return $false }
}

Write-Host "[RavaFit] Checking local runtime/body assets..."
& (Join-Path $RepoRoot "scripts\Prepare-DevEnvironment.ps1")

if (-not (Test-Path -LiteralPath (Join-Path $Runtime "python.exe") -PathType Leaf)) { throw "Runtime is not ready at $Runtime" }
if (-not (Test-Path -LiteralPath (Join-Path $Runtime "solver\production_b14.py") -PathType Leaf)) { throw "Runtime is missing solver\production_b14.py" }
if (-not (Test-Path -LiteralPath $Bodies -PathType Leaf)) { throw "Bodies.rbody not found: $Bodies" }
if (Test-GitLfsPointer $Bodies) { throw "Bodies.rbody is still a Git LFS pointer. Run git lfs pull or restore the real catalogue before publishing." }

New-Item -ItemType Directory -Force -Path $AssetDirectory | Out-Null
Write-Host "[RavaFit] Packaging hosted runtime..."
& (Join-Path $RepoRoot "scripts\Package-PrivateRuntime.ps1") -Runtime $Runtime -Output $RuntimeZip
if (-not [string]::Equals([IO.Path]::GetFullPath($Bodies), [IO.Path]::GetFullPath($BodiesOut), [StringComparison]::OrdinalIgnoreCase)) {
    Copy-Item -LiteralPath $Bodies -Destination $BodiesOut -Force
}

$productionText = Get-Content -LiteralPath (Join-Path $Runtime "solver\production_b14.py") -Raw
$revisionMatch = [regex]::Match($productionText, 'PRODUCTION_REVISION\s*=\s*["''](?<revision>[^"'']+)["'']')
if (-not $revisionMatch.Success) { throw "Could not read PRODUCTION_REVISION from runtime\solver\production_b14.py" }
$productionRevision = $revisionMatch.Groups['revision'].Value

$runtimeHash = (Get-FileHash -LiteralPath $RuntimeZip -Algorithm SHA256).Hash.ToLowerInvariant()
$bodiesHash = (Get-FileHash -LiteralPath $BodiesOut -Algorithm SHA256).Hash.ToLowerInvariant()
$runtimeSize = [int64](Get-Item -LiteralPath $RuntimeZip).Length
$bodiesSize = [int64](Get-Item -LiteralPath $BodiesOut).Length
$mediaBase = "https://media.githubusercontent.com/media/$GitHubRepository/$Branch/distribution/assets"

$manifest = [ordered]@{
    schema = 1
    runtime = [ordered]@{
        version = $RuntimeVersion
        url = "$mediaBase/$RuntimeFileName"
        sha256 = $runtimeHash
        size = $runtimeSize
        productionRevision = $productionRevision
        minimumPluginVersion = $MinimumPluginVersion
    }
    bodies = [ordered]@{
        version = $BodiesVersion
        url = "$mediaBase/Bodies.rbody"
        sha256 = $bodiesHash
        size = $bodiesSize
        minimumPluginVersion = $MinimumPluginVersion
    }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ManifestOut -Encoding UTF8
@(
    "$runtimeHash  distribution/assets/$RuntimeFileName",
    "$bodiesHash  distribution/assets/Bodies.rbody"
) | Set-Content -LiteralPath $ChecksumsOut -Encoding ascii

Write-Host ""
Write-Host "[RavaFit] GitHub/LFS assets are ready." -ForegroundColor Green
Write-Host "  Runtime: $RuntimeZip"
Write-Host "  Bodies : $BodiesOut"
Write-Host "  Manifest: $ManifestOut"
Write-Host ""
Write-Host "Commit the generated files to master. Git LFS will store the runtime ZIP and Bodies.rbody." -ForegroundColor Cyan
