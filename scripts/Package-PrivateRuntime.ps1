param(
    [string]$Runtime = "$env:LOCALAPPDATA\RavaFit\Runtime\current",
    [string]$Output = ".\artifacts\RavaFit-Runtime.zip"
)

$ErrorActionPreference = "Stop"
$Runtime = [IO.Path]::GetFullPath($Runtime)
$Output = [IO.Path]::GetFullPath($Output)

$required = @(
    "python.exe",
    ".ravafit-runtime.json",
    "solver\server.py",
    "solver\production_b14.py",
    "solver\coverage_analysis.py",
    "solver\customise_mod.py",
    "rbody\rbody_v3_loader.py",
    "packages\numpy\__init__.py",
    "packages\scipy\__init__.py",
    "packages\torch\__init__.py",
    "packages\numba\__init__.py",
    "packages\llvmlite\__init__.py"
)
foreach ($relative in $required) {
    $path = Join-Path $Runtime $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Runtime is incomplete; missing $relative in $Runtime"
    }
}

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
Remove-Item -LiteralPath $Output -Force -ErrorAction SilentlyContinue

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $Runtime,
    $Output,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)

if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw "Runtime package was not created: $Output" }
$hash = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
$size = [int64](Get-Item -LiteralPath $Output).Length
Write-Host "$hash  $Output"
Write-Host "Runtime package size: $([Math]::Round($size / 1MB, 2)) MiB"
