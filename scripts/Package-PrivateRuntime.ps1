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

# Produce byte-stable hosted runtime ZIPs. GitHub Actions may hydrate the same
# runtime on different machines/dates; filesystem mtimes must not turn identical
# runtime content into a brand-new ~500 MB Git LFS object on every source push.
$fixedTimestamp = [DateTimeOffset]::new(2000, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
$runtimeRoot = $Runtime.TrimEnd([char[]]@('\','/'))
$files = @(Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Force -File |
    Sort-Object { $_.FullName.Substring($runtimeRoot.Length).Replace('\','/') })

$stream = [IO.File]::Open($Output, [IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($runtimeRoot.Length).TrimStart([char[]]@('\','/')).Replace('\','/')
            $entry = $archive.CreateEntry($relative, [IO.Compression.CompressionLevel]::Optimal)
            $entry.LastWriteTime = $fixedTimestamp

            $input = [IO.File]::OpenRead($file.FullName)
            try {
                $outputStream = $entry.Open()
                try { $input.CopyTo($outputStream) }
                finally { $outputStream.Dispose() }
            }
            finally { $input.Dispose() }
        }
    }
    finally { $archive.Dispose() }
}
finally { $stream.Dispose() }

if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) { throw "Runtime package was not created: $Output" }
$hash = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
$size = [int64](Get-Item -LiteralPath $Output).Length
Write-Host "$hash  $Output"
Write-Host "Runtime package size: $([Math]::Round($size / 1MB, 2)) MiB"
