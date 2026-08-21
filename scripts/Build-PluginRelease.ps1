param(
    [string]$Configuration = "Release",
    [string]$Output = "",
    [int]$MaxPackageMiB = 50
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path $RepoRoot "artifacts\latest.zip" }
$Output = [IO.Path]::GetFullPath($Output)
$PluginProject = Join-Path $RepoRoot "src\RavaFit.Plugin\RavaFit.Plugin.csproj"
$PluginBin = Join-Path $RepoRoot "src\RavaFit.Plugin\bin"
$RepositoryZip = Join-Path $RepoRoot "distribution\latest.zip"

if (Test-Path -LiteralPath $PluginBin) {
    Get-ChildItem -LiteralPath $PluginBin -Filter "latest.zip" -File -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
}
Remove-Item -LiteralPath $RepositoryZip -Force -ErrorAction SilentlyContinue

Write-Host "[RavaFit] Building plugin $Configuration..."
& dotnet build $PluginProject --configuration $Configuration -p:Platform=x64
if ($LASTEXITCODE -ne 0) { throw "dotnet build failed with exit code $LASTEXITCODE." }

$latest = Get-ChildItem -Path $PluginBin -Filter "latest.zip" -File -Recurse |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if ($null -eq $latest) { throw "DalamudPackager did not produce latest.zip." }

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
Copy-Item -LiteralPath $latest.FullName -Destination $Output -Force
Copy-Item -LiteralPath $latest.FullName -Destination $RepositoryZip -Force

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($Output)
try {
    $entries = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\\','/') })
    $forbidden = @($entries | Where-Object {
        $_ -match '(^|/)Runtime/' -or $_ -match '(^|/)Bodies/' -or $_ -match '\.rbody$' -or $_ -match '(^|/)python(w)?\.exe$'
    })
    if ($forbidden.Count -gt 0) { throw "Heavy assets leaked into latest.zip: $($forbidden -join ', ')" }
    foreach ($required in @('RavaFit.dll','RavaFit.Core.dll','Penumbra.Api.dll','RavaFit.json')) {
        if (-not ($entries | Where-Object { $_ -ieq $required })) { throw "latest.zip is missing $required." }
    }
}
finally { $archive.Dispose() }

$hash = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
$size = (Get-Item -LiteralPath $Output).Length
if ($MaxPackageMiB -gt 0 -and $size -gt ($MaxPackageMiB * 1MB)) {
    throw "RavaFit latest.zip is $([Math]::Round($size / 1MB, 2)) MiB, above the $MaxPackageMiB MiB safety ceiling."
}
Write-Host "[RavaFit] Plugin-only Dalamud package: $RepositoryZip" -ForegroundColor Green
Write-Host "[RavaFit] Size: $([Math]::Round($size / 1MB, 2)) MiB"
Write-Host "[RavaFit] SHA256: $hash"
