param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$RepoRoot = [IO.Path]::GetFullPath((Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))).TrimEnd([char[]]@('\','/'))
$roots = @(
    (Join-Path $RepoRoot "src\RavaFit.Plugin\bin"),
    (Join-Path $RepoRoot "src\RavaFit.Core\bin")
)
$ownNames = @('RavaFit.dll', 'RavaFit.pdb', 'RavaFit.Core.dll', 'RavaFit.Core.pdb')
$files = @()
foreach ($root in $roots) {
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { continue }
    $files += @(Get-ChildItem -LiteralPath $root -Recurse -Force -File | Where-Object {
        $ownNames -contains $_.Name -and $_.FullName -match [regex]::Escape("\$Configuration\")
    })
}
$files = @($files | Sort-Object FullName -Unique)
if ($files.Count -eq 0) { throw "No RavaFit Release DLL/PDB outputs were found for source-path privacy validation." }

$repoRootSlash = $RepoRoot.Replace('\','/')
$leaks = New-Object System.Collections.Generic.List[string]
foreach ($file in $files) {
    $bytes = [IO.File]::ReadAllBytes($file.FullName)
    $utf8 = [Text.Encoding]::UTF8.GetString($bytes)
    $utf16 = [Text.Encoding]::Unicode.GetString($bytes)

    foreach ($needle in @($RepoRoot, $repoRootSlash)) {
        if ([string]::IsNullOrWhiteSpace($needle)) { continue }
        if ($utf8.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $utf16.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            $leaks.Add("$($file.FullName) contains build checkout path '$needle'.")
        }
    }
}

if ($leaks.Count -gt 0) {
    throw "Release source-path privacy validation failed:`n$($leaks -join [Environment]::NewLine)"
}

Write-Host "[RavaFit] Release source-path privacy validation passed; checkout root is not embedded in RavaFit DLL/PDB outputs." -ForegroundColor Green
