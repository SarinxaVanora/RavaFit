param(
    [string]$Destination = "external\Xande-306003a"
)

$ErrorActionPreference = "Stop"
$Commit = "306003a7fd591cc6854e98165dd9cea00b66d16b"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Target = Join-Path $RepoRoot $Destination
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
git clone https://github.com/xivdev/Xande.git $Target
git -C $Target checkout $Commit
Write-Host "Pinned historical Xande source at $Commit to $Target"
Write-Warning "DO NOT enable this directly in RavaFit. Its Havok signatures were last checked in 2023. Retarget to Dalamud API 15/current FFXIV and validate Diagnostics round-trip first."
