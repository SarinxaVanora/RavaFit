param(
    [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
    [string[]]$Libraries,
    [string]$Destination = "$env:LOCALAPPDATA\RavaFit\Bodies"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
foreach ($Library in $Libraries) {
    if (-not (Test-Path $Library)) { throw "Body library not found: $Library" }
    if ([IO.Path]::GetExtension($Library) -ne ".rbody") { throw "Not an .rbody file: $Library" }
    Copy-Item -Force $Library (Join-Path $Destination ([IO.Path]::GetFileName($Library)))
}
Write-Host "Installed $($Libraries.Count) RBODY libraries to $Destination"
