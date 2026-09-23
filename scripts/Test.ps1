param([string]$Configuration = "Release")

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $RepoRoot
try {
    dotnet test .\src\RavaFit.Core.Tests\RavaFit.Core.Tests.csproj -c $Configuration -p:Platform=x64
    if ($LASTEXITCODE -ne 0) { throw "dotnet test failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
