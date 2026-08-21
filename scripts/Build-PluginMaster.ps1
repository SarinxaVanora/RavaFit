param(
    [string]$GitHubRepository = "SarinxaVanora/RavaFit",
    [string]$Branch = "master",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ProjectPath = Join-Path $RepoRoot "src\RavaFit.Plugin\RavaFit.Plugin.csproj"
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path $RepoRoot "distribution\pluginmaster.json" }

[xml]$project = Get-Content -LiteralPath $ProjectPath -Raw
$versionText = [string]$project.Project.PropertyGroup.Version
if ([string]::IsNullOrWhiteSpace($versionText)) { throw "RavaFit.Plugin.csproj does not contain a Version." }
$version = [Version]$versionText
$assemblyVersion = '{0}.{1}.{2}.{3}' -f $version.Major, $version.Minor, [Math]::Max(0, $version.Build), [Math]::Max(0, $version.Revision)
$pluginUrl = "https://raw.githubusercontent.com/$GitHubRepository/$Branch/distribution/latest.zip"

$plugin = [ordered]@{
    Author = "RavaFit"
    Name = "RavaFit"
    InternalName = "RavaFit"
    AssemblyVersion = $assemblyVersion
    Description = "Port outfits and animations between FFXIV races, genders and supported body shapes."
    ApplicableVersion = "any"
    DalamudApiLevel = 15
    Punchline = "Port outfits and animations without the faff."
    Tags = @("penumbra", "modding", "outfit", "animation")
    RepoUrl = "https://github.com/$GitHubRepository"
    DownloadLinkInstall = $pluginUrl
    DownloadLinkUpdate = $pluginUrl
    LastUpdate = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
@($plugin) | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Host "[RavaFit] Plugin repository metadata: $Output" -ForegroundColor Green
Write-Host "[RavaFit] Plugin ZIP URL: $pluginUrl"
