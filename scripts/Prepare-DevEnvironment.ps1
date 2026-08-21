param(
    [switch]$SkipRuntime,
    [switch]$SkipBodies
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$DevAssets = Join-Path $RepoRoot "DevAssets"
$BodiesDir = Join-Path $DevAssets "Bodies"
$RuntimeDir = Join-Path $DevAssets "Runtime"
$IncomingDir = Join-Path $DevAssets "Incoming"
$ExpectedRuntimeVersion = "1.0.0"
$ExpectedProductionRevision = "1.0.0-runtime-1"

New-Item -ItemType Directory -Force -Path $BodiesDir, $IncomingDir | Out-Null

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Assert-Hash([string]$Path, [string]$Expected, [string]$Label) {
    if (-not (Test-Path $Path)) { throw "$Label is missing: $Path" }
    $actual = Get-Sha256 $Path
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch. Expected $Expected but got $actual for $Path"
    }
}

function Test-GitLfsPointer([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -gt 4096) { return $false }
    try {
        $first = Get-Content -LiteralPath $Path -TotalCount 1 -ErrorAction Stop
        return [string]::Equals([string]$first, 'version https://git-lfs.github.com/spec/v1', [StringComparison]::Ordinal)
    }
    catch { return $false }
}

function Get-HostedBodies([string]$Destination) {
    $manifestPath = Join-Path $RepoRoot "distribution\ravafit-assets.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $false }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $url = [string]$manifest.bodies.url
        $hash = [string]$manifest.bodies.sha256
        if ([string]::IsNullOrWhiteSpace($url) -or $url -notmatch '^https://') { return $false }
        if ($hash -notmatch '^[0-9a-fA-F]{64}$') { return $false }
        $temp = $Destination + ".download-" + [Guid]::NewGuid().ToString("N")
        try {
            Write-Host "  Downloading Bodies.rbody from the configured GitHub/LFS asset..." -ForegroundColor Yellow
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $temp
            Assert-Hash $temp $hash "Downloaded Bodies.rbody"
            Move-Item -LiteralPath $temp -Destination $Destination -Force
            return $true
        }
        finally { Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue }
    }
    catch {
        Write-Host "  Hosted Bodies.rbody could not be hydrated: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

function Sync-RavaFitRuntimeCode([string]$Target) {
    $solverTarget = Join-Path $Target "solver"
    $rbodyTarget = Join-Path $Target "rbody"
    $b14Target = Join-Path $Target "b14_frozen\scripts"
    New-Item -ItemType Directory -Force -Path $solverTarget, $rbodyTarget, $b14Target | Out-Null

    foreach ($name in @(
        'server.py','production_b14.py','b14_compat.py','coverage_analysis.py','native_body_graft.py','customise_mod.py',
        'peer_shell_worker.py','peer_group_supervisor.py','shell_solve_worker.py','assembly_worker.py','coverage_clearance_worker.py',
        'garment_mesh_worker.py','garment_finalize_worker.py','garment_group_supervisor.py'
    )) {
        Copy-Item -LiteralPath (Join-Path $RepoRoot "runtime\solver\$name") -Destination (Join-Path $solverTarget $name) -Force
    }
    foreach ($name in @('rbody_v3_loader.py','rbody_b14_adapter.py','prepare_b14_rbody_cache.py','rbody_v3_core.py')) {
        Copy-Item -LiteralPath (Join-Path $RepoRoot "runtime\rbody\$name") -Destination (Join-Path $rbodyTarget $name) -Force
    }
    foreach ($name in @('ffxiv_lobofit.py','b14_mesh_worker.py','structural_refine.py','construction_fields.py','collision_eval.py','lobofit_official_refine.py','glb_patch_legacy.py')) {
        Copy-Item -LiteralPath (Join-Path $RepoRoot "runtime\b14_frozen\scripts\$name") -Destination (Join-Path $b14Target $name) -Force
    }
}

function Update-RuntimeMarker([string]$Path) {
    try {
        $marker = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $marker | Add-Member -NotePropertyName RuntimeVersion -NotePropertyValue $ExpectedRuntimeVersion -Force
        $marker | Add-Member -NotePropertyName ProductionRevision -NotePropertyValue $ExpectedProductionRevision -Force
        $marker | Add-Member -NotePropertyName SolverSource -NotePropertyValue "RavaFit $ExpectedProductionRevision" -Force
        $marker | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
    }
    catch { throw "Could not update the local runtime marker: $($_.Exception.Message)" }
}

function Find-CompanionFile([string]$Name) {
    $candidates = @(
        $IncomingDir,
        (Join-Path $RepoRoot "distribution\assets"),
        $RepoRoot,
        (Split-Path -Parent $RepoRoot),
        (Join-Path $env:USERPROFILE "Downloads")
    ) | Select-Object -Unique

    foreach ($directory in $candidates) {
        if ([string]::IsNullOrWhiteSpace($directory) -or -not (Test-Path $directory)) { continue }
        $exact = Join-Path $directory $Name
        if ((Test-Path $exact) -and -not (Test-GitLfsPointer $exact)) { return $exact }
        $stem = [IO.Path]::GetFileNameWithoutExtension($Name)
        $extension = [IO.Path]::GetExtension($Name)
        $alternate = Get-ChildItem -Path $directory -File -Filter "$stem*$extension" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($alternate -and -not (Test-GitLfsPointer $alternate.FullName)) { return $alternate.FullName }
    }
    return $null
}

if (-not $SkipBodies) {
    Write-Host "[RavaFit Prepare] Validating/hydrating unified RBODY catalogue..."
    $unifiedPath = Join-Path $BodiesDir "Bodies.rbody"
    if ((Test-Path -LiteralPath $unifiedPath) -and (Test-GitLfsPointer $unifiedPath)) {
        Remove-Item -LiteralPath $unifiedPath -Force
    }
    if (-not (Test-Path -LiteralPath $unifiedPath -PathType Leaf)) {
        $found = Find-CompanionFile "Bodies.rbody"
        if ($found) {
            Copy-Item -LiteralPath $found -Destination $unifiedPath -Force
        }
        elseif (-not (Get-HostedBodies $unifiedPath)) {
            throw "Bodies.rbody is missing. Run git lfs pull, put the catalogue in DevAssets\Incoming/Downloads, or make sure distribution\ravafit-assets.json points at a published body asset."
        }
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($unifiedPath)
    try {
        $requiredEntries = @("manifest.json", "catalogue.json", "payload_index.json", "target_options.json")
        $names = @($archive.Entries | ForEach-Object { $_.FullName })
        foreach ($entry in $requiredEntries) {
            if ($names -notcontains $entry) { throw "Bodies.rbody is missing required entry $entry." }
        }
        $manifestEntry = $archive.GetEntry("manifest.json")
        $reader = New-Object System.IO.StreamReader($manifestEntry.Open())
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json }
        finally { $reader.Dispose() }
        if ([string]$manifest.format -ne "RBODY" -or [int]$manifest.version -lt 4) {
            throw "Bodies.rbody is not the unified RBODY V4+ catalogue required by this RavaFit build."
        }
    }
    finally { $archive.Dispose() }
    Write-Host "  Unified bodies: $unifiedPath"
}

if (-not $SkipRuntime) {
    Write-Host "[RavaFit Prepare] Validating/hydrating private SolverHost runtime..."
    $python = Join-Path $RuntimeDir "python.exe"
    $server = Join-Path $RuntimeDir "solver\server.py"
    $coverage = Join-Path $RuntimeDir "solver\coverage_analysis.py"
    $nativeGraft = Join-Path $RuntimeDir "solver\native_body_graft.py"
    $markerPath = Join-Path $RuntimeDir ".ravafit-runtime.json"
    $runtimeNeedsProvisioning = -not (Test-Path $python) -or -not (Test-Path $server) -or -not (Test-Path $coverage) -or -not (Test-Path $nativeGraft) -or -not (Test-Path $markerPath)

    if (-not $runtimeNeedsProvisioning) {
        try {
            $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
            if (-not [string]::Equals([string]$marker.RuntimeVersion, $ExpectedRuntimeVersion, [StringComparison]::Ordinal)) {
                Write-Host "Cached RavaFit runtime is from $($marker.RuntimeVersion); rebuilding as $ExpectedRuntimeVersion..." -ForegroundColor Yellow
                $runtimeNeedsProvisioning = $true
            }
        }
        catch {
            Write-Host "Cached RavaFit runtime marker is unreadable; rebuilding it..." -ForegroundColor Yellow
            $runtimeNeedsProvisioning = $true
        }
    }

    if (-not $runtimeNeedsProvisioning) {
        Write-Host "  Refreshing RavaFit-owned SolverHost code in the cached runtime..."
        Sync-RavaFitRuntimeCode $RuntimeDir

        # Imports are not enough; run the real SolverHost self-test so bad pruning is caught here.
        try {
            $oldPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                $healthOutput = & $python -I -u $server --health-json 2>&1
                $healthExit = $LASTEXITCODE
            }
            finally { $ErrorActionPreference = $oldPreference }
            $healthText = (($healthOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
            if ($healthExit -ne 0) { throw "SolverHost health process exited with $healthExit. Output: $healthText" }
            $healthLine = $healthOutput | ForEach-Object { [string]$_ } | Where-Object { $_ -match '"id"\s*:\s*1' } | Select-Object -Last 1
            if ([string]::IsNullOrWhiteSpace($healthLine)) { throw "SolverHost returned no health response. Output: $healthText" }
            $health = $healthLine | ConvertFrom-Json
            if (-not $health.ok) { throw "SolverHost production self-test returned ok=false." }
            $requiredCapabilities = @(
                'convert', 'graft_native_body', 'analyze_coverage', 'detect_source_bodies',
                'inspect_mdl_parts', 'hide_mdl_parts', 'tag_mdl_parts', 'split_mdl_parts', 'piercing_customisation'
            )
            foreach ($capability in $requiredCapabilities) {
                if (-not [bool]$health.capabilities.$capability) {
                    throw "SolverHost production self-test is missing required capability '$capability'."
                }
            }
            if (-not [string]::Equals([string]$health.production_revision, $ExpectedProductionRevision, [StringComparison]::Ordinal)) {
                throw "Cached SolverHost production revision is '$($health.production_revision)', expected '$ExpectedProductionRevision'."
            }

            $expectedVersions = @{
                numpy = '2.3.5'
                scipy = '1.17.0'
                trimesh = '4.11.1'
                torch = '2.10.0'
                numba = '0.65.1'
                llvmlite = '0.47.0'
            }
            foreach ($name in $expectedVersions.Keys) {
                $actual = [string]$health.versions.$name
                if ([string]::IsNullOrWhiteSpace($actual) -or $actual -eq 'unknown' -or $actual.StartsWith('missing:', [StringComparison]::OrdinalIgnoreCase)) {
                    throw "SolverHost health reports unusable $name`: $actual"
                }
                $public = $actual.Split('+')[0]
                if (-not [string]::Equals($public, [string]$expectedVersions[$name], [StringComparison]::Ordinal)) {
                    throw "SolverHost health reports $name=$actual; expected $($expectedVersions[$name])."
                }
            }
            Update-RuntimeMarker $markerPath
        }
        catch {
            Write-Host "Private RavaFit runtime failed its production self-test; rebuilding the cached runtime. $($_.Exception.Message)" -ForegroundColor Yellow
            $runtimeNeedsProvisioning = $true
        }
    }

    if ($runtimeNeedsProvisioning) {
        Write-Host "Private RavaFit runtime is not healthy/current yet; provisioning the slim runtime now..." -ForegroundColor Yellow
        & (Join-Path $RepoRoot "scripts\Install-DevRuntime.ps1") -Destination $RuntimeDir
    }
    if (-not (Test-Path $python)) { throw "Private RavaFit runtime provisioning did not produce python.exe." }
    if (-not (Test-Path $server)) { throw "Private RavaFit runtime provisioning did not produce solver\server.py." }
    if (-not (Test-Path $coverage)) { throw "Private RavaFit runtime provisioning did not produce solver\coverage_analysis.py." }
    if (-not (Test-Path $nativeGraft)) { throw "Private RavaFit runtime provisioning did not produce solver\native_body_graft.py." }
    if (-not (Test-Path $markerPath)) { throw "Private RavaFit runtime provisioning did not produce .ravafit-runtime.json." }
}

Write-Host "RavaFit development assets are ready." -ForegroundColor Green
Write-Host "  Bodies : $BodiesDir"
if (-not $SkipRuntime) { Write-Host "  Runtime: $RuntimeDir" }
