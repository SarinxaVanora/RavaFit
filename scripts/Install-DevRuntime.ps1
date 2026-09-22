param(
    [string]$Destination = "$env:LOCALAPPDATA\RavaFit\Runtime\current"
)

$ErrorActionPreference = "Stop"
$RuntimeVersion = "1.1.1"
$ExpectedProductionRevision = "1.1.1-multi-region-support-source-preserve"
$PythonVersion = "3.13.5"
$PipZipappVersion = "26.2.1"
$PythonEmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$PipZipappUrl = "https://bootstrap.pypa.io/pip/zipapp/pip-$PipZipappVersion.pyz"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Temp = Join-Path $env:TEMP "RavaFitRuntimeBuild-$([Guid]::NewGuid().ToString('N'))"
$Destination = [IO.Path]::GetFullPath($Destination)
$DestinationParent = Split-Path -Parent $Destination
$RuntimeBuild = Join-Path $DestinationParent (".RavaFitRuntime-build-" + [Guid]::NewGuid().ToString('N'))
$RuntimeBackup = Join-Path $DestinationParent (".RavaFitRuntime-backup-" + [Guid]::NewGuid().ToString('N'))

function Get-TreeBytes([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
    if (Test-Path -LiteralPath $Path -PathType Leaf) { return [int64](Get-Item -LiteralPath $Path).Length }
    $sum = [int64]0
    foreach ($file in Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue) {
        $sum += [int64]$file.Length
    }
    return $sum
}

function Format-MiB([int64]$Bytes) { return ('{0:N1} MiB' -f ($Bytes / 1MB)) }

function Remove-TreeIfPresent([string]$Path) {
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
}

function Remove-TestTrees([string]$PackageRoot) {
    if (-not (Test-Path -LiteralPath $PackageRoot)) { return }
    $dirs = Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('tests','test') } |
        Sort-Object { $_.FullName.Length } -Descending
    foreach ($dir in $dirs) {
        if (Test-Path -LiteralPath $dir.FullName) { Remove-Item -LiteralPath $dir.FullName -Recurse -Force }
    }
}

function Remove-Bytecode([string]$Root) {
    if (-not (Test-Path -LiteralPath $Root)) { return }
    $cacheDirs = Get-ChildItem -LiteralPath $Root -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq '__pycache__' } |
        Sort-Object { $_.FullName.Length } -Descending
    foreach ($dir in $cacheDirs) {
        if (Test-Path -LiteralPath $dir.FullName) { Remove-Item -LiteralPath $dir.FullName -Recurse -Force }
    }
    Get-ChildItem -LiteralPath $Root -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @('.pyc','.pyo') } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Remove-FileIfPresent([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return [int64]0 }
    $bytes = [int64](Get-Item -LiteralPath $Path).Length
    Remove-Item -LiteralPath $Path -Force
    return $bytes
}

function Try-PruneValidatedFilesByExtension([string]$Root, [string[]]$Extensions, [string]$PythonExe, [string]$ServerPath, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Root)) { return [int64]0 }
    $normalised = @($Extensions | ForEach-Object { $_.ToLowerInvariant() })
    $files = @(Get-ChildItem -LiteralPath $Root -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $normalised -contains $_.Extension.ToLowerInvariant() })
    if ($files.Count -eq 0) { return [int64]0 }

    $renamed = @()
    $bytes = [int64]0
    try {
        foreach ($file in $files) {
            $backup = "$($file.FullName).ravafit-prune"
            if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Force }
            $bytes += [int64]$file.Length
            Move-Item -LiteralPath $file.FullName -Destination $backup
            $renamed += [pscustomobject]@{ Original = $file.FullName; Backup = $backup }
        }
        Invoke-SolverHealth $PythonExe $ServerPath "validated file prune check: $Label" | Out-Null
        foreach ($row in $renamed) {
            if (Test-Path -LiteralPath $row.Backup) { Remove-Item -LiteralPath $row.Backup -Force }
        }
        Write-Host "[RavaFit Runtime] Pruned $Label ($(Format-MiB $bytes)); production health still passes."
        return [int64]$bytes
    }
    catch {
        $failure = $_.Exception.Message
        foreach ($row in $renamed) {
            if (Test-Path -LiteralPath $row.Original) { Remove-Item -LiteralPath $row.Original -Force -ErrorAction SilentlyContinue }
            if (Test-Path -LiteralPath $row.Backup) { Move-Item -LiteralPath $row.Backup -Destination $row.Original -Force }
        }
        Write-Host "[RavaFit Runtime] Kept $Label; production health needs one or more of those files. $failure" -ForegroundColor DarkGray
        return [int64]0
    }
}

function Try-PruneValidatedTree([string]$Path, [string]$PythonExe, [string]$ServerPath, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
    $bytes = Get-TreeBytes $Path
    $backup = "$Path.ravafit-prune-$([Guid]::NewGuid().ToString('N'))"
    Move-Item -LiteralPath $Path -Destination $backup
    try {
        Invoke-SolverHealth $PythonExe $ServerPath "adaptive prune check: $Label" | Out-Null
        Remove-Item -LiteralPath $backup -Recurse -Force
        Write-Host "[RavaFit Runtime] Pruned $Label ($(Format-MiB $bytes)); production health still passes."
        return [int64]$bytes
    }
    catch {
        $failure = $_.Exception.Message
        if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path -LiteralPath $backup) { Move-Item -LiteralPath $backup -Destination $Path }
        Write-Host "[RavaFit Runtime] Kept $Label; production health needs it. $failure" -ForegroundColor DarkGray
        return [int64]0
    }
}

function Copy-RavaFitRuntimeCode([string]$Target) {
    $solverSource = Join-Path $RepoRoot 'runtime\solver'
    $solverTarget = Join-Path $Target 'solver'
    $rbodyTarget = Join-Path $Target 'rbody'
    $b14Target = Join-Path $Target 'b14_frozen\scripts'

    New-Item -ItemType Directory -Force -Path $solverTarget, $rbodyTarget, $b14Target | Out-Null

    # SolverHost is an internally-coupled Python module set.
    # Copy every Python source file while preserving any package subdirectories.
    # Do not maintain a brittle per-file allowlist here.
    if (-not (Test-Path -LiteralPath $solverSource -PathType Container)) {
        throw "SolverHost source directory is missing: $solverSource"
    }

    $solverFiles = @(
        Get-ChildItem -LiteralPath $solverSource -Recurse -Force -File -Filter '*.py' |
            Where-Object {
                $_.FullName -notmatch '[\\/](?:__pycache__|tests?)[\\/]'
            } |
            Sort-Object FullName
    )

    if ($solverFiles.Count -eq 0) {
        throw "SolverHost source directory contains no Python files: $solverSource"
    }

    foreach ($source in $solverFiles) {
        $relative = $source.FullName.Substring($solverSource.Length).TrimStart('\', '/')
        $destination = Join-Path $solverTarget $relative
        $destinationDirectory = Split-Path -Parent $destination

        New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
        Copy-Item -LiteralPath $source.FullName -Force -Destination $destination
    }

    foreach ($name in @(
        'rbody_v3_loader.py',
        'rbody_b14_adapter.py',
        'prepare_b14_rbody_cache.py',
        'rbody_v3_core.py'
    )) {
        $source = Join-Path $RepoRoot "runtime\rbody\$name"

        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Required RBODY runtime source is missing: $source"
        }

        Copy-Item -LiteralPath $source -Force -Destination (Join-Path $rbodyTarget $name)
    }

    foreach ($name in @(
        'ffxiv_lobofit.py',
        'b14_mesh_worker.py',
        'structural_refine.py',
        'construction_fields.py',
        'collision_eval.py',
        'lobofit_official_refine.py',
        'glb_patch_legacy.py'
    )) {
        $source = Join-Path $RepoRoot "runtime\b14_frozen\scripts\$name"

        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Required B14 production source is missing: $source"
        }

        Copy-Item -LiteralPath $source -Force -Destination (Join-Path $b14Target $name)
    }
}

function Get-NativeText($Output) {
    return (($Output | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
}

function Invoke-NativeCaptured([string]$Label, [string]$Executable, [string[]]$Arguments) {
    $output = @()
    $exitCode = $null
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    catch {
        throw "$Label could not launch '$Executable': $($_.Exception.Message)"
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }

    $text = Get-NativeText $output
    if ($exitCode -ne 0) {
        if ([string]::IsNullOrWhiteSpace($text)) { $text = '<no stdout/stderr captured>' }
        throw "$Label failed with exit code $exitCode. Output: $text"
    }
    return $output
}

function Assert-PrivatePython([string]$PythonExe, [string]$Stage) {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        $parent = Split-Path -Parent $PythonExe
        $contents = if (Test-Path -LiteralPath $parent) {
            (Get-ChildItem -LiteralPath $parent -Force -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name) -join ', '
        }
        else { '<parent directory missing>' }
        throw "Private CPython is missing $Stage. Expected executable: $PythonExe. Runtime-root contents: $contents"
    }

    $output = Invoke-NativeCaptured "Private CPython executable probe $Stage" $PythonExe @('-I','-c','import sys; print(sys.version.split()[0])')
    $version = (Get-NativeText $output).Trim()
    if ([string]::IsNullOrWhiteSpace($version)) { throw "Private CPython returned no version $Stage." }
}

function Invoke-PrivatePythonProbe([string]$PythonExe, [string]$Label, [string[]]$Arguments) {
    Write-Host "[RavaFit Runtime] Probe: $Label"
    $output = Invoke-NativeCaptured $Label $PythonExe $Arguments
    $text = Get-NativeText $output
    if (-not [string]::IsNullOrWhiteSpace($text)) {
        foreach ($line in ($text -split '\r?\n')) { Write-Host "[RavaFit Runtime]   $line" }
    }
    return $output
}

function Invoke-SolverHealth([string]$PythonExe, [string]$ServerPath, [string]$Label) {
    $healthLines = Invoke-PrivatePythonProbe $PythonExe $Label @('-I','-u',$ServerPath,'--health-json')
    $healthLine = $healthLines | ForEach-Object { [string]$_ } | Where-Object { $_ -match '"id"\s*:\s*1' } | Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($healthLine)) { throw "$Label did not return a health response." }
    try { $health = $healthLine | ConvertFrom-Json }
    catch { throw "$Label returned invalid health JSON: $healthLine" }
    if (-not $health.ok) { throw "$Label returned ok=false. error=$($health.error)" }
    $requiredCapabilities = @(
        'convert', 'graft_native_body', 'analyze_coverage', 'detect_source_bodies',
        'inspect_mdl_parts', 'hide_mdl_parts', 'tag_mdl_parts', 'split_mdl_parts', 'piercing_customisation'
    )
    foreach ($capability in $requiredCapabilities) {
        if (-not [bool]$health.capabilities.$capability) {
            throw "$Label does not expose required capability '$capability'."
        }
    }
    if (-not [string]::Equals([string]$health.production_revision, $ExpectedProductionRevision, [StringComparison]::Ordinal)) {
        throw "$Label reports production revision '$($health.production_revision)'; expected '$ExpectedProductionRevision'."
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
            throw "$Label reports unusable $name`: $actual"
        }
        $public = $actual.Split('+')[0]
        if (-not [string]::Equals($public, [string]$expectedVersions[$name], [StringComparison]::Ordinal)) {
            throw "$Label reports $name=$actual; expected $($expectedVersions[$name])."
        }
    }
    return $health
}

Write-Host "[RavaFit Runtime] Creating private production runtime for $Destination"
Write-Host "[RavaFit Runtime] Strategy: exact embedded CPython installs exact binary wheels into Runtime/packages using a temporary pip zipapp; pip is never shipped."
New-Item -ItemType Directory -Force -Path $Temp, $DestinationParent | Out-Null
if (Test-Path -LiteralPath $RuntimeBuild) { Remove-Item -LiteralPath $RuntimeBuild -Recurse -Force }
if (Test-Path -LiteralPath $RuntimeBackup) { Remove-Item -LiteralPath $RuntimeBackup -Recurse -Force }
New-Item -ItemType Directory -Force -Path $RuntimeBuild | Out-Null

try {
    Write-Host "[RavaFit Runtime] Copying exact production SolverHost/B14/RBODY code allowlist..."
    Copy-RavaFitRuntimeCode $RuntimeBuild

    # Build and install wheels with the exact interpreter we ship.
    Write-Host "[RavaFit Runtime] Downloading CPython $PythonVersion embeddable x64..."
    $EmbedPackage = Join-Path $Temp 'python-embed.zip'
    Invoke-WebRequest -UseBasicParsing $PythonEmbedUrl -OutFile $EmbedPackage
    if (-not (Test-Path -LiteralPath $EmbedPackage -PathType Leaf)) { throw "CPython embeddable download did not produce $EmbedPackage" }
    Expand-Archive -LiteralPath $EmbedPackage -DestinationPath $RuntimeBuild -Force

    $PythonExe = Join-Path $RuntimeBuild 'python.exe'
    $pth = Get-ChildItem -LiteralPath $RuntimeBuild -File -Filter 'python*._pth' | Select-Object -First 1
    $stdlibZip = Get-ChildItem -LiteralPath $RuntimeBuild -File -Filter 'python*.zip' | Select-Object -First 1
    if (-not $pth) { throw "CPython embeddable package did not contain python*._pth." }
    if (-not $stdlibZip) { throw "CPython embeddable package did not contain its standard-library ZIP." }
    @($stdlibZip.Name, '.', 'packages', 'import site') | Set-Content -LiteralPath $pth.FullName -Encoding ASCII
    Assert-PrivatePython $PythonExe 'immediately after extraction'

    # pip is build-only; the private interpreter handles the wheel layout for its own ABI.
    Write-Host "[RavaFit Runtime] Downloading pinned pip $PipZipappVersion zipapp (temporary build tool only)..."
    $PipZipApp = Join-Path $Temp "pip-$PipZipappVersion.pyz"
    Invoke-WebRequest -UseBasicParsing $PipZipappUrl -OutFile $PipZipApp
    if (-not (Test-Path -LiteralPath $PipZipApp -PathType Leaf)) { throw "Pinned pip zipapp download did not produce $PipZipApp" }
    Invoke-PrivatePythonProbe $PythonExe 'pip zipapp bootstrap' @('-I',$PipZipApp,'--version') | Out-Null

    $Packages = Join-Path $RuntimeBuild 'packages'
    New-Item -ItemType Directory -Force -Path $Packages | Out-Null
    Write-Host "[RavaFit Runtime] Installing exact production wheel closure with the private CPython interpreter..."
    $pipInstallArgs = @(
        '-I', $PipZipApp, 'install',
        '--disable-pip-version-check', '--no-cache-dir', '--no-compile', '--only-binary=:all:',
        '--target', $Packages,
        '--index-url', 'https://pypi.org/simple',
        '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
        'numpy==2.3.5', 'scipy==1.17.0', 'trimesh==4.11.1', 'torch==2.10.0+cpu', 'numba==0.65.1', 'llvmlite==0.47.0'
    )
    $pipOutput = Invoke-NativeCaptured 'private-runtime dependency installation' $PythonExe $pipInstallArgs
    $pipText = Get-NativeText $pipOutput
    if (-not [string]::IsNullOrWhiteSpace($pipText)) {
        foreach ($line in ($pipText -split '\r?\n')) { Write-Host "[RavaFit Runtime]   $line" }
    }

    foreach ($requiredInit in @('numpy\__init__.py','scipy\__init__.py','trimesh\__init__.py','torch\__init__.py','numba\__init__.py','llvmlite\__init__.py')) {
        $requiredPath = Join-Path $Packages $requiredInit
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Dependency installation is incomplete; required package entry point is missing: $requiredPath"
        }
    }

    $ServerPath = Join-Path $RuntimeBuild 'solver\server.py'
    if (-not (Test-Path -LiteralPath $ServerPath -PathType Leaf)) { throw "SolverHost entry point was not copied: $ServerPath" }

    # Use the file probe; it avoids PowerShell/Python quoting nonsense.
    $RuntimeProbe = Join-Path $RepoRoot 'scripts\PrivateRuntimeProbe.py'
    if (-not (Test-Path -LiteralPath $RuntimeProbe -PathType Leaf)) { throw "Private runtime probe is missing: $RuntimeProbe" }

    Invoke-PrivatePythonProbe $PythonExe 'pre-prune package/optimiser provenance' @('-I','-u',$RuntimeProbe) | Out-Null
    $prePruneHealth = Invoke-SolverHealth $PythonExe $ServerPath 'pre-prune SolverHost production health'

    Write-Host "[RavaFit Runtime] Pruning non-runtime development payloads after successful health validation..."
    $prunedBytes = [int64]0
    foreach ($tree in @(
        (Join-Path $Packages 'torch\test'),
        (Join-Path $Packages 'torch\include'),
        (Join-Path $Packages 'torch\share'),
        (Join-Path $Packages 'bin')
    )) {
        if (Test-Path -LiteralPath $tree) {
            $prunedBytes += Get-TreeBytes $tree
            Remove-TreeIfPresent $tree
        }
    }
    foreach ($packageName in @('numpy','scipy','trimesh','sympy','mpmath','networkx','jinja2','fsspec','filelock')) {
        $before = Get-TreeBytes (Join-Path $Packages $packageName)
        Remove-TestTrees (Join-Path $Packages $packageName)
        $after = Get-TreeBytes (Join-Path $Packages $packageName)
        $prunedBytes += [Math]::Max([int64]0, [int64]($before - $after))
    }

    # Remove build metadata, not DLL/PYD runtime files. Keep torch_shm_manager.
    $prunedBytes += (Try-PruneValidatedFilesByExtension $Packages @('.lib','.a','.exp','.pdb','.pyi','.pyx','.pxd','.h','.hpp','.c','.cc','.cpp','.cu','.cuh') $PythonExe $ServerPath 'linker/debug/header/Cython/type-development files')
    $prunedBytes += (Remove-FileIfPresent (Join-Path $Packages 'torch\bin\protoc.exe'))

    # These trees are optional; every removal is checked against the production self-test.
    foreach ($candidate in @(
        @{ Path = (Join-Path $Packages 'torch\onnx'); Label = 'torch/onnx' },
        @{ Path = (Join-Path $Packages 'functorch'); Label = 'functorch' },
        @{ Path = (Join-Path $Packages 'jinja2'); Label = 'jinja2' },
        @{ Path = (Join-Path $Packages 'fsspec'); Label = 'fsspec' },
        @{ Path = (Join-Path $Packages 'filelock'); Label = 'filelock' },
        @{ Path = (Join-Path $Packages 'setuptools'); Label = 'setuptools' },
        @{ Path = (Join-Path $Packages 'scipy\optimize'); Label = 'scipy/optimize' },
        @{ Path = (Join-Path $Packages 'scipy\stats'); Label = 'scipy/stats' },
        @{ Path = (Join-Path $Packages 'scipy\interpolate'); Label = 'scipy/interpolate' },
        @{ Path = (Join-Path $Packages 'scipy\io'); Label = 'scipy/io' },
        @{ Path = (Join-Path $Packages 'scipy\signal'); Label = 'scipy/signal' },
        @{ Path = (Join-Path $Packages 'scipy\integrate'); Label = 'scipy/integrate' },
        @{ Path = (Join-Path $Packages 'scipy\fft'); Label = 'scipy/fft' },
        @{ Path = (Join-Path $Packages 'scipy\ndimage'); Label = 'scipy/ndimage' },
        @{ Path = (Join-Path $Packages 'scipy\odr'); Label = 'scipy/odr' },
        @{ Path = (Join-Path $Packages 'scipy\cluster'); Label = 'scipy/cluster' },
        @{ Path = (Join-Path $Packages 'scipy\fftpack'); Label = 'scipy/fftpack' },
        @{ Path = (Join-Path $Packages 'scipy\differentiate'); Label = 'scipy/differentiate' },
        @{ Path = (Join-Path $Packages 'scipy\datasets'); Label = 'scipy/datasets' }
    )) {
        $prunedBytes += (Try-PruneValidatedTree ([string]$candidate.Path) $PythonExe $ServerPath ([string]$candidate.Label))
    }

    Assert-PrivatePython $PythonExe 'after dependency pruning'
    Invoke-PrivatePythonProbe $PythonExe 'post-prune package/optimiser provenance' @('-I','-u',$RuntimeProbe) | Out-Null
    $health = Invoke-SolverHealth $PythonExe $ServerPath 'post-prune SolverHost production health'

    $expectedProduction = [IO.Path]::GetFullPath((Join-Path $RuntimeBuild 'solver\production_b14.py'))
    $actualProduction = [IO.Path]::GetFullPath([string]$health.conversion_module)
    if (-not [string]::Equals($expectedProduction, $actualProduction, [StringComparison]::OrdinalIgnoreCase)) {
        throw "SolverHost loaded the wrong production module. Expected $expectedProduction but got $actualProduction"
    }

    # Final Python filesystem step: strip bytecode and do not run Python again before publish.
    $beforeBytecode = Get-TreeBytes $RuntimeBuild
    Remove-Bytecode $Packages
    Remove-Bytecode (Join-Path $RuntimeBuild 'solver')
    Remove-Bytecode (Join-Path $RuntimeBuild 'rbody')
    Remove-Bytecode (Join-Path $RuntimeBuild 'b14_frozen')
    $afterBytecode = Get-TreeBytes $RuntimeBuild
    $bytecodePruned = [Math]::Max([int64]0, [int64]($beforeBytecode - $afterBytecode))
    $prunedBytes += $bytecodePruned
    Write-Host "[RavaFit Runtime] Final bytecode cleanup removed $(Format-MiB $bytecodePruned). No Python process runs after this point."

    $components = [ordered]@{
        PythonCoreBytes = (Get-TreeBytes $RuntimeBuild) - (Get-TreeBytes $Packages) - (Get-TreeBytes (Join-Path $RuntimeBuild 'solver')) - (Get-TreeBytes (Join-Path $RuntimeBuild 'rbody')) - (Get-TreeBytes (Join-Path $RuntimeBuild 'b14_frozen'))
        PackagesBytes = Get-TreeBytes $Packages
        TorchBytes = Get-TreeBytes (Join-Path $Packages 'torch')
        NumpyBytes = Get-TreeBytes (Join-Path $Packages 'numpy')
        ScipyBytes = Get-TreeBytes (Join-Path $Packages 'scipy')
        TrimeshBytes = Get-TreeBytes (Join-Path $Packages 'trimesh')
        SympyBytes = Get-TreeBytes (Join-Path $Packages 'sympy')
        MpmathBytes = Get-TreeBytes (Join-Path $Packages 'mpmath')
        RavaFitPythonBytes = (Get-TreeBytes (Join-Path $RuntimeBuild 'solver')) + (Get-TreeBytes (Join-Path $RuntimeBuild 'rbody')) + (Get-TreeBytes (Join-Path $RuntimeBuild 'b14_frozen'))
    }

    [ordered]@{
        RuntimeVersion = $RuntimeVersion
        Python = "$PythonVersion-embed-amd64"
        BuildInstaller = "pip-$PipZipappVersion.pyz executed by private CPython; not shipped"
        Numpy = '2.3.5'
        Scipy = '1.17.0'
        Trimesh = '4.11.1'
        Torch = '2.10.0+cpu'
        Layout = 'private-python-wheel-install+validated-runtime-diet+final-bytecode-prune'
        SolverSource = 'RavaFit 1.1.1-multi-region-support-source-preserve'
        ProductionRevision = $ExpectedProductionRevision
        PrunedBytes = $prunedBytes
        Components = $components
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RuntimeBuild '.ravafit-runtime.json') -Encoding UTF8

    $runtimeBytes = Get-TreeBytes $RuntimeBuild
    Write-Host "[RavaFit Runtime] Publishing validated runtime atomically..."
    try {
        if (Test-Path -LiteralPath $Destination) { Move-Item -LiteralPath $Destination -Destination $RuntimeBackup }
        Move-Item -LiteralPath $RuntimeBuild -Destination $Destination
        if (Test-Path -LiteralPath $RuntimeBackup) { Remove-Item -LiteralPath $RuntimeBackup -Recurse -Force }
    }
    catch {
        $publishError = $_
        if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path -LiteralPath $RuntimeBackup) { Move-Item -LiteralPath $RuntimeBackup -Destination $Destination -ErrorAction SilentlyContinue }
        throw "Validated runtime could not be published to $Destination. $($publishError.Exception.Message)"
    }

    Write-Host "[RavaFit Runtime] Private runtime installed successfully." -ForegroundColor Green
    Write-Host "[RavaFit Runtime] Total:      $(Format-MiB $runtimeBytes)"
    Write-Host "[RavaFit Runtime] PyTorch:    $(Format-MiB $components.TorchBytes)"
    Write-Host "[RavaFit Runtime] SciPy:      $(Format-MiB $components.ScipyBytes)"
    Write-Host "[RavaFit Runtime] NumPy:      $(Format-MiB $components.NumpyBytes)"
    Write-Host "[RavaFit Runtime] trimesh:    $(Format-MiB $components.TrimeshBytes)"
    Write-Host "[RavaFit Runtime] SymPy:      $(Format-MiB $components.SympyBytes)"
    Write-Host "[RavaFit Runtime] mpmath:     $(Format-MiB $components.MpmathBytes)"
    Write-Host "[RavaFit Runtime] RavaFit py: $(Format-MiB $components.RavaFitPythonBytes)"
    Write-Host "[RavaFit Runtime] Pruned:     $(Format-MiB $prunedBytes)"
}
catch {
    Write-Host "[RavaFit Runtime] FAILED: $($_.Exception.Message)" -ForegroundColor Red
    throw
}
finally {
    if (Test-Path -LiteralPath $RuntimeBuild) { Remove-Item -LiteralPath $RuntimeBuild -Recurse -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $RuntimeBackup) {
        if (-not (Test-Path -LiteralPath $Destination)) { Move-Item -LiteralPath $RuntimeBackup -Destination $Destination -ErrorAction SilentlyContinue }
        elseif (Test-Path -LiteralPath $RuntimeBackup) { Remove-Item -LiteralPath $RuntimeBackup -Recurse -Force -ErrorAction SilentlyContinue }
    }
    if (Test-Path -LiteralPath $Temp) { Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue }
}
