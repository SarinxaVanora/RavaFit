using System.Diagnostics;
using System.IO.Compression;
using System.Net;
using System.Net.Http.Headers;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using Dalamud.Plugin.Services;

namespace RavaFit.Services;

internal sealed record RavaFitAssetStatus(
    bool Ready,
    bool Busy,
    bool UpdateAvailable,
    string Stage,
    string Detail,
    double Progress,
    long CurrentBytes,
    long TotalBytes,
    string RuntimeVersion,
    string BodiesVersion,
    string RemoteRuntimeVersion,
    string RemoteBodiesVersion,
    string LastError);

internal sealed record RavaFitAssetActivation(
    string RuntimeDirectory,
    string BodyLibraryDirectory,
    string RuntimeProductionRevision,
    string RuntimeVersion,
    string BodiesVersion);

internal sealed class RavaFitAssetService : IDisposable
{
    private const int SupportedManifestSchema = 1;
    private static readonly string[] RequiredRuntimeCapabilities =
    [
        "convert",
        "graft_native_body",
        "analyze_coverage",
        "detect_source_bodies",
        "inspect_mdl_parts",
        "hide_mdl_parts",
        "tag_mdl_parts",
        "split_mdl_parts",
        "piercing_customisation",
    ];

    private readonly IPluginLog _log;
    private readonly HttpClient _http;
    private readonly SemaphoreSlim _operationGate = new(1, 1);
    private readonly object _statusGate = new();
    private readonly string _assetRoot;
    private readonly string _runtimeRoot;
    private readonly string _bodiesRoot;
    private readonly string _downloadRoot;
    private readonly string _statePath;
    private RavaFitAssetManifest? _manifest;
    private InstalledAssetState _installed = new();
    private RavaFitAssetStatus _status = new(false, false, false, "Not checked", string.Empty, 0d, 0, 0, string.Empty, string.Empty, string.Empty, string.Empty, string.Empty);
    private bool _disposed;

    internal RavaFitAssetService(string localRoot, IPluginLog log)
    {
        _log = log;
        _assetRoot = Path.Combine(localRoot, "Assets");
        _runtimeRoot = Path.Combine(_assetRoot, "Runtime");
        _bodiesRoot = Path.Combine(_assetRoot, "Bodies");
        _downloadRoot = Path.Combine(_assetRoot, "Downloads");
        _statePath = Path.Combine(_assetRoot, "installed-assets.json");
        Directory.CreateDirectory(_runtimeRoot);
        Directory.CreateDirectory(_bodiesRoot);
        Directory.CreateDirectory(_downloadRoot);
        Directory.CreateDirectory(FallbackRuntimeDirectory);
        Directory.CreateDirectory(FallbackBodyDirectory);

        _http = new HttpClient(new HttpClientHandler { AutomaticDecompression = DecompressionMethods.All, AllowAutoRedirect = true })
        {
            Timeout = TimeSpan.FromMinutes(30),
        };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("RavaFit/1.0");
    }

    internal event Action<RavaFitAssetActivation>? ActivationReady;

    internal string FallbackRuntimeDirectory => Path.Combine(_assetRoot, "UnavailableRuntime");
    internal string FallbackBodyDirectory => Path.Combine(_assetRoot, "UnavailableBodies");

    internal RavaFitAssetStatus GetStatus()
    {
        lock (_statusGate) return _status;
    }

    internal void ApplyInstalledState(Configuration configuration)
    {
        _installed = ReadInstalledState();
        var runtimeValid = ValidateInstalledRuntime(_installed.Runtime, out var runtimeDirectory);
        var bodiesValid = ValidateInstalledBodies(_installed.Bodies, out var bodiesDirectory);

        configuration.RuntimeDirectory = runtimeValid ? runtimeDirectory : FallbackRuntimeDirectory;
        configuration.BodyLibraryDirectory = bodiesValid ? bodiesDirectory : FallbackBodyDirectory;
        configuration.RuntimeProductionRevision = runtimeValid ? _installed.Runtime!.ProductionRevision : string.Empty;

        if (runtimeValid || bodiesValid)
            _ = PruneInactiveAssetsAsync();

        UpdateStatus(status => status with
        {
            Ready = runtimeValid && bodiesValid,
            RuntimeVersion = runtimeValid ? _installed.Runtime!.Version : string.Empty,
            BodiesVersion = bodiesValid ? _installed.Bodies!.Version : string.Empty,
            Stage = runtimeValid && bodiesValid ? "Ready" : "Setup required",
            Detail = runtimeValid && bodiesValid ? "RavaFit assets are installed." : "RavaFit needs its runtime and body catalogue.",
            LastError = string.Empty,
        });
    }

    internal async Task InitialiseAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            await RefreshManifestAsync(cancellationToken).ConfigureAwait(false);
            if (!GetStatus().Ready)
                await InstallAvailableAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        catch (Exception ex)
        {
            SetFailure("Could not check RavaFit assets", ex);
        }
    }

    internal async Task RefreshManifestAsync(CancellationToken cancellationToken = default)
    {
        if (!RavaFitDistribution.IsConfigured)
        {
            UpdateStatus(status => status with
            {
                Stage = "GitHub setup required",
                Detail = "RavaFit has no valid GitHub asset manifest URL configured.",
                LastError = "The configured GitHub asset manifest URL is invalid.",
                UpdateAvailable = false,
            });
            return;
        }

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            SetBusy("Checking GitHub", "Checking runtime and body catalogue versions...", 0d, 0, 0);
            using var request = new HttpRequestMessage(HttpMethod.Get, RavaFitDistribution.AssetManifestUrl);
            using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();
            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            var manifest = await JsonSerializer.DeserializeAsync<RavaFitAssetManifest>(stream, JsonOptions, cancellationToken).ConfigureAwait(false)
                ?? throw new InvalidDataException("GitHub returned an empty RavaFit asset manifest.");
            ValidateManifest(manifest);
            _manifest = manifest;
            var updateAvailable = NeedsInstall(manifest.Runtime, _installed.Runtime) || NeedsInstall(manifest.Bodies, _installed.Bodies);
            var ready = ValidateInstalledRuntime(_installed.Runtime, out _) && ValidateInstalledBodies(_installed.Bodies, out _);
            UpdateStatus(status => status with
            {
                Ready = ready,
                Busy = false,
                UpdateAvailable = updateAvailable,
                Stage = ready ? (updateAvailable ? "Update available" : "Ready") : "Setup required",
                Detail = ready
                    ? (updateAvailable ? "A newer RavaFit runtime or body catalogue is available." : "RavaFit assets are up to date.")
                    : "RavaFit needs its runtime and body catalogue.",
                Progress = 0d,
                CurrentBytes = 0,
                TotalBytes = 0,
                RemoteRuntimeVersion = manifest.Runtime.Version,
                RemoteBodiesVersion = manifest.Bodies.Version,
                LastError = string.Empty,
            });
        }
        catch
        {
            UpdateStatus(status => status with { Busy = false });
            throw;
        }
        finally
        {
            _operationGate.Release();
        }
    }

    internal async Task InstallAvailableAsync(CancellationToken cancellationToken = default)
    {
        if (!RavaFitDistribution.IsConfigured)
        {
            UpdateStatus(status => status with
            {
                Stage = "GitHub setup required",
                Detail = "RavaFit has no valid GitHub asset manifest URL configured.",
                LastError = "The configured GitHub asset manifest URL is invalid.",
            });
            return;
        }

        if (_manifest is null)
        {
            await RefreshManifestAsync(cancellationToken).ConfigureAwait(false);
            if (_manifest is null) return;
        }

        await _operationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            var manifest = _manifest;
            ValidateManifest(manifest);

            var runtime = _installed.Runtime;
            var bodies = _installed.Bodies;
            if (NeedsInstall(manifest.Runtime, runtime) || !ValidateInstalledRuntime(runtime, out _))
                runtime = await InstallRuntimeAsync(manifest.Runtime, cancellationToken).ConfigureAwait(false);
            if (NeedsInstall(manifest.Bodies, bodies) || !ValidateInstalledBodies(bodies, out _))
                bodies = await InstallBodiesAsync(manifest.Bodies, cancellationToken).ConfigureAwait(false);

            if (runtime is null || bodies is null)
                throw new InvalidOperationException("RavaFit could not establish a complete runtime + body catalogue installation.");

            _installed = new InstalledAssetState { Schema = 1, Runtime = runtime, Bodies = bodies };
            WriteInstalledState(_installed);

            if (!ValidateInstalledRuntime(runtime, out var runtimeDirectory) || !ValidateInstalledBodies(bodies, out var bodiesDirectory))
                throw new InvalidDataException("RavaFit assets were downloaded but failed their final installation validation.");

            UpdateStatus(status => status with
            {
                Ready = true,
                Busy = false,
                UpdateAvailable = false,
                Stage = "Ready",
                Detail = "RavaFit assets are installed and ready.",
                Progress = 1d,
                CurrentBytes = 0,
                TotalBytes = 0,
                RuntimeVersion = runtime.Version,
                BodiesVersion = bodies.Version,
                RemoteRuntimeVersion = manifest.Runtime.Version,
                RemoteBodiesVersion = manifest.Bodies.Version,
                LastError = string.Empty,
            });

            ActivationReady?.Invoke(new RavaFitAssetActivation(runtimeDirectory, bodiesDirectory, runtime.ProductionRevision, runtime.Version, bodies.Version));
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            UpdateStatus(status => status with { Busy = false, Stage = status.Ready ? "Ready" : "Setup required", Detail = "Asset download cancelled." });
        }
        catch (Exception ex)
        {
            SetFailure("RavaFit asset installation failed", ex);
        }
        finally
        {
            _operationGate.Release();
        }
    }

    private async Task<InstalledRuntimeAsset> InstallRuntimeAsync(RavaFitAssetDescriptor asset, CancellationToken cancellationToken)
    {
        var finalDirectory = Path.Combine(_runtimeRoot, $"{SafeSegment(asset.Version)}-{asset.Sha256[..12].ToLowerInvariant()}");
        if (Directory.Exists(finalDirectory))
        {
            try
            {
                ValidateRuntimeLayout(finalDirectory, asset.ProductionRevision);
                await ValidateRuntimeHealthAsync(finalDirectory, asset.ProductionRevision!, cancellationToken).ConfigureAwait(false);
                return CreateInstalledRuntime(asset, finalDirectory);
            }
            catch { TryDeleteDirectory(finalDirectory); }
        }

        var archiveName = $"runtime-{SafeSegment(asset.Version)}-{asset.Sha256[..12].ToLowerInvariant()}.zip";
        var archivePath = Path.Combine(_downloadRoot, archiveName + ".part");
        await DownloadAsync(asset, archivePath, "Downloading runtime", cancellationToken).ConfigureAwait(false);
        await VerifyHashAsync(archivePath, asset.Sha256, "runtime", cancellationToken).ConfigureAwait(false);

        if (!Directory.Exists(finalDirectory))
        {
            var staging = finalDirectory + ".staging-" + Guid.NewGuid().ToString("N");
            try
            {
                Directory.CreateDirectory(staging);
                UpdateStatus(status => status with { Stage = "Installing runtime", Detail = "Extracting the private RavaFit runtime...", Progress = 0d, CurrentBytes = 0, TotalBytes = 0 });
                ZipFile.ExtractToDirectory(archivePath, staging, overwriteFiles: true);
                ValidateRuntimeLayout(staging, asset.ProductionRevision);
                await ValidateRuntimeHealthAsync(staging, asset.ProductionRevision!, cancellationToken).ConfigureAwait(false);
                Directory.Move(staging, finalDirectory);
            }
            finally
            {
                TryDeleteDirectory(staging);
            }
        }

        TryDeleteFile(archivePath);
        return CreateInstalledRuntime(asset, finalDirectory);
    }

    private static InstalledRuntimeAsset CreateInstalledRuntime(RavaFitAssetDescriptor asset, string directory)
        => new()
        {
            Version = asset.Version,
            Sha256 = NormaliseHash(asset.Sha256),
            Directory = directory,
            ProductionRevision = asset.ProductionRevision ?? string.Empty,
            InstalledUtc = DateTime.UtcNow,
        };

    private async Task<InstalledBodiesAsset> InstallBodiesAsync(RavaFitAssetDescriptor asset, CancellationToken cancellationToken)
    {
        var finalDirectory = Path.Combine(_bodiesRoot, $"{SafeSegment(asset.Version)}-{asset.Sha256[..12].ToLowerInvariant()}");
        var finalPath = Path.Combine(finalDirectory, "Bodies.rbody");
        if (File.Exists(finalPath))
        {
            try
            {
                ValidateRBody(finalPath);
                return CreateInstalledBodies(asset, finalDirectory, finalPath);
            }
            catch { TryDeleteFile(finalPath); }
        }

        var partialPath = Path.Combine(_downloadRoot, $"Bodies-{SafeSegment(asset.Version)}-{asset.Sha256[..12].ToLowerInvariant()}.rbody.part");
        await DownloadAsync(asset, partialPath, "Downloading body catalogue", cancellationToken).ConfigureAwait(false);
        await VerifyHashAsync(partialPath, asset.Sha256, "Bodies.rbody", cancellationToken).ConfigureAwait(false);

        if (!File.Exists(finalPath))
        {
            Directory.CreateDirectory(finalDirectory);
            var staging = finalPath + ".staging-" + Guid.NewGuid().ToString("N");
            try
            {
                File.Move(partialPath, staging, true);
                ValidateRBody(staging);
                File.Move(staging, finalPath, true);
            }
            finally
            {
                TryDeleteFile(staging);
            }
        }
        else
        {
            TryDeleteFile(partialPath);
        }

        return CreateInstalledBodies(asset, finalDirectory, finalPath);
    }

    private static InstalledBodiesAsset CreateInstalledBodies(RavaFitAssetDescriptor asset, string directory, string path)
        => new()
        {
            Version = asset.Version,
            Sha256 = NormaliseHash(asset.Sha256),
            Directory = directory,
            Size = new FileInfo(path).Length,
            InstalledUtc = DateTime.UtcNow,
        };

    private async Task DownloadAsync(RavaFitAssetDescriptor asset, string partialPath, string stage, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(partialPath)!);
        var existing = File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0L;
        if (asset.Size > 0 && existing > asset.Size)
        {
            TryDeleteFile(partialPath);
            existing = 0;
        }
        if (asset.Size > 0 && existing == asset.Size)
        {
            UpdateStatus(status => status with
            {
                Busy = true,
                Stage = stage,
                Detail = $"Resuming from a complete {FormatBytes(existing)} download; verifying locally...",
                Progress = 1d,
                CurrentBytes = existing,
                TotalBytes = asset.Size,
                LastError = string.Empty,
            });
            return;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, asset.Url);
        if (existing > 0)
            request.Headers.Range = new RangeHeaderValue(existing, null);

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        var append = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        if (!append)
            existing = 0;
        response.EnsureSuccessStatusCode();

        var responseBytes = response.Content.Headers.ContentLength ?? 0L;
        var total = asset.Size > 0 ? asset.Size : existing + responseBytes;
        var mode = append ? FileMode.Append : FileMode.Create;
        await using var output = new FileStream(partialPath, mode, FileAccess.Write, FileShare.None, 1024 * 1024, useAsync: true);
        await using var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        var buffer = new byte[1024 * 1024];
        var downloaded = existing;
        while (true)
        {
            var read = await input.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken).ConfigureAwait(false);
            if (read <= 0) break;
            await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            downloaded += read;
            var progress = total > 0 ? Math.Clamp((double)downloaded / total, 0d, 1d) : 0d;
            UpdateStatus(status => status with
            {
                Busy = true,
                Stage = stage,
                Detail = $"{FormatBytes(downloaded)} / {(total > 0 ? FormatBytes(total) : "unknown")}",
                Progress = progress,
                CurrentBytes = downloaded,
                TotalBytes = total,
                LastError = string.Empty,
            });
        }
        await output.FlushAsync(cancellationToken).ConfigureAwait(false);

        if (asset.Size > 0 && downloaded != asset.Size)
            throw new InvalidDataException($"{stage} completed at {downloaded} bytes, but the manifest requires {asset.Size} bytes.");
    }

    private async Task VerifyHashAsync(string path, string expected, string label, CancellationToken cancellationToken)
    {
        UpdateStatus(status => status with { Stage = $"Verifying {label}", Detail = "Checking SHA-256...", Progress = 0d, CurrentBytes = 0, TotalBytes = 0 });
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, useAsync: true);
        var actual = Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false)).ToLowerInvariant();
        if (!string.Equals(actual, NormaliseHash(expected), StringComparison.Ordinal))
        {
            TryDeleteFile(path);
            throw new InvalidDataException($"{label} SHA-256 mismatch. The partial download was discarded.");
        }
    }

    private static void ValidateRuntimeLayout(string directory, string? expectedProductionRevision)
    {
        var required = new[]
        {
            "python.exe",
            Path.Combine("solver", "server.py"),
            Path.Combine("solver", "production_b14.py"),
            Path.Combine("solver", "b14_compat.py"),
            Path.Combine("solver", "native_body_graft.py"),
            Path.Combine("rbody", "rbody_v3_loader.py"),
            Path.Combine("b14_frozen", "scripts", "b14_mesh_worker.py"),
        };
        foreach (var relative in required)
        {
            var path = Path.Combine(directory, relative);
            if (!File.Exists(path)) throw new InvalidDataException($"Runtime archive is incomplete: missing {relative}.");
        }

        if (string.IsNullOrWhiteSpace(expectedProductionRevision))
            throw new InvalidDataException("Runtime manifest is missing productionRevision.");

        var markerPath = Path.Combine(directory, ".ravafit-runtime.json");
        if (!File.Exists(markerPath))
            throw new InvalidDataException("Runtime archive is missing .ravafit-runtime.json.");
        using var marker = JsonDocument.Parse(File.ReadAllText(markerPath));
        var markerRevision = TryReadString(marker.RootElement, "ProductionRevision")
            ?? TryReadString(marker.RootElement, "SolverSource")?.Replace("RavaFit ", string.Empty, StringComparison.OrdinalIgnoreCase);
        if (!string.Equals(markerRevision, expectedProductionRevision, StringComparison.Ordinal))
            throw new InvalidDataException($"Runtime marker revision '{markerRevision ?? "missing"}' does not match manifest revision '{expectedProductionRevision}'.");
    }

    private async Task ValidateRuntimeHealthAsync(string directory, string expectedProductionRevision, CancellationToken cancellationToken)
    {
        UpdateStatus(status => status with
        {
            Busy = true,
            Stage = "Validating runtime",
            Detail = "Starting SolverHost from the staged GitHub runtime...",
            Progress = 0d,
            CurrentBytes = 0,
            TotalBytes = 0,
            LastError = string.Empty,
        });

        var python = Path.Combine(directory, "python.exe");
        var server = Path.Combine(directory, "solver", "server.py");
        var start = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-I -u \"{server}\"",
            WorkingDirectory = directory,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        start.Environment["PYTHONUTF8"] = "1";
        start.Environment["PYTHONHASHSEED"] = "0";
        start.Environment["OMP_NUM_THREADS"] = "1";
        start.Environment["OPENBLAS_NUM_THREADS"] = "1";
        start.Environment["MKL_NUM_THREADS"] = "1";
        start.Environment["NUMEXPR_NUM_THREADS"] = "1";

        using var process = Process.Start(start) ?? throw new InvalidOperationException("Could not launch the staged RavaFit SolverHost.");
        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(TimeSpan.FromMinutes(2));
            const long requestId = 1;
            var request = JsonSerializer.Serialize(new { id = requestId, method = "health", payload = new { } });
            await process.StandardInput.WriteLineAsync(request.AsMemory(), timeout.Token).ConfigureAwait(false);
            await process.StandardInput.FlushAsync(timeout.Token).ConfigureAwait(false);

            while (true)
            {
                var line = await process.StandardOutput.ReadLineAsync(timeout.Token).ConfigureAwait(false);
                if (line is null)
                    throw new InvalidDataException("The staged SolverHost exited before returning health information.");
                using var response = JsonDocument.Parse(line);
                if (!response.RootElement.TryGetProperty("id", out var idElement) || idElement.GetInt64() != requestId)
                    continue;
                if (response.RootElement.TryGetProperty("error", out var error) && error.ValueKind == JsonValueKind.String)
                    throw new InvalidDataException($"The staged SolverHost rejected its health check: {error.GetString()}");
                if (!response.RootElement.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
                    throw new InvalidDataException("The staged SolverHost health check did not return ok=true.");
                if (!response.RootElement.TryGetProperty("capabilities", out var capabilities))
                    throw new InvalidDataException("The staged SolverHost did not report runtime capabilities.");
                foreach (var capabilityName in RequiredRuntimeCapabilities)
                {
                    if (!capabilities.TryGetProperty(capabilityName, out var capability) || capability.ValueKind != JsonValueKind.True)
                        throw new InvalidDataException($"The staged SolverHost is missing required capability '{capabilityName}'.");
                }
                var revision = response.RootElement.TryGetProperty("production_revision", out var revisionElement) && revisionElement.ValueKind == JsonValueKind.String
                    ? revisionElement.GetString() ?? string.Empty
                    : string.Empty;
                if (!string.Equals(revision, expectedProductionRevision, StringComparison.Ordinal))
                    throw new InvalidDataException($"The staged SolverHost reports production revision '{revision}', expected '{expectedProductionRevision}'.");
                break;
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException("The staged RavaFit SolverHost did not become healthy within two minutes.");
        }
        finally
        {
            try { process.StandardInput.Close(); } catch { }
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(true);
                    await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
                }
            }
            catch { }
        }

        var stderr = await stderrTask.ConfigureAwait(false);
        if (!string.IsNullOrWhiteSpace(stderr))
            _log.Debug("[Staged SolverHost] {Stderr}", stderr.Trim());
    }

    private static void ValidateRBody(string path)
    {
        using var archive = ZipFile.OpenRead(path);
        var manifestEntry = archive.GetEntry("manifest.json");
        var catalogueEntry = archive.GetEntry("catalogue.json");
        if (manifestEntry is null || catalogueEntry is null)
            throw new InvalidDataException("Bodies.rbody is not a valid RavaFit RBODY catalogue.");
        using var manifestStream = manifestEntry.Open();
        using var manifest = JsonDocument.Parse(manifestStream);
        var format = TryReadString(manifest.RootElement, "format");
        if (!string.Equals(format, "RBODY", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Bodies.rbody manifest does not report RBODY format.");
        using var catalogueStream = catalogueEntry.Open();
        using var catalogue = JsonDocument.Parse(catalogueStream);
        if (catalogue.RootElement.ValueKind is not JsonValueKind.Object and not JsonValueKind.Array)
            throw new InvalidDataException("Bodies.rbody catalogue.json is not a valid catalogue payload.");
    }

    private void ValidateManifest(RavaFitAssetManifest manifest)
    {
        if (manifest.Schema != SupportedManifestSchema)
            throw new InvalidDataException($"Unsupported RavaFit asset manifest schema {manifest.Schema}; this plugin supports {SupportedManifestSchema}.");
        ValidateDescriptor("runtime", manifest.Runtime, requireProductionRevision: true);
        ValidateDescriptor("bodies", manifest.Bodies, requireProductionRevision: false);
        EnsurePluginCompatible(manifest.Runtime.MinimumPluginVersion, "runtime");
        EnsurePluginCompatible(manifest.Bodies.MinimumPluginVersion, "body catalogue");
    }

    private static void ValidateDescriptor(string name, RavaFitAssetDescriptor descriptor, bool requireProductionRevision)
    {
        if (string.IsNullOrWhiteSpace(descriptor.Version)) throw new InvalidDataException($"Asset manifest {name}.version is empty.");
        if (!Uri.TryCreate(descriptor.Url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps)
            throw new InvalidDataException($"Asset manifest {name}.url must be an absolute HTTPS URL.");
        var hash = NormaliseHash(descriptor.Sha256);
        if (hash.Length != 64 || hash.Any(c => !Uri.IsHexDigit(c)))
            throw new InvalidDataException($"Asset manifest {name}.sha256 is not a SHA-256 hex digest.");
        if (descriptor.Size < 0) throw new InvalidDataException($"Asset manifest {name}.size cannot be negative.");
        if (requireProductionRevision && string.IsNullOrWhiteSpace(descriptor.ProductionRevision))
            throw new InvalidDataException("Asset manifest runtime.productionRevision is required.");
    }

    private static void EnsurePluginCompatible(string? minimumPluginVersion, string assetName)
    {
        if (string.IsNullOrWhiteSpace(minimumPluginVersion)) return;
        if (!Version.TryParse(minimumPluginVersion, out var minimum))
            throw new InvalidDataException($"Asset manifest {assetName} minimumPluginVersion '{minimumPluginVersion}' is invalid.");
        var current = Assembly.GetExecutingAssembly().GetName().Version ?? new Version(1, 0, 0, 0);
        if (current < minimum)
            throw new InvalidDataException($"The available {assetName} requires RavaFit {minimum} or newer. Update the plugin through Dalamud first.");
    }

    private bool NeedsInstall(RavaFitAssetDescriptor remote, InstalledAsset? local)
        => local is null
        || !string.Equals(local.Version, remote.Version, StringComparison.OrdinalIgnoreCase)
        || !string.Equals(NormaliseHash(local.Sha256), NormaliseHash(remote.Sha256), StringComparison.Ordinal);

    private bool ValidateInstalledRuntime(InstalledRuntimeAsset? runtime, out string directory)
    {
        directory = runtime?.Directory ?? string.Empty;
        if (runtime is null || string.IsNullOrWhiteSpace(runtime.Directory) || !Directory.Exists(runtime.Directory)) return false;
        try { ValidateRuntimeLayout(runtime.Directory, runtime.ProductionRevision); return true; }
        catch (Exception ex) { _log.Warning(ex, "RavaFit ignored an invalid installed runtime at {Directory}", runtime.Directory); return false; }
    }

    private bool ValidateInstalledBodies(InstalledBodiesAsset? bodies, out string directory)
    {
        directory = bodies?.Directory ?? string.Empty;
        if (bodies is null || string.IsNullOrWhiteSpace(bodies.Directory)) return false;
        var path = Path.Combine(bodies.Directory, "Bodies.rbody");
        if (!File.Exists(path)) return false;
        if (bodies.Size > 0 && new FileInfo(path).Length != bodies.Size) return false;
        try { ValidateRBody(path); return true; }
        catch (Exception ex) { _log.Warning(ex, "RavaFit ignored an invalid installed body catalogue at {Path}", path); return false; }
    }

    private InstalledAssetState ReadInstalledState()
    {
        try
        {
            if (!File.Exists(_statePath)) return new InstalledAssetState();
            var state = JsonSerializer.Deserialize<InstalledAssetState>(File.ReadAllText(_statePath), JsonOptions);
            return state is { Schema: 1 } ? state : new InstalledAssetState();
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "RavaFit could not read installed asset state; assets will be checked again.");
            return new InstalledAssetState();
        }
    }

    private void WriteInstalledState(InstalledAssetState state)
    {
        Directory.CreateDirectory(_assetRoot);
        var temp = _statePath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            File.WriteAllText(temp, JsonSerializer.Serialize(state, JsonOptionsIndented));
            File.Move(temp, _statePath, true);
        }
        finally { TryDeleteFile(temp); }
    }

    private void SetBusy(string stage, string detail, double progress, long current, long total)
        => UpdateStatus(status => status with { Busy = true, Stage = stage, Detail = detail, Progress = progress, CurrentBytes = current, TotalBytes = total, LastError = string.Empty });

    private void SetFailure(string stage, Exception ex)
    {
        _log.Error(ex, "{Stage}", stage);
        UpdateStatus(status => status with
        {
            Busy = false,
            Stage = status.Ready ? "Asset update failed" : "Setup failed",
            Detail = ex.Message,
            Progress = 0d,
            CurrentBytes = 0,
            TotalBytes = 0,
            LastError = ex.Message,
        });
    }

    private void UpdateStatus(Func<RavaFitAssetStatus, RavaFitAssetStatus> update)
    {
        lock (_statusGate) _status = update(_status);
    }

    private static string SafeSegment(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = value.Select(c => invalid.Contains(c) || char.IsWhiteSpace(c) ? '_' : c).ToArray();
        var result = new string(chars).Trim('.', '_');
        return string.IsNullOrWhiteSpace(result) ? "asset" : result;
    }

    private static string NormaliseHash(string? value) => (value ?? string.Empty).Trim().ToLowerInvariant();
    private static string FormatBytes(long bytes) => bytes >= 1024L * 1024L * 1024L ? $"{bytes / (1024d * 1024d * 1024d):0.00} GB" : $"{bytes / (1024d * 1024d):0.0} MB";

    private static string? TryReadString(JsonElement element, string name)
    {
        foreach (var property in element.EnumerateObject())
            if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase) && property.Value.ValueKind == JsonValueKind.String)
                return property.Value.GetString();
        return null;
    }

    private async Task PruneInactiveAssetsAsync()
    {
        try
        {
            await _operationGate.WaitAsync().ConfigureAwait(false);
            if (_disposed) return;
            if (ValidateInstalledRuntime(_installed.Runtime, out var runtimeDirectory))
                PruneInactiveAssetDirectories(_runtimeRoot, runtimeDirectory);
            if (ValidateInstalledBodies(_installed.Bodies, out var bodiesDirectory))
                PruneInactiveAssetDirectories(_bodiesRoot, bodiesDirectory);
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "RavaFit could not prune inactive downloaded assets.");
        }
        finally
        {
            try { _operationGate.Release(); } catch { }
        }
    }

    private void PruneInactiveAssetDirectories(string root, string activeDirectory)
    {
        try
        {
            var active = Path.GetFullPath(activeDirectory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            foreach (var directory in Directory.EnumerateDirectories(root))
            {
                var candidate = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                if (string.Equals(candidate, active, StringComparison.OrdinalIgnoreCase)) continue;
                TryDeleteDirectory(directory);
            }
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "RavaFit could not prune every inactive asset directory under {Root}.", root);
        }
    }

    private static void TryDeleteFile(string path) { try { if (File.Exists(path)) File.Delete(path); } catch { } }
    private static void TryDeleteDirectory(string path) { try { if (Directory.Exists(path)) Directory.Delete(path, true); } catch { } }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(nameof(RavaFitAssetService)); }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _http.Dispose();
    }

    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };
    private static readonly JsonSerializerOptions JsonOptionsIndented = new() { PropertyNameCaseInsensitive = true, WriteIndented = true };

    private sealed class RavaFitAssetManifest
    {
        public int Schema { get; set; }
        public RavaFitAssetDescriptor Runtime { get; set; } = new();
        public RavaFitAssetDescriptor Bodies { get; set; } = new();
    }

    private sealed class RavaFitAssetDescriptor
    {
        public string Version { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public string Sha256 { get; set; } = string.Empty;
        public long Size { get; set; }
        public string? ProductionRevision { get; set; }
        public string? MinimumPluginVersion { get; set; }
    }

    private class InstalledAsset
    {
        public string Version { get; set; } = string.Empty;
        public string Sha256 { get; set; } = string.Empty;
        public string Directory { get; set; } = string.Empty;
        public DateTime InstalledUtc { get; set; }
    }

    private sealed class InstalledRuntimeAsset : InstalledAsset
    {
        public string ProductionRevision { get; set; } = string.Empty;
    }

    private sealed class InstalledBodiesAsset : InstalledAsset
    {
        public long Size { get; set; }
    }

    private sealed class InstalledAssetState
    {
        public int Schema { get; set; } = 1;
        public InstalledRuntimeAsset? Runtime { get; set; }
        public InstalledBodiesAsset? Bodies { get; set; }
    }
}
