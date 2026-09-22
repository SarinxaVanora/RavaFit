using System.Diagnostics;
using System.Text.Json;
using Dalamud.Plugin.Services;

namespace RavaFit.Services;

internal sealed class SolverHostService : IDisposable
{
    internal const string DefaultProductionRevision = "1.1.1-multi-region-support-source-preserve";
    private static readonly string[] RequiredCapabilities =
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

    private readonly Configuration _config;
    private readonly IPluginLog _log;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _process;
    private StreamWriter? _stdin;
    private StreamReader? _stdout;
    private long _nextId;

    public SolverHostService(Configuration config, IPluginLog log)
    {
        _config = config;
        _log = log;
    }

    public bool Ready { get; private set; }
    public string Status { get; private set; } = "Offline";
    public string LastError { get; private set; } = string.Empty;
    public bool ConversionReady { get; private set; }

    public async Task ProbeAsync()
    {
        try
        {
            var reply = await CallAsync("health", new { }).ConfigureAwait(false);
            Ready = reply.RootElement.TryGetProperty("ok", out var ok) && ok.GetBoolean();
            var missingCapability = string.Empty;
            var capabilitiesReady = Ready
                && reply.RootElement.TryGetProperty("capabilities", out var capabilities)
                && HasRequiredCapabilities(capabilities, out missingCapability);
            var productionRevision = reply.RootElement.TryGetProperty("production_revision", out var revisionElement)
                && revisionElement.ValueKind == JsonValueKind.String
                    ? revisionElement.GetString() ?? string.Empty
                    : string.Empty;
            var runtimeRoot = reply.RootElement.TryGetProperty("runtime_root", out var runtimeRootElement)
                && runtimeRootElement.ValueKind == JsonValueKind.String
                    ? runtimeRootElement.GetString() ?? string.Empty
                    : string.Empty;
            var expectedProductionRevision = string.IsNullOrWhiteSpace(_config.RuntimeProductionRevision)
                ? DefaultProductionRevision
                : _config.RuntimeProductionRevision;
            ConversionReady = capabilitiesReady
                && string.Equals(productionRevision, expectedProductionRevision, StringComparison.Ordinal);
            if (Ready && ConversionReady)
            {
                Status = "Ready";
                LastError = string.Empty;
                _log.Information("RavaFit SolverHost ready. Runtime={RuntimeRoot}, ProductionRevision={ProductionRevision}", runtimeRoot, productionRevision);
            }
            else if (Ready && capabilitiesReady)
            {
                Status = "Runtime mismatch";
                LastError = $"RavaFit runtime does not match the active GitHub asset manifest. Loaded production revision '{(string.IsNullOrWhiteSpace(productionRevision) ? "missing" : productionRevision)}', expected '{expectedProductionRevision}'. Re-download the RavaFit runtime asset.";
                _log.Error("{Message} Runtime={RuntimeRoot}", LastError, runtimeRoot);
            }
            else if (Ready)
            {
                Status = "Incomplete";
                LastError = !string.IsNullOrWhiteSpace(missingCapability)
                    ? $"Solver runtime is missing required capability '{missingCapability}'."
                    : reply.RootElement.TryGetProperty("conversion_error", out var conversionError)
                        && conversionError.ValueKind == JsonValueKind.String
                        && !string.IsNullOrWhiteSpace(conversionError.GetString())
                            ? $"Solver runtime could not load: {conversionError.GetString()}"
                            : "Solver runtime is incomplete.";
            }
            else
            {
                Status = "Unavailable";
                LastError = "Solver health check did not return ok=true.";
            }
        }
        catch (Exception ex)
        {
            Ready = false;
            ConversionReady = false;
            Status = "Offline";
            LastError = ex.Message;
            _log.Debug(ex, "RavaFit SolverHost probe failed.");
        }
    }

    private static bool HasRequiredCapabilities(JsonElement capabilities, out string missingCapability)
    {
        foreach (var name in RequiredCapabilities)
        {
            if (!capabilities.TryGetProperty(name, out var capability) || capability.ValueKind != JsonValueKind.True)
            {
                missingCapability = name;
                return false;
            }
        }

        missingCapability = string.Empty;
        return true;
    }

    public async Task<JsonDocument> CallAsync(string method, object payload, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            EnsureStarted();
            var id = Interlocked.Increment(ref _nextId);
            var request = JsonSerializer.Serialize(new { id, method, payload });
            await _stdin!.WriteLineAsync(request.AsMemory(), cancellationToken).ConfigureAwait(false);
            await _stdin.FlushAsync(cancellationToken).ConfigureAwait(false);

            while (true)
            {
                var line = await _stdout!.ReadLineAsync(cancellationToken).ConfigureAwait(false);
                if (line is null)
                    throw new IOException("SolverHost closed its output stream.");
                using var response = JsonDocument.Parse(line);
                if (!response.RootElement.TryGetProperty("id", out var responseId) || responseId.GetInt64() != id)
                    continue;
                if (response.RootElement.TryGetProperty("error", out var error) && error.ValueKind == JsonValueKind.String)
                    throw new InvalidOperationException(error.GetString());
                return JsonDocument.Parse(response.RootElement.GetRawText());
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    private void EnsureStarted()
    {
        if (_process is { HasExited: false })
            return;
        DisposeProcess();

        var runtime = !string.IsNullOrWhiteSpace(_config.DeveloperPythonPath) && File.Exists(_config.DeveloperPythonPath)
            ? Path.GetDirectoryName(Path.GetFullPath(_config.DeveloperPythonPath)) ?? Path.GetFullPath(_config.RuntimeDirectory)
            : Path.GetFullPath(_config.RuntimeDirectory);
        var python = ResolvePython(runtime);
        var server = Path.Combine(runtime, "solver", "server.py");
        if (!File.Exists(server))
            throw new FileNotFoundException("RavaFit solver runtime is not installed. Missing solver/server.py.", server);

        var start = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-I -u \"{server}\"",
            WorkingDirectory = runtime,
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

        var process = Process.Start(start) ?? throw new InvalidOperationException("Could not launch RavaFit SolverHost.");
        _process = process;
        _stdin = process.StandardInput;
        _stdout = process.StandardOutput;
        _ = Task.Run(async () =>
        {
            try
            {
                while (!process.HasExited && await process.StandardError.ReadLineAsync().ConfigureAwait(false) is { } line)
                    _log.Debug("[SolverHost] {Line}", line);
            }
            catch { }
        });
    }

    private string ResolvePython(string runtime)
    {
        if (!string.IsNullOrWhiteSpace(_config.DeveloperPythonPath) && File.Exists(_config.DeveloperPythonPath))
            return _config.DeveloperPythonPath;
        foreach (var candidate in new[] { "python.exe", "pythonw.exe" })
        {
            var path = Path.Combine(runtime, candidate);
            if (File.Exists(path))
                return path;
        }
        throw new FileNotFoundException("RavaFit private Python runtime is missing. Open RavaFit and install the GitHub runtime asset.");
    }

    private void DisposeProcess()
    {
        try { _stdin?.Dispose(); } catch { }
        try { _stdout?.Dispose(); } catch { }
        try
        {
            if (_process is { HasExited: false })
            {
                _process.Kill(true);
                try { _process.WaitForExit(5000); } catch { }
            }
        }
        catch { }
        _process?.Dispose();
        _process = null;
        _stdin = null;
        _stdout = null;
    }

    public async Task RestartAsync()
    {
        await _gate.WaitAsync().ConfigureAwait(false);
        try
        {
            DisposeProcess();
            Ready = false;
            ConversionReady = false;
            Status = "Restarting";
            LastError = string.Empty;
        }
        finally
        {
            _gate.Release();
        }

        await ProbeAsync().ConfigureAwait(false);
    }

    public void Dispose()
    {
        DisposeProcess();
    }
}
