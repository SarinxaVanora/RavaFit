using Dalamud.Plugin;
using Dalamud.Plugin.Services;
using Penumbra.Api.Enums;
using Penumbra.Api.IpcSubscribers;
using Penumbra.Api.Helpers;
using RavaFit.Core.Models;

namespace RavaFit.Services;

internal sealed class PenumbraService : IDisposable
{
    private readonly IPluginLog _log;
    private readonly ApiVersion _apiVersion;
    private readonly GetEnabledState _enabled;
    private readonly GetModList _getModList;
    private readonly GetModDirectory _getModDirectory;
    private readonly AddMod _addMod;
    private readonly ReloadMod _reloadMod;
    private readonly GetCollection _getCollection;
    private readonly GetCurrentModSettings _getCurrentModSettings;
    private readonly EventSubscriber _initialized;
    private readonly EventSubscriber _disposed;
    private readonly EventSubscriber<string, bool> _modDirectoryChanged;
    private readonly Func<string, string> _resolvePlayerPath;

    public PenumbraService(IDalamudPluginInterface pi, IPluginLog log)
    {
        _log = log;
        _apiVersion = new ApiVersion(pi);
        _enabled = new GetEnabledState(pi);
        _getModList = new GetModList(pi);
        _getModDirectory = new GetModDirectory(pi);
        _addMod = new AddMod(pi);
        _reloadMod = new ReloadMod(pi);
        _getCollection = new GetCollection(pi);
        _getCurrentModSettings = new GetCurrentModSettings(pi);
        var resolvePlayerPath = pi.GetIpcSubscriber<string, string>("Penumbra.ResolvePlayerPath");
        _resolvePlayerPath = resolvePlayerPath.InvokeFunc;
        _initialized = Initialized.Subscriber(pi, Refresh);
        _disposed = Disposed.Subscriber(pi, Refresh);
        _modDirectoryChanged = ModDirectoryChanged.Subscriber(pi, (_, _) => Refresh());
    }

    public bool Available { get; private set; }
    public bool ModsEnabled { get; private set; }
    public (int Breaking, int Features)? Version { get; private set; }
    public string LastError { get; private set; } = string.Empty;
    public string ModDirectoryRoot { get; private set; } = string.Empty;
    public IReadOnlyList<PenumbraModInfo> Mods { get; private set; } = [];

    public void Refresh()
    {
        try
        {
            Version = _apiVersion.Invoke();
            ModsEnabled = _enabled.Invoke();
            var modDirectory = _getModDirectory.Invoke();
            if (string.IsNullOrWhiteSpace(modDirectory))
                throw new DirectoryNotFoundException("Penumbra did not provide its mod directory.");

            var modDirectoryRoot = Path.GetFullPath(modDirectory);
            if (!Directory.Exists(modDirectoryRoot))
                throw new DirectoryNotFoundException($"Penumbra mod directory does not exist: {modDirectoryRoot}");

            var mods = _getModList.Invoke();
            var list = new List<PenumbraModInfo>(mods.Count);
            foreach (var (directory, name) in mods)
            {
                try
                {
                    var modRoot = ResolvePhysicalModRoot(modDirectoryRoot, directory);
                    if (!Directory.Exists(modRoot))
                    {
                        _log.Debug("Penumbra listed {Mod} at {Directory}, but the physical directory does not exist: {ModRoot}", name, directory, modRoot);
                        continue;
                    }

                    list.Add(new PenumbraModInfo(directory, name, modRoot));
                }
                catch (Exception ex)
                {
                    _log.Debug(ex, "Could not resolve physical Penumbra mod directory for {Directory}", directory);
                }
            }
            Mods = list.OrderBy(m => m.Name, StringComparer.OrdinalIgnoreCase).ToArray();
            ModDirectoryRoot = modDirectoryRoot;
            Available = true;
            LastError = string.Empty;
        }
        catch (Exception ex)
        {
            Available = false;
            ModsEnabled = false;
            Version = null;
            ModDirectoryRoot = string.Empty;
            Mods = [];
            LastError = ex.Message;
        }
    }

    public IReadOnlyDictionary<string, IReadOnlyCollection<string>>? GetCurrentOptionSettings(PenumbraModInfo mod)
    {
        try
        {
            var collection = _getCollection.Invoke(ApiCollectionType.Yourself);
            if (collection is null || collection.Value.Id == Guid.Empty)
                return null;

            var (ec, settings) = _getCurrentModSettings.Invoke(collection.Value.Id, mod.Directory, mod.Name, false);
            if (ec != PenumbraApiEc.Success || settings is null)
            {
                _log.Debug("Could not read current Penumbra settings for {Mod}: {Error}", mod.Name, ec);
                return null;
            }

            var selected = settings.Value.Item3;
            return selected.ToDictionary(
                x => x.Key,
                x => (IReadOnlyCollection<string>)x.Value.ToArray(),
                StringComparer.OrdinalIgnoreCase);
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "Could not read current Penumbra option settings for {Mod}.", mod.Name);
            return null;
        }
    }

    public string ResolvePlayerPath(string gamePath)
    {
        try
        {
            return _resolvePlayerPath(gamePath);
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "Penumbra ResolvePlayerPath failed for {GamePath}; using the game path.", gamePath);
            return gamePath;
        }
    }

    public bool AddMod(string directory, out string error)
    {
        try
        {
            var ec = _addMod.Invoke(directory);
            if (ec == PenumbraApiEc.Success)
            {
                error = string.Empty;
                return true;
            }
            error = $"Penumbra AddMod returned {ec}.";
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            _log.Error(ex, "Penumbra add-mod failed for {ModDirectory}", directory);
            return false;
        }
    }

    public bool Reload(PenumbraModInfo mod, out string error)
    {
        try
        {
            var ec = _reloadMod.Invoke(mod.Directory, mod.Name);
            if (ec == PenumbraApiEc.Success)
            {
                error = string.Empty;
                return true;
            }
            error = $"Penumbra ReloadMod returned {ec}.";
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            _log.Error(ex, "Penumbra reload failed for {Mod}", mod.Name);
            return false;
        }
    }

    private static string ResolvePhysicalModRoot(string modDirectoryRoot, string modDirectoryName)
    {
        if (string.IsNullOrWhiteSpace(modDirectoryName))
            throw new ArgumentException("Penumbra returned an empty mod directory name.", nameof(modDirectoryName));

        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(modDirectoryRoot));
        var candidate = Path.GetFullPath(Path.Combine(root, modDirectoryName));
        var relative = Path.GetRelativePath(root, candidate);
        if (relative == ".." || relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) || Path.IsPathRooted(relative))
            throw new InvalidDataException($"Penumbra mod directory escaped the configured mod root: {modDirectoryName}");

        return candidate;
    }

    public void Dispose()
    {
        _initialized.Dispose();
        _disposed.Dispose();
        _modDirectoryChanged.Dispose();
    }
}
