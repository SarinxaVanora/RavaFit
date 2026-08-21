using System.Text.Json.Nodes;
using Dalamud.Plugin.Services;
using RavaFit.Core.Models;
using RavaFit.Core.Penumbra;

namespace RavaFit.Services;

internal sealed class ModCleanupService
{
    private readonly PenumbraService _penumbra;
    private readonly IFramework _framework;
    private readonly IPluginLog _log;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public ModCleanupService(PenumbraService penumbra, IFramework framework, IPluginLog log)
    {
        _penumbra = penumbra;
        _framework = framework;
        _log = log;
    }

    public bool Busy => _gate.CurrentCount == 0;
    public string Status { get; private set; } = string.Empty;

    public IReadOnlyList<ModOptionCleanupCandidate> Inspect(PenumbraModInfo? mod)
    {
        if (mod is null) return [];
        var document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
        return document.Groups
            .SelectMany(group => group.Options.Select(option => BuildCandidate(group, option)))
            .OrderBy(candidate => candidate.GroupName, StringComparer.OrdinalIgnoreCase)
            .ThenBy(candidate => candidate.OptionName, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    public async Task<ModOptionCleanupResult> RemoveOptionsAsync(PenumbraModInfo mod, IReadOnlyList<ModOptionCleanupSelection> selections, CancellationToken cancellationToken = default)
    {
        if (selections.Count == 0) throw new InvalidOperationException("Select at least one option to remove.");
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            Status = "Reading mod";
            var metaPath = Path.Combine(mod.ModRoot, "meta.json");
            var originalMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
            var document = PenumbraV4Document.Load(metaPath);
            var beforeFiles = CollectLocalFiles(document.Root);
            var groups = PenumbraV4Document.FindProperty(document.Root, "Groups") as JsonArray
                ?? throw new InvalidDataException("Penumbra V4 metadata has no Groups array.");

            var requested = selections.Select(selection => $"{selection.GroupKey}\u001f{selection.OptionKey}").ToHashSet(StringComparer.OrdinalIgnoreCase);
            var removedOptionIds = new HashSet<Guid>();
            var groupsMadeEmpty = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var removedOptions = 0;
            var removedGroups = 0;

            foreach (var group in document.Groups)
            {
                var optionArray = PenumbraV4Document.FindProperty(group.Node, "Options") as JsonArray;
                if (optionArray is null) continue;
                var removeIndices = new List<int>();
                for (var index = 0; index < group.Options.Count; index++)
                {
                    var option = group.Options[index];
                    if (!requested.Contains($"{group.StableKey}\u001f{option.StableKey}")) continue;
                    removeIndices.Add(index);
                    if (option.Id is { } id) removedOptionIds.Add(id);
                }
                if (removeIndices.Count == 0) continue;
                foreach (var index in removeIndices.OrderByDescending(value => value))
                    optionArray.RemoveAt(index);
                AdjustDefaultSettings(group.Node, group.Type, removeIndices, group.Options.Count);
                if (optionArray.Count == 0) groupsMadeEmpty.Add(group.StableKey);
                removedOptions += removeIndices.Count;
            }

            var changed = true;
            while (changed)
            {
                changed = false;
                for (var index = groups.Count - 1; index >= 0; index--)
                {
                    if (groups[index] is not JsonObject groupNode) continue;
                    var optionArray = PenumbraV4Document.FindProperty(groupNode, "Options") as JsonArray;
                    var groupKey = ReadString(groupNode, "Id") ?? ReadString(groupNode, "Name") ?? string.Empty;
                    var selectedGroupBecameEmpty = optionArray is not null && optionArray.Count == 0 && groupsMadeEmpty.Contains(groupKey);
                    var dependsOnRemoved = ContainsAnySettingReference(PenumbraV4Document.FindProperty(groupNode, "Condition"), removedOptionIds);
                    if (!selectedGroupBecameEmpty && !dependsOnRemoved) continue;
                    if (optionArray is not null)
                    {
                        foreach (var option in optionArray.OfType<JsonObject>())
                        {
                            var idText = ReadString(option, "Id");
                            if (Guid.TryParse(idText, out var id)) removedOptionIds.Add(id);
                        }
                    }
                    groups.RemoveAt(index);
                    removedGroups++;
                    changed = true;
                }
            }

            if (removedOptions == 0)
                throw new InvalidOperationException("None of the selected options still exist. Refresh the mod and try again.");

            RemoveUnusedPageNames(document.Root, groups);
            Status = "Writing metadata";
            SetProperty(document.Root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));
            var tempPath = metaPath + $".ravafit-remove-{Guid.NewGuid():N}.tmp";
            try
            {
                await File.WriteAllTextAsync(tempPath, document.Root.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
                _ = PenumbraV4Document.Load(tempPath);
                var liveMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
                if (!originalMetaBytes.AsSpan().SequenceEqual(liveMetaBytes))
                    throw new InvalidDataException("meta.json changed while RavaFit was preparing removal. Nothing was written; refresh and retry.");
                File.Move(tempPath, metaPath, true);
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }

            Status = "Refreshing Penumbra";
            var reloadResult = await _framework.RunOnTick(() =>
            {
                var success = _penumbra.Reload(mod, out var error);
                _penumbra.Refresh();
                return (Success: success, Error: error);
            }, delayTicks: 2, cancellationToken: cancellationToken).ConfigureAwait(false);

            var warnings = new List<string>();
            var removedFiles = 0;
            var orphanDeleteFailures = 0;
            if (!reloadResult.Success)
            {
                warnings.Add($"Penumbra reload failed after the metadata was committed: {reloadResult.Error}. The removed option metadata is safe, and its orphaned files were deliberately left on disk. Refresh the mod manually.");
                _log.Warning("RavaFit removed the selected options from {Mod}, but the deferred Penumbra reload failed: {Error}. Orphaned files were preserved for safety.", mod.Name, reloadResult.Error);
            }
            else
            {
                Status = "Removing orphaned files";
                var finalDocument = PenumbraV4Document.Load(metaPath);
                var afterFiles = CollectLocalFiles(finalDocument.Root);
                foreach (var relative in beforeFiles.Except(afterFiles, StringComparer.OrdinalIgnoreCase))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var physical = SafeResolve(mod.ModRoot, relative);
                    if (!File.Exists(physical)) continue;
                    try
                    {
                        File.Delete(physical);
                        removedFiles++;
                        DeleteEmptyParents(Path.GetDirectoryName(physical), mod.ModRoot);
                    }
                    catch (Exception ex)
                    {
                        orphanDeleteFailures++;
                        _log.Warning(ex, "RavaFit could not delete orphaned file {File} after option removal; leaving it on disk", physical);
                    }
                }
                if (orphanDeleteFailures > 0)
                    warnings.Add($"{orphanDeleteFailures} orphaned file(s) could not be deleted and were left unreferenced on disk.");
            }

            Status = warnings.Count == 0 ? "Complete" : "Complete with warning";
            return new ModOptionCleanupResult(removedOptions, removedGroups, removedFiles, warnings.Count == 0 ? null : string.Join(" ", warnings));
        }
        catch (Exception ex)
        {
            Status = "Failed";
            _log.Error(ex, "RavaFit mod option removal failed for {Mod}", mod.Name);
            throw;
        }
        finally
        {
            _gate.Release();
        }
    }

    private static ModOptionCleanupCandidate BuildCandidate(V4GroupInfo group, V4OptionInfo option)
    {
        var localFiles = CollectLocalFiles(option.Node);
        var paths = CollectPathLikeStrings(option.Node);
        var kinds = ModOptionKind.None;
        var label = $"{group.Name} {ReadString(group.Node, "Description")} {option.Name} {ReadString(option.Node, "Description")}";
        if (paths.Any(path => HasExtension(path, ".scd")) || ContainsWord(label, "sound") || ContainsWord(label, "sfx") || ContainsWord(label, "audio")) kinds |= ModOptionKind.Sound;
        if (paths.Any(path => HasExtension(path, ".avfx", ".atex")) || ContainsWord(label, "vfx") || ContainsWord(label, "effect") || ContainsWord(label, "effects")) kinds |= ModOptionKind.Vfx;
        if (paths.Any(path => HasExtension(path, ".pap", ".tmb"))) kinds |= ModOptionKind.Animation;
        if (paths.Any(path => HasExtension(path, ".mdl", ".mtrl", ".tex", ".imc") || path.Contains("chara/equipment/", StringComparison.OrdinalIgnoreCase) || path.Contains("chara/accessory/", StringComparison.OrdinalIgnoreCase))) kinds |= ModOptionKind.Gear;
        if (localFiles.Any(path => path.Replace('\\', '/').Contains("RavaFit/", StringComparison.OrdinalIgnoreCase))
            || ($"{group.Name} {ReadString(group.Node, "Description")} {option.Name} {ReadString(option.Node, "Description")}").Contains("RavaFit", StringComparison.OrdinalIgnoreCase))
            kinds |= ModOptionKind.RavaFitGenerated;
        if (kinds == ModOptionKind.None) kinds = ModOptionKind.Other;
        return new ModOptionCleanupCandidate(group.StableKey, group.Name, group.Type, option.StableKey, option.Name, kinds, localFiles.Count);
    }

    private static void RemoveUnusedPageNames(JsonObject root, JsonArray groups)
    {
        var pageNames = PenumbraV4Document.FindProperty(root, "PageNames") as JsonObject;
        if (pageNames is null) return;
        var used = groups.OfType<JsonObject>()
            .Select(group => PenumbraV4Document.FindProperty(group, "Page"))
            .Select(ReadInt)
            .Where(page => page >= 0)
            .ToHashSet();
        foreach (var key in pageNames.Select(pair => pair.Key).ToArray())
            if (int.TryParse(key, out var page) && !used.Contains(page)) pageNames.Remove(key);
    }

    private static void AdjustDefaultSettings(JsonObject group, string groupType, IReadOnlyList<int> removedIndices, int oldOptionCount)
    {
        if (removedIndices.Count == 0) return;
        var node = PenumbraV4Document.FindProperty(group, "DefaultSettings");
        if (node is null) return;
        var defaultName = PenumbraV4Document.FindPropertyName(group, "DefaultSettings") ?? "DefaultSettings";
        if (string.Equals(groupType, "Multi", StringComparison.OrdinalIgnoreCase))
        {
            var value = ReadUInt64(node);
            foreach (var index in removedIndices.OrderByDescending(value => value))
            {
                var lower = index == 0 ? 0UL : value & ((1UL << index) - 1UL);
                var upper = index >= 63 ? 0UL : (value >> (index + 1)) << index;
                value = lower | upper;
            }
            group[defaultName] = JsonValue.Create(value);
            return;
        }

        var current = ReadInt(node);
        if (current < 0) current = 0;
        foreach (var index in removedIndices.OrderBy(value => value))
        {
            if (index < current) current--;
            else if (index == current) current = Math.Min(current, Math.Max(0, oldOptionCount - removedIndices.Count - 1));
        }
        group[defaultName] = JsonValue.Create(Math.Max(0, current));
    }

    private static bool ContainsAnySettingReference(JsonNode? node, IReadOnlySet<Guid> removed)
    {
        if (node is null || removed.Count == 0) return false;
        if (node is JsonObject obj)
        {
            foreach (var pair in obj)
            {
                if (string.Equals(pair.Key, "Setting", StringComparison.OrdinalIgnoreCase)
                    && pair.Value is JsonValue value
                    && value.TryGetValue<string>(out var text)
                    && Guid.TryParse(text, out var id)
                    && removed.Contains(id)) return true;
                if (ContainsAnySettingReference(pair.Value, removed)) return true;
            }
        }
        else if (node is JsonArray array)
        {
            foreach (var child in array)
                if (ContainsAnySettingReference(child, removed)) return true;
        }
        return false;
    }

    private static HashSet<string> CollectLocalFiles(JsonNode node)
    {
        var output = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        CollectLocalFiles(node, output);
        return output;
    }

    private static void CollectLocalFiles(JsonNode? node, HashSet<string> output)
    {
        if (node is JsonArray array)
        {
            foreach (var child in array) CollectLocalFiles(child, output);
            return;
        }
        if (node is not JsonObject obj) return;
        foreach (var pair in obj)
        {
            if (string.Equals(pair.Key, "Files", StringComparison.OrdinalIgnoreCase) && pair.Value is JsonObject files)
            {
                foreach (var mapping in files)
                {
                    if (mapping.Value is JsonValue value && value.TryGetValue<string>(out var relative) && !string.IsNullOrWhiteSpace(relative) && !Path.IsPathRooted(relative))
                        output.Add(relative.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar));
                }
            }
            CollectLocalFiles(pair.Value, output);
        }
    }

    private static List<string> CollectPathLikeStrings(JsonNode node)
    {
        var output = new List<string>();
        CollectPathLikeStrings(node, output);
        return output;
    }

    private static void CollectPathLikeStrings(JsonNode? node, List<string> output)
    {
        if (node is JsonValue value && value.TryGetValue<string>(out var text))
        {
            if (text.Contains('.') || text.Contains('/') || text.Contains('\\')) output.Add(text.Replace('\\', '/'));
            return;
        }
        if (node is JsonArray array)
        {
            foreach (var child in array) CollectPathLikeStrings(child, output);
            return;
        }
        if (node is not JsonObject obj) return;
        foreach (var pair in obj)
        {
            if (pair.Key.Contains('.') || pair.Key.Contains('/') || pair.Key.Contains('\\')) output.Add(pair.Key.Replace('\\', '/'));
            CollectPathLikeStrings(pair.Value, output);
        }
    }

    private static bool ContainsWord(string text, string word)
        => System.Text.RegularExpressions.Regex.IsMatch(text, $@"(?<![A-Za-z0-9]){System.Text.RegularExpressions.Regex.Escape(word)}(?![A-Za-z0-9])", System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.CultureInvariant);

    private static bool HasExtension(string path, params string[] extensions)
        => extensions.Any(extension => path.EndsWith(extension, StringComparison.OrdinalIgnoreCase));

    private static int ReadInt(JsonNode? node)
    {
        if (node is JsonValue value)
        {
            if (value.TryGetValue<int>(out var integer)) return integer;
            if (value.TryGetValue<long>(out var longer)) return checked((int)longer);
        }
        return 0;
    }

    private static ulong ReadUInt64(JsonNode? node)
    {
        if (node is JsonValue value)
        {
            if (value.TryGetValue<ulong>(out var unsigned)) return unsigned;
            if (value.TryGetValue<long>(out var signed) && signed >= 0) return (ulong)signed;
            if (value.TryGetValue<int>(out var integer) && integer >= 0) return (ulong)integer;
        }
        return 0;
    }

    private static string? ReadString(JsonObject obj, string name)
    {
        var node = PenumbraV4Document.FindProperty(obj, name);
        return node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
    }

    private static void SetProperty(JsonObject obj, string name, JsonNode? value)
    {
        var actual = PenumbraV4Document.FindPropertyName(obj, name) ?? name;
        obj[actual] = value;
    }

    private static string SafeResolve(string root, string relative)
    {
        var fullRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        var full = Path.GetFullPath(Path.Combine(fullRoot, relative));
        var rel = Path.GetRelativePath(fullRoot, full);
        if (rel == ".." || rel.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal) || Path.IsPathRooted(rel))
            throw new InvalidDataException($"Referenced file escaped the mod directory: {relative}");
        return full;
    }

    private static void DeleteEmptyParents(string? directory, string modRoot)
    {
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(modRoot));
        while (!string.IsNullOrWhiteSpace(directory) && Directory.Exists(directory))
        {
            var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory));
            if (string.Equals(full, root, StringComparison.OrdinalIgnoreCase) || !full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) break;
            if (Directory.EnumerateFileSystemEntries(full).Any()) break;
            Directory.Delete(full);
            directory = Path.GetDirectoryName(full);
        }
    }
}
