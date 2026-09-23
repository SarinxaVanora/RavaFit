using System.Text.Json.Nodes;
using Dalamud.Plugin.Services;
using RavaFit.Core;
using RavaFit.Core.Models;

namespace RavaFit.Services;

internal sealed record PreparedTargetBodyOptionGroup(string BodyName, string Slot, string SourceKey, JsonObject Group);
internal sealed record PreparedTargetBodyOptions(IReadOnlyList<PreparedTargetBodyOptionGroup> Groups, IReadOnlyList<string> CreatedFiles);
internal sealed record CustomBodyImportModel(string Slot, string RaceCode, string GamePath, string RelativePath, string PhysicalPath);
internal sealed record CustomBodyImportRequest(string BodyName, string VariantName, string SourceMod, string SourceGroup, string SourceOption, IReadOnlyList<CustomBodyImportModel> Models);
internal sealed record CustomBodyImportResult(string BodyName, string VariantName, int ImportedSlots, int ImportedPayloads, string LibraryPath);

internal sealed class BodyLibraryService
{
    private readonly IPluginLog _log;
    private readonly SolverHostService _solver;
    private readonly BodyCatalogueSet _catalogue = new();
    private readonly object _profileCacheGate = new();
    private readonly Dictionary<string, IReadOnlyDictionary<string, TargetBodyOptionProfileInfo>> _targetProfileArchiveCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IReadOnlyList<BodyVariantInfo>> _piercingVariantCache = new(StringComparer.OrdinalIgnoreCase);

    public BodyLibraryService(string directory, string userDirectory, SolverHostService solver, IPluginLog log)
    {
        Directory = directory;
        UserDirectory = userDirectory;
        _solver = solver;
        _log = log;
    }

    public string Directory { get; private set; }
    public string UserDirectory { get; }
    public string UserLibraryPath => Path.Combine(UserDirectory, "UserBodies.rbody");
    public IReadOnlyList<BodyLibraryInfo> Libraries => _catalogue.Libraries;
    public IReadOnlyList<BodyVariantInfo> Variants => _catalogue.Variants;
    public IReadOnlyList<BodyVariantInfo> UserVariants => Variants.Where(v => IsWithinDirectory(v.RBodyPath, UserDirectory)).ToArray();
    public IReadOnlyList<string> Errors => _catalogue.Errors;
    public bool Ready => Libraries.Count > 0;
    public string CatalogueStamp => string.Join(";", Libraries.Select(l =>
    {
        var info = new FileInfo(l.Path);
        return $"{Path.GetFullPath(l.Path)}:{info.Length}:{info.LastWriteTimeUtc.Ticks}";
    }));

    public void SetDirectory(string directory)
    {
        Directory = directory;
        Refresh();
    }

    public void Refresh()
    {
        try
        {
            _catalogue.Scan(Directory, UserDirectory);
            lock (_profileCacheGate)
            {
                _targetProfileArchiveCache.Clear();
                _piercingVariantCache.Clear();
            }
        }
        catch (Exception ex)
        {
            _log.Error(ex, "Failed to scan RBODY libraries in {Directory} and custom body directory {UserDirectory}", Directory, UserDirectory);
        }
    }

    public IReadOnlyList<BodyVariantInfo> ForSlot(string slot, string? raceCode = null) => _catalogue.ForSlot(slot, raceCode);


    public BodyVariantInfo? ResolveVariant(string rbodyPath, string bodyId, string slot, string variantId)
    {
        var fullPath = Path.GetFullPath(rbodyPath);
        return Variants.FirstOrDefault(v => string.Equals(Path.GetFullPath(v.RBodyPath), fullPath, StringComparison.OrdinalIgnoreCase)
            && string.Equals(v.BodyId, bodyId, StringComparison.OrdinalIgnoreCase)
            && string.Equals(v.Slot, slot, StringComparison.OrdinalIgnoreCase)
            && string.Equals(v.VariantId, variantId, StringComparison.OrdinalIgnoreCase));
    }

    public BodyVariantInfo? FindSibling(BodyVariantInfo source, string slot, string? raceCode = null)
    {
        var candidates = ForSlot(slot, raceCode)
            .Where(v => string.Equals(v.Collection, source.Collection, StringComparison.OrdinalIgnoreCase)
                     && string.Equals(v.BodyId, source.BodyId, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        if (candidates.Length == 0)
            return null;
        if (candidates.Length == 1)
            return candidates[0];

        var byName = candidates.Where(v => string.Equals(v.VariantName, source.VariantName, StringComparison.OrdinalIgnoreCase)).ToArray();
        if (byName.Length == 1)
            return byName[0];

        var sourceSuffix = VariantSuffix(source.VariantId);
        if (!string.IsNullOrWhiteSpace(sourceSuffix))
        {
            var bySuffix = candidates.Where(v => string.Equals(VariantSuffix(v.VariantId), sourceSuffix, StringComparison.OrdinalIgnoreCase)).ToArray();
            if (bySuffix.Length == 1)
                return bySuffix[0];
        }

        return null;
    }

    public async Task<CustomBodyImportResult> ImportCustomBodyAsync(CustomBodyImportRequest request, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(request.BodyName))
            throw new ArgumentException("Body name is required.", nameof(request));
        if (string.IsNullOrWhiteSpace(request.VariantName))
            throw new ArgumentException("Variant name is required.", nameof(request));
        if (request.Models.Count == 0)
            throw new InvalidOperationException("The selected Penumbra option does not contain any body models RavaFit can import.");

        System.IO.Directory.CreateDirectory(UserDirectory);
        using var reply = await _solver.CallAsync("import_body", new
        {
            output = UserLibraryPath,
            collection = "Custom",
            body_name = request.BodyName.Trim(),
            variant_name = request.VariantName.Trim(),
            source_mod = request.SourceMod,
            source_group = request.SourceGroup,
            source_option = request.SourceOption,
            models = request.Models.Select(model => new
            {
                slot = model.Slot,
                race_code = model.RaceCode,
                game_path = model.GamePath,
                relative_path = model.RelativePath,
                physical_path = model.PhysicalPath,
            }).ToArray(),
        }, cancellationToken).ConfigureAwait(false);

        if (!reply.RootElement.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
            throw new InvalidOperationException("RavaFit could not add the body to the custom catalogue.");

        var bodyName = reply.RootElement.TryGetProperty("body_name", out var body) ? body.GetString() ?? request.BodyName.Trim() : request.BodyName.Trim();
        var variantName = reply.RootElement.TryGetProperty("variant_name", out var variant) ? variant.GetString() ?? request.VariantName.Trim() : request.VariantName.Trim();
        var importedSlots = reply.RootElement.TryGetProperty("imported_slots", out var slots) && slots.TryGetInt32(out var slotCount) ? slotCount : request.Models.Select(m => m.Slot).Distinct(StringComparer.OrdinalIgnoreCase).Count();
        var importedPayloads = reply.RootElement.TryGetProperty("imported_payloads", out var payloads) && payloads.TryGetInt32(out var payloadCount) ? payloadCount : request.Models.Count;
        Refresh();
        return new CustomBodyImportResult(bodyName, variantName, importedSlots, importedPayloads, UserLibraryPath);
    }

    public bool HasPiercingOptions(BodyVariantInfo variant)
    {
        var profile = ReadTargetProfileCached(variant);
        return profile is not null && profile.Groups.Any(IsPiercingProfileGroup) && variant.PiercingRaceCodes is { Count: > 0 };
    }

    public IReadOnlyList<BodyVariantInfo> PiercingVariants(string slot, string? raceCode = null)
    {
        var key = $"{slot}|{raceCode ?? "*"}";
        lock (_profileCacheGate)
            if (_piercingVariantCache.TryGetValue(key, out var cached))
                return cached;

        var result = ForSlot(slot).Where(HasPiercingOptions)
            .Where(variant => string.IsNullOrWhiteSpace(raceCode) || CharacterRaceCatalog.BodySupports(variant, raceCode))
            .Where(variant => string.IsNullOrWhiteSpace(raceCode) || HasPiercingGeometryForRace(variant, raceCode))
            .ToArray();
        lock (_profileCacheGate)
            _piercingVariantCache[key] = result;
        return result;
    }

    private TargetBodyOptionProfileInfo? ReadTargetProfileCached(BodyVariantInfo variant)
    {
        if (string.IsNullOrWhiteSpace(variant.TargetOptionProfileId)) return null;
        var archivePath = Path.GetFullPath(variant.RBodyPath);
        IReadOnlyDictionary<string, TargetBodyOptionProfileInfo>? profiles;
        lock (_profileCacheGate)
            _targetProfileArchiveCache.TryGetValue(archivePath, out profiles);

        if (profiles is null)
        {
            var loaded = TargetBodyOptionReader.ReadProfiles(archivePath);
            lock (_profileCacheGate)
            {
                if (!_targetProfileArchiveCache.TryGetValue(archivePath, out profiles))
                {
                    _targetProfileArchiveCache[archivePath] = loaded;
                    profiles = loaded;
                }
            }
        }
        return profiles.TryGetValue(variant.TargetOptionProfileId, out var profile) ? profile : null;
    }

    private static bool HasPiercingGeometryForRace(BodyVariantInfo variant, string raceCode)
    {
        var payloadRace = CharacterRaceCatalog.ResolveBodyPayloadRace(variant, raceCode);
        return !string.IsNullOrWhiteSpace(payloadRace)
            && variant.PiercingRaceCodes is not null
            && variant.PiercingRaceCodes.Contains(payloadRace, StringComparer.OrdinalIgnoreCase);
    }

    public async Task<PreparedTargetBodyOptions> PreparePiercingOptionsAsync(BodyVariantInfo variant, string slot, string? raceCode, string modRoot, CancellationToken cancellationToken = default)
        => await PrepareTargetBodyOptionsCoreAsync(variant, slot, raceCode, modRoot, piercingOnly: true, cancellationToken).ConfigureAwait(false);

    public async Task<PreparedTargetBodyOptions> PrepareTargetBodyOptionsAsync(BodyVariantInfo variant, string slot, string? raceCode, string modRoot, CancellationToken cancellationToken = default)
        => await PrepareTargetBodyOptionsCoreAsync(variant, slot, raceCode, modRoot, piercingOnly: false, cancellationToken).ConfigureAwait(false);

    private async Task<PreparedTargetBodyOptions> PrepareTargetBodyOptionsCoreAsync(BodyVariantInfo variant, string slot, string? raceCode, string modRoot, bool piercingOnly, CancellationToken cancellationToken)
    {
        var profile = ReadTargetProfileCached(variant);
        if (profile is null)
            return new PreparedTargetBodyOptions([], []);

        var createdFiles = new List<string>();
        var prepared = new List<PreparedTargetBodyOptionGroup>();
        var root = Path.GetFullPath(modRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        try
        {
            foreach (var source in profile.Groups)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (piercingOnly && !IsPiercingProfileGroup(source))
                    continue;
                if (source.Slots.Count > 0 && !source.Slots.Contains(slot, StringComparer.OrdinalIgnoreCase))
                    continue;
                if (source.RaceCodes.Count > 0 && (string.IsNullOrWhiteSpace(raceCode) || !source.RaceCodes.Contains(raceCode, StringComparer.OrdinalIgnoreCase)))
                    continue;

                var group = source.Group.DeepClone().AsObject();
                if (FindProperty(group, "Options") is JsonArray options)
                {
                    foreach (var option in options.OfType<JsonObject>())
                    {
                        if (FindProperty(option, "Files") is not JsonObject files)
                            continue;
                        foreach (var gamePath in files.Select(x => x.Key).ToArray())
                        {
                            if (files[gamePath] is not JsonValue value || !value.TryGetValue<string>(out var originalRelative) || string.IsNullOrWhiteSpace(originalRelative))
                                continue;
                            var sourceRelative = NormalizeRelative(originalRelative);
                            if (!source.FileAssets.TryGetValue(sourceRelative, out var assetId))
                                throw new InvalidDataException($"RBODY target-body group '{ReadString(group, "Name") ?? source.SourceGroupFile}' references '{sourceRelative}', but the asset was not captured during re-processing.");

                            var relative = Path.Combine("RavaFit", "TargetBodyOptions", SafeSegment(profile.Id), assetId).Replace('\\', '/');
                            var absolute = Path.GetFullPath(Path.Combine(modRoot, relative.Replace('/', Path.DirectorySeparatorChar)));
                            if (!absolute.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                                throw new InvalidDataException("Target-body support asset escaped the Penumbra mod directory.");
                            var existed = File.Exists(absolute);
                            await TargetBodyOptionReader.ExtractAssetAsync(profile.RBodyPath, assetId, absolute, cancellationToken).ConfigureAwait(false);
                            if (!existed)
                                createdFiles.Add(absolute);
                            files[gamePath] = relative;
                        }
                    }
                }

                var sourceKey = $"{profile.Id}:{source.SourcePackage}:{source.SourceGroupFile}";
                prepared.Add(new PreparedTargetBodyOptionGroup(profile.BodyName, slot, sourceKey, group));
            }
            return new PreparedTargetBodyOptions(prepared, createdFiles.Distinct(StringComparer.OrdinalIgnoreCase).ToArray());
        }
        catch
        {
            foreach (var file in createdFiles.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                try { if (File.Exists(file)) File.Delete(file); }
                catch (Exception ex) { _log.Warning(ex, "RavaFit could not remove target-body support asset after preparation failed: {Path}", file); }
            }
            throw;
        }
    }

    private static bool IsPiercingProfileGroup(TargetBodyOptionGroupInfo group)
    {
        var text = $"{group.SourcePackage} {group.SourceGroupFile} {ReadString(group.Group, "Name")} {group.Group.ToJsonString()}".ToLowerInvariant();
        return text.Contains("pierc", StringComparison.Ordinal)
            || text.Contains("jewel", StringComparison.Ordinal)
            || text.Contains("dermal", StringComparison.Ordinal)
            || text.Contains("barbell", StringComparison.Ordinal)
            || text.Contains("bellyring", StringComparison.Ordinal)
            || text.Contains("nipple", StringComparison.Ordinal);
    }

    private static JsonNode? FindProperty(JsonObject obj, string name)
    {
        foreach (var pair in obj)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase))
                return pair.Value;
        return null;
    }

    private static string? ReadString(JsonObject obj, string name)
        => FindProperty(obj, name) is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;

    private static string NormalizeRelative(string value)
    {
        var normalized = value.Trim().Replace('\\', '/').TrimStart('/');
        if (normalized.Contains(':', StringComparison.Ordinal) || normalized.Split('/').Any(x => x == ".."))
            throw new InvalidDataException($"Invalid target-body support asset path '{value}'.");
        return normalized;
    }

    private static string SafeSegment(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var safe = new string(value.Trim().Select(c => invalid.Contains(c) ? '_' : c).ToArray()).Trim().Trim('.');
        return string.IsNullOrWhiteSpace(safe) ? "Body" : safe;
    }

    private static string VariantSuffix(string variantId)
    {
        var parts = variantId.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length >= 3 ? string.Join(".", parts.Skip(2)) : variantId;
    }

    private static bool IsWithinDirectory(string path, string directory)
    {
        try
        {
            var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory)) + Path.DirectorySeparatorChar;
            return Path.GetFullPath(path).StartsWith(root, StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }
}
