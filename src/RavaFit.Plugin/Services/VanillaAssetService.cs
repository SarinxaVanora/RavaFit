using System.Collections;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Dalamud.Plugin.Services;
using Dalamud.Utility;
using Lumina.Excel.Sheets;
using RavaFit.Core.Models;
using RavaFit.Services.Animation;

namespace RavaFit.Services;

internal sealed record VanillaOutfitPiece(uint ItemId, string Name, string Slot);

internal sealed record VanillaAccessoryAsset(uint ItemId, ushort ModelSetId, ushort VariantId, string Name, string Slot)
{
    public string DisplayName => $"{Slot} · {Name}";
    public string SearchText => $"{Name} {Slot} a{ModelSetId:D4} variant {VariantId}";
}


internal sealed record VanillaOutfitAsset(ushort ModelSetId, string Name, IReadOnlyList<VanillaOutfitPiece> Pieces)
{
    public string DisplayName => $"{Name} · {Pieces.Count} piece{(Pieces.Count == 1 ? string.Empty : "s")}";
    public string SearchText => string.Join(" ", Pieces.Select(piece => $"{piece.Name} {piece.Slot}"));
}

internal sealed record VanillaAnimationVariant(string SourceRaceCode, IReadOnlyList<string> GamePaths)
{
    public CharacterRaceIdentity? SourceRace => CharacterRaceCatalog.FromCode(SourceRaceCode);
    public string PrimaryGamePath => GamePaths.FirstOrDefault() ?? string.Empty;
}

internal sealed record VanillaAnimationAsset(string Id, string Name, string Category, IReadOnlyList<VanillaAnimationVariant> Variants, string SearchText)
{
    public string DisplayName => $"{Category} · {Name}";
}

internal sealed class VanillaAssetService
{
    private static readonly string[] JobAnimationRoots =
    [
        "bt_2ax_emp", "bt_swd_sld", "bt_2gb_emp", "bt_2sw_emp", "bt_2gl_emp", "bt_2ff_emp",
        "bt_2bk_emp", "bt_stf_sld", "bt_2gn_emp", "bt_chk_chk", "bt_2bw_emp", "bt_2kt_emp",
        "bt_2sp_emp", "bt_clw_clw", "bt_dgr_dgr", "bt_2km_emp", "bt_2rp_emp", "bt_jst_sld",
        "bt_rod_emp", "bt_brs_plt", "bt_bld_bld",
    ];

    private readonly IDataManager _dataManager;
    private readonly PenumbraService _penumbra;
    private readonly object _catalogueGate = new();
    private IReadOnlyList<VanillaOutfitAsset>? _outfits;
    private IReadOnlyList<VanillaAccessoryAsset>? _accessories;
    private IReadOnlyList<VanillaAnimationAsset>? _animations;
    public string AnimationCatalogueStatus { get; private set; } = "Not scanned yet.";

    public VanillaAssetService(IDataManager dataManager, PenumbraService penumbra)
    {
        _dataManager = dataManager;
        _penumbra = penumbra;
    }

    private sealed class VanillaAnimationFamilyBuilder
    {
        private readonly Dictionary<string, HashSet<string>> _pathsByRace = new(StringComparer.OrdinalIgnoreCase);

        public VanillaAnimationFamilyBuilder(string id, string name, string category)
        {
            Id = id;
            Name = name;
            Category = category;
        }

        public string Id { get; }
        public string Name { get; }
        public string Category { get; }
        public string SearchText { get; set; } = string.Empty;
        public bool HasPaths => _pathsByRace.Values.Any(paths => paths.Count > 0);

        public void Add(string sourceRaceCode, string gamePath)
        {
            if (!_pathsByRace.TryGetValue(sourceRaceCode, out var paths))
            {
                paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                _pathsByRace[sourceRaceCode] = paths;
            }
            paths.Add(CollapseGamePath(gamePath));
        }

        public VanillaAnimationAsset Build()
        {
            var variants = _pathsByRace
                .Where(pair => CharacterRaceCatalog.FromCode(pair.Key) is not null && pair.Value.Count > 0)
                .OrderBy(pair => CharacterRaceOrder(pair.Key))
                .Select(pair => new VanillaAnimationVariant(
                    pair.Key,
                    pair.Value.OrderBy(AnimationPhaseOrder).ThenBy(path => path, StringComparer.OrdinalIgnoreCase).ToArray()))
                .ToArray();
            return new VanillaAnimationAsset(Id, Name, Category, variants, SearchText);
        }
    }

    public IReadOnlyList<VanillaOutfitAsset> GetOutfitAssets()
    {
        lock (_catalogueGate)
        {
            if (_outfits is not null) return _outfits;
            var rows = new List<(uint ItemId, string Name, string Slot, ushort ModelSetId)>();
            foreach (var item in _dataManager.GetExcelSheet<Item>())
            {
                try
                {
                    var name = item.Name.ExtractText().Trim();
                    if (string.IsNullOrWhiteSpace(name)) continue;
                    var setId = ReadModelSetId(GetMemberValue(item, "ModelMain"));
                    if (setId == 0) continue;
                    var slotCategory = ResolveRowReference(GetMemberValue(item, "EquipSlotCategory"));
                    var slot = ResolveEquipmentSlot(slotCategory);
                    if (slot is null) continue;
                    rows.Add((item.RowId, name, slot, setId));
                }
                catch { /* One malformed/non-equipment row must not poison the human-readable catalogue. */ }
            }

            _outfits = rows.GroupBy(row => row.ModelSetId)
                .Select(group =>
                {
                    var pieces = group.GroupBy(row => row.Slot, StringComparer.OrdinalIgnoreCase)
                        .Select(slotGroup => slotGroup.OrderBy(row => row.Name, StringComparer.CurrentCultureIgnoreCase).First())
                        .OrderBy(row => Array.IndexOf(BodySlots.All.ToArray(), row.Slot))
                        .Select(row => new VanillaOutfitPiece(row.ItemId, row.Name, row.Slot))
                        .ToArray();
                    var name = pieces.FirstOrDefault(piece => string.Equals(piece.Slot, BodySlots.Chest, StringComparison.OrdinalIgnoreCase))?.Name
                        ?? pieces.FirstOrDefault()?.Name
                        ?? $"Equipment set {group.Key:D4}";
                    return new VanillaOutfitAsset(group.Key, name, pieces);
                })
                .OrderBy(asset => asset.Name, StringComparer.CurrentCultureIgnoreCase)
                .ThenBy(asset => asset.ModelSetId)
                .ToArray();
            return _outfits;
        }
    }

    public IReadOnlyList<VanillaAccessoryAsset> GetAccessoryAssets()
    {
        lock (_catalogueGate)
        {
            if (_accessories is not null) return _accessories;
            var rows = new List<VanillaAccessoryAsset>();
            foreach (var item in _dataManager.GetExcelSheet<Item>())
            {
                try
                {
                    var name = item.Name.ExtractText().Trim();
                    if (string.IsNullOrWhiteSpace(name)) continue;
                    var modelMain = GetMemberValue(item, "ModelMain");
                    var setId = ReadModelSetId(modelMain);
                    if (setId == 0) continue;
                    var variantId = ReadModelVariantId(modelMain);
                    var slotCategory = ResolveRowReference(GetMemberValue(item, "EquipSlotCategory"));
                    foreach (var slot in ResolveAccessorySlots(slotCategory))
                        rows.Add(new VanillaAccessoryAsset(item.RowId, setId, variantId, name, slot));
                }
                catch { }
            }
            _accessories = rows
                .GroupBy(row => $"{row.Slot}|{row.ModelSetId}|{row.VariantId}", StringComparer.OrdinalIgnoreCase)
                .Select(group => group.OrderBy(row => row.Name, StringComparer.CurrentCultureIgnoreCase).First())
                .OrderBy(row => Array.IndexOf(AccessoryModelSlots.All, row.Slot))
                .ThenBy(row => row.Name, StringComparer.CurrentCultureIgnoreCase)
                .ThenBy(row => row.ModelSetId)
                .ToArray();
            return _accessories;
        }
    }

    public string ResolveAccessoryModelPath(VanillaAccessoryAsset asset, string sourceRaceCode)
    {
        var race = CharacterRaceCatalog.FromCode(sourceRaceCode) ?? throw new InvalidDataException($"Unknown player race c{sourceRaceCode}.");
        var suffix = AccessoryModelSlots.Suffix(asset.Slot);
        foreach (var candidateRace in new[] { race.Code, race.MidlanderFallbackCode }.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var path = $"chara/accessory/a{asset.ModelSetId:D4}/model/c{candidateRace}a{asset.ModelSetId:D4}_{suffix}.mdl";
            if (_dataManager.FileExists(path)) return path;
        }
        throw new FileNotFoundException($"XIV game data does not contain a {race.DisplayName} model for {asset.Name} ({asset.Slot}, a{asset.ModelSetId:D4}).");
    }

    public byte? ResolveAccessoryMaterialId(VanillaAccessoryAsset asset)
    {
        var resource = _dataManager.GetFile($"chara/accessory/a{asset.ModelSetId:D4}/a{asset.ModelSetId:D4}.imc");
        if (resource?.Data is not { Length: >= 4 } data) return null;
        var count = BitConverter.ToUInt16(data, 0);
        var partMask = BitConverter.ToUInt16(data, 2);
        var partBit = AccessoryImcPartBit(asset.Slot);
        if (partBit < 0 || (partMask & (1 << partBit)) == 0) return null;

        var partCount = 0;
        var compactPartIndex = 0;
        for (var bit = 0; bit < 8; bit++)
        {
            if ((partMask & (1 << bit)) == 0) continue;
            if (bit < partBit) compactPartIndex++;
            partCount++;
        }
        if (partCount == 0) return null;

        var entrySize = 6;
        long offset = asset.VariantId == 0
            ? 4L + compactPartIndex * entrySize
            : asset.VariantId <= count
                ? 4L + partCount * entrySize + ((long)(asset.VariantId - 1) * partCount + compactPartIndex) * entrySize
                : -1;
        return offset >= 0 && offset < data.Length ? data[(int)offset] : null;
    }

    public IReadOnlyList<VanillaAnimationAsset> GetAnimationAssets()
    {
        lock (_catalogueGate)
        {
            if (_animations is not null) return _animations;

            var families = new Dictionary<string, VanillaAnimationFamilyBuilder>(StringComparer.OrdinalIgnoreCase);
            var emotePapFamilies = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var scannedEmotes = 0;
            var resolvedEmotes = 0;
            var manifestCandidates = 0;
            var manifestAccepted = 0;

            foreach (var emote in _dataManager.GetExcelSheet<Emote>())
            {
                string name;
                try { name = emote.Name.ExtractText().Trim(); }
                catch { continue; }
                if (string.IsNullOrWhiteSpace(name)) continue;
                scannedEmotes++;

                var builder = new VanillaAnimationFamilyBuilder($"emote:{emote.RowId}", name, "Emote");
                var searchParts = new List<string> { name };
                var command = TryGetEmoteCommand(emote);
                if (!string.IsNullOrWhiteSpace(command)) searchParts.Add(command);

                foreach (var timeline in EnumerateEmoteActionTimelines(emote))
                {
                    var key = ReadableText(GetMemberValue(timeline, "Key"));
                    if (string.IsNullOrWhiteSpace(key)) continue;
                    searchParts.Add(key);
                    if (!TryResolveEmotePapRelatives(timeline, key, out var relatives)) continue;

                    foreach (var relative in relatives)
                    {
                        foreach (var path in ResolveHumanPapCandidatePaths(relative))
                        {
                            var sourceRaceCode = ExtractRaceCode(path);
                            if (sourceRaceCode is null || CharacterRaceCatalog.FromCode(sourceRaceCode) is null) continue;
                            foreach (var sibling in ExpandAnimationPhaseFamily(path))
                            {
                                builder.Add(sourceRaceCode, sibling);
                                emotePapFamilies.Add(GetAnimationPhaseFamilyKey(sibling));
                            }
                        }
                    }
                }

                if (!builder.HasPaths) continue;
                builder.SearchText = string.Join(' ', searchParts.Distinct(StringComparer.OrdinalIgnoreCase));
                families[builder.Id] = builder;
                resolvedEmotes++;
            }

            SeedVfxEditorCharacterPapFamilies(families);

            // Shamelessly taken from VFXEditor's MIT-licensed common_pap list; see VFXEDITOR_LICENSE.txt.
            foreach (var rawPath in EnumerateBundledVfxEditorCommonPapPaths())
            {
                manifestCandidates++;
                var path = CollapseGamePath(rawPath);
                if (!path.Contains("/animation/a0001/", StringComparison.OrdinalIgnoreCase)) continue;
                if (!XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(path) || !_dataManager.FileExists(path)) continue;
                var sourceRaceCode = ExtractRaceCode(path);
                if (sourceRaceCode is null || CharacterRaceCatalog.FromCode(sourceRaceCode) is null) continue;
                if (emotePapFamilies.Contains(GetAnimationPhaseFamilyKey(path))) continue;

                var marker = path.IndexOf("/animation/a0001/", StringComparison.OrdinalIgnoreCase);
                if (marker < 0) continue;
                var relative = path[(marker + "/animation/a0001/".Length)..];
                if (!TryClassifyNonCombatCommonPap(relative, out var familyKey, out var name, out var category)) continue;

                var id = $"common:{familyKey}";
                if (!families.TryGetValue(id, out var builder))
                {
                    builder = new VanillaAnimationFamilyBuilder(id, name, category)
                    {
                        SearchText = $"{name} {category} {relative.Replace('/', ' ')}",
                    };
                    families[id] = builder;
                }

                foreach (var sibling in ExpandAnimationPhaseFamily(path))
                    builder.Add(sourceRaceCode, sibling);
                manifestAccepted++;
            }

            _animations = families.Values
                .Where(builder => builder.HasPaths)
                .Select(builder => builder.Build())
                .OrderBy(asset => AnimationCategoryOrder(asset.Category))
                .ThenBy(asset => asset.Name, StringComparer.CurrentCultureIgnoreCase)
                .ThenBy(asset => asset.Id, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            AnimationCatalogueStatus = $"Built {_animations.Count:N0} emote/idle/pose/movement families: {resolvedEmotes:N0}/{scannedEmotes:N0} named XIV emotes, resident Character PAP families, plus {manifestAccepted:N0} verified common PAP variants from {manifestCandidates:N0} manifest candidates. Combat, interaction, activity, crafting, gathering and performance families are excluded.";
            return _animations;
        }
    }

    public async Task<PenumbraModInfo> CreateModelSourceAsync(VanillaOutfitAsset asset, string sourceRaceCode, CancellationToken cancellationToken = default)
    {
        var race = CharacterRaceCatalog.FromCode(sourceRaceCode) ?? throw new InvalidDataException($"Unknown player race c{sourceRaceCode}.");
        if (string.IsNullOrWhiteSpace(_penumbra.ModDirectoryRoot) || !Directory.Exists(_penumbra.ModDirectoryRoot)) throw new DirectoryNotFoundException("Penumbra did not provide its mod directory.");

        var models = new List<(string Slot, string GamePath, byte[] Data)>();
        foreach (var slot in BodySlots.All)
        {
            if (!TryResolveOutfitModelPath(asset.ModelSetId, slot, race, out var gamePath)) continue;
            var resource = _dataManager.GetFile(gamePath);
            if (resource is null) continue;
            models.Add((slot, gamePath, resource.Data));
        }
        if (models.Count == 0)
            throw new FileNotFoundException($"XIV game data does not contain any {race.DisplayName} models for {asset.Name} (e{asset.ModelSetId:D4}).");

        var sourceName = $"RavaFit Vanilla - {SanitiseDisplayName(asset.Name)}";
        var folderName = MakeUniqueDirectoryName(sourceName);
        var root = Path.Combine(_penumbra.ModDirectoryRoot, folderName);
        Directory.CreateDirectory(root);
        try
        {
            var files = new JsonObject();
            foreach (var model in models)
            {
                var file = Path.GetFileName(model.GamePath.Replace('/', Path.DirectorySeparatorChar));
                var relative = Path.Combine("Vanilla", model.Slot, file).Replace('\\', '/');
                var physical = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
                Directory.CreateDirectory(Path.GetDirectoryName(physical)!);
                await File.WriteAllBytesAsync(physical, model.Data, cancellationToken).ConfigureAwait(false);
                files[model.GamePath] = relative;
            }

            var groupId = Guid.NewGuid();
            var optionId = Guid.NewGuid();
            var meta = new JsonObject
            {
                ["FileVersion"] = 4, ["Name"] = sourceName, ["Author"] = "RavaFit", ["Version"] = "1.1.0", ["Website"] = string.Empty,
                ["Description"] = $"Vanilla XIV outfit staged by RavaFit from equipment set e{asset.ModelSetId:D4} for {race.DisplayName}.", ["Tags"] = new JsonArray(), ["LastWrite"] = DateTimeOffset.UtcNow.ToString("O"),
                ["DefaultData"] = EmptyOptionData(), ["PageNames"] = new JsonObject { ["0"] = "Source" },
                ["Groups"] = new JsonArray
                {
                    new JsonObject
                    {
                        ["Type"] = "Single", ["Id"] = groupId.ToString("D"), ["Name"] = "Vanilla Outfit", ["Description"] = string.Empty, ["Page"] = 0, ["Priority"] = 0, ["DefaultSettings"] = 0,
                        ["Options"] = new JsonArray
                        {
                            new JsonObject { ["Id"] = optionId.ToString("D"), ["Name"] = "Original", ["Description"] = string.Empty, ["Priority"] = 0, ["Files"] = files, ["FileSwaps"] = new JsonObject(), ["Manipulations"] = new JsonArray() },
                        },
                    },
                },
            };
            await File.WriteAllTextAsync(Path.Combine(root, "meta.json"), meta.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
            if (!_penumbra.AddMod(folderName, out var addError))
                throw new InvalidOperationException($"RavaFit staged the vanilla outfit safely, but Penumbra could not register the new mod: {addError}");
            _penumbra.Refresh();
            return _penumbra.Mods.FirstOrDefault(mod => string.Equals(Path.GetFullPath(mod.ModRoot), Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase)) ?? new PenumbraModInfo(folderName, sourceName, root);
        }
        catch
        {
            try { Directory.Delete(root, true); } catch { }
            throw;
        }
    }

    public string ResolveOutfitModelPath(VanillaOutfitAsset asset, string sourceRaceCode, string slot)
    {
        var race = CharacterRaceCatalog.FromCode(sourceRaceCode) ?? throw new InvalidDataException($"Unknown player race c{sourceRaceCode}.");
        return TryResolveOutfitModelPath(asset.ModelSetId, slot, race, out var path)
            ? path
            : throw new FileNotFoundException($"XIV game data does not contain a {race.DisplayName} model for {asset.Name} ({slot}).");
    }

    private bool TryResolveOutfitModelPath(ushort modelSetId, string slot, CharacterRaceIdentity race, out string path)
    {
        var modelRace = string.Equals(race.Code, "1201", StringComparison.OrdinalIgnoreCase) && string.Equals(slot, BodySlots.Feet, StringComparison.OrdinalIgnoreCase) ? "1101" : race.Code;
        var suffix = slot switch
        {
            BodySlots.Chest => "top",
            BodySlots.Legs => "dwn",
            BodySlots.Hands => "glv",
            BodySlots.Feet => "sho",
            _ => string.Empty,
        };
        if (suffix.Length == 0) { path = string.Empty; return false; }
        path = $"chara/equipment/e{modelSetId:D4}/model/c{modelRace}e{modelSetId:D4}_{suffix}.mdl";
        if (_dataManager.FileExists(path)) return true;
        if (!string.Equals(race.Race, "Lalafell", StringComparison.OrdinalIgnoreCase))
        {
            var fallback = $"chara/equipment/e{modelSetId:D4}/model/c{race.MidlanderFallbackCode}e{modelSetId:D4}_{suffix}.mdl";
            if (_dataManager.FileExists(fallback)) { path = fallback; return true; }
        }
        path = string.Empty;
        return false;
    }

    public async Task<PenumbraModInfo> CreateModelSourceAsync(string gamePathRaw, CancellationToken cancellationToken = default)
        => await CreateModelSourceAsync(gamePathRaw, null, cancellationToken).ConfigureAwait(false);

    private async Task<PenumbraModInfo> CreateModelSourceAsync(string gamePathRaw, string? displayName, CancellationToken cancellationToken)
    {
        var gamePath = NormalizeGamePath(gamePathRaw);
        if (!gamePath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Select an XIV equipment model.");
        var file = Path.GetFileName(gamePath.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(file)) throw new InvalidDataException("The vanilla model path has no file name.");
        var resource = _dataManager.GetFile(gamePath) ?? throw new FileNotFoundException($"XIV game data does not contain '{gamePath}'.");
        if (string.IsNullOrWhiteSpace(_penumbra.ModDirectoryRoot) || !Directory.Exists(_penumbra.ModDirectoryRoot)) throw new DirectoryNotFoundException("Penumbra did not provide its mod directory.");

        var sourceName = $"RavaFit Vanilla - {SanitiseDisplayName(displayName ?? Path.GetFileNameWithoutExtension(file))}";
        var folderName = MakeUniqueDirectoryName(sourceName);
        var root = Path.Combine(_penumbra.ModDirectoryRoot, folderName);
        Directory.CreateDirectory(root);
        try
        {
            var relative = Path.Combine("Vanilla", file).Replace('\\', '/');
            var physical = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(physical)!);
            await File.WriteAllBytesAsync(physical, resource.Data, cancellationToken).ConfigureAwait(false);
            var groupId = Guid.NewGuid();
            var optionId = Guid.NewGuid();
            var meta = new JsonObject
            {
                ["FileVersion"] = 4, ["Name"] = sourceName, ["Author"] = "RavaFit", ["Version"] = "1.1.0", ["Website"] = string.Empty,
                ["Description"] = $"Vanilla XIV model staged by RavaFit from {gamePath}", ["Tags"] = new JsonArray(), ["LastWrite"] = DateTimeOffset.UtcNow.ToString("O"),
                ["DefaultData"] = EmptyOptionData(), ["PageNames"] = new JsonObject { ["0"] = "Source" },
                ["Groups"] = new JsonArray
                {
                    new JsonObject
                    {
                        ["Type"] = "Single", ["Id"] = groupId.ToString("D"), ["Name"] = "Vanilla", ["Description"] = string.Empty, ["Page"] = 0, ["Priority"] = 0, ["DefaultSettings"] = 0,
                        ["Options"] = new JsonArray
                        {
                            new JsonObject { ["Id"] = optionId.ToString("D"), ["Name"] = "Original", ["Description"] = string.Empty, ["Priority"] = 0, ["Files"] = new JsonObject { [gamePath] = relative }, ["FileSwaps"] = new JsonObject(), ["Manipulations"] = new JsonArray() },
                        },
                    },
                },
            };
            await File.WriteAllTextAsync(Path.Combine(root, "meta.json"), meta.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
            if (!_penumbra.AddMod(folderName, out var addError))
                throw new InvalidOperationException($"RavaFit staged the vanilla outfit safely, but Penumbra could not register the new mod: {addError}");
            _penumbra.Refresh();
            return _penumbra.Mods.FirstOrDefault(mod => string.Equals(Path.GetFullPath(mod.ModRoot), Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase)) ?? new PenumbraModInfo(folderName, sourceName, root);
        }
        catch
        {
            try { Directory.Delete(root, true); } catch { }
            throw;
        }
    }

    private static int CharacterRaceOrder(string code)
    {
        var index = 0;
        foreach (var race in CharacterRaceCatalog.All)
        {
            if (string.Equals(race.Code, code, StringComparison.OrdinalIgnoreCase)) return index;
            index++;
        }
        return int.MaxValue;
    }

    private static string TryGetEmoteCommand(Emote emote)
    {
        try
        {
            var textCommand = ResolveRowReference(GetMemberValue(emote, "TextCommand"));
            return ReadableText(GetMemberValue(textCommand, "Command"));
        }
        catch
        {
            return string.Empty;
        }
    }

    private static IEnumerable<object> EnumerateEmoteActionTimelines(Emote emote)
    {
        var value = GetMemberValue(emote, "ActionTimeline");
        if (value is not IEnumerable enumerable || value is string) yield break;
        foreach (var item in enumerable)
        {
            var timeline = ResolveRowReference(item);
            if (timeline is not null) yield return timeline;
        }
    }

    private IEnumerable<string> ExpandAnimationPhaseFamily(string selectedGamePath)
    {
        var normalized = CollapseGamePath(selectedGamePath);
        if (!normalized.EndsWith(".pap", StringComparison.OrdinalIgnoreCase)) yield break;

        var suffix = AnimationPhaseSuffixes.FirstOrDefault(candidate => normalized.EndsWith(candidate, StringComparison.OrdinalIgnoreCase));
        if (suffix is not null)
        {
            var familyBase = normalized[..^suffix.Length];
            foreach (var candidateSuffix in AnimationPhaseSuffixes)
            {
                var path = familyBase + candidateSuffix;
                if (_dataManager.FileExists(path)) yield return path;
            }
            yield break;
        }

        if (_dataManager.FileExists(normalized)) yield return normalized;
        var baseWithoutExtension = normalized[..^4];
        foreach (var candidateSuffix in AnimationPhaseSuffixes)
        {
            var path = baseWithoutExtension + candidateSuffix;
            if (_dataManager.FileExists(path)) yield return path;
        }
    }

    private static IEnumerable<string> EnumerateBundledVfxEditorCommonPapPaths()
    {
        Stream? stream = null;
        StreamReader? reader = null;
        try
        {
            stream = typeof(VanillaAssetService).Assembly.GetManifestResourceStream("RavaFit.Assets.VFXEditor.common_pap");
            if (stream is null) yield break;
            reader = new StreamReader(stream, Encoding.UTF8, true, 4096, leaveOpen: false);
            stream = null;
            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (!string.IsNullOrWhiteSpace(line)) yield return line.Trim();
            }
        }
        finally
        {
            reader?.Dispose();
            stream?.Dispose();
        }
    }

    private static bool TryClassifyNonCombatCommonPap(string rawRelative, out string familyKey, out string name, out string category)
    {
        familyKey = string.Empty;
        name = string.Empty;
        category = string.Empty;

        var relative = CollapseGamePath(rawRelative);
        if (string.IsNullOrWhiteSpace(relative) || !relative.EndsWith(".pap", StringComparison.OrdinalIgnoreCase)) return false;
        var slash = relative.IndexOf('/');
        if (slash <= 0) return false;
        var root = relative[..slash];
        var rest = relative[(slash + 1)..];

        if (JobAnimationRoots.Contains(root, StringComparer.OrdinalIgnoreCase)
            || string.Equals(root, "bt_nin_nin", StringComparison.OrdinalIgnoreCase)
            || string.Equals(root, "bt_stf_sld", StringComparison.OrdinalIgnoreCase))
            return false;

        if (!string.Equals(root, "bt_common", StringComparison.OrdinalIgnoreCase)) return false;
        if (rest.StartsWith("pc_contentsaction/", StringComparison.OrdinalIgnoreCase)
            || rest.StartsWith("human_sp/", StringComparison.OrdinalIgnoreCase))
            return false;

        if (rest.StartsWith("music/", StringComparison.OrdinalIgnoreCase)
            || rest.StartsWith("gs/", StringComparison.OrdinalIgnoreCase))
            return false;

        if (rest.StartsWith("idle_sp/", StringComparison.OrdinalIgnoreCase))
        {
            var leaf = rest["idle_sp/".Length..];
            if (leaf.Contains("battle", StringComparison.OrdinalIgnoreCase) || leaf.Contains("dead", StringComparison.OrdinalIgnoreCase)) return false;
            category = "Idle";
            familyKey = $"bt_common/{GetRelativeAnimationPhaseFamilyKey(rest)}";
            name = HumaniseCommonPapName(leaf, "Special");
            return true;
        }

        if (rest.StartsWith("normal/", StringComparison.OrdinalIgnoreCase))
        {
            var leaf = rest["normal/".Length..];
            if (LooksCombatLikeCommonPap(leaf) || !leaf.Contains("idle", StringComparison.OrdinalIgnoreCase)) return false;
            category = "Idle";
            familyKey = $"bt_common/{GetRelativeAnimationPhaseFamilyKey(rest)}";
            name = HumaniseCommonPapName(leaf);
            return true;
        }

        if (rest.StartsWith("event/", StringComparison.OrdinalIgnoreCase)
            || rest.StartsWith("event_base/", StringComparison.OrdinalIgnoreCase))
            return false;

        return false;
    }

    private void SeedVfxEditorCharacterPapFamilies(Dictionary<string, VanillaAnimationFamilyBuilder> families)
    {
        var idle = new VanillaAnimationFamilyBuilder("character:resident/idle", "Default Idle", "Idle")
        {
            SearchText = "idle default resident character",
        };
        var moveA = new VanillaAnimationFamilyBuilder("character:resident/move_a", "Movement A · Walk / Run / Jump / Sprint", "Movement")
        {
            SearchText = "move a movement walk slow walk run sprint jump fall landing resident locomotion umbrella ornament",
        };
        var moveB = new VanillaAnimationFamilyBuilder("character:resident/move_b", "Movement B · Diagonal / Secondary", "Movement")
        {
            SearchText = "move b movement diagonal strafe secondary resident locomotion",
        };

        foreach (var race in CharacterRaceCatalog.All)
        {
            AddCharacterPapIfPresent(idle, race.Code, "bt_common/resident/idle.pap");

            if (AddCharacterPapIfPresent(moveA, race.Code, "bt_common/resident/move_a.pap"))
                AddCharacterPapIfPresent(moveA, race.Code, "ot_m6001/resident/ornament.pap");

            AddCharacterPapIfPresent(moveB, race.Code, "bt_common/resident/move_b.pap");
        }

        AddBuiltFamily(families, idle);
        AddBuiltFamily(families, moveA);
        AddBuiltFamily(families, moveB);

        for (var pose = 1; pose <= 6; pose++)
        {
            SeedCharacterPoseFamily(families, $"pose:{pose}", $"Pose {pose}", $"emote/pose{pose:D2}", "Pose", $"pose {pose} change pose");
            SeedCharacterPoseFamily(families, $"ground-pose:{pose}", $"Ground Sit Pose {pose}", $"emote/j_pose{pose:D2}", "Pose", $"ground sit pose {pose}");
            SeedCharacterPoseFamily(families, $"chair-pose:{pose}", $"Chair Sit Pose {pose}", $"emote/s_pose{pose:D2}", "Pose", $"chair sit pose {pose}");
            SeedCharacterPoseFamily(families, $"umbrella-pose:{pose}", $"Umbrella Pose {pose}", $"ornament_sp/m6001/onm_pose{pose:D2}", "Pose", $"umbrella ornament pose {pose}");
        }
    }

    private void SeedCharacterPoseFamily(Dictionary<string, VanillaAnimationFamilyBuilder> families, string id, string name, string relativeBase, string category, string searchText)
    {
        var builder = new VanillaAnimationFamilyBuilder($"character:{id}", name, category) { SearchText = searchText };
        foreach (var race in CharacterRaceCatalog.All)
        {
            var start = BuildCharacterPapPath(race.Code, $"bt_common/{relativeBase}_start.pap");
            var loop = BuildCharacterPapPath(race.Code, $"bt_common/{relativeBase}_loop.pap");
            if (!_dataManager.FileExists(start) || !_dataManager.FileExists(loop)) continue;
            builder.Add(race.Code, start);
            builder.Add(race.Code, loop);
        }
        AddBuiltFamily(families, builder);
    }

    private bool AddCharacterPapIfPresent(VanillaAnimationFamilyBuilder builder, string raceCode, string relative)
    {
        var path = BuildCharacterPapPath(raceCode, relative);
        if (!_dataManager.FileExists(path)) return false;
        builder.Add(raceCode, path);
        return true;
    }

    private static string BuildCharacterPapPath(string raceCode, string relative)
        => $"chara/human/{raceCode}/animation/a0001/{relative}";

    private static void AddBuiltFamily(Dictionary<string, VanillaAnimationFamilyBuilder> families, VanillaAnimationFamilyBuilder builder)
    {
        if (builder.HasPaths) families[builder.Id] = builder;
    }

    private static bool LooksCombatLikeCommonPap(string value)
    {
        var normalized = value.ToLowerInvariant();
        string[] blocked =
        [
            "attack", "battle", "damage", "guard", "weapon", "weaponskill", "limitbreak", "knockback",
            "magic_heal", "revive", "air_fall_attack", "blowaway", "cannon", "2sw_", "sword", "katana",
            "ninja", "gun_", "rifle", "bow_", "spear", "shield", "parry", "defend", "ws_",
        ];
        return blocked.Any(normalized.Contains);
    }

    private static string GetRelativeAnimationPhaseFamilyKey(string relativePath)
    {
        var normalized = CollapseGamePath(relativePath);
        foreach (var suffix in AnimationPhaseSuffixes)
            if (normalized.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) return normalized[..^suffix.Length];
        return normalized.EndsWith(".pap", StringComparison.OrdinalIgnoreCase) ? normalized[..^4] : normalized;
    }

    private static string HumaniseCommonPapName(string path, string? prefix = null, string? stripPrefix = null)
    {
        var value = CollapseGamePath(path);
        if (value.EndsWith(".pap", StringComparison.OrdinalIgnoreCase)) value = value[..^4];
        foreach (var suffix in new[] { "_start", "_loop", "_end", "_stop" })
            if (value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) { value = value[..^suffix.Length]; break; }

        value = value.Replace('/', ' ').Replace('_', ' ').Replace('-', ' ');
        value = Regex.Replace(value, @"\s+", " ").Trim();
        if (!string.IsNullOrWhiteSpace(stripPrefix) && value.StartsWith(stripPrefix.Replace('_', ' '), StringComparison.OrdinalIgnoreCase))
            value = value[stripPrefix.Replace('_', ' ').Length..].Trim();
        if (string.IsNullOrWhiteSpace(value)) value = "Player animation";

        value = string.Join(' ', value.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Select(word => word.Length == 1 ? word.ToUpperInvariant() : char.ToUpperInvariant(word[0]) + word[1..]));
        return string.IsNullOrWhiteSpace(prefix) ? value : $"{prefix} {value}";
    }

    private static bool TryResolveEmotePapRelatives(object timeline, string rawKey, out IReadOnlyList<string> relatives)
    {
        var output = new List<string>();
        relatives = output;
        var key = NormalizeEmbeddedPath(rawKey);
        if (string.IsNullOrWhiteSpace(key) || key.Contains("[skl_id]", StringComparison.OrdinalIgnoreCase)) return true;
        if (key.EndsWith(".pap", StringComparison.OrdinalIgnoreCase)) key = key[..^4];
        if (key.EndsWith(".tmb", StringComparison.OrdinalIgnoreCase)) key = key[..^4];

        if (!TryUInt64(GetMemberValue(timeline, "LoadType"), out var loadType))
            return false;

        switch (loadType)
        {
            case 2:
                output.Add($"bt_common/{key}.pap");
                return true;
            case 1:
                foreach (var root in JobAnimationRoots) output.Add($"{root}/{key}.pap");
                return true;
            case 0:
                if (!key.StartsWith("facial/pose/", StringComparison.OrdinalIgnoreCase))
                    output.Add($"bt_common/{key}.pap");
                return true;
            default:
                return false;
        }
    }

    private IEnumerable<string> ResolveHumanPapCandidatePaths(string raw)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        void AddIfPresent(string value)
        {
            var clean = CollapseGamePath(value);
            if (clean.Length == 0 || !XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(clean)) return;
            if (_dataManager.FileExists(clean)) seen.Add(clean);
        }

        var direct = NormalizePapCandidate(raw);
        if (direct is not null) AddIfPresent(direct);

        var relative = NormalizeEmbeddedPath(raw);
        if (relative.EndsWith(".tmb", StringComparison.OrdinalIgnoreCase))
            relative = relative[..^4] + ".pap";
        else if (!relative.EndsWith(".pap", StringComparison.OrdinalIgnoreCase))
            relative += ".pap";

        var humanMarker = Regex.Match(relative, @"(?:^|/)c\d{4}/(?<rest>animation/.+\.pap)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        if (humanMarker.Success) relative = humanMarker.Groups["rest"].Value;
        if (relative.StartsWith("animation/a0001/", StringComparison.OrdinalIgnoreCase))
            relative = relative["animation/a0001/".Length..];
        else if (relative.StartsWith("a0001/", StringComparison.OrdinalIgnoreCase))
            relative = relative["a0001/".Length..];

        foreach (var race in CharacterRaceCatalog.All)
        {
            var rooted = relative.StartsWith("bt_", StringComparison.OrdinalIgnoreCase) ? relative : $"bt_common/{relative}";
            AddIfPresent($"chara/human/c{race.Code}/animation/a0001/{rooted}");
        }

        return seen;
    }

    private static string? NormalizePapCandidate(string raw)
    {
        var value = NormalizeEmbeddedPath(raw);
        var marker = value.IndexOf("chara/human/c", StringComparison.OrdinalIgnoreCase);
        if (marker >= 0) return CollapseGamePath(value[marker..]);
        marker = value.IndexOf("human/c", StringComparison.OrdinalIgnoreCase);
        if (marker >= 0) return CollapseGamePath("chara/" + value[marker..]);
        var match = Regex.Match(value, @"(?:^|/)c(?<race>\d{4})/(?<rest>[a-z0-9_./-]+\.pap)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        return match.Success ? CollapseGamePath($"chara/human/c{match.Groups["race"].Value}/{match.Groups["rest"].Value}") : null;
    }

    private static string CollapseGamePath(string value)
    {
        var output = new List<string>();
        foreach (var segment in value.Replace('\\', '/').Trim().TrimStart('/').Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            if (segment == ".") continue;
            if (segment == "..") { if (output.Count > 0) output.RemoveAt(output.Count - 1); continue; }
            output.Add(segment);
        }
        return string.Join('/', output).ToLowerInvariant();
    }


    private static readonly string[] AnimationPhaseSuffixes = ["_start.pap", "_loop.pap", "_end.pap", "_stop.pap"];

    private static string GetAnimationPhaseFamilyKey(string gamePath)
    {
        var normalized = gamePath.Replace('\\', '/');
        foreach (var suffix in AnimationPhaseSuffixes)
            if (normalized.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) return normalized[..^suffix.Length];
        return normalized.EndsWith(".pap", StringComparison.OrdinalIgnoreCase) ? normalized[..^4] : normalized;
    }

    private static int AnimationPhaseOrder(string gamePath)
    {
        if (gamePath.EndsWith("_start.pap", StringComparison.OrdinalIgnoreCase)) return 0;
        if (gamePath.EndsWith("_loop.pap", StringComparison.OrdinalIgnoreCase)) return 1;
        if (gamePath.EndsWith("_end.pap", StringComparison.OrdinalIgnoreCase)) return 2;
        if (gamePath.EndsWith("_stop.pap", StringComparison.OrdinalIgnoreCase)) return 3;
        return 4;
    }

    private static int AnimationCategoryOrder(string category) => category switch
    {
        "Emote" => 0,
        "Idle" => 1,
        "Pose" => 2,
        "Movement" => 3,
        _ => 4,
    };

    private static string? ExtractRaceCode(string gamePath)
    {
        var match = Regex.Match(gamePath, @"(?:^|/)c(?<race>\d{4})(?:/|$)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        return match.Success ? match.Groups["race"].Value : null;
    }

    private static string NormalizeEmbeddedPath(string value) => value.Trim().Trim('\0').Replace('\\', '/').TrimStart('/').ToLowerInvariant();

    private static IEnumerable<string> ResolveAccessorySlots(object? slotCategory)
    {
        if (slotCategory is null) yield break;
        if (ReadBooleanMember(slotCategory, "Ears") || ReadBooleanMember(slotCategory, "Ear")) yield return AccessoryModelSlots.Earrings;
        if (ReadBooleanMember(slotCategory, "Neck")) yield return AccessoryModelSlots.Necklace;
        if (ReadBooleanMember(slotCategory, "Wrists") || ReadBooleanMember(slotCategory, "Wrist")) yield return AccessoryModelSlots.Wrists;
        if (ReadBooleanMember(slotCategory, "FingerR") || ReadBooleanMember(slotCategory, "FingerRight")) yield return AccessoryModelSlots.RightRing;
        if (ReadBooleanMember(slotCategory, "FingerL") || ReadBooleanMember(slotCategory, "FingerLeft")) yield return AccessoryModelSlots.LeftRing;
    }

    private static string? ResolveEquipmentSlot(object? slotCategory)
    {
        if (slotCategory is null) return null;
        if (ReadBooleanMember(slotCategory, "Body")) return BodySlots.Chest;
        if (ReadBooleanMember(slotCategory, "Legs")) return BodySlots.Legs;
        if (ReadBooleanMember(slotCategory, "Gloves")) return BodySlots.Hands;
        if (ReadBooleanMember(slotCategory, "Feet")) return BodySlots.Feet;
        return null;
    }

    private static ushort ReadModelSetId(object? model)
    {
        if (model is null) return 0;
        foreach (var name in new[] { "Id", "PrimaryId", "ModelId" })
        {
            var value = GetMemberValue(model, name);
            if (TryUInt64(value, out var number) && number is > 0 and <= ushort.MaxValue) return (ushort)number;
        }
        if (TryUInt64(model, out var packed) && packed > 0) return (ushort)(packed & 0xFFFF);
        return 0;
    }

    private static ushort ReadModelVariantId(object? model)
    {
        if (model is null) return 0;
        foreach (var name in new[] { "Variant", "VariantId", "ModelVariant" })
        {
            var value = GetMemberValue(model, name);
            if (TryUInt64(value, out var number) && number <= ushort.MaxValue) return (ushort)number;
        }
        return TryUInt64(model, out var packed) ? (ushort)((packed >> 32) & 0xFFFF) : (ushort)0;
    }

    private static int AccessoryImcPartBit(string slot)
        => slot switch
        {
            AccessoryModelSlots.Earrings => 0,
            AccessoryModelSlots.Necklace => 1,
            AccessoryModelSlots.Wrists => 2,
            AccessoryModelSlots.RightRing => 3,
            AccessoryModelSlots.LeftRing => 4,
            _ => -1,
        };

    private static object? ResolveRowReference(object? value)
    {
        if (value is null) return null;
        return GetMemberValue(value, "Value") ?? value;
    }

    private static object? GetMemberValue(object? value, string name)
    {
        if (value is null) return null;
        var type = value.GetType();
        return type.GetProperty(name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase)?.GetValue(value)
            ?? type.GetField(name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase)?.GetValue(value);
    }

    private static bool ReadBooleanMember(object value, string name)
    {
        var member = GetMemberValue(value, name);
        if (member is bool flag) return flag;
        return TryUInt64(member, out var number) && number != 0;
    }

    private static bool TryUInt64(object? value, out ulong number)
    {
        try
        {
            if (value is null) { number = 0; return false; }
            number = Convert.ToUInt64(value);
            return true;
        }
        catch { number = 0; return false; }
    }

    private static string ReadableText(object? value)
    {
        var text = value?.ToString()?.Trim() ?? string.Empty;
        return text.Replace("\0", string.Empty).Trim();
    }

    private static string SanitiseDisplayName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var cleaned = new string(value.Select(c => invalid.Contains(c) ? '_' : c).ToArray()).Trim();
        return string.IsNullOrWhiteSpace(cleaned) ? "Vanilla Outfit" : cleaned;
    }

    private static JsonObject EmptyOptionData() => new() { ["Files"] = new JsonObject(), ["FileSwaps"] = new JsonObject(), ["Manipulations"] = new JsonArray() };

    private string MakeUniqueDirectoryName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var safe = new string(name.Select(c => invalid.Contains(c) ? '_' : c).ToArray()).Trim().Trim('.');
        if (safe.Length == 0) safe = "RavaFit Vanilla";
        var candidate = safe;
        var suffix = 2;
        while (Directory.Exists(Path.Combine(_penumbra.ModDirectoryRoot, candidate))) candidate = $"{safe} {suffix++}";
        return candidate;
    }

    private static string NormalizeGamePath(string value)
    {
        var normalized = value.Trim().Replace('\\', '/').TrimStart('/');
        if (normalized.Length == 0 || normalized.Contains("..", StringComparison.Ordinal) || normalized.Contains(':', StringComparison.Ordinal)) throw new InvalidDataException($"Invalid XIV game path '{value}'.");
        return normalized;
    }

}
