using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Dalamud.Plugin.Services;
using RavaFit.Core.Models;
using RavaFit.Services;

namespace RavaFit.Services.Animation;

internal sealed partial class AnimationPortService
{
    [GeneratedRegex(@"^chara/human/c(?<race>\d{4})/.*\.pap$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex HumanPapPathRegex();

    private readonly PenumbraService _penumbra;
    private readonly AnimationSkeletonService _skeletons;
    private readonly AnimationPapRetargeter _retargeter;
    private readonly IPluginLog _log;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly object _inspectionGate = new();
    private string _inspectionMetaPath = string.Empty;
    private DateTime _inspectionWriteUtc;
    private long _inspectionLength = -1;
    private AnimationModInspection? _inspection;

    private sealed record AnimationModInspection(IReadOnlyList<CharacterRaceIdentity> Sources, string? Issue);
    private enum AnimationSourceKind { ModFile, GameFile }
    private sealed record SelectedAnimationSource(AnimationSourceKind Kind, string SourceReference, string SourceRaceCode, string TargetRelative);

    public AnimationPortService(PenumbraService penumbra, AnimationSkeletonService skeletons, AnimationPapRetargeter retargeter, IPluginLog log)
    {
        _penumbra = penumbra;
        _skeletons = skeletons;
        _retargeter = retargeter;
        _log = log;
    }

    public bool Busy { get; private set; }
    public string LastError { get; private set; } = string.Empty;
    public AnimationPortProgress Progress { get; private set; } = new(AnimationPortStage.Idle, 0f);

    public IReadOnlyList<CharacterRaceIdentity> GetSourceCharacters(PenumbraModInfo? mod)
        => mod is null ? [] : Inspect(mod).Sources;

    public string? GetPortabilityIssue(PenumbraModInfo? mod)
        => mod is null ? null : Inspect(mod).Issue;

    public CharacterRaceIdentity? GetVanillaSourceCharacter(string? gamePath)
    {
        if (string.IsNullOrWhiteSpace(gamePath)) return null;
        var normalized = gamePath.Trim().Replace('\\', '/').TrimStart('/');
        var match = HumanPapPathRegex().Match(normalized);
        return match.Success && XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(normalized)
            ? CharacterRaceCatalog.FromCode(match.Groups["race"].Value)
            : null;
    }

    public void ResetStatus()
    {
        if (Busy) return;
        LastError = string.Empty;
        Progress = new(AnimationPortStage.Idle, 0f);
    }

    private AnimationModInspection Inspect(PenumbraModInfo mod)
    {
        var metaPath = Path.Combine(mod.ModRoot, "meta.json");
        try
        {
            var info = new FileInfo(metaPath);
            if (!info.Exists) return new AnimationModInspection([], "The selected mod has no meta.json.");
            lock (_inspectionGate)
            {
                if (_inspection is not null
                    && string.Equals(_inspectionMetaPath, info.FullName, StringComparison.OrdinalIgnoreCase)
                    && _inspectionWriteUtc == info.LastWriteTimeUtc
                    && _inspectionLength == info.Length)
                {
                    return _inspection;
                }

                var root = ReadMeta(mod);
                var codes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                CollectHumanPapRaceCodes(root, codes);
                var sources = codes.Select(CharacterRaceCatalog.FromCode)
                    .Where(identity => identity is not null)
                    .Cast<CharacterRaceIdentity>()
                    .OrderBy(identity => identity.DisplayName, StringComparer.OrdinalIgnoreCase)
                    .ToArray();
                var fileVersion = ReadInt(root, "FileVersion");
                var issue = fileVersion != 4 ? "Select a Penumbra V4 mod." : null;

                _inspectionMetaPath = info.FullName;
                _inspectionWriteUtc = info.LastWriteTimeUtc;
                _inspectionLength = info.Length;
                _inspection = new AnimationModInspection(sources, issue);
                return _inspection;
            }
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "RavaFit could not inspect animation metadata in {Mod}", mod.Name);
            return new AnimationModInspection([], ex.Message);
        }
    }

    public async Task<AnimationPortResult> PortAsync(PenumbraModInfo sourceMod, CharacterRaceIdentity sourceCharacter, CharacterRaceIdentity targetCharacter, string skeletonChoiceId, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        Busy = true;
        string? destinationRoot = null;
        string? work = null;
        try
        {
            LastError = string.Empty;
            var root = ReadMeta(sourceMod);
            var fileVersion = ReadInt(root, "FileVersion");
            if (fileVersion != 4)
                throw new InvalidDataException($"Animation Porting supports Penumbra V4 mods. Found FileVersion={fileVersion?.ToString() ?? "<missing>"}.");
            if (LalafellContentGuard.IsLalafell(targetCharacter))
            {
                var guard = LalafellContentGuard.InspectAnimation(sourceMod);
                if (guard.Blocked)
                    throw new NotSupportedException("RavaFit will not port explicit/NSFW animation content to Lalafell. Evidence: " + string.Join("; ", guard.Reasons));
            }
            var destinationName = $"{sourceMod.Name} - Converted for {targetCharacter.DisplayName}";
            var sourceParent = Directory.GetParent(sourceMod.ModRoot)?.FullName ?? throw new InvalidOperationException("The selected mod has no parent directory.");
            var directoryName = SanitizeDirectoryName($"{Path.GetFileName(sourceMod.ModRoot)} - Converted for {targetCharacter.DisplayName}");
            destinationRoot = Path.Combine(sourceParent, directoryName);
            if (Directory.Exists(destinationRoot))
                throw new InvalidOperationException($"'{destinationName}' already exists. Rename or remove it before converting again.");

            SetProperty(root, "Name", JsonValue.Create(destinationName));
            SetProperty(root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));
            AppendRavaFitProvenance(root, sourceMod.Name, targetCharacter.DisplayName);

            var selectedAnimations = new Dictionary<string, SelectedAnimationSource>(StringComparer.OrdinalIgnoreCase);
            RewriteAnimationMappings(root, sourceCharacter.Code, targetCharacter.Code, selectedAnimations);
            if (selectedAnimations.Count == 0)
                throw new InvalidOperationException($"No {sourceCharacter.DisplayName} player animations were found in this mod.");

            Progress = new(AnimationPortStage.ReadingSkeleton, 0.04f, $"Loading {sourceCharacter.DisplayName} and {targetCharacter.DisplayName} skeletons...");
            var sourceSkeletonCache = new Dictionary<string, IReadOnlyList<AnimationTargetSkeletonSnapshot>>(StringComparer.OrdinalIgnoreCase);
            var sourceSkeletons = await _skeletons.LoadAsync(sourceCharacter, AnimationSkeletonService.StandardChoiceId, cancellationToken).ConfigureAwait(false);
            var targetSkeletons = await _skeletons.LoadAsync(targetCharacter, skeletonChoiceId, cancellationToken).ConfigureAwait(false);
            if (sourceSkeletons.Count == 0) throw new InvalidOperationException("The source skeleton could not be loaded.");
            if (targetSkeletons.Count == 0) throw new InvalidOperationException("The selected target skeleton could not be loaded.");
            sourceSkeletonCache[sourceCharacter.Code] = sourceSkeletons;

            Progress = new(AnimationPortStage.Copying, 0.09f, "Copying mod...");
            CopyDirectory(sourceMod.ModRoot, destinationRoot, cancellationToken);
            var destinationMeta = Path.Combine(destinationRoot, "meta.json");

            var converted = 0;
            var unchanged = 0;
            var droppedTracks = 0;
            var bindings = 0;
            var gamePapCache = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var files = selectedAnimations.Values.OrderBy(source => source.SourceReference, StringComparer.OrdinalIgnoreCase).ThenBy(source => source.TargetRelative, StringComparer.OrdinalIgnoreCase).ToArray();
            for (var i = 0; i < files.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var source = files[i];
                string sourceFull;
                if (source.Kind == AnimationSourceKind.ModFile)
                {
                    sourceFull = SafeResolve(destinationRoot, source.SourceReference);
                    if (!File.Exists(sourceFull))
                        throw new FileNotFoundException($"Animation file referenced by the mod was not found: {source.SourceReference}", sourceFull);
                }
                else
                {
                    if (!gamePapCache.TryGetValue(source.SourceReference, out var cachedSource))
                    {
                        work ??= Path.Combine(Path.GetTempPath(), "RavaFit", "animation-port", Guid.NewGuid().ToString("N"));
                        Directory.CreateDirectory(work);
                        var bytes = _skeletons.ReadGameFileBytes(source.SourceReference);
                        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(source.SourceReference.ToLowerInvariant()))).ToLowerInvariant();
                        sourceFull = Path.Combine(work, $"{hash[..24]}-{Path.GetFileName(source.SourceReference)}");
                        await File.WriteAllBytesAsync(sourceFull, bytes, cancellationToken).ConfigureAwait(false);
                        gamePapCache[source.SourceReference] = sourceFull;
                    }
                    else
                    {
                        sourceFull = cachedSource;
                    }
                }

                if (!sourceSkeletonCache.TryGetValue(source.SourceRaceCode, out var animationSourceSkeletons))
                {
                    var actualSourceCharacter = CharacterRaceCatalog.FromCode(source.SourceRaceCode)
                        ?? throw new InvalidDataException($"Animation source race c{source.SourceRaceCode} is not a supported player race/gender.");
                    Progress = new(AnimationPortStage.ReadingSkeleton, 0.10f, $"Loading {actualSourceCharacter.DisplayName} source skeleton...");
                    animationSourceSkeletons = await _skeletons.LoadAsync(actualSourceCharacter, AnimationSkeletonService.StandardChoiceId, cancellationToken).ConfigureAwait(false);
                    if (animationSourceSkeletons.Count == 0)
                        throw new InvalidOperationException($"The {actualSourceCharacter.DisplayName} source skeleton could not be loaded.");
                    sourceSkeletonCache[source.SourceRaceCode] = animationSourceSkeletons;
                }

                var targetFull = SafeResolve(destinationRoot, source.TargetRelative);
                var progress = 0.12f + (0.78f * ((float)i / Math.Max(1, files.Length)));
                Progress = new(AnimationPortStage.Converting, progress, $"Converting animation {i + 1}/{files.Length}...");
                var result = await _retargeter.RewriteFileAsync(sourceFull, targetFull, animationSourceSkeletons, targetSkeletons, cancellationToken).ConfigureAwait(false);
                if (result.Status == AnimationPapRewriteStatus.Blocked)
                    throw new InvalidDataException($"Could not safely convert {Path.GetFileName(sourceFull)}: {result.Reason}");

                if (result.Status == AnimationPapRewriteStatus.Converted) converted++; else unchanged++;
                droppedTracks += result.DroppedTrackCount;
                bindings += result.BindingCount;
                await Task.Yield();
            }

            Progress = new(AnimationPortStage.Writing, 0.92f, "Writing new mod...");
            var options = new JsonSerializerOptions { WriteIndented = true };
            var tempMeta = destinationMeta + $".ravafit-{Guid.NewGuid():N}.tmp";
            await File.WriteAllTextAsync(tempMeta, root.ToJsonString(options), cancellationToken).ConfigureAwait(false);
            _ = JsonNode.Parse(await File.ReadAllTextAsync(tempMeta, cancellationToken).ConfigureAwait(false))
                ?? throw new InvalidDataException("Generated animation mod metadata could not be verified.");
            File.Move(tempMeta, destinationMeta, true);

            _penumbra.Refresh();
            Progress = new(AnimationPortStage.Complete, 1f, "Done");
            return new AnimationPortResult(destinationName, destinationRoot, converted, unchanged, droppedTracks, bindings);
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            Progress = new(AnimationPortStage.Failed, 1f, "Failed");
            _log.Error(ex, "RavaFit animation port failed.");
            if (!string.IsNullOrWhiteSpace(destinationRoot) && Directory.Exists(destinationRoot))
            {
                try { Directory.Delete(destinationRoot, true); }
                catch (Exception cleanup) { _log.Warning(cleanup, "RavaFit could not remove incomplete animation port {Path}", destinationRoot); }
            }
            throw;
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(work))
            {
                try { Directory.Delete(work, true); } catch { }
            }
            Busy = false;
            _gate.Release();
        }
    }

    public Task<AnimationPortResult> PortVanillaAsync(string gamePathRaw, CharacterRaceIdentity targetCharacter, string skeletonChoiceId, string? sourceDisplayName = null, CancellationToken cancellationToken = default)
    {
        var gamePath = gamePathRaw.Trim().Replace('\\', '/').TrimStart('/');
        var sourceCharacter = GetVanillaSourceCharacter(gamePath)
            ?? throw new InvalidDataException("Enter a vanilla player animation path under chara/human/c####/.../animation/...pap.");
        return PortVanillaAsync([gamePath], sourceCharacter, targetCharacter, skeletonChoiceId, sourceDisplayName, cancellationToken);
    }

    public async Task<AnimationPortResult> PortVanillaAsync(IReadOnlyCollection<string> gamePathsRaw, CharacterRaceIdentity sourceCharacter, CharacterRaceIdentity targetCharacter, string skeletonChoiceId, string? sourceDisplayName = null, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        Busy = true;
        string? destinationRoot = null;
        string? work = null;
        try
        {
            LastError = string.Empty;
            if (string.IsNullOrWhiteSpace(_penumbra.ModDirectoryRoot) || !Directory.Exists(_penumbra.ModDirectoryRoot))
                throw new DirectoryNotFoundException("Penumbra did not provide its mod directory.");

            var selectedPaths = gamePathsRaw
                .Where(path => !string.IsNullOrWhiteSpace(path))
                .Select(path => path.Trim().Replace('\\', '/').TrimStart('/'))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (selectedPaths.Length == 0)
                throw new InvalidDataException("The selected vanilla animation family has no PAP files for this base race.");

            var expandedSelections = selectedPaths
                .Select(path => (SelectedPath: path, ResolvedPaths: ResolveVanillaAnimationPhaseFamily(path)))
                .ToArray();
            var unresolvedSelections = expandedSelections
                .Where(item => item.ResolvedPaths.Count == 0)
                .Select(item => item.SelectedPath)
                .ToArray();
            if (unresolvedSelections.Length > 0)
                throw new FileNotFoundException($"The selected vanilla animation family contains PAPs that could not be read safely: {string.Join(", ", unresolvedSelections)}");

            var animationPaths = expandedSelections
                .SelectMany(item => item.ResolvedPaths)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(AnimationPhaseOrder)
                .ThenBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (animationPaths.Length == 0)
                throw new FileNotFoundException("None of the selected vanilla player animation PAPs could be read from XIV game data.");

            foreach (var animationPath in animationPaths)
            {
                var pathSource = GetVanillaSourceCharacter(animationPath);
                if (pathSource is null || !string.Equals(pathSource.Code, sourceCharacter.Code, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException($"Vanilla animation '{animationPath}' does not use the selected base race {sourceCharacter.DisplayName}.");
            }

            Progress = new(AnimationPortStage.ReadingSkeleton, 0.05f, $"Loading {sourceCharacter.DisplayName} and {targetCharacter.DisplayName} skeletons...");
            var sourceSkeletons = await _skeletons.LoadAsync(sourceCharacter, AnimationSkeletonService.StandardChoiceId, cancellationToken).ConfigureAwait(false);
            var targetSkeletons = await _skeletons.LoadAsync(targetCharacter, skeletonChoiceId, cancellationToken).ConfigureAwait(false);
            if (sourceSkeletons.Count == 0 || targetSkeletons.Count == 0)
                throw new InvalidOperationException("The source or target player skeleton could not be loaded.");

            work = Path.Combine(Path.GetTempPath(), "RavaFit", "vanilla-animation", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(work);

            var friendlyName = string.IsNullOrWhiteSpace(sourceDisplayName)
                ? Path.GetFileNameWithoutExtension(animationPaths[0])
                : sourceDisplayName.Trim();
            var destinationName = $"RavaFit Vanilla Animation - {friendlyName} - Converted for {targetCharacter.DisplayName}";
            var directoryName = SanitizeDirectoryName(destinationName);
            destinationRoot = Path.Combine(_penumbra.ModDirectoryRoot, directoryName);
            var suffix = 2;
            while (Directory.Exists(destinationRoot))
                destinationRoot = Path.Combine(_penumbra.ModDirectoryRoot, SanitizeDirectoryName($"{destinationName} {suffix++}"));
            Directory.CreateDirectory(destinationRoot);

            var files = new JsonObject();
            var targetMappings = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var converted = 0;
            var unchanged = 0;
            var droppedTracks = 0;
            var bindings = 0;
            var crossRaceRetarget = !string.Equals(sourceCharacter.Code, targetCharacter.Code, StringComparison.OrdinalIgnoreCase);

            for (var i = 0; i < animationPaths.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var animationPath = animationPaths[i];
                var raw = _skeletons.ReadGameFileBytes(animationPath);
                var sourcePap = Path.Combine(work, $"{i:D3}-{Path.GetFileName(animationPath)}");
                await File.WriteAllBytesAsync(sourcePap, raw, cancellationToken).ConfigureAwait(false);

                var targetGamePath = RewriteHumanAnimationRace(animationPath, targetCharacter.Code);
                if (!XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(targetGamePath))
                    throw new InvalidDataException($"Retargeted vanilla animation path '{targetGamePath}' is outside the supported player a0001 animation tree.");
                if (!targetMappings.Add(targetGamePath))
                    throw new InvalidDataException($"Vanilla animation family produced more than one source PAP for target path '{targetGamePath}'.");

                var targetRelative = GetGeneratedPapRelativePath(animationPath, targetCharacter.Code);
                var targetPhysical = SafeResolve(destinationRoot, targetRelative);
                Directory.CreateDirectory(Path.GetDirectoryName(targetPhysical)!);

                Progress = new(AnimationPortStage.Converting, 0.25f + (0.60f * ((float)i / Math.Max(1, animationPaths.Length))), $"Retargeting vanilla animation {i + 1}/{animationPaths.Length}...");
                var result = await _retargeter.RewriteFileAsync(sourcePap, targetPhysical, sourceSkeletons, targetSkeletons, cancellationToken).ConfigureAwait(false);
                if (result.Status == AnimationPapRewriteStatus.Blocked)
                    throw new InvalidDataException($"Could not safely retarget {Path.GetFileName(animationPath)}: {result.Reason}");

                if (crossRaceRetarget && result.Status != AnimationPapRewriteStatus.Converted)
                    throw new InvalidDataException($"Could not safely retarget {Path.GetFileName(animationPath)}: no rewritten target-race animation was produced ({result.Reason}).");

                files[targetGamePath] = targetRelative;
                if (result.Status == AnimationPapRewriteStatus.Converted) converted++; else unchanged++;
                droppedTracks += result.DroppedTrackCount;
                bindings += result.BindingCount;
            }

            Progress = new(AnimationPortStage.Writing, 0.90f, "Creating Penumbra mod...");
            var sourceSummary = animationPaths.Length == 1 ? Path.GetFileName(animationPaths[0]) : $"{animationPaths.Length} PAP files";
            var meta = new JsonObject
            {
                ["FileVersion"] = 4,
                ["Name"] = destinationName,
                ["Author"] = "RavaFit",
                ["Version"] = "1.1.0",
                ["Website"] = string.Empty,
                ["Description"] = $"RavaFit vanilla animation port · {friendlyName} · {sourceCharacter.DisplayName} → {targetCharacter.DisplayName}.",
                ["Tags"] = new JsonArray(),
                ["LastWrite"] = DateTimeOffset.UtcNow.ToString("O"),
                ["DefaultData"] = new JsonObject
                {
                    ["Files"] = files,
                    ["FileSwaps"] = new JsonObject(),
                    ["Manipulations"] = new JsonArray(),
                },
                ["PageNames"] = new JsonObject(),
                ["Groups"] = new JsonArray(),
            };
            await File.WriteAllTextAsync(Path.Combine(destinationRoot, "meta.json"), meta.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
            var generatedModDirectory = Path.GetFileName(destinationRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.IsNullOrWhiteSpace(generatedModDirectory))
                throw new InvalidOperationException("RavaFit wrote the converted vanilla animation safely, but Penumbra could not register the generated mod: generated mod directory was invalid");
            if (!_penumbra.AddMod(generatedModDirectory, out var addError))
                throw new InvalidOperationException($"RavaFit wrote the converted vanilla animation safely, but Penumbra could not register the generated mod: {addError}");
            _penumbra.Refresh();
            Progress = new(AnimationPortStage.Complete, 1f, "Done");
            return new AnimationPortResult(destinationName, destinationRoot, converted, unchanged, droppedTracks, bindings);
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            Progress = new(AnimationPortStage.Failed, 1f, "Failed");
            _log.Error(ex, "RavaFit vanilla animation port failed.");
            if (!string.IsNullOrWhiteSpace(destinationRoot) && Directory.Exists(destinationRoot))
            {
                try { Directory.Delete(destinationRoot, true); } catch { }
            }
            throw;
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(work))
            {
                try { Directory.Delete(work, true); } catch { }
            }
            Busy = false;
            _gate.Release();
        }
    }

    private static int AnimationPhaseOrder(string gamePath)
    {
        if (gamePath.EndsWith("_start.pap", StringComparison.OrdinalIgnoreCase)) return 0;
        if (gamePath.EndsWith("_loop.pap", StringComparison.OrdinalIgnoreCase)) return 1;
        if (gamePath.EndsWith("_end.pap", StringComparison.OrdinalIgnoreCase)) return 2;
        if (gamePath.EndsWith("_stop.pap", StringComparison.OrdinalIgnoreCase)) return 3;
        return 4;
    }

    private IReadOnlyList<string> ResolveVanillaAnimationPhaseFamily(string selectedGamePath)
    {
        var normalized = selectedGamePath.Trim().Replace('\\', '/').TrimStart('/');
        var suffixes = new[] { "_start.pap", "_loop.pap", "_end.pap", "_stop.pap" };
        var matchedSuffix = suffixes.FirstOrDefault(suffix => normalized.EndsWith(suffix, StringComparison.OrdinalIgnoreCase));
        if (matchedSuffix is not null)
        {
            var familyBase = normalized[..^matchedSuffix.Length];
            var phased = new List<string>(suffixes.Length);
            foreach (var suffix in suffixes)
            {
                var path = familyBase + suffix;
                if (!XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(path) || !_skeletons.GameFileExists(path)) continue;
                phased.Add(path);
            }
            return phased;
        }

        if (!XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(normalized) || !_skeletons.GameFileExists(normalized))
            return [];

        var output = new List<string>(1 + suffixes.Length) { normalized };
        var baseWithoutExtension = normalized.EndsWith(".pap", StringComparison.OrdinalIgnoreCase) ? normalized[..^4] : normalized;
        foreach (var suffix in suffixes)
        {
            var path = baseWithoutExtension + suffix;
            if (!XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(path) || !_skeletons.GameFileExists(path)) continue;
            output.Add(path);
        }
        return output;
    }

    private static void AppendRavaFitProvenance(JsonObject root, string sourceName, string targetName)
    {
        var existing = ReadString(root, "Description")?.Trim() ?? string.Empty;
        var author = ReadString(root, "Author")?.Trim();
        var notice = $"RavaFit port of '{sourceName}' for {targetName}."
            + (string.IsNullOrWhiteSpace(author) ? string.Empty : $" Credit: {author}.")
            + " Check the original creator's permissions before sharing.";
        SetProperty(root, "Description", JsonValue.Create(string.IsNullOrWhiteSpace(existing) ? notice : $"{existing}\n\n{notice}"));
    }

    private static JsonObject ReadMeta(PenumbraModInfo mod)
    {
        var metaPath = Path.Combine(mod.ModRoot, "meta.json");
        if (!File.Exists(metaPath)) throw new FileNotFoundException("The selected mod has no meta.json.", metaPath);
        return JsonNode.Parse(File.ReadAllText(metaPath), documentOptions: new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip })?.AsObject()
            ?? throw new InvalidDataException("The selected mod meta.json is not a JSON object.");
    }

    private static void CollectHumanPapRaceCodes(JsonNode node, HashSet<string> output)
    {
        if (node is JsonArray array)
        {
            foreach (var child in array) if (child is not null) CollectHumanPapRaceCodes(child, output);
            return;
        }
        if (node is not JsonObject obj) return;

        foreach (var pair in obj)
        {
            if ((string.Equals(pair.Key, "Files", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(pair.Key, "FileSwaps", StringComparison.OrdinalIgnoreCase))
                && pair.Value is JsonObject mappings)
            {
                foreach (var mapping in mappings)
                {
                    var normalized = mapping.Key.Replace('\\', '/');
                    var match = HumanPapPathRegex().Match(normalized);
                    if (match.Success && XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(normalized)
                        && CharacterRaceCatalog.FromCode(match.Groups["race"].Value) is not null)
                    {
                        output.Add(match.Groups["race"].Value);
                    }
                }
                continue;
            }
            if (pair.Value is not null) CollectHumanPapRaceCodes(pair.Value, output);
        }
    }

    private static void RewriteAnimationMappings(JsonNode node, string sourceRaceCode, string targetRaceCode, Dictionary<string, SelectedAnimationSource> selectedAnimations)
    {
        if (node is JsonArray array)
        {
            foreach (var child in array) if (child is not null) RewriteAnimationMappings(child, sourceRaceCode, targetRaceCode, selectedAnimations);
            return;
        }
        if (node is not JsonObject obj) return;

        var filesProperty = obj.FirstOrDefault(pair => string.Equals(pair.Key, "Files", StringComparison.OrdinalIgnoreCase));
        var files = filesProperty.Value as JsonObject;
        if (!string.IsNullOrWhiteSpace(filesProperty.Key) && files is null)
            throw new InvalidDataException("Penumbra animation container has a Files property that is not an object.");
        if (files is not null)
            RewriteFilesMap(files, sourceRaceCode, targetRaceCode, selectedAnimations);

        var swapsProperty = obj.FirstOrDefault(pair => string.Equals(pair.Key, "FileSwaps", StringComparison.OrdinalIgnoreCase));
        if (!string.IsNullOrWhiteSpace(swapsProperty.Key) && swapsProperty.Value is not JsonObject)
            throw new InvalidDataException("Penumbra animation container has a FileSwaps property that is not an object.");
        if (swapsProperty.Value is JsonObject swaps)
        {
            if (files is null)
            {
                files = new JsonObject();
                obj[string.IsNullOrWhiteSpace(filesProperty.Key) ? "Files" : filesProperty.Key] = files;
            }
            RewriteFileSwapsMap(files, swaps, sourceRaceCode, targetRaceCode, selectedAnimations);
        }

        foreach (var pair in obj.ToArray())
        {
            if (string.Equals(pair.Key, "Files", StringComparison.OrdinalIgnoreCase)
                || string.Equals(pair.Key, "FileSwaps", StringComparison.OrdinalIgnoreCase))
                continue;
            if (pair.Value is not null) RewriteAnimationMappings(pair.Value, sourceRaceCode, targetRaceCode, selectedAnimations);
        }
    }

    private static void RewriteFilesMap(JsonObject files, string sourceRaceCode, string targetRaceCode, Dictionary<string, SelectedAnimationSource> selectedAnimations)
    {
        var changes = new List<(string OldKey, string? NewKey, JsonNode? Value)>();
        foreach (var pair in files.ToArray())
        {
            var normalizedKey = pair.Key.Replace('\\', '/');
            var match = HumanPapPathRegex().Match(normalizedKey);
            if (!match.Success || !XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(normalizedKey)) continue;

            var race = match.Groups["race"].Value;
            if (!string.Equals(race, sourceRaceCode, StringComparison.OrdinalIgnoreCase))
            {
                // This copy is target-specific, so drop other player race/gender PAP mappings.
                changes.Add((pair.Key, null, null));
                continue;
            }

            if (pair.Value is not JsonValue value || !value.TryGetValue<string>(out var sourceRelative) || string.IsNullOrWhiteSpace(sourceRelative))
                throw new InvalidDataException($"Animation mapping '{pair.Key}' did not point to a physical PAP file.");

            var generatedRelative = GetGeneratedPapRelativePath(sourceRelative, targetRaceCode);
            RegisterSelectedAnimation(selectedAnimations, $"mod:{sourceRelative}", new SelectedAnimationSource(AnimationSourceKind.ModFile, sourceRelative, sourceRaceCode, generatedRelative));
            var newKey = RewriteHumanAnimationRace(normalizedKey, targetRaceCode);
            changes.Add((pair.Key, newKey, JsonValue.Create(generatedRelative)));
        }

        ApplyMappingChanges(files, changes);
    }

    private static void RewriteFileSwapsMap(JsonObject files, JsonObject swaps, string sourceRaceCode, string targetRaceCode, Dictionary<string, SelectedAnimationSource> selectedAnimations)
    {
        var removals = new List<string>();
        var additions = new List<(string Key, JsonNode Value)>();
        foreach (var pair in swaps.ToArray())
        {
            var originalGamePath = pair.Key.Replace('\\', '/');
            var originalMatch = HumanPapPathRegex().Match(originalGamePath);
            if (!originalMatch.Success || !XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(originalGamePath)) continue;

            removals.Add(pair.Key);
            var targetedRace = originalMatch.Groups["race"].Value;
            if (!string.Equals(targetedRace, sourceRaceCode, StringComparison.OrdinalIgnoreCase))
                continue;

            if (pair.Value is not JsonValue value || !value.TryGetValue<string>(out var actualRaw) || string.IsNullOrWhiteSpace(actualRaw))
                throw new InvalidDataException($"Animation file swap '{pair.Key}' did not point at a vanilla game PAP.");
            var actualGamePath = actualRaw.Replace('\\', '/').TrimStart('/');
            var actualMatch = HumanPapPathRegex().Match(actualGamePath);
            if (!actualMatch.Success || !XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(actualGamePath))
                throw new NotSupportedException($"Animation file swap '{pair.Key}' points at '{actualGamePath}', which is not a supported player PAP.");

            var actualSourceRaceCode = actualMatch.Groups["race"].Value;
            if (CharacterRaceCatalog.FromCode(actualSourceRaceCode) is null)
                throw new NotSupportedException($"Animation file swap '{pair.Key}' uses unsupported source race c{actualSourceRaceCode}.");

            var generatedRelative = GetGeneratedPapRelativePath($"game:{actualGamePath}", targetRaceCode);
            RegisterSelectedAnimation(selectedAnimations, $"game:{actualGamePath}", new SelectedAnimationSource(AnimationSourceKind.GameFile, actualGamePath, actualSourceRaceCode, generatedRelative));
            additions.Add((RewriteHumanAnimationRace(originalGamePath, targetRaceCode), JsonValue.Create(generatedRelative)!));
        }

        foreach (var key in removals) swaps.Remove(key);
        foreach (var addition in additions)
            SetMappingChecked(files, addition.Key, addition.Value);
    }

    private static void RegisterSelectedAnimation(Dictionary<string, SelectedAnimationSource> selectedAnimations, string sourceId, SelectedAnimationSource source)
    {
        if (selectedAnimations.TryGetValue(sourceId, out var existing))
        {
            if (!string.Equals(existing.TargetRelative, source.TargetRelative, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(existing.SourceRaceCode, source.SourceRaceCode, StringComparison.OrdinalIgnoreCase)
                || existing.Kind != source.Kind)
                throw new InvalidDataException($"Animation source '{source.SourceReference}' resolved to conflicting target conversions.");
            return;
        }
        selectedAnimations[sourceId] = source;
    }

    private static void ApplyMappingChanges(JsonObject mappings, IReadOnlyList<(string OldKey, string? NewKey, JsonNode? Value)> changes)
    {
        foreach (var change in changes)
        {
            mappings.Remove(change.OldKey);
            if (change.NewKey is null || change.Value is null) continue;
            SetMappingChecked(mappings, change.NewKey, change.Value);
        }
    }

    private static void SetMappingChecked(JsonObject mappings, string key, JsonNode value)
    {
        if (mappings.TryGetPropertyValue(key, out var existing) && existing is not null
            && !string.Equals(existing.ToJsonString(), value.ToJsonString(), StringComparison.Ordinal))
            throw new InvalidDataException($"Two source animation mappings would both become '{key}'. Choose a different source animation set.");
        mappings[key] = value;
    }

    private static string RewriteHumanAnimationRace(string gamePath, string targetRaceCode)
    {
        var normalized = gamePath.Replace('\\', '/');
        var match = HumanPapPathRegex().Match(normalized);
        if (!match.Success) return normalized;
        return normalized[..match.Groups["race"].Index] + targetRaceCode + normalized[(match.Groups["race"].Index + match.Groups["race"].Length)..];
    }

    private static string GetGeneratedPapRelativePath(string sourceRelative, string targetRaceCode)
    {
        var normalized = sourceRelative.Replace('\\', '/').TrimStart('/');
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(normalized.ToLowerInvariant()))).ToLowerInvariant();
        var fileName = Path.GetFileName(normalized);
        if (string.IsNullOrWhiteSpace(fileName) || !fileName.EndsWith(".pap", StringComparison.OrdinalIgnoreCase)) fileName = "animation.pap";
        return $"RavaFit/AnimationPort/{targetRaceCode}/{hash[..24]}-{fileName}";
    }

    private static void CopyDirectory(string source, string destination, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(destination);
        foreach (var directory in Directory.EnumerateDirectories(source, "*", SearchOption.AllDirectories))
        {
            cancellationToken.ThrowIfCancellationRequested();
            Directory.CreateDirectory(Path.Combine(destination, Path.GetRelativePath(source, directory)));
        }
        foreach (var file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var relative = Path.GetRelativePath(source, file);
            if (relative.EndsWith(".ravafit-last.bak", StringComparison.OrdinalIgnoreCase)
                || Path.GetFileName(relative).Contains(".ravafit-", StringComparison.OrdinalIgnoreCase) && relative.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            var target = Path.Combine(destination, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(file, target, false);
        }
    }

    private static string SafeResolve(string root, string relative)
    {
        var fullRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)) + Path.DirectorySeparatorChar;
        var full = Path.GetFullPath(Path.Combine(root, relative));
        if (!full.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException($"Mod file path escapes its folder: {relative}");
        return full;
    }

    private static string SanitizeDirectoryName(string value)
    {
        foreach (var c in Path.GetInvalidFileNameChars()) value = value.Replace(c, '_');
        return value.Trim().TrimEnd('.');
    }

    private static string? ReadString(JsonObject obj, string name)
    {
        foreach (var pair in obj)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase) && pair.Value is JsonValue value && value.TryGetValue<string>(out var result))
                return result;
        return null;
    }

    private static int? ReadInt(JsonObject obj, string name)
    {
        foreach (var pair in obj)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase) && pair.Value is JsonValue value && value.TryGetValue<int>(out var result))
                return result;
        return null;
    }

    private static void SetProperty(JsonObject obj, string name, JsonNode? value)
    {
        var existing = obj.FirstOrDefault(pair => string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase)).Key;
        obj[string.IsNullOrWhiteSpace(existing) ? name : existing] = value;
    }
}
