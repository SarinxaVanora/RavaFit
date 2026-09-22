using System.Text.Json;
using Dalamud.Plugin.Services;
using RavaFit.Core.Models;
using RavaFit.Core.Penumbra;
using RavaFit.ModelBridge;

namespace RavaFit.Services;

internal sealed class ConversionService
{
    private readonly PenumbraService _penumbra;
    private readonly IDataManager _dataManager;
    private readonly SolverHostService _solver;
    private readonly ModelBridgeService _bridge;
    private readonly BodyLibraryService _bodies;
    private readonly IPluginLog _log;
    private readonly GeneratedModelStore _store = new();
    private readonly PenumbraV4Writer _writer = new();
    private readonly SemaphoreSlim _conversionGate = new(1, 1);
    private readonly object _coverageCacheLock = new();
    private readonly Dictionary<string, GarmentCoverageResult> _coverageCache = new(StringComparer.OrdinalIgnoreCase);

    private sealed record TargetBodyArtifacts(Dictionary<string, string> Glbs, Dictionary<string, string> Mdls);
    private sealed record TargetBodyArtifact(string Glb, string Mdl);
    private sealed record VanillaSourceBodyArtifact(string Glb, string Mdl, string GamePath, string RaceCode);

    public ConversionService(PenumbraService penumbra, IDataManager dataManager, SolverHostService solver, ModelBridgeService bridge, BodyLibraryService bodies, IPluginLog log)
    {
        _penumbra = penumbra;
        _dataManager = dataManager;
        _solver = solver;
        _bridge = bridge;
        _bodies = bodies;
        _log = log;
    }

    public bool Busy => _conversionGate.CurrentCount == 0;
    public ConversionProgress Progress { get; private set; } = new(ConversionStage.Idle, 0);
    public string LastError { get; private set; } = string.Empty;
    public event Action? ProgressChanged;

    public async Task<GarmentCoverageResult> AnalyseCoverageAsync(PenumbraModInfo mod, string groupKey, string optionKey, ModelRedirect model, CancellationToken cancellationToken = default, bool detectSourceBodies = true)
    {
        if (!_solver.Ready)
            throw new InvalidOperationException("RavaFit is still getting ready.");
        RequireBridgeAvailable();

        var metaPath = Path.Combine(mod.ModRoot, "meta.json");
        var document = PenumbraV4Document.Load(metaPath);
        var selectedGroup = document.GetGroup(groupKey);
        var selectedOption = document.GetOption(selectedGroup, optionKey);
        var physicalMdl = document.ResolvePhysicalPath(model);
        if (!File.Exists(physicalMdl))
            throw new FileNotFoundException("The selected Penumbra model redirection does not exist.", physicalMdl);

        var info = new FileInfo(physicalMdl);
        var metaInfo = new FileInfo(metaPath);
        var cacheKey = string.Join("|", mod.Directory, groupKey, optionKey, selectedGroup.Name, selectedOption.Name, model.GamePath, info.Length, info.LastWriteTimeUtc.Ticks, metaInfo.LastWriteTimeUtc.Ticks, _bodies.CatalogueStamp, detectSourceBodies ? "source-detect" : "coverage-only");
        lock (_coverageCacheLock)
        {
            if (_coverageCache.TryGetValue(cacheKey, out var cached))
                return cached;
        }

        await _conversionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        string? work = null;
        try
        {
            // Recheck after the wait in case another request already filled the cache.
            lock (_coverageCacheLock)
            {
                if (_coverageCache.TryGetValue(cacheKey, out var cached))
                    return cached;
            }

            work = CreateWorkDirectory("coverage");
            var sourceGlb = Path.Combine(work, "source.glb");
            var activeSettings = _penumbra.GetCurrentOptionSettings(mod);
            var estOverrides = document.GetEstSkeletonOverrides(groupKey, optionKey, model.GamePath, activeSettings)
                .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source))
                .ToArray();
            var exportRequest = new ModelExportRequest(model.GamePath, physicalMdl, sourceGlb, estOverrides);
            var skeletonPaths = _bridge.ResolveSkeletonPaths(exportRequest);
            await _bridge.ExportAsync(exportRequest with { SkeletonPathsOverride = skeletonPaths }, cancellationToken).ConfigureAwait(false);
            if (!File.Exists(sourceGlb) || new FileInfo(sourceGlb).Length == 0)
                throw new InvalidDataException("RavaFit couldn't prepare the selected model for fitting.");

            using var reply = await _solver.CallAsync("analyze_coverage", new { glb = sourceGlb, game_path = model.GamePath }, cancellationToken).ConfigureAwait(false);
            var root = reply.RootElement;
            var physicalBodySlot = TryResolveBodyModelSlot(model.GamePath);
            var primary = root.TryGetProperty("primary_slot", out var primaryNode) && primaryNode.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(primaryNode.GetString())
                ? primaryNode.GetString()!
                : physicalBodySlot ?? throw new InvalidDataException($"RavaFit couldn't work out which body shape {model.FileName} should fit against.");
            var slots = new Dictionary<string, GarmentCoverageSlot>(StringComparer.OrdinalIgnoreCase);
            if (root.TryGetProperty("slots", out var slotsNode) && slotsNode.ValueKind == JsonValueKind.Object)
            {
                foreach (var slot in BodySlots.All)
                {
                    if (!slotsNode.TryGetProperty(slot, out var slotNode) || slotNode.ValueKind != JsonValueKind.Object)
                        continue;
                    var recommended = slotNode.TryGetProperty("recommended", out var recommendedNode)
                        && (recommendedNode.ValueKind is JsonValueKind.True or JsonValueKind.False)
                        && recommendedNode.GetBoolean();
                    var isPrimary = slotNode.TryGetProperty("primary", out var primaryFlagNode)
                        && (primaryFlagNode.ValueKind is JsonValueKind.True or JsonValueKind.False)
                        && primaryFlagNode.GetBoolean();
                    var confidence = slotNode.TryGetProperty("confidence", out var confidenceNode) && confidenceNode.TryGetSingle(out var value) ? value : 0f;
                    var reason = slotNode.TryGetProperty("reason", out var reasonNode) && reasonNode.ValueKind == JsonValueKind.String ? reasonNode.GetString() ?? string.Empty : string.Empty;
                    slots[slot] = new GarmentCoverageSlot(slot, recommended, isPrimary, confidence, reason);
                }
            }

            foreach (var slot in BodySlots.All)
                if (!slots.ContainsKey(slot))
                    slots[slot] = new GarmentCoverageSlot(slot, string.Equals(slot, primary, StringComparison.OrdinalIgnoreCase), string.Equals(slot, primary, StringComparison.OrdinalIgnoreCase), string.Equals(slot, primary, StringComparison.OrdinalIgnoreCase) ? 1f : 0f, string.Equals(slot, primary, StringComparison.OrdinalIgnoreCase) ? "Primary XIV model slot" : "No strong interaction detected");

            var sourceMatches = new Dictionary<string, SourceBodyMatch>(StringComparer.OrdinalIgnoreCase);
            if (detectSourceBodies)
            {
                try
                {
                    var requiredSlots = BodySlots.All.Where(slot => slots[slot].Primary || slots[slot].Recommended).ToArray();
                    var preferredRaceCode = ResolveRaceCode(model.GamePath);
                    var preferredGender = CharacterRaceCatalog.FromCode(preferredRaceCode)?.Gender;
                    var candidates = requiredSlots.SelectMany(slot => _bodies.ForSlot(slot)).Where(variant => variant.SupportsGender(preferredGender)).ToArray();
                    if (candidates.Length > 0)
                    {
                        using var sourceReply = await _solver.CallAsync("detect_source_bodies", new
                        {
                            glb = sourceGlb,
                            game_path = model.GamePath,
                            preferred_race_code = preferredRaceCode,
                            required_slots = requiredSlots,
                            primary_slot = primary,
                            source_context = new
                            {
                                mod = mod.Name,
                                group = selectedGroup.Name,
                                option = selectedOption.Name,
                            },
                            candidates = candidates.Select(v => new
                            {
                                rbody = v.RBodyPath,
                                collection = v.Collection,
                                body_id = v.BodyId,
                                body = v.BodyName,
                                slot = v.Slot,
                                variant_id = v.VariantId,
                                variant = v.VariantName,
                                race_codes = v.RaceCodes,
                            }).ToArray(),
                        }, cancellationToken).ConfigureAwait(false);
                        var sourceRoot = sourceReply.RootElement;
                        if (sourceRoot.TryGetProperty("matches", out var matchesNode) && matchesNode.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var matchNode in matchesNode.EnumerateArray())
                            {
                                if (!matchNode.TryGetProperty("slot", out var slotNode) || slotNode.ValueKind != JsonValueKind.String
                                    || !matchNode.TryGetProperty("rbody", out var rbodyNode) || rbodyNode.ValueKind != JsonValueKind.String
                                    || !matchNode.TryGetProperty("body_id", out var bodyNode) || bodyNode.ValueKind != JsonValueKind.String
                                    || !matchNode.TryGetProperty("variant_id", out var variantNode) || variantNode.ValueKind != JsonValueKind.String)
                                    continue;
                                var slot = slotNode.GetString()!;
                                var variant = _bodies.ResolveVariant(rbodyNode.GetString()!, bodyNode.GetString()!, slot, variantNode.GetString()!);
                                if (variant is null)
                                    continue;
                                var confidence = matchNode.TryGetProperty("confidence", out var confidenceNode) && confidenceNode.TryGetSingle(out var confidenceValue) ? confidenceValue : 0f;
                                var reason = matchNode.TryGetProperty("reason", out var reasonNode) && reasonNode.ValueKind == JsonValueKind.String ? reasonNode.GetString() ?? string.Empty : string.Empty;
                                sourceMatches[slot] = new SourceBodyMatch(slot, variant, confidence, reason);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Source inference helps, but it is not required for conversion.
                    _log.Debug(ex, "Automatic source-body matching could not resolve {GamePath}", model.GamePath);
                }
            }

            var sourceContainsBody = !root.TryGetProperty("source_contains_body", out var sourceContainsBodyNode)
                || sourceContainsBodyNode.ValueKind != JsonValueKind.False;
            var result = new GarmentCoverageResult(primary, slots, sourceMatches, sourceContainsBody);
            lock (_coverageCacheLock)
                _coverageCache[cacheKey] = result;
            return result;
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(work))
            {
                try { Directory.Delete(work, true); }
                catch (Exception ex) { _log.Debug(ex, "RavaFit could not remove coverage-analysis work directory {Directory}", work); }
            }
            _conversionGate.Release();
        }
    }

    public async Task<V4AppendResult> RunRoundTripAsync(RoundTripRequest request, CancellationToken cancellationToken = default)
    {
        await _conversionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            LastError = string.Empty;
            RequireBridgeAvailable();
            SetProgress(ConversionStage.Preparing, 0.05f, "Preparing");
            var work = CreateWorkDirectory("roundtrip");
            var glb = Path.Combine(work, "source.glb");

            var document = PenumbraV4Document.Load(Path.Combine(request.ModRoot, "meta.json"));
            var mod = new PenumbraModInfo(request.ModDirectory, request.ModName, request.ModRoot);
            var activeSettings = _penumbra.GetCurrentOptionSettings(mod);
            var estOverrides = document.GetEstSkeletonOverrides(request.GroupKey, request.SourceOptionKey, request.GamePath, activeSettings)
                .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source))
                .ToArray();
            var exportRequest = new ModelExportRequest(request.GamePath, request.SourceMdlPath, glb, estOverrides);
            var skeletonPaths = _bridge.ResolveSkeletonPaths(exportRequest);

            SetProgress(ConversionStage.Exporting, 0.25f, skeletonPaths.Count > 1 ? "Exporting model + supplemental skeleton" : "Exporting model");
            await _bridge.ExportAsync(exportRequest with { SkeletonPathsOverride = skeletonPaths }, cancellationToken).ConfigureAwait(false);
            if (!File.Exists(glb) || new FileInfo(glb).Length == 0)
                throw new InvalidDataException("Model bridge did not produce a GLB.");

            SetProgress(ConversionStage.Building, 0.55f, "Building model");
            var mdl = await _bridge.ImportAsync(new ModelImportRequest(request.GamePath, request.SourceMdlPath, glb), cancellationToken).ConfigureAwait(false);
            var generated = await _store.WriteAsync(request.ModRoot, request.GamePath, request.OutputOptionName, mdl, cancellationToken).ConfigureAwait(false);

            SetProgress(ConversionStage.UpdatingPenumbra, 0.80f, "Updating Penumbra");
            V4AppendResult result;
            try
            {
                result = await _writer.AppendClonedOptionAsync(new V4AppendRequest(
                    Path.Combine(request.ModRoot, "meta.json"),
                    request.GroupKey,
                    request.SourceOptionKey,
                    request.OutputOptionName,
                    new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        [generated.GamePath] = generated.RelativePath,
                    }), cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                DeleteUnreferencedGeneratedFile(generated);
                throw;
            }

            var reloadResult = await _penumbra.ReloadAsync(mod, cancellationToken).ConfigureAwait(false);
            if (!reloadResult.Success)
                throw new InvalidOperationException($"The new option was written safely, but Penumbra reload failed: {reloadResult.Error}");

            SetProgress(ConversionStage.Complete, 1.0f, "Complete");
            return result;
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            SetProgress(ConversionStage.Failed, 1.0f, "Failed");
            _log.Error(ex, "RavaFit round-trip failed.");
            throw;
        }
        finally
        {
            _conversionGate.Release();
        }
    }

    public async Task<V4AppendResult> RunConversionAsync(ConversionRequest request, CancellationToken cancellationToken = default)
    {
        await _conversionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            LastError = string.Empty;
            RequireBridgeAvailable();
            if (!_solver.Ready || !_solver.ConversionReady)
                throw new InvalidOperationException("Solver conversion engine is not ready.");
            if (request.Models.Count != 1)
                throw new NotSupportedException("The first vertical slice converts one MDL at a time. Multi-MDL transactions come after the single-model path is verified.");
            if (request.Slots.Count == 0)
                throw new InvalidOperationException("Select at least one body slot.");

            var model = request.Models[0];
            var document = PenumbraV4Document.Load(Path.Combine(request.ModRoot, "meta.json"));
            var physicalMdl = document.ResolvePhysicalPath(model);
            if (!File.Exists(physicalMdl))
                throw new FileNotFoundException("The selected Penumbra model redirection does not exist.", physicalMdl);

            var physicalBodySlot = TryResolveBodyModelSlot(model.GamePath);
            var modelSlot = ResolvePrimaryBodySlot(model.GamePath, request.PrimaryBodySlot);
            var modelRaceCode = ResolveRaceCode(model.GamePath);
            // Accessory slots are fit-only; coverage chooses their body support region.
            var slots = request.Slots.ToArray();
            var useVanillaSourceBody = slots.Length > 0 && slots.All(selection => selection.Source is null);
            var (sourceContainsBody, transplantTargetBody, fitOnly) = ResolveBodyOutputPolicy(model.GamePath, request.SourceContainsBody, request.TransplantTargetBody);
            ValidateStandardBodySelections(slots, modelRaceCode, useVanillaSourceBody);
            if (slots.Any(selection => IsResolvedLalafellTarget(selection.Target, modelRaceCode)))
            {
                var guard = LalafellContentGuard.InspectOutfit(
                    new PenumbraModInfo(request.ModDirectory, request.ModName, request.ModRoot),
                    [(request.GroupKey, request.SourceOptionKey)],
                    slots.SelectMany(selection => new BodyVariantInfo?[] { selection.Source, selection.Target }).OfType<BodyVariantInfo>());
                if (guard.Blocked)
                    throw new NotSupportedException("RavaFit will not create an explicit/NSFW Lalafell outfit conversion. Evidence: " + string.Join("; ", guard.Reasons));
            }
            if (!slots.Any(s => string.Equals(s.Slot, modelSlot, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException($"The selected {modelSlot} model requires a {modelSlot} source and target body selection.");

            var work = CreateWorkDirectory("convert");
            var sourceGlb = Path.Combine(work, "source.glb");
            var solvedGlb = Path.Combine(work, "solved.glb");
            var specPath = Path.Combine(work, "conversion.json");

            var penumbraMod = new PenumbraModInfo(request.ModDirectory, request.ModName, request.ModRoot);
            var activeSettings = _penumbra.GetCurrentOptionSettings(penumbraMod);
            var estOverrides = document.GetEstSkeletonOverrides(request.GroupKey, request.SourceOptionKey, model.GamePath, activeSettings)
                .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source))
                .ToArray();
            var sourceExport = new ModelExportRequest(model.GamePath, physicalMdl, sourceGlb, estOverrides);
            var skeletonPaths = _bridge.ResolveSkeletonPaths(sourceExport);

            SetProgress(ConversionStage.Exporting, 0.10f, skeletonPaths.Count > 1 ? "Exporting model + supplemental skeleton" : "Exporting model");
            await _bridge.ExportAsync(sourceExport with { SkeletonPathsOverride = skeletonPaths }, cancellationToken).ConfigureAwait(false);

            SetProgress(ConversionStage.Preparing, 0.20f, "Preparing target body regions");
            var targetBodies = await PrepareTargetBodiesAsync(slots, work, modelRaceCode, estOverrides, skeletonPaths, null, cancellationToken).ConfigureAwait(false);
            var vanillaSourceBodies = useVanillaSourceBody
                ? await PrepareVanillaSourceBodiesAsync(slots, work, modelRaceCode, estOverrides, skeletonPaths, cancellationToken).ConfigureAwait(false)
                : null;

            var spec = new
            {
                game_path = model.GamePath,
                model_slot = modelSlot,
                source_glb = sourceGlb,
                source_body_glbs = vanillaSourceBodies?.ToDictionary(pair => pair.Key, pair => pair.Value.Glb, StringComparer.OrdinalIgnoreCase),
                target_body_glbs = targetBodies.Glbs,
                output_glb = solvedGlb,
                source_body_mode = useVanillaSourceBody ? "vanilla" : "rbody",
                source_contains_body = sourceContainsBody,
                transplant_target_body = transplantTargetBody,
                fit_only = fitOnly,
                slots = slots.Select(s => new
                {
                    slot = s.Slot,
                    source_mode = useVanillaSourceBody ? "vanilla" : "rbody",
                    source_race_code = useVanillaSourceBody ? vanillaSourceBodies![s.Slot].RaceCode : ResolveBodyRaceCode(s.Source!, modelRaceCode),
                    target_race_code = ResolveBodyRaceCode(s.Target, modelRaceCode),
                    target_support_surface = s.Target.IsSmallclothesSupport ? "smallclothes" : "body",
                    source = useVanillaSourceBody || s.Source is null ? null : new { rbody = s.Source!.RBodyPath, body = s.Source!.BodyId, variant = s.Source!.VariantId },
                    target = new { rbody = s.Target.RBodyPath, body = s.Target.BodyId, variant = s.Target.VariantId },
                }).ToArray(),
            };
            await File.WriteAllTextAsync(specPath, JsonSerializer.Serialize(spec, new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);

            SetProgress(ConversionStage.Fitting, 0.35f, "Fitting");
            using var reply = await _solver.CallAsync("convert", new { spec = specPath }, cancellationToken).ConfigureAwait(false);
            if (!reply.RootElement.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
                throw new InvalidOperationException("Solver conversion did not complete successfully.");
            if (!File.Exists(solvedGlb))
                throw new FileNotFoundException("Solver did not produce solved.glb.", solvedGlb);

            SetProgress(ConversionStage.Building, 0.68f, transplantTargetBody ? "Building model + restoring native body" : "Building garment-only model");
            var mdl = await _bridge.ImportAsync(new ModelImportRequest(model.GamePath, physicalMdl, solvedGlb), cancellationToken).ConfigureAwait(false);
            if (transplantTargetBody)
                mdl = await RestoreNativeTargetBodyAsync(mdl, physicalMdl, work, reply.RootElement, targetBodies.Mdls, cancellationToken).ConfigureAwait(false);
            var generated = await _store.WriteAsync(request.ModRoot, model.GamePath, request.OutputOptionName, mdl, cancellationToken).ConfigureAwait(false);
            var generatedSupportFiles = new List<string>();

            SetProgress(ConversionStage.UpdatingPenumbra, 0.84f, transplantTargetBody ? "Updating Penumbra + target body options" : "Updating Penumbra");
            V4AppendResult result;
            try
            {
                var targetBodyGroups = Array.Empty<V4TargetBodyGroupRequest>();
                if (transplantTargetBody && physicalBodySlot is not null)
                {
                    var primarySelection = slots.Single(selection => string.Equals(selection.Slot, modelSlot, StringComparison.OrdinalIgnoreCase));
                    var preparedBodyOptions = await _bodies.PrepareTargetBodyOptionsAsync(primarySelection.Target, modelSlot, ResolveBodyRaceCode(primarySelection.Target, modelRaceCode), request.ModRoot, cancellationToken).ConfigureAwait(false);
                    generatedSupportFiles.AddRange(preparedBodyOptions.CreatedFiles);
                    targetBodyGroups = preparedBodyOptions.Groups.Select(group => new V4TargetBodyGroupRequest(
                        request.GroupKey, request.OutputOptionName, group.BodyName, group.Slot, model.GamePath, group.SourceKey, group.Group)).ToArray();
                }
                var append = new V4AppendRequest(
                    Path.Combine(request.ModRoot, "meta.json"), request.GroupKey, request.SourceOptionKey, request.OutputOptionName,
                    new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { [generated.GamePath] = generated.RelativePath });
                result = targetBodyGroups.Length > 0
                    ? (await _writer.AppendClonedOptionsWithTargetBodyGroupsAsync([append], targetBodyGroups, cancellationToken).ConfigureAwait(false)).Single()
                    : await _writer.AppendClonedOptionAsync(append, cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                DeleteUnreferencedGeneratedFile(generated);
                foreach (var support in generatedSupportFiles) DeleteUnreferencedSupportFile(support);
                throw;
            }
            var reloadResult = await _penumbra.ReloadAsync(penumbraMod, cancellationToken).ConfigureAwait(false);
            if (!reloadResult.Success)
                throw new InvalidOperationException($"Conversion was written safely, but Penumbra reload failed: {reloadResult.Error}");

            SetProgress(ConversionStage.Complete, 1.0f, "Complete");
            return result;
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            SetProgress(ConversionStage.Failed, 1.0f, "Failed");
            _log.Error(ex, "RavaFit conversion failed.");
            throw;
        }
        finally
        {
            _conversionGate.Release();
        }
    }


    public async Task<IReadOnlyList<V4AppendResult>> RunOutfitConversionAsync(IReadOnlyList<ConversionRequest> requests, CancellationToken cancellationToken = default)
    {
        await _conversionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        var generatedFiles = new List<GeneratedModelFile>();
        var generatedSupportFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var targetBodyGroups = new List<V4TargetBodyGroupRequest>();
        var metaCommitted = false;
        try
        {
            LastError = string.Empty;
            RequireBridgeAvailable();
            if (!_solver.Ready || !_solver.ConversionReady)
                throw new InvalidOperationException("Solver conversion engine is not ready.");
            if (requests.Count == 0)
                throw new InvalidOperationException("Select at least one outfit row to convert.");

            var request = requests[0];
            if (requests.Any(candidate => !string.Equals(candidate.ModDirectory, request.ModDirectory, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(candidate.ModRoot, request.ModRoot, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("All outfit rows must belong to the same Penumbra mod.");

            var workItems = requests.SelectMany(candidate => candidate.Models.Select(model => (Request: candidate, Model: model))).ToArray();
            if (workItems.Length == 0)
                throw new InvalidOperationException("Select at least one outfit model to convert.");

            var document = PenumbraV4Document.Load(Path.Combine(request.ModRoot, "meta.json"));
            var penumbraMod = new PenumbraModInfo(request.ModDirectory, request.ModName, request.ModRoot);
            var standardTargetsLalafell = requests.Any(candidate => candidate.Slots.Any(selection =>
            {
                var preferredRaceCode = selection.RaceCode ?? candidate.Models.Select(model => TryResolveRaceCode(model.GamePath)).FirstOrDefault(code => !string.IsNullOrWhiteSpace(code));
                return !string.IsNullOrWhiteSpace(preferredRaceCode) && IsResolvedLalafellTarget(selection.Target, preferredRaceCode!);
            }));
            if (standardTargetsLalafell)
            {
                var guard = LalafellContentGuard.InspectOutfit(
                    penumbraMod,
                    requests.Select(candidate => (candidate.GroupKey, candidate.SourceOptionKey)).Distinct().ToArray(),
                    requests.SelectMany(candidate => candidate.Slots).SelectMany(selection => new BodyVariantInfo?[] { selection.Source, selection.Target }).OfType<BodyVariantInfo>());
                if (guard.Blocked)
                    throw new NotSupportedException("RavaFit will not create an explicit/NSFW Lalafell outfit conversion. Evidence: " + string.Join("; ", guard.Reasons));
            }
            var activeSettings = _penumbra.GetCurrentOptionSettings(penumbraMod);
            var appendMaps = new Dictionary<string, (string GroupKey, string SourceOptionKey, string OutputOptionName, Dictionary<string, string> Redirects)>(StringComparer.OrdinalIgnoreCase);
            var targetBodyArtifactCache = new Dictionary<string, TargetBodyArtifact>(StringComparer.Ordinal);

            for (var index = 0; index < workItems.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var item = workItems[index];
                var itemRequest = item.Request;
                var model = item.Model;
                if (itemRequest.Slots.Count == 0)
                    throw new InvalidOperationException($"{model.FileName} has no source/target body context.");
                var physicalMdl = document.ResolvePhysicalPath(model);
                if (!File.Exists(physicalMdl))
                    throw new FileNotFoundException("The selected Penumbra model redirection does not exist.", physicalMdl);

                var physicalBodySlot = TryResolveBodyModelSlot(model.GamePath);
                var modelSlot = ResolvePrimaryBodySlot(model.GamePath, itemRequest.PrimaryBodySlot);
                var modelRaceCode = ResolveRaceCode(model.GamePath);
                var slots = itemRequest.Slots.ToArray();
                var useVanillaSourceBody = slots.Length > 0 && slots.All(selection => selection.Source is null);
                var (sourceContainsBody, transplantTargetBody, fitOnly) = ResolveBodyOutputPolicy(model.GamePath, itemRequest.SourceContainsBody, itemRequest.TransplantTargetBody);
                ValidateStandardBodySelections(slots, modelRaceCode, useVanillaSourceBody);
                if (!slots.Any(s => string.Equals(s.Slot, modelSlot, StringComparison.OrdinalIgnoreCase)))
                    throw new InvalidOperationException($"The selected {modelSlot} model requires a {modelSlot} source and target body selection.");

                var fractionStart = (float)index / workItems.Length;
                var fractionEnd = (float)(index + 1) / workItems.Length;
                float Step(float local) => 0.04f + ((fractionStart + ((fractionEnd - fractionStart) * local)) * 0.82f);
                var prefix = $"{index + 1}/{workItems.Length} {model.FileName}";

                var work = CreateWorkDirectory("outfit");
                var sourceGlb = Path.Combine(work, "source.glb");
                var solvedGlb = Path.Combine(work, "solved.glb");
                var specPath = Path.Combine(work, "conversion.json");

                var estOverrides = document.GetEstSkeletonOverrides(itemRequest.GroupKey, itemRequest.SourceOptionKey, model.GamePath, activeSettings)
                    .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source))
                    .ToArray();
                var sourceExport = new ModelExportRequest(model.GamePath, physicalMdl, sourceGlb, estOverrides);
                var skeletonPaths = _bridge.ResolveSkeletonPaths(sourceExport);

                SetProgress(ConversionStage.Exporting, Step(0.08f), $"{prefix} — exporting");
                await _bridge.ExportAsync(sourceExport with { SkeletonPathsOverride = skeletonPaths }, cancellationToken).ConfigureAwait(false);
                if (!File.Exists(sourceGlb) || new FileInfo(sourceGlb).Length == 0)
                    throw new InvalidDataException($"Model bridge did not produce a GLB for {model.FileName}.");

                SetProgress(ConversionStage.Preparing, Step(0.22f), $"{prefix} — preparing body support regions");
                var targetBodies = await PrepareTargetBodiesAsync(slots, work, modelRaceCode, estOverrides, skeletonPaths, targetBodyArtifactCache, cancellationToken).ConfigureAwait(false);
                var vanillaSourceBodies = useVanillaSourceBody
                    ? await PrepareVanillaSourceBodiesAsync(slots, work, modelRaceCode, estOverrides, skeletonPaths, cancellationToken).ConfigureAwait(false)
                    : null;

                var spec = new
                {
                    game_path = model.GamePath,
                    model_slot = modelSlot,
                    source_glb = sourceGlb,
                    source_body_glbs = vanillaSourceBodies?.ToDictionary(pair => pair.Key, pair => pair.Value.Glb, StringComparer.OrdinalIgnoreCase),
                    target_body_glbs = targetBodies.Glbs,
                    output_glb = solvedGlb,
                    source_body_mode = useVanillaSourceBody ? "vanilla" : "rbody",
                    source_contains_body = sourceContainsBody,
                    transplant_target_body = transplantTargetBody,
                    fit_only = fitOnly,
                    slots = slots.Select(slot => new
                    {
                        slot = slot.Slot,
                        source_mode = useVanillaSourceBody ? "vanilla" : "rbody",
                        source_race_code = useVanillaSourceBody ? vanillaSourceBodies![slot.Slot].RaceCode : ResolveBodyRaceCode(slot.Source!, modelRaceCode),
                        target_race_code = ResolveBodyRaceCode(slot.Target, modelRaceCode),
                        target_support_surface = slot.Target.IsSmallclothesSupport ? "smallclothes" : "body",
                        source = useVanillaSourceBody || slot.Source is null ? null : new { rbody = slot.Source!.RBodyPath, body = slot.Source!.BodyId, variant = slot.Source!.VariantId },
                        target = new { rbody = slot.Target.RBodyPath, body = slot.Target.BodyId, variant = slot.Target.VariantId },
                    }).ToArray(),
                };
                await File.WriteAllTextAsync(specPath, JsonSerializer.Serialize(spec, new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);

                SetProgress(ConversionStage.Fitting, Step(0.40f), $"{prefix} — fitting");
                using var reply = await _solver.CallAsync("convert", new { spec = specPath }, cancellationToken).ConfigureAwait(false);
                if (!reply.RootElement.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
                    throw new InvalidOperationException($"Solver conversion did not complete successfully for {model.FileName}.");
                if (!File.Exists(solvedGlb))
                    throw new FileNotFoundException("Solver did not produce solved.glb.", solvedGlb);

                SetProgress(ConversionStage.Building, Step(0.72f), $"{prefix} — {(transplantTargetBody ? "building model + restoring native body" : "building garment-only model")}");
                var mdl = await _bridge.ImportAsync(new ModelImportRequest(model.GamePath, physicalMdl, solvedGlb), cancellationToken).ConfigureAwait(false);
                if (transplantTargetBody)
                    mdl = await RestoreNativeTargetBodyAsync(mdl, physicalMdl, work, reply.RootElement, targetBodies.Mdls, cancellationToken).ConfigureAwait(false);
                var generated = await _store.WriteAsync(request.ModRoot, model.GamePath, itemRequest.OutputOptionName, mdl, cancellationToken).ConfigureAwait(false);
                generatedFiles.Add(generated);

                var appendKey = string.Join("|", itemRequest.GroupKey, itemRequest.SourceOptionKey, itemRequest.OutputOptionName);
                if (!appendMaps.TryGetValue(appendKey, out var append))
                {
                    append = (itemRequest.GroupKey, itemRequest.SourceOptionKey, itemRequest.OutputOptionName, new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase));
                    appendMaps[appendKey] = append;
                }
                if (!append.Redirects.TryAdd(generated.GamePath, generated.RelativePath))
                    throw new InvalidOperationException($"The outfit transaction attempted to generate '{generated.GamePath}' more than once.");

                // Fit-only accessories do not inherit target-body option groups.
                if (transplantTargetBody && physicalBodySlot is not null)
                {
                    var primarySelection = slots.Single(selection => string.Equals(selection.Slot, modelSlot, StringComparison.OrdinalIgnoreCase));
                    var preparedBodyOptions = await _bodies.PrepareTargetBodyOptionsAsync(primarySelection.Target, modelSlot, ResolveBodyRaceCode(primarySelection.Target, modelRaceCode), request.ModRoot, cancellationToken).ConfigureAwait(false);
                    foreach (var created in preparedBodyOptions.CreatedFiles) generatedSupportFiles.Add(created);
                    foreach (var group in preparedBodyOptions.Groups)
                        targetBodyGroups.Add(new V4TargetBodyGroupRequest(
                            itemRequest.GroupKey, itemRequest.OutputOptionName, group.BodyName, group.Slot, model.GamePath, group.SourceKey, group.Group));
                }
            }

            var conflicting = appendMaps.Values
                .GroupBy(x => $"{x.GroupKey}\u001f{x.OutputOptionName}", StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault(group => group.Select(x => x.SourceOptionKey).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1);
            if (conflicting is not null)
                throw new InvalidOperationException($"Selected outfit rows map target '{conflicting.First().OutputOptionName}' from different source options in the same Penumbra group. Choose one source option for that group.");

            SetProgress(ConversionStage.UpdatingPenumbra, 0.90f, "Writing converted outfit options");
            IReadOnlyList<V4AppendResult> results;
            try
            {
                var appendRequests = appendMaps.Values.Select(append => new V4AppendRequest(
                    Path.Combine(request.ModRoot, "meta.json"), append.GroupKey, append.SourceOptionKey, append.OutputOptionName, append.Redirects)).ToArray();
                results = await _writer.AppendClonedOptionsWithTargetBodyGroupsAsync(appendRequests, targetBodyGroups, cancellationToken).ConfigureAwait(false);
                metaCommitted = true;
            }
            catch
            {
                foreach (var generated in generatedFiles)
                    DeleteUnreferencedGeneratedFile(generated);
                foreach (var support in generatedSupportFiles)
                    DeleteUnreferencedSupportFile(support);
                generatedFiles.Clear();
                generatedSupportFiles.Clear();
                throw;
            }

            var reloadResult = await _penumbra.ReloadAsync(penumbraMod, cancellationToken).ConfigureAwait(false);
            if (!reloadResult.Success)
                throw new InvalidOperationException($"The converted outfit was written safely, but Penumbra reload failed: {reloadResult.Error}");

            SetProgress(ConversionStage.Complete, 1.0f, "Complete");
            return results;
        }
        catch (Exception ex)
        {
            if (!metaCommitted)
            {
                foreach (var generated in generatedFiles)
                    DeleteUnreferencedGeneratedFile(generated);
                foreach (var support in generatedSupportFiles)
                    DeleteUnreferencedSupportFile(support);
            }
            LastError = ex.Message;
            SetProgress(ConversionStage.Failed, 1.0f, "Failed");
            _log.Error(ex, "RavaFit outfit conversion failed.");
            throw;
        }
        finally
        {
            _conversionGate.Release();
        }
    }

    public async Task<IReadOnlyList<V4AppendResult>> RunRaceSwapOutfitConversionAsync(IReadOnlyList<RaceSwapConversionRequest> requests, CancellationToken cancellationToken = default)
    {
        await _conversionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        var generatedFiles = new List<GeneratedModelFile>();
        var generatedSupportFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var targetBodyGroups = new List<V4TargetBodyGroupRequest>();
        var metaCommitted = false;
        try
        {
            LastError = string.Empty;
            RequireBridgeAvailable();
            if (!_solver.Ready || !_solver.ConversionReady)
                throw new InvalidOperationException("Solver conversion engine is not ready.");
            if (requests.Count == 0)
                throw new InvalidOperationException("Select at least one outfit row to convert.");

            var request = requests[0];
            if (requests.Any(candidate => !string.Equals(candidate.ModDirectory, request.ModDirectory, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(candidate.ModRoot, request.ModRoot, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("All outfit rows must belong to the same Penumbra mod.");
            if (requests.Any(candidate => !string.Equals(candidate.TargetRaceCode, request.TargetRaceCode, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException("All race-swap rows must use the same target character.");

            var targetIdentity = CharacterRaceCatalog.FromCode(request.TargetRaceCode)
                ?? throw new InvalidOperationException($"Unknown target character code c{request.TargetRaceCode}.");
            var workItems = requests.SelectMany(candidate => candidate.Models.Select(model => (Request: candidate, Model: model))).ToArray();
            if (workItems.Length == 0)
                throw new InvalidOperationException("Select at least one outfit model to convert.");

            var document = PenumbraV4Document.Load(Path.Combine(request.ModRoot, "meta.json"));
            var penumbraMod = new PenumbraModInfo(request.ModDirectory, request.ModName, request.ModRoot);
            if (LalafellContentGuard.IsLalafell(targetIdentity))
            {
                var guard = LalafellContentGuard.InspectOutfit(
                    penumbraMod,
                    requests.Select(candidate => (candidate.GroupKey, candidate.SourceOptionKey)).Distinct().ToArray(),
                    requests.SelectMany(candidate => candidate.Slots).SelectMany(selection => new BodyVariantInfo?[] { selection.Source, selection.Target }).OfType<BodyVariantInfo>());
                if (guard.Blocked)
                    throw new NotSupportedException("RavaFit will not port explicit/NSFW outfit content to Lalafell. Evidence: " + string.Join("; ", guard.Reasons));
            }
            var activeSettings = _penumbra.GetCurrentOptionSettings(penumbraMod);
            var appendMaps = new Dictionary<string, (string GroupKey, string SourceOptionKey, string OutputOptionName, Dictionary<string, string> Redirects)>(StringComparer.OrdinalIgnoreCase);
            var targetBodyArtifactCache = new Dictionary<string, TargetBodyArtifact>(StringComparer.Ordinal);

            for (var index = 0; index < workItems.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var item = workItems[index];
                var itemRequest = item.Request;
                var model = item.Model;
                if (itemRequest.Slots.Count == 0)
                    throw new InvalidOperationException($"{model.FileName} has no source/target body context.");
                var physicalMdl = document.ResolvePhysicalPath(model);
                if (!File.Exists(physicalMdl))
                    throw new FileNotFoundException("The selected Penumbra model redirection does not exist.", physicalMdl);

                var physicalBodySlot = TryResolveBodyModelSlot(model.GamePath);
                var modelSlot = ResolvePrimaryBodySlot(model.GamePath, itemRequest.PrimaryBodySlot);
                var (sourceContainsBody, transplantTargetBody, fitOnly) = ResolveBodyOutputPolicy(model.GamePath, itemRequest.SourceContainsBody, itemRequest.TransplantTargetBody);
                var sourceRaceCode = ResolveRaceCode(model.GamePath);
                var targetGamePath = CharacterRaceCatalog.RewriteHumanRaceCode(model.GamePath, targetIdentity.Code);
                var targetTemplateRedirect = document.GetOptionModelRedirects(itemRequest.GroupKey, itemRequest.SourceOptionKey)
                    .FirstOrDefault(candidate => string.Equals(candidate.GamePath, targetGamePath, StringComparison.OrdinalIgnoreCase));
                var targetTemplateMdl = targetTemplateRedirect is null ? physicalMdl : document.ResolvePhysicalPath(targetTemplateRedirect);
                if (!File.Exists(targetTemplateMdl)) targetTemplateMdl = physicalMdl;
                var slots = itemRequest.Slots.ToArray();
                if (!slots.Any(s => string.Equals(s.Slot, modelSlot, StringComparison.OrdinalIgnoreCase)))
                    throw new InvalidOperationException($"The selected {modelSlot} model requires a {modelSlot} source and target body selection.");
                ValidateRaceSwapBodySelections(slots, sourceRaceCode, targetIdentity);

                var fractionStart = (float)index / workItems.Length;
                var fractionEnd = (float)(index + 1) / workItems.Length;
                float Step(float local) => 0.04f + ((fractionStart + ((fractionEnd - fractionStart) * local)) * 0.82f);
                var prefix = $"{index + 1}/{workItems.Length} {model.FileName}";

                var work = CreateWorkDirectory("race-swap");
                var sourceGlb = Path.Combine(work, "source.glb");
                var sourceRestGlb = Path.Combine(work, "source-rest.glb");
                var solvedGlb = Path.Combine(work, "solved.glb");
                var specPath = Path.Combine(work, "conversion.json");

                var sourceEstOverrides = document.GetEstSkeletonOverrides(itemRequest.GroupKey, itemRequest.SourceOptionKey, model.GamePath, activeSettings)
                    .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source))
                    .ToArray();
                var targetEstOverrides = document.GetEstSkeletonOverrides(itemRequest.GroupKey, itemRequest.SourceOptionKey, targetGamePath, activeSettings)
                    .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source))
                    .ToArray();
                if (!string.Equals(sourceRaceCode, targetIdentity.Code, StringComparison.OrdinalIgnoreCase)
                    && sourceEstOverrides.Any(x => x.Entry > 0)
                    && !targetEstOverrides.Any(x => x.Entry > 0))
                    throw new NotSupportedException($"This outfit uses a race-specific body skeleton but has no matching {targetIdentity.DisplayName} skeleton mapping.");

                var sourceSkeletonRequest = new ModelExportRequest(model.GamePath, physicalMdl, sourceRestGlb, sourceEstOverrides);
                var sourceSkeletonPaths = _bridge.ResolveSkeletonPaths(sourceSkeletonRequest);
                var resolvedSourceSkeletonRequest = sourceSkeletonRequest with { SkeletonPathsOverride = sourceSkeletonPaths };
                var targetSkeletonRequest = new ModelExportRequest(targetGamePath, physicalMdl, sourceGlb, targetEstOverrides);
                var targetSkeletonPaths = _bridge.ResolveSkeletonPaths(targetSkeletonRequest);
                var resolvedTargetSkeletonRequest = targetSkeletonRequest with { SkeletonPathsOverride = targetSkeletonPaths };
                var skeletonRetarget = CharacterRaceCatalog.RequiresSkeletonProportionRetarget(sourceRaceCode, targetIdentity.Code);

                SetProgress(ConversionStage.Preparing, Step(0.06f), $"{prefix} — checking {targetIdentity.DisplayName} skeleton");
                var missingBones = await _bridge.FindMissingBonesAsync(resolvedTargetSkeletonRequest, cancellationToken).ConfigureAwait(false);
                if (missingBones.Count > 0)
                {
                    var shown = string.Join(", ", missingBones.Take(12).Select(name => $"\"{name}\""));
                    var remainder = missingBones.Count > 12 ? $" (+{missingBones.Count - 12} more)" : string.Empty;
                    throw new NotSupportedException(
                        $"Cannot port {model.FileName} to {targetIdentity.DisplayName}: the target armature is missing {missingBones.Count} bone(s) referenced by the source model: {shown}{remainder}. " +
                        "Enable or provide the matching target EST/IVCS skeleton, or choose a compatible target.");
                }

                SetProgress(ConversionStage.Exporting, Step(0.08f), $"{prefix} — reading for {targetIdentity.DisplayName}");
                if (skeletonRetarget)
                {
                    await _bridge.ExportAsync(resolvedSourceSkeletonRequest, cancellationToken).ConfigureAwait(false);
                    if (!File.Exists(sourceRestGlb) || new FileInfo(sourceRestGlb).Length == 0)
                        throw new InvalidDataException($"Model bridge did not produce the source-rest GLB for {model.FileName}.");
                }
                await _bridge.ExportAsync(resolvedTargetSkeletonRequest, cancellationToken).ConfigureAwait(false);
                if (!File.Exists(sourceGlb) || new FileInfo(sourceGlb).Length == 0)
                    throw new InvalidDataException($"Model bridge did not produce a GLB for {model.FileName}.");

                SetProgress(ConversionStage.Preparing, Step(0.22f), $"{prefix} — preparing {targetIdentity.DisplayName}");
                var targetBodies = await PrepareRaceSwapTargetBodiesAsync(slots, work, targetIdentity.Code, targetEstOverrides, targetSkeletonPaths, targetBodyArtifactCache, cancellationToken).ConfigureAwait(false);

                var spec = new
                {
                    game_path = targetGamePath,
                    model_slot = modelSlot,
                    source_glb = sourceGlb,
                    source_rest_glb = skeletonRetarget ? sourceRestGlb : null,
                    race_skeleton_retarget = skeletonRetarget,
                    source_race_code = sourceRaceCode,
                    target_race_code = targetIdentity.Code,
                    target_body_glbs = targetBodies.Glbs,
                    output_glb = solvedGlb,
                    source_contains_body = sourceContainsBody,
                    transplant_target_body = transplantTargetBody,
                    fit_only = fitOnly,
                    slots = slots.Select(slot => new
                    {
                        slot = slot.Slot,
                        source_race_code = ResolveRaceSwapBodyRaceCode(slot.Source!, sourceRaceCode),
                        target_race_code = ResolveRaceSwapBodyRaceCode(slot.Target, targetIdentity.Code),
                        source_support_surface = slot.Source!.IsSmallclothesSupport ? "smallclothes" : "body",
                        target_support_surface = slot.Target.IsSmallclothesSupport ? "smallclothes" : "body",
                        cross_sex_smallclothes_bridge = CharacterRaceCatalog.RequiresCrossSexSmallclothesBridge(CharacterRaceCatalog.FromCode(sourceRaceCode)!, targetIdentity, slot.Slot),
                        source = new { rbody = slot.Source!.RBodyPath, body = slot.Source!.BodyId, variant = slot.Source!.VariantId },
                        target = new { rbody = slot.Target.RBodyPath, body = slot.Target.BodyId, variant = slot.Target.VariantId },
                    }).ToArray(),
                };
                await File.WriteAllTextAsync(specPath, JsonSerializer.Serialize(spec, new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);

                SetProgress(ConversionStage.Fitting, Step(0.40f), $"{prefix} — fitting");
                using var reply = await _solver.CallAsync("convert", new { spec = specPath }, cancellationToken).ConfigureAwait(false);
                if (!reply.RootElement.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
                    throw new InvalidOperationException($"Solver conversion did not complete successfully for {model.FileName}.");
                if (!File.Exists(solvedGlb))
                    throw new FileNotFoundException("Solver did not produce solved.glb.", solvedGlb);

                SetProgress(ConversionStage.Building, Step(0.72f), $"{prefix} — rebuilding for {targetIdentity.DisplayName}");
                var mdl = await _bridge.ImportAsync(new ModelImportRequest(targetGamePath, targetTemplateMdl, solvedGlb), cancellationToken).ConfigureAwait(false);
                if (transplantTargetBody)
                    mdl = await RestoreNativeTargetBodyAsync(mdl, targetTemplateMdl, work, reply.RootElement, targetBodies.Mdls, cancellationToken).ConfigureAwait(false);
                var generated = await _store.WriteAsync(request.ModRoot, targetGamePath, itemRequest.OutputOptionName, mdl, cancellationToken).ConfigureAwait(false);
                generatedFiles.Add(generated);

                var appendKey = string.Join("|", itemRequest.GroupKey, itemRequest.SourceOptionKey, itemRequest.OutputOptionName);
                if (!appendMaps.TryGetValue(appendKey, out var append))
                {
                    append = (itemRequest.GroupKey, itemRequest.SourceOptionKey, itemRequest.OutputOptionName, new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase));
                    appendMaps[appendKey] = append;
                }
                if (!append.Redirects.TryAdd(generated.GamePath, generated.RelativePath))
                    throw new InvalidOperationException($"The race-swap transaction attempted to generate '{generated.GamePath}' more than once.");

                // Accessory body slots provide fitting support only.
                if (transplantTargetBody && physicalBodySlot is not null)
                {
                    var primarySelection = slots.Single(selection => string.Equals(selection.Slot, modelSlot, StringComparison.OrdinalIgnoreCase));
                    var effectiveBodyRace = ResolveRaceSwapBodyRaceCode(primarySelection.Target, targetIdentity.Code);
                    var preparedBodyOptions = await _bodies.PrepareTargetBodyOptionsAsync(primarySelection.Target, modelSlot, effectiveBodyRace, request.ModRoot, cancellationToken).ConfigureAwait(false);
                    foreach (var created in preparedBodyOptions.CreatedFiles) generatedSupportFiles.Add(created);
                    foreach (var group in preparedBodyOptions.Groups)
                        targetBodyGroups.Add(new V4TargetBodyGroupRequest(
                            itemRequest.GroupKey, itemRequest.OutputOptionName, group.BodyName, group.Slot, targetGamePath, group.SourceKey, group.Group));
                }
            }

            var conflicting = appendMaps.Values
                .GroupBy(x => $"{x.GroupKey}\u001f{x.OutputOptionName}", StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault(group => group.Select(x => x.SourceOptionKey).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1);
            if (conflicting is not null)
                throw new InvalidOperationException($"Selected outfit rows map target '{conflicting.First().OutputOptionName}' from different source options in the same Penumbra group. Choose one source option for that group.");

            SetProgress(ConversionStage.UpdatingPenumbra, 0.90f, $"Writing {targetIdentity.DisplayName} option");
            IReadOnlyList<V4AppendResult> results;
            try
            {
                var appendRequests = appendMaps.Values.Select(append => new V4AppendRequest(
                    Path.Combine(request.ModRoot, "meta.json"), append.GroupKey, append.SourceOptionKey, append.OutputOptionName, append.Redirects)).ToArray();
                results = await _writer.AppendClonedOptionsWithTargetBodyGroupsAsync(appendRequests, targetBodyGroups, cancellationToken).ConfigureAwait(false);
                metaCommitted = true;
            }
            catch
            {
                foreach (var generated in generatedFiles) DeleteUnreferencedGeneratedFile(generated);
                foreach (var support in generatedSupportFiles) DeleteUnreferencedSupportFile(support);
                generatedFiles.Clear();
                generatedSupportFiles.Clear();
                throw;
            }

            var reloadResult = await _penumbra.ReloadAsync(penumbraMod, cancellationToken).ConfigureAwait(false);
            if (!reloadResult.Success)
                throw new InvalidOperationException($"The converted outfit was written safely, but Penumbra reload failed: {reloadResult.Error}");

            SetProgress(ConversionStage.Complete, 1.0f, "Complete");
            return results;
        }
        catch (Exception ex)
        {
            if (!metaCommitted)
            {
                foreach (var generated in generatedFiles) DeleteUnreferencedGeneratedFile(generated);
                foreach (var support in generatedSupportFiles) DeleteUnreferencedSupportFile(support);
            }
            LastError = ex.Message;
            SetProgress(ConversionStage.Failed, 1.0f, "Failed");
            _log.Error(ex, "RavaFit gender/race swap conversion failed.");
            throw;
        }
        finally
        {
            _conversionGate.Release();
        }
    }

    private async Task<TargetBodyArtifacts> PrepareRaceSwapTargetBodiesAsync(IReadOnlyList<SlotConversionSelection> slots, string work, string targetRaceCode, IReadOnlyList<EstSkeletonOverride> targetEstOverrides, IReadOnlyList<string> targetSkeletonPaths, Dictionary<string, TargetBodyArtifact>? reuseCache, CancellationToken cancellationToken)
    {
        var duplicate = slots.GroupBy(x => x.Slot, StringComparer.OrdinalIgnoreCase).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
            throw new InvalidOperationException($"Body context contains the {duplicate.Key} slot more than once.");

        var targetBodyGlbs = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var targetBodyMdls = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var selection in slots)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var safeSlot = selection.Slot.ToLowerInvariant();
            var effectiveRaceCode = ResolveRaceSwapBodyRaceCode(selection.Target, targetRaceCode)
                ?? throw new InvalidOperationException($"{selection.Target.BodyName} / {selection.Target.VariantName} has no payload compatible with c{targetRaceCode}.");
            var cacheKey = BuildTargetBodyArtifactCacheKey(selection, $"swap:{targetRaceCode}:{effectiveRaceCode}", targetEstOverrides, targetSkeletonPaths);
            if (reuseCache is not null && reuseCache.TryGetValue(cacheKey, out var cached)
                && File.Exists(cached.Mdl) && new FileInfo(cached.Mdl).Length > 0
                && File.Exists(cached.Glb) && new FileInfo(cached.Glb).Length > 0)
            {
                targetBodyGlbs.Add(selection.Slot, cached.Glb);
                targetBodyMdls.Add(selection.Slot, cached.Mdl);
                continue;
            }

            var targetBodyMdl = Path.Combine(work, $"target-body-{safeSlot}.mdl");
            var targetBodyGlb = Path.Combine(work, $"target-body-{safeSlot}.glb");
            using var targetReply = await _solver.CallAsync("extract_payload", new
            {
                rbody = selection.Target.RBodyPath,
                body = selection.Target.BodyId,
                slot = selection.Target.Slot,
                variant = selection.Target.VariantId,
                race_code = effectiveRaceCode,
                destination = targetBodyMdl,
            }, cancellationToken).ConfigureAwait(false);
            var targetRoot = targetReply.RootElement;
            if (!targetRoot.TryGetProperty("target_path", out var targetPathNode) || targetPathNode.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(targetPathNode.GetString()))
                throw new InvalidDataException($"RBODY {selection.Slot} target payload did not provide its original XIV model path.");

            var targetGamePath = CharacterRaceCatalog.RewriteHumanRaceCode(targetPathNode.GetString()!, targetRaceCode);
            await _bridge.ExportAsync(new ModelExportRequest(targetGamePath, targetBodyMdl, targetBodyGlb, targetEstOverrides, targetSkeletonPaths), cancellationToken).ConfigureAwait(false);
            if (!File.Exists(targetBodyGlb) || new FileInfo(targetBodyGlb).Length == 0)
                throw new InvalidDataException($"Model bridge did not produce the {selection.Slot} target-body GLB.");
            targetBodyGlbs.Add(selection.Slot, targetBodyGlb);
            targetBodyMdls.Add(selection.Slot, targetBodyMdl);
            if (reuseCache is not null) reuseCache[cacheKey] = new TargetBodyArtifact(targetBodyGlb, targetBodyMdl);
        }
        return new TargetBodyArtifacts(targetBodyGlbs, targetBodyMdls);
    }

    private static string? ResolveRaceSwapBodyRaceCode(BodyVariantInfo variant, string targetRaceCode)
        => CharacterRaceCatalog.ResolveBodyPayloadRace(variant, targetRaceCode);

    private async Task<Dictionary<string, VanillaSourceBodyArtifact>> PrepareVanillaSourceBodiesAsync(IReadOnlyList<SlotConversionSelection> slots, string work, string modelRaceCode, IReadOnlyList<EstSkeletonOverride> estOverrides, IReadOnlyList<string> skeletonPaths, CancellationToken cancellationToken)
    {
        var duplicate = slots.GroupBy(x => x.Slot, StringComparer.OrdinalIgnoreCase).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
            throw new InvalidOperationException($"Vanilla body context contains the {duplicate.Key} slot more than once.");

        var output = new Dictionary<string, VanillaSourceBodyArtifact>(StringComparer.OrdinalIgnoreCase);
        foreach (var selection in slots)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (gamePath, sourceRaceCode) = ResolveVanillaBodyGamePath(modelRaceCode, selection.Slot);
            var resource = _dataManager.GetFile(gamePath)
                ?? throw new FileNotFoundException($"XIV game data did not return the vanilla {selection.Slot} body model '{gamePath}'.");

            var safeSlot = selection.Slot.ToLowerInvariant();
            var sourceBodyMdl = Path.Combine(work, $"vanilla-source-body-{safeSlot}.mdl");
            var sourceBodyGlb = Path.Combine(work, $"vanilla-source-body-{safeSlot}.glb");
            await File.WriteAllBytesAsync(sourceBodyMdl, resource.Data, cancellationToken).ConfigureAwait(false);

            // Use complete XIV body support for fitting; outfit fragments only control replacement.
            await _bridge.ExportAsync(new ModelExportRequest(gamePath, sourceBodyMdl, sourceBodyGlb, estOverrides, skeletonPaths), cancellationToken).ConfigureAwait(false);
            if (!File.Exists(sourceBodyGlb) || new FileInfo(sourceBodyGlb).Length == 0)
                throw new InvalidDataException($"Model bridge did not produce the vanilla {selection.Slot} source-body GLB.");

            output.Add(selection.Slot, new VanillaSourceBodyArtifact(sourceBodyGlb, sourceBodyMdl, gamePath, sourceRaceCode));
        }
        return output;
    }

    private (string GamePath, string RaceCode) ResolveVanillaBodyGamePath(string modelRaceCode, string slot)
    {
        var suffix = slot switch
        {
            BodySlots.Chest => "top",
            BodySlots.Legs => "dwn",
            BodySlots.Hands => "glv",
            BodySlots.Feet => "sho",
            _ => throw new InvalidOperationException($"Unsupported body slot '{slot}'."),
        };

        var candidates = new List<string>();
        void AddCandidate(string? raceCode)
        {
            if (string.IsNullOrWhiteSpace(raceCode)) return;
            if (string.Equals(raceCode, "1201", StringComparison.OrdinalIgnoreCase) && string.Equals(slot, BodySlots.Feet, StringComparison.OrdinalIgnoreCase))
                raceCode = "1101"; // Female Lalafell feet use the shared male Lalafell body payload.
            if (!candidates.Contains(raceCode, StringComparer.OrdinalIgnoreCase)) candidates.Add(raceCode);
        }

        AddCandidate(modelRaceCode);
        AddCandidate(CharacterRaceCatalog.FromCode(modelRaceCode)?.MidlanderFallbackCode);
        foreach (var raceCode in candidates)
        {
            var path = $"chara/equipment/e0000/model/c{raceCode}e0000_{suffix}.mdl";
            if (_dataManager.FileExists(path)) return (path, raceCode);
        }

        throw new FileNotFoundException($"XIV game data contains no canonical vanilla {slot} body model for c{modelRaceCode}. Tried: {string.Join(", ", candidates.Select(raceCode => $"c{raceCode}e0000_{suffix}.mdl"))}.");
    }

    private async Task<TargetBodyArtifacts> PrepareTargetBodiesAsync(IReadOnlyList<SlotConversionSelection> slots, string work, string modelRaceCode, IReadOnlyList<EstSkeletonOverride> estOverrides, IReadOnlyList<string> skeletonPaths, Dictionary<string, TargetBodyArtifact>? reuseCache, CancellationToken cancellationToken)
    {
        var duplicate = slots.GroupBy(x => x.Slot, StringComparer.OrdinalIgnoreCase).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
            throw new InvalidOperationException($"Body context contains the {duplicate.Key} slot more than once.");

        var targetBodyGlbs = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var targetBodyMdls = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var selection in slots)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var safeSlot = selection.Slot.ToLowerInvariant();
            var targetRaceCode = ResolveBodyRaceCode(selection.Target, modelRaceCode);
            var cacheKey = BuildTargetBodyArtifactCacheKey(selection, targetRaceCode, estOverrides, skeletonPaths);
            if (reuseCache is not null && reuseCache.TryGetValue(cacheKey, out var cached)
                && File.Exists(cached.Mdl) && new FileInfo(cached.Mdl).Length > 0
                && File.Exists(cached.Glb) && new FileInfo(cached.Glb).Length > 0)
            {
                targetBodyGlbs.Add(selection.Slot, cached.Glb);
                targetBodyMdls.Add(selection.Slot, cached.Mdl);
                _log.Debug("RavaFit reused prepared {Slot} target body {Body}/{Variant} for this outfit transaction.", selection.Slot, selection.Target.BodyName, selection.Target.VariantName);
                continue;
            }

            var targetBodyMdl = Path.Combine(work, $"target-body-{safeSlot}.mdl");
            var targetBodyGlb = Path.Combine(work, $"target-body-{safeSlot}.glb");
            using var targetReply = await _solver.CallAsync("extract_payload", new
            {
                rbody = selection.Target.RBodyPath,
                body = selection.Target.BodyId,
                slot = selection.Target.Slot,
                variant = selection.Target.VariantId,
                race_code = targetRaceCode,
                destination = targetBodyMdl,
            }, cancellationToken).ConfigureAwait(false);
            var targetRoot = targetReply.RootElement;
            if (!targetRoot.TryGetProperty("target_path", out var targetPathNode) || targetPathNode.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(targetPathNode.GetString()))
                throw new InvalidDataException($"RBODY {selection.Slot} target payload did not provide its original XIV model path.");

            var targetGamePath = targetPathNode.GetString()!;
            // Export every target region against the outfit armature so incompatible skeletons fail cleanly.
            await _bridge.ExportAsync(new ModelExportRequest(targetGamePath, targetBodyMdl, targetBodyGlb, estOverrides, skeletonPaths), cancellationToken).ConfigureAwait(false);
            if (!File.Exists(targetBodyGlb) || new FileInfo(targetBodyGlb).Length == 0)
                throw new InvalidDataException($"Model bridge did not produce the {selection.Slot} target-body GLB.");
            targetBodyGlbs.Add(selection.Slot, targetBodyGlb);
            targetBodyMdls.Add(selection.Slot, targetBodyMdl);
            if (reuseCache is not null) reuseCache[cacheKey] = new TargetBodyArtifact(targetBodyGlb, targetBodyMdl);
        }

        return new TargetBodyArtifacts(targetBodyGlbs, targetBodyMdls);
    }

    private static string BuildTargetBodyArtifactCacheKey(SlotConversionSelection selection, string? targetRaceCode, IReadOnlyList<EstSkeletonOverride> estOverrides, IReadOnlyList<string> skeletonPaths)
    {
        var rbodyPath = Path.GetFullPath(selection.Target.RBodyPath);
        var rbodyInfo = new FileInfo(rbodyPath);
        return JsonSerializer.Serialize(new
        {
            rbodyPath,
            rbodyInfo.Length,
            RBodyWriteTicks = rbodyInfo.LastWriteTimeUtc.Ticks,
            selection.Slot,
            selection.Target.BodyId,
            selection.Target.VariantId,
            TargetRaceCode = targetRaceCode ?? string.Empty,
            EstOverrides = estOverrides.Select(value => new { value.Slot, value.Entry, value.Gender, value.Race, value.SetId, value.Source }).ToArray(),
            SkeletonPaths = skeletonPaths.Select(value => Path.GetFullPath(value)).ToArray(),
        });
    }

    private async Task<byte[]> RestoreNativeTargetBodyAsync(byte[] importedMdl, string sourceMdl, string work, JsonElement conversionReply, IReadOnlyDictionary<string, string> targetBodyMdls, CancellationToken cancellationToken)
    {
        // Bodyless garments use body surfaces for fitting context without transplanting a body.
        if (conversionReply.TryGetProperty("transplant", out var transplantNode) && transplantNode.ValueKind == JsonValueKind.Object)
        {
            if (transplantNode.TryGetProperty("enabled", out var enabledNode)
                && enabledNode.ValueKind is JsonValueKind.True or JsonValueKind.False
                && !enabledNode.GetBoolean())
            {
                _log.Information("RavaFit conversion used body support for fitting only; no native target-body graft was required.");
                return importedMdl;
            }
            if (transplantNode.TryGetProperty("inserted", out var insertedNode)
                && insertedNode.ValueKind == JsonValueKind.Array
                && insertedNode.GetArrayLength() == 0)
            {
                _log.Information("RavaFit conversion produced no transplanted body meshes; no native target-body graft was required.");
                return importedMdl;
            }
        }

        if (!conversionReply.TryGetProperty("report", out var reportNode) || reportNode.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(reportNode.GetString()))
            throw new InvalidDataException("Solver conversion did not provide the body-transplant report required for native RBODY restoration.");

        var importedPath = Path.Combine(work, "imported-before-native-body-graft.mdl");
        var graftedPath = Path.Combine(work, "native-body-grafted.mdl");
        await File.WriteAllBytesAsync(importedPath, importedMdl, cancellationToken).ConfigureAwait(false);
        using var graftReply = await _solver.CallAsync("graft_native_body", new
        {
            imported_mdl = importedPath,
            output_mdl = graftedPath,
            conversion_report = reportNode.GetString(),
            target_body_mdls = targetBodyMdls,
            source_model_mdl = sourceMdl,
        }, cancellationToken).ConfigureAwait(false);
        if (!graftReply.RootElement.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
            throw new InvalidOperationException("Native RBODY restoration did not complete successfully.");
        if (!File.Exists(graftedPath) || new FileInfo(graftedPath).Length == 0)
            throw new FileNotFoundException("Native RBODY restoration did not produce the final MDL.", graftedPath);

        var bodyCount = graftReply.RootElement.TryGetProperty("body_meshes", out var bodyMeshes) && bodyMeshes.ValueKind == JsonValueKind.Array ? bodyMeshes.GetArrayLength() : 0;
        _log.Information("RavaFit restored {BodyMeshCount} native RBODY mesh group(s) after Penumbra model import.", bodyCount);
        return await File.ReadAllBytesAsync(graftedPath, cancellationToken).ConfigureAwait(false);
    }

    private static void ValidateStandardBodySelections(IEnumerable<SlotConversionSelection> slots, string modelRaceCode, bool useVanillaSourceBody = false)
    {
        var identity = CharacterRaceCatalog.FromCode(modelRaceCode)
            ?? throw new InvalidOperationException($"Unknown source character code c{modelRaceCode}.");
        foreach (var slot in slots)
        {
            if (!useVanillaSourceBody)
            {
                if (slot.Source is null)
                    throw new InvalidOperationException($"{slot.Slot} requires a source body.");
                if (!slot.Source.SupportsGender(identity.Gender))
                    throw new InvalidOperationException($"{slot.Source.BodyName} / {slot.Source.VariantName} is not a {identity.Gender.ToLowerInvariant()} source body.");
            }
            if (!slot.Target.SupportsGender(identity.Gender))
                throw new InvalidOperationException($"{slot.Target.BodyName} / {slot.Target.VariantName} is not a {identity.Gender.ToLowerInvariant()} target body.");
        }
    }

    private static void ValidateRaceSwapBodySelections(IEnumerable<SlotConversionSelection> slots, string sourceRaceCode, CharacterRaceIdentity targetIdentity)
    {
        var sourceIdentity = CharacterRaceCatalog.FromCode(sourceRaceCode)
            ?? throw new InvalidOperationException($"Unknown source character code c{sourceRaceCode}.");
        foreach (var slot in slots)
        {
            if (slot.Source is null)
                throw new InvalidOperationException($"{slot.Slot} requires a source body for race/gender swap.");
            if (!slot.Source.SupportsGender(sourceIdentity.Gender) || ResolveRaceSwapBodyRaceCode(slot.Source, sourceRaceCode) is null)
                throw new InvalidOperationException($"{slot.Source.BodyName} / {slot.Source.VariantName} is not a {sourceIdentity.Gender.ToLowerInvariant()} source body compatible with c{sourceRaceCode}.");
            if (!slot.Target.SupportsGender(targetIdentity.Gender) || ResolveRaceSwapBodyRaceCode(slot.Target, targetIdentity.Code) is null)
                throw new InvalidOperationException($"{slot.Target.BodyName} / {slot.Target.VariantName} is not a {targetIdentity.Gender.ToLowerInvariant()} target body compatible with {targetIdentity.DisplayName}.");
            if (CharacterRaceCatalog.RequiresSmallclothesTarget(sourceIdentity.Gender, targetIdentity, slot.Slot) && !slot.Target.IsSmallclothesSupport)
                throw new InvalidOperationException($"Female-to-male gender ports require an authored SFW Smallclothes Legs target; {slot.Target.BodyName} / {slot.Target.VariantName} is not marked as one.");
            if (CharacterRaceCatalog.RequiresSmallclothesSource(sourceIdentity, targetIdentity.Gender, slot.Slot) && !slot.Source.IsSmallclothesSupport)
                throw new InvalidOperationException($"Male-to-female gender ports require an authored SFW Smallclothes Legs source; {slot.Source.BodyName} / {slot.Source.VariantName} is not marked as one.");
        }
    }

    private void RequireBridgeAvailable()
    {
        if (!_bridge.Status.Available)
            throw new InvalidOperationException(_bridge.Status.Detail);
    }

    private void DeleteUnreferencedGeneratedFile(GeneratedModelFile generated)
    {
        try
        {
            if (File.Exists(generated.AbsolutePath))
                File.Delete(generated.AbsolutePath);
            var directory = Path.GetDirectoryName(generated.AbsolutePath);
            if (!string.IsNullOrWhiteSpace(directory) && Directory.Exists(directory) && !Directory.EnumerateFileSystemEntries(directory).Any())
                Directory.Delete(directory);
        }
        catch (Exception cleanupError)
        {
            _log.Warning(cleanupError, "RavaFit could not remove an unreferenced generated model after the metadata transaction failed: {Path}", generated.AbsolutePath);
        }
    }

    private void DeleteUnreferencedSupportFile(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
            var directory = Path.GetDirectoryName(path);
            while (!string.IsNullOrWhiteSpace(directory) && Directory.Exists(directory) && !Directory.EnumerateFileSystemEntries(directory).Any())
            {
                var parent = Path.GetDirectoryName(directory);
                Directory.Delete(directory);
                if (string.Equals(Path.GetFileName(directory), "TargetBodyOptions", StringComparison.OrdinalIgnoreCase)) break;
                directory = parent;
            }
        }
        catch (Exception cleanupError)
        {
            _log.Warning(cleanupError, "RavaFit could not remove an unreferenced target-body support asset after the metadata transaction failed: {Path}", path);
        }
    }

    private string CreateWorkDirectory(string kind)
    {
        var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "RavaFit", "Work");
        var path = Path.Combine(root, $"{DateTime.UtcNow:yyyyMMdd-HHmmss}-{kind}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }

    private static string? ResolveBodyRaceCode(BodyVariantInfo variant, string preferredRaceCode)
        => variant.RaceCodes.Contains(preferredRaceCode, StringComparer.OrdinalIgnoreCase) ? preferredRaceCode : variant.CanonicalRaceCode;

    private static bool IsResolvedLalafellTarget(BodyVariantInfo variant, string preferredRaceCode)
        => LalafellContentGuard.IsLalafell(CharacterRaceCatalog.FromCode(ResolveBodyRaceCode(variant, preferredRaceCode)));

    private static string? TryResolveRaceCode(string gamePath)
    {
        try { return ResolveRaceCode(gamePath); }
        catch { return null; }
    }

    private static string ResolveRaceCode(string gamePath)
    {
        var path = gamePath.Replace('\\', '/');
        var slash = path.LastIndexOf('/');
        var file = slash >= 0 ? path[(slash + 1)..] : path;
        if (file.Length >= 5 && char.ToLowerInvariant(file[0]) == 'c')
        {
            var value = file.AsSpan(1, 4);
            if (value[0] is >= '0' and <= '9' && value[1] is >= '0' and <= '9' && value[2] is >= '0' and <= '9' && value[3] is >= '0' and <= '9')
                return value.ToString();
        }
        throw new NotSupportedException($"RavaFit cannot determine the human race code for model path {gamePath}.");
    }

    private static string? TryResolveBodyModelSlot(string gamePath)
    {
        var path = gamePath.Replace('\\', '/');
        if (path.EndsWith("_top.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Chest;
        if (path.EndsWith("_dwn.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Legs;
        if (path.EndsWith("_glv.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Hands;
        if (path.EndsWith("_sho.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Feet;
        return null;
    }

    private static string ResolvePrimaryBodySlot(string gamePath, string? requestedPrimaryBodySlot)
    {
        var physical = TryResolveBodyModelSlot(gamePath);
        if (physical is not null) return physical;
        var accessory = AccessoryModelSlots.FromGamePath(gamePath);
        if (accessory is null)
            throw new NotSupportedException($"RavaFit cannot determine a supported equipment/accessory container for model path {gamePath}.");
        if (string.IsNullOrWhiteSpace(requestedPrimaryBodySlot) || !BodySlots.All.Contains(requestedPrimaryBodySlot, StringComparer.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Accessory model {gamePath} needs a coverage-inferred Chest/Legs/Hands/Feet fitting region.");
        return BodySlots.All.First(slot => string.Equals(slot, requestedPrimaryBodySlot, StringComparison.OrdinalIgnoreCase));
    }

    private static (bool SourceContainsBody, bool TransplantTargetBody, bool FitOnly) ResolveBodyOutputPolicy(string gamePath, bool? requestedSourceContainsBody, bool? requestedTransplantTargetBody)
    {
        var fitOnly = AccessoryModelSlots.FromGamePath(gamePath) is not null;
        // Detecting embedded body geometry does not grant accessories body-transplant authority.
        var sourceContainsBody = requestedSourceContainsBody ?? !fitOnly;
        var transplantTargetBody = !fitOnly && (requestedTransplantTargetBody ?? sourceContainsBody);
        return (sourceContainsBody, transplantTargetBody, fitOnly);
    }

    private void SetProgress(ConversionStage stage, float progress, string detail)
    {
        Progress = new ConversionProgress(stage, Math.Clamp(progress, 0, 1), detail);
        ProgressChanged?.Invoke();
    }
}
