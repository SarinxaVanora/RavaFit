using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Text.Json.Nodes;
using Dalamud.Plugin.Services;
using RavaFit.Core.Models;
using RavaFit.Core.Penumbra;
using RavaFit.ModelBridge;

namespace RavaFit.Services;

internal sealed record CustomiseModelPart(int PartIndex, int MeshIndex, int SubmeshIndex, string Material, int VertexCount, int IndexCount, IReadOnlyList<string> Attributes, bool PiercingLike, bool AttributeCapable);

internal sealed record CustomisePreviewTriangle(int PartIndex, Vector3 A, Vector3 B, Vector3 C, Vector3 Normal, Vector2 UvA, Vector2 UvB, Vector2 UvC);
internal sealed record CustomisePreviewEdge(int PartIndex, Vector3 A, Vector3 B);
internal sealed record CustomiseModelInspection(IReadOnlyList<CustomiseModelPart> Parts, IReadOnlyList<CustomisePreviewTriangle> Triangles, IReadOnlyList<CustomisePreviewEdge> Edges, Vector3 BoundsMin, Vector3 BoundsMax);

internal sealed record PiercingAssignmentRequest(PenumbraModInfo Mod, string GroupKey, string OptionKey, ModelRedirect Model, BodyVariantInfo PiercingBody);
internal sealed record VisibilityToggleDefinition(string ToggleName, IReadOnlyList<int> PartIndices);
internal sealed record VisibilityToggleRequest(PenumbraModInfo Mod, string GroupKey, string OptionKey, ModelRedirect Model, IReadOnlyList<int> PartIndices, string ToggleName);
internal sealed record VisibilityToggleBatchRequest(PenumbraModInfo Mod, string GroupKey, string OptionKey, ModelRedirect Model, IReadOnlyList<VisibilityToggleDefinition> Toggles);
internal sealed record AccessorySplitRequest(PenumbraModInfo Mod, string GroupKey, string OptionKey, ModelRedirect Model, IReadOnlyList<int> PartIndices, string TargetGamePath, ushort TargetVariantId, byte? TargetVanillaMaterialId, string TargetDisplayName, string OutputOptionName);

internal sealed class CustomiseModService
{
    private readonly PenumbraService _penumbra;
    private readonly IDataManager _dataManager;
    private readonly SolverHostService _solver;
    private readonly ModelBridgeService _bridge;
    private readonly BodyLibraryService _bodies;
    private readonly ConversionService _conversion;
    private readonly IPluginLog _log;
    private readonly GeneratedModelStore _store = new();
    private readonly PenumbraV4Writer _writer = new();
    private readonly SemaphoreSlim _gate = new(1, 1);

    public CustomiseModService(PenumbraService penumbra, IDataManager dataManager, SolverHostService solver, ModelBridgeService bridge, BodyLibraryService bodies, ConversionService conversion, IPluginLog log)
    {
        _penumbra = penumbra;
        _dataManager = dataManager;
        _solver = solver;
        _bridge = bridge;
        _bodies = bodies;
        _conversion = conversion;
        _log = log;
    }

    public bool Busy => _gate.CurrentCount == 0;
    public string Status { get; private set; } = string.Empty;

    public IReadOnlyList<BodyVariantInfo> GetPiercingBodies(string slot, string? raceCode)
        => _bodies.PiercingVariants(slot, raceCode)
            .OrderBy(v => v.BodyName, StringComparer.OrdinalIgnoreCase)
            .ThenBy(v => v.VariantName, StringComparer.OrdinalIgnoreCase)
            .ToArray();

    public async Task<CustomiseModelInspection> InspectPartsAsync(PenumbraModInfo mod, string groupKey, string optionKey, ModelRedirect model, CancellationToken cancellationToken = default)
    {
        var document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
        var liveModel = document.GetModelRedirects(groupKey, optionKey).FirstOrDefault(candidate => string.Equals(candidate.GamePath, model.GamePath, StringComparison.OrdinalIgnoreCase)) ?? model;
        var physical = document.ResolvePhysicalPath(liveModel);
        if (!File.Exists(physical)) throw new FileNotFoundException("The selected model does not exist.", physical);
        using var reply = await _solver.CallAsync("inspect_mdl_parts", new { mdl = physical }, cancellationToken).ConfigureAwait(false);
        var output = new List<CustomiseModelPart>();
        if (reply.RootElement.TryGetProperty("parts", out var parts) && parts.ValueKind == JsonValueKind.Array)
        {
            foreach (var part in parts.EnumerateArray())
            {
                var attributes = part.TryGetProperty("attributes", out var attrs) && attrs.ValueKind == JsonValueKind.Array ? attrs.EnumerateArray().Select(x => x.GetString() ?? string.Empty).Where(x => x.Length > 0).ToArray() : [];
                output.Add(new CustomiseModelPart(
                    part.GetProperty("part_index").GetInt32(),
                    part.GetProperty("mesh_index").GetInt32(),
                    part.TryGetProperty("submesh_index", out var submesh) ? submesh.GetInt32() : 0,
                    part.TryGetProperty("material", out var material) ? material.GetString() ?? string.Empty : string.Empty,
                    part.TryGetProperty("vertex_count", out var vertices) ? vertices.GetInt32() : 0,
                    part.TryGetProperty("index_count", out var indices) ? indices.GetInt32() : 0,
                    attributes,
                    part.TryGetProperty("piercing_like", out var piercing) && piercing.ValueKind is JsonValueKind.True,
                    !part.TryGetProperty("attribute_capable", out var attributeCapable) || attributeCapable.ValueKind is JsonValueKind.True));
            }
        }
        var triangles = new List<CustomisePreviewTriangle>();
        var edges = new List<CustomisePreviewEdge>();
        var boundsMin = Vector3.Zero;
        var boundsMax = Vector3.One;
        if (reply.RootElement.TryGetProperty("preview", out var preview) && preview.ValueKind == JsonValueKind.Object)
        {
            boundsMin = ReadVector3(preview, "bounds_min", Vector3.Zero);
            boundsMax = ReadVector3(preview, "bounds_max", Vector3.One);
            if (preview.TryGetProperty("triangles", out var triangleArray) && triangleArray.ValueKind == JsonValueKind.Array)
            {
                foreach (var triangle in triangleArray.EnumerateArray())
                {
                    if (!triangle.TryGetProperty("p", out var points) || points.ValueKind != JsonValueKind.Array) continue;
                    var values = points.EnumerateArray().Select(value => value.GetSingle()).ToArray();
                    if (values.Length != 9) continue;
                    var normal = ReadVector3(triangle, "n", Vector3.UnitZ);
                    var uvs = triangle.TryGetProperty("uv", out var uvNode) && uvNode.ValueKind == JsonValueKind.Array ? uvNode.EnumerateArray().Select(value => value.GetSingle()).ToArray() : [];
                    var uvA = uvs.Length >= 6 ? new Vector2(uvs[0], uvs[1]) : Vector2.Zero;
                    var uvB = uvs.Length >= 6 ? new Vector2(uvs[2], uvs[3]) : Vector2.Zero;
                    var uvC = uvs.Length >= 6 ? new Vector2(uvs[4], uvs[5]) : Vector2.Zero;
                    triangles.Add(new CustomisePreviewTriangle(triangle.GetProperty("part_index").GetInt32(), new Vector3(values[0], values[1], values[2]), new Vector3(values[3], values[4], values[5]), new Vector3(values[6], values[7], values[8]), normal, uvA, uvB, uvC));
                }
            }
            if (preview.TryGetProperty("edges", out var edgeArray) && edgeArray.ValueKind == JsonValueKind.Array)
            {
                foreach (var edge in edgeArray.EnumerateArray())
                {
                    if (!edge.TryGetProperty("p", out var points) || points.ValueKind != JsonValueKind.Array) continue;
                    var values = points.EnumerateArray().Select(value => value.GetSingle()).ToArray();
                    if (values.Length != 6) continue;
                    edges.Add(new CustomisePreviewEdge(edge.GetProperty("part_index").GetInt32(), new Vector3(values[0], values[1], values[2]), new Vector3(values[3], values[4], values[5])));
                }
            }
        }
        return new CustomiseModelInspection(output, triangles, edges, boundsMin, boundsMax);
    }

    public async Task AssignPiercingsAsync(PiercingAssignmentRequest request, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        var createdFiles = new List<string>();
        string? work = null;
        var metaCommitted = false;
        try
        {
            Status = "Detecting destination body";
            RequireReady();
            var slot = ResolveModelSlot(request.Model.GamePath);
            if (slot is not (BodySlots.Chest or BodySlots.Legs))
                throw new NotSupportedException("Piercing assignment currently applies to chest and legs body models, where the catalogue contains piercing geometry.");
            var modelRaceCode = ResolveRaceCode(request.Model.GamePath);
            if (!_bodies.HasPiercingOptions(request.PiercingBody))
                throw new InvalidOperationException("The selected body does not contain captured piercing controls.");
            var donorRace = CharacterRaceCatalog.ResolveBodyPayloadRace(request.PiercingBody, modelRaceCode)
                ?? throw new InvalidOperationException($"{request.PiercingBody.BodyName} / {request.PiercingBody.VariantName} has no body payload compatible with c{modelRaceCode}.");

            var coverage = await _conversion.AnalyseCoverageAsync(request.Mod, request.GroupKey, request.OptionKey, request.Model, cancellationToken).ConfigureAwait(false);
            if (!coverage.SourceMatches.TryGetValue(slot, out var destinationMatch))
                throw new InvalidOperationException($"RavaFit could not identify the {slot.ToLowerInvariant()} body inside the selected model. Import that body into the RBody catalogue first, then retry piercing assignment.");
            var destinationBody = destinationMatch.Variant;
            var destinationRace = CharacterRaceCatalog.ResolveBodyPayloadRace(destinationBody, modelRaceCode)
                ?? throw new InvalidOperationException($"Detected destination body {destinationBody.BodyName} / {destinationBody.VariantName} has no payload compatible with c{modelRaceCode}.");

            var document = PenumbraV4Document.Load(Path.Combine(request.Mod.ModRoot, "meta.json"));
            var physical = document.ResolvePhysicalPath(request.Model);
            if (!File.Exists(physical)) throw new FileNotFoundException("The selected model does not exist.", physical);
            var sourcePiercingResources = (await InspectPartsAsync(request.Mod, request.GroupKey, request.OptionKey, request.Model, cancellationToken).ConfigureAwait(false)).Parts
                .Where(part => part.PiercingLike && !string.IsNullOrWhiteSpace(part.Material))
                .Select(part => part.Material)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            work = CreateWorkDirectory("piercings");
            var destinationGlb = Path.Combine(work, "destination.glb");
            var donorMdl = Path.Combine(work, "donor-body.mdl");
            var donorBodyGlb = Path.Combine(work, "donor-body.glb");
            var donorPiercingGlb = Path.Combine(work, "donor-piercings.glb");
            var destinationBodyMdl = Path.Combine(work, "destination-body.mdl");
            var destinationBodyGlb = Path.Combine(work, "destination-body.glb");
            var fittedPiercingGlb = Path.Combine(work, "fitted-piercings.glb");
            var mergedGlb = Path.Combine(work, "merged.glb");
            var specPath = Path.Combine(work, "piercing-fit.json");

            var activeSettings = _penumbra.GetCurrentOptionSettings(request.Mod);
            var estOverrides = document.GetEstSkeletonOverrides(request.GroupKey, request.OptionKey, request.Model.GamePath, activeSettings)
                .Select(x => new EstSkeletonOverride(x.Slot, x.Entry, x.Gender, x.Race, x.SetId, x.Source)).ToArray();
            var sourceExport = new ModelExportRequest(request.Model.GamePath, physical, destinationGlb, estOverrides);
            var skeletonPaths = _bridge.ResolveSkeletonPaths(sourceExport);
            Status = "Exporting selected model";
            await _bridge.ExportAsync(sourceExport with { SkeletonPathsOverride = skeletonPaths }, cancellationToken).ConfigureAwait(false);

            Status = "Preparing piercing body";
            var donorGamePath = await ExtractBodyAsync(request.PiercingBody, donorRace, donorMdl, cancellationToken).ConfigureAwait(false);
            donorGamePath = CharacterRaceCatalog.RewriteHumanRaceCode(donorGamePath, modelRaceCode);
            await _bridge.ExportAsync(new ModelExportRequest(donorGamePath, donorMdl, donorBodyGlb, estOverrides, skeletonPaths), cancellationToken).ConfigureAwait(false);
            using (var extracted = await _solver.CallAsync("extract_piercings", new { source_glb = donorBodyGlb, output_glb = donorPiercingGlb }, cancellationToken).ConfigureAwait(false))
                EnsureOk(extracted.RootElement, "Could not isolate target piercing geometry.");

            Status = "Preparing destination body";
            var destinationGamePath = await ExtractBodyAsync(destinationBody, destinationRace, destinationBodyMdl, cancellationToken).ConfigureAwait(false);
            destinationGamePath = CharacterRaceCatalog.RewriteHumanRaceCode(destinationGamePath, modelRaceCode);
            await _bridge.ExportAsync(new ModelExportRequest(destinationGamePath, destinationBodyMdl, destinationBodyGlb, estOverrides, skeletonPaths), cancellationToken).ConfigureAwait(false);

            var spec = new
            {
                game_path = request.Model.GamePath,
                model_slot = slot,
                source_glb = donorPiercingGlb,
                target_body_glbs = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { [slot] = destinationBodyGlb },
                output_glb = fittedPiercingGlb,
                source_contains_body = false,
                transplant_target_body = false,
                slots = new[]
                {
                    new
                    {
                        slot,
                        source_race_code = donorRace,
                        target_race_code = destinationRace,
                        target_support_surface = destinationBody.IsSmallclothesSupport ? "smallclothes" : "body",
                        source = new { rbody = request.PiercingBody.RBodyPath, body = request.PiercingBody.BodyId, variant = request.PiercingBody.VariantId },
                        target = new { rbody = destinationBody.RBodyPath, body = destinationBody.BodyId, variant = destinationBody.VariantId },
                    },
                },
            };
            await File.WriteAllTextAsync(specPath, JsonSerializer.Serialize(spec, new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
            Status = $"Fitting {request.PiercingBody.BodyName} piercings to {destinationBody.BodyName}";
            using (var fit = await _solver.CallAsync("convert", new { spec = specPath }, cancellationToken).ConfigureAwait(false))
                EnsureOk(fit.RootElement, "Piercing fit failed.");

            Status = "Replacing piercing geometry";
            using (var merged = await _solver.CallAsync("replace_piercings", new { source_glb = destinationGlb, donor_glb = fittedPiercingGlb, output_glb = mergedGlb }, cancellationToken).ConfigureAwait(false))
                EnsureOk(merged.RootElement, "Piercing replacement failed.");
            var mdl = await _bridge.ImportAsync(new ModelImportRequest(request.Model.GamePath, physical, mergedGlb), cancellationToken).ConfigureAwait(false);
            var generated = await _store.WriteAsync(request.Mod.ModRoot, request.Model.GamePath, $"{request.PiercingBody.BodyName} Piercings", mdl, cancellationToken).ConfigureAwait(false);
            createdFiles.Add(generated.AbsolutePath);

            Status = "Replacing piercing controls";
            var prepared = await _bodies.PreparePiercingOptionsAsync(request.PiercingBody, slot, modelRaceCode, request.Mod.ModRoot, cancellationToken).ConfigureAwait(false);
            createdFiles.AddRange(prepared.CreatedFiles);
            var controlRequests = prepared.Groups.Select(group => new V4CustomisationGroupRequest(group.BodyName, group.Slot, request.Model.GamePath, group.SourceKey, group.Group)).ToArray();
            await _writer.ReplaceModelAndPiercingControlsAsync(Path.Combine(request.Mod.ModRoot, "meta.json"), request.GroupKey, request.OptionKey, request.Model.GamePath, generated.RelativePath, controlRequests, sourcePiercingResources, cancellationToken).ConfigureAwait(false);
            metaCommitted = true;
            var reloadResult = await _penumbra.ReloadAsync(request.Mod, cancellationToken).ConfigureAwait(false);
            if (!reloadResult.Success)
                throw new InvalidOperationException($"Piercings were written safely, but Penumbra reload failed: {reloadResult.Error}");
            Status = $"Assigned {request.PiercingBody.BodyName} piercings";
        }
        catch
        {
            if (!metaCommitted)
            {
                foreach (var file in createdFiles.Distinct(StringComparer.OrdinalIgnoreCase))
                {
                    try { if (File.Exists(file)) File.Delete(file); } catch { }
                }
            }
            throw;
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(work))
            {
                try { Directory.Delete(work, true); } catch (Exception ex) { _log.Debug(ex, "RavaFit could not remove piercing work directory {Directory}", work); }
            }
            _gate.Release();
        }
    }

    public Task AddVisibilityToggleAsync(VisibilityToggleRequest request, CancellationToken cancellationToken = default)
        => AddVisibilityTogglesAsync(new VisibilityToggleBatchRequest(request.Mod, request.GroupKey, request.OptionKey, request.Model, new[] { new VisibilityToggleDefinition(request.ToggleName, request.PartIndices) }), cancellationToken);

    public async Task AddVisibilityTogglesAsync(VisibilityToggleBatchRequest request, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        GeneratedModelFile? generated = null;
        string? work = null;
        var metaCommitted = false;
        try
        {
            RequireReady();
            if (request.Toggles.Count == 0) throw new InvalidOperationException("Create at least one visibility toggle.");
            if (request.Toggles.Any(toggle => toggle.PartIndices.Count == 0)) throw new InvalidOperationException("Every visibility toggle must control at least one model part.");
            if (request.Toggles.Any(toggle => string.IsNullOrWhiteSpace(toggle.ToggleName))) throw new InvalidOperationException("Give every visibility toggle a name.");
            if (request.Toggles.Select(toggle => toggle.ToggleName.Trim()).Distinct(StringComparer.OrdinalIgnoreCase).Count() != request.Toggles.Count)
                throw new InvalidOperationException("Visibility toggle names must be unique.");

            var document = PenumbraV4Document.Load(Path.Combine(request.Mod.ModRoot, "meta.json"));
            var liveModel = document.GetModelRedirects(request.GroupKey, request.OptionKey).FirstOrDefault(candidate => string.Equals(candidate.GamePath, request.Model.GamePath, StringComparison.OrdinalIgnoreCase))
                ?? throw new InvalidOperationException("The selected model is no longer present in this option. Refresh the model picker and retry.");
            var physical = document.ResolvePhysicalPath(liveModel);
            if (!File.Exists(physical)) throw new FileNotFoundException("The selected model does not exist.", physical);

            work = CreateWorkDirectory("visibility");
            var currentMdl = physical;
            var controls = new List<(string ToggleName, string AttributeName)>();
            for (var i = 0; i < request.Toggles.Count; i++)
            {
                var toggle = request.Toggles[i];
                var attributeName = "atrx_ravafit_" + Guid.NewGuid().ToString("N")[..12];
                var taggedMdl = Path.Combine(work, $"tagged-{i + 1:D2}.mdl");
                Status = request.Toggles.Count == 1
                    ? $"Tagging {toggle.ToggleName.Trim()}"
                    : $"Tagging {toggle.ToggleName.Trim()} ({i + 1}/{request.Toggles.Count})";
                using (var tagged = await _solver.CallAsync("tag_mdl_parts", new { mdl = currentMdl, output = taggedMdl, part_indices = toggle.PartIndices, attribute = attributeName }, cancellationToken).ConfigureAwait(false))
                    EnsureOk(tagged.RootElement, $"Could not add the visibility attribute for '{toggle.ToggleName.Trim()}'.");
                currentMdl = taggedMdl;
                controls.Add((toggle.ToggleName.Trim(), attributeName));
            }

            var generatedLabel = request.Toggles.Count == 1 ? $"visibility-{controls[0].ToggleName}" : $"visibility-batch-{request.Toggles.Count}";
            generated = await _store.WriteAsync(request.Mod.ModRoot, request.Model.GamePath, generatedLabel, await File.ReadAllBytesAsync(currentMdl, cancellationToken).ConfigureAwait(false), cancellationToken).ConfigureAwait(false);
            Status = request.Toggles.Count == 1 ? "Adding visibility toggle" : $"Adding {request.Toggles.Count} visibility toggles";
            await _writer.AddAttributeVisibilityTogglesAsync(Path.Combine(request.Mod.ModRoot, "meta.json"), request.GroupKey, request.OptionKey, request.Model.GamePath, generated.RelativePath, controls, cancellationToken).ConfigureAwait(false);
            metaCommitted = true;
            var reloadResult = await _penumbra.ReloadAsync(request.Mod, cancellationToken).ConfigureAwait(false);
            if (!reloadResult.Success)
                throw new InvalidOperationException($"Visibility toggles were written safely, but Penumbra reload failed: {reloadResult.Error}");
            Status = request.Toggles.Count == 1 ? $"Added {controls[0].ToggleName}" : $"Added {request.Toggles.Count} visibility toggles";
        }
        catch
        {
            if (!metaCommitted && generated is not null)
            {
                try { if (File.Exists(generated.AbsolutePath)) File.Delete(generated.AbsolutePath); } catch { }
            }
            throw;
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(work))
            {
                try { Directory.Delete(work, true); } catch (Exception ex) { _log.Debug(ex, "RavaFit could not remove visibility work directory {Directory}", work); }
            }
            _gate.Release();
        }
    }

    public async Task<PenumbraModInfo> SplitToAccessoryAsync(AccessorySplitRequest request, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        string? work = null;
        string? createdRoot = null;
        try
        {
            RequireReady();
            if (request.PartIndices.Count == 0) throw new InvalidOperationException("Select at least one model part to split into the accessory.");
            if (string.IsNullOrWhiteSpace(request.OutputOptionName)) throw new InvalidOperationException("Give the new mod a name.");
            var targetAccessorySlot = AccessoryModelSlots.FromGamePath(request.TargetGamePath)
                ?? throw new InvalidOperationException("Choose a supported XIV accessory target (earrings, necklace, wrists or ring).");
            if (string.Equals(request.Model.GamePath.Replace('\\', '/'), request.TargetGamePath.Replace('\\', '/'), StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("The accessory target must be a different XIV model slot from the source model.");
            var sourceRaceCode = ResolveRaceCode(request.Model.GamePath);
            var targetRaceCode = ResolveRaceCode(request.TargetGamePath);
            if (!string.Equals(sourceRaceCode, targetRaceCode, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("Choose an accessory for the same race and gender. You can use Race/Gender Swap on the finished mod afterwards.");

            var document = PenumbraV4Document.Load(Path.Combine(request.Mod.ModRoot, "meta.json"));
            var liveModel = document.GetModelRedirects(request.GroupKey, request.OptionKey).FirstOrDefault(candidate => string.Equals(candidate.GamePath, request.Model.GamePath, StringComparison.OrdinalIgnoreCase))
                ?? throw new InvalidOperationException("The selected model is no longer present in this option. Refresh the model picker and retry.");
            var physical = document.ResolvePhysicalPath(liveModel);
            if (!File.Exists(physical)) throw new FileNotFoundException("The selected model does not exist.", physical);

            var selectedMaterialReferences = await GetAccessorySupportMaterialReferencesAsync(physical, request.PartIndices, cancellationToken).ConfigureAwait(false);
            var targetMaterialIds = ResolveTargetAccessoryMaterialIds(document, request.TargetGamePath, request.TargetVariantId, request.TargetVanillaMaterialId);
            var materialPayloads = ResolveAccessoryMaterialPayloads(document, request.GroupKey, request.OptionKey, request.Model.GamePath, selectedMaterialReferences);
            var texturePayloads = ResolveAccessoryTexturePayloads(document, request.GroupKey, request.OptionKey, materialPayloads);

            work = CreateWorkDirectory("split-accessory");
            var remainingMdl = Path.Combine(work, "remaining.mdl");
            var accessoryMdl = Path.Combine(work, "accessory.mdl");
            Status = $"Splitting selected parts to {request.TargetDisplayName}";
            using (var split = await _solver.CallAsync("split_mdl_parts", new
            {
                mdl = physical,
                remaining_output = remainingMdl,
                accessory_output = accessoryMdl,
                part_indices = request.PartIndices,
            }, cancellationToken).ConfigureAwait(false))
                EnsureOk(split.RootElement, "Could not split the selected model parts into an accessory.");

            if (!File.Exists(remainingMdl) || !File.Exists(accessoryMdl))
                throw new InvalidDataException("Accessory split completed without producing both native MDL outputs.");
            var sourceLength = new FileInfo(physical).Length;
            if (new FileInfo(remainingMdl).Length != sourceLength || new FileInfo(accessoryMdl).Length != sourceLength)
                throw new InvalidDataException("Accessory split changed native MDL size; nothing was written.");
            var remainingBytes = await File.ReadAllBytesAsync(remainingMdl, cancellationToken).ConfigureAwait(false);
            var accessoryBytes = await File.ReadAllBytesAsync(accessoryMdl, cancellationToken).ConfigureAwait(false);
            if (remainingBytes.AsSpan().SequenceEqual(accessoryBytes))
                throw new InvalidDataException("Accessory split did not produce distinct source and accessory models.");

            Status = "Creating new mod...";
            var created = await CreateStandaloneAccessoryModAsync(request, document, remainingBytes, accessoryBytes, targetMaterialIds, materialPayloads, texturePayloads, cancellationToken).ConfigureAwait(false);
            createdRoot = created.ModRoot;
            Status = $"Created {created.Name}";
            return created;
        }
        catch
        {
            if (!string.IsNullOrWhiteSpace(createdRoot))
            {
                try { if (Directory.Exists(createdRoot)) Directory.Delete(createdRoot, true); } catch { }
            }
            throw;
        }
        finally
        {
            if (!string.IsNullOrWhiteSpace(work))
            {
                try { Directory.Delete(work, true); } catch (Exception ex) { _log.Debug(ex, "RavaFit could not remove accessory-split work directory {Directory}", work); }
            }
            _gate.Release();
        }
    }

    private sealed record AccessoryMaterialPayload(string FileName, string SourceGamePath, byte[] Bytes);
    private sealed record AccessoryTexturePayload(string GamePath, byte[] Bytes);

    private async Task<PenumbraModInfo> CreateStandaloneAccessoryModAsync(AccessorySplitRequest request, PenumbraV4Document sourceDocument, byte[] remainingMdl, byte[] accessoryMdl,
        IReadOnlyList<byte> targetMaterialIds, IReadOnlyList<AccessoryMaterialPayload> materials, IReadOnlyList<AccessoryTexturePayload> textures, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(_penumbra.ModDirectoryRoot) || !Directory.Exists(_penumbra.ModDirectoryRoot))
            throw new DirectoryNotFoundException("Penumbra did not provide its mod directory.");

        var displayName = MakeUniqueStandaloneModName(request.OutputOptionName.Trim());
        var directoryName = MakeUniqueStandaloneDirectoryName(displayName);
        var root = Path.Combine(_penumbra.ModDirectoryRoot, directoryName);
        Directory.CreateDirectory(root);
        try
        {
            var data = BuildEffectiveStandaloneData(sourceDocument, request.GroupKey, request.OptionKey);
            var files = PenumbraV4Document.FindProperty(data, "Files") as JsonObject;
            if (files is null)
            {
                files = new JsonObject();
                SetJsonProperty(data, "Files", files);
            }

            var sourceMappings = CollectEffectiveFileMappings(sourceDocument, request.GroupKey, request.OptionKey);
            foreach (var (gamePath, relative) in sourceMappings)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var source = sourceDocument.ResolvePhysicalPath(new ModelRedirect(gamePath, relative, false));
                if (!File.Exists(source))
                    throw new FileNotFoundException($"The selected option points to a missing file for '{gamePath}'.", source);
                var copiedRelative = await CopyStandaloneFileAsync(root, gamePath, source, cancellationToken).ConfigureAwait(false);
                files[gamePath] = copiedRelative;
            }

            files[NormalizeResourcePath(request.Model.GamePath)] = await WriteStandaloneBytesAsync(root, request.Model.GamePath, remainingMdl, cancellationToken).ConfigureAwait(false);
            files[NormalizeResourcePath(request.TargetGamePath)] = await WriteStandaloneBytesAsync(root, request.TargetGamePath, accessoryMdl, cancellationToken).ConfigureAwait(false);

            foreach (var material in materials)
            {
                var relative = await WriteStandaloneBytesAsync(root, material.SourceGamePath, material.Bytes, cancellationToken).ConfigureAwait(false);
                files[NormalizeResourcePath(material.SourceGamePath)] = relative;
                foreach (var materialId in targetMaterialIds)
                    files[BuildAccessoryMaterialGamePath(request.TargetGamePath, materialId, material.FileName)] = relative;
            }

            foreach (var texture in textures)
                files[NormalizeResourcePath(texture.GamePath)] = await WriteStandaloneBytesAsync(root, texture.GamePath, texture.Bytes, cancellationToken).ConfigureAwait(false);

            var meta = new JsonObject
            {
                ["FileVersion"] = 4,
                ["Name"] = displayName,
                ["Author"] = string.IsNullOrWhiteSpace(sourceDocument.Author) ? "RavaFit" : sourceDocument.Author,
                ["Version"] = "1.0.0",
                ["Website"] = sourceDocument.Website,
                ["Description"] = $"Accessory split from {request.Mod.Name}. {request.TargetDisplayName} carries the selected parts; the original outfit piece is kept without them.",
                ["Tags"] = new JsonArray(),
                ["LastWrite"] = DateTimeOffset.UtcNow.ToString("O"),
                ["DefaultData"] = data,
                ["PageNames"] = new JsonObject(),
                ["Groups"] = new JsonArray(),
            };

            var metaPath = Path.Combine(root, "meta.json");
            await File.WriteAllTextAsync(metaPath, meta.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
            _ = PenumbraV4Document.Load(metaPath);
            var addResult = await _penumbra.AddModAsync(directoryName, cancellationToken).ConfigureAwait(false);
            if (!addResult.Success)
                throw new InvalidOperationException($"The new mod was created, but Penumbra could not add it: {addResult.Error}");
            await _penumbra.RefreshAsync(cancellationToken).ConfigureAwait(false);
            return _penumbra.Mods.FirstOrDefault(mod => string.Equals(Path.GetFullPath(mod.ModRoot), Path.GetFullPath(root), StringComparison.OrdinalIgnoreCase))
                ?? new PenumbraModInfo(directoryName, displayName, root);
        }
        catch
        {
            try { Directory.Delete(root, true); } catch { }
            throw;
        }
    }

    private IReadOnlyList<AccessoryTexturePayload> ResolveAccessoryTexturePayloads(PenumbraV4Document document, string groupKey, string optionKey, IReadOnlyList<AccessoryMaterialPayload> materials)
    {
        var output = new Dictionary<string, AccessoryTexturePayload>(StringComparer.OrdinalIgnoreCase);
        foreach (var material in materials)
        {
            var ascii = Encoding.ASCII.GetString(material.Bytes);
            foreach (Match match in Regex.Matches(ascii, @"[a-z0-9_./\\-]{3,}?\.tex", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
            {
                var gamePath = ResolveTextureGamePath(material.SourceGamePath, match.Value);
                if (string.IsNullOrWhiteSpace(gamePath) || output.ContainsKey(gamePath)) continue;
                var bytes = ReadResourceBytes(document, groupKey, optionKey, gamePath);
                if (bytes is null || bytes.Length == 0) continue;
                output[gamePath] = new AccessoryTexturePayload(gamePath, bytes);
            }
        }
        return output.Values.OrderBy(value => value.GamePath, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    private static string? ResolveTextureGamePath(string materialGamePath, string rawReference)
    {
        var value = NormalizeResourcePath(rawReference);
        var chara = value.IndexOf("chara/", StringComparison.OrdinalIgnoreCase);
        if (chara >= 0) return value[chara..];
        if (value.StartsWith("common/", StringComparison.OrdinalIgnoreCase)) return value;
        var fileName = Path.GetFileName(value.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(fileName)) return null;
        var material = NormalizeResourcePath(materialGamePath);
        var marker = material.IndexOf("/material/", StringComparison.OrdinalIgnoreCase);
        if (marker <= 0) return null;
        return $"{material[..marker]}/texture/{ExtractVersionFolder(material) ?? "v0001"}/{fileName}";
    }

    private static JsonObject BuildEffectiveStandaloneData(PenumbraV4Document document, string groupKey, string optionKey)
    {
        var result = EmptyOptionData();
        MergeOptionData(result, PenumbraV4Document.FindProperty(document.Root, "DefaultData") as JsonObject);
        MergeOptionData(result, document.GetOption(document.GetGroup(groupKey), optionKey).Node);
        return result;
    }

    private static Dictionary<string, string> CollectEffectiveFileMappings(PenumbraV4Document document, string groupKey, string optionKey)
    {
        var mappings = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        AddFileMappings(mappings, PenumbraV4Document.FindProperty(document.Root, "DefaultData"));
        AddFileMappings(mappings, document.GetOption(document.GetGroup(groupKey), optionKey).Node);
        return mappings;
    }

    private static void AddFileMappings(Dictionary<string, string> output, JsonNode? node)
    {
        if (node is not JsonObject obj || PenumbraV4Document.FindProperty(obj, "Files") is not JsonObject files) return;
        foreach (var pair in files)
            if (pair.Value is JsonValue value && value.TryGetValue<string>(out var relative) && !string.IsNullOrWhiteSpace(relative))
                output[NormalizeResourcePath(pair.Key)] = relative;
    }

    private static JsonObject EmptyOptionData() => new()
    {
        ["Files"] = new JsonObject(),
        ["FileSwaps"] = new JsonObject(),
        ["Manipulations"] = new JsonArray(),
    };

    private static void MergeOptionData(JsonObject destination, JsonObject? source)
    {
        if (source is null) return;
        MergeJsonObjectMap(destination, source, "Files");
        MergeJsonObjectMap(destination, source, "FileSwaps");
        var destinationManipulations = PenumbraV4Document.FindProperty(destination, "Manipulations") as JsonArray;
        if (destinationManipulations is null)
        {
            destinationManipulations = new JsonArray();
            SetJsonProperty(destination, "Manipulations", destinationManipulations);
        }
        if (PenumbraV4Document.FindProperty(source, "Manipulations") is JsonArray sourceManipulations)
            foreach (var manipulation in sourceManipulations)
                destinationManipulations.Add(manipulation?.DeepClone());
    }

    private static void MergeJsonObjectMap(JsonObject destination, JsonObject source, string name)
    {
        var target = PenumbraV4Document.FindProperty(destination, name) as JsonObject;
        if (target is null)
        {
            target = new JsonObject();
            SetJsonProperty(destination, name, target);
        }
        if (PenumbraV4Document.FindProperty(source, name) is not JsonObject sourceMap) return;
        foreach (var pair in sourceMap) target[pair.Key] = pair.Value?.DeepClone();
    }

    private static void SetJsonProperty(JsonObject obj, string name, JsonNode? value)
    {
        var existing = PenumbraV4Document.FindPropertyName(obj, name);
        obj[existing ?? name] = value;
    }

    private static async Task<string> CopyStandaloneFileAsync(string root, string gamePath, string source, CancellationToken cancellationToken)
    {
        var bytes = await File.ReadAllBytesAsync(source, cancellationToken).ConfigureAwait(false);
        return await WriteStandaloneBytesAsync(root, gamePath, bytes, cancellationToken).ConfigureAwait(false);
    }

    private static async Task<string> WriteStandaloneBytesAsync(string root, string gamePath, ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken)
    {
        if (bytes.IsEmpty) throw new InvalidDataException($"'{gamePath}' is empty.");
        var normalized = NormalizeResourcePath(gamePath);
        if (normalized.Split('/', StringSplitOptions.RemoveEmptyEntries).Any(segment => segment == "..")) throw new InvalidDataException($"Invalid game path '{gamePath}'.");
        var relative = Path.Combine("Files", normalized.Replace('/', Path.DirectorySeparatorChar)).Replace('\\', '/');
        var absolute = Path.GetFullPath(Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar)));
        var rootPrefix = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root)) + Path.DirectorySeparatorChar;
        if (!absolute.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException($"Invalid game path '{gamePath}'.");
        Directory.CreateDirectory(Path.GetDirectoryName(absolute)!);
        await File.WriteAllBytesAsync(absolute, bytes.ToArray(), cancellationToken).ConfigureAwait(false);
        return relative;
    }

    private string MakeUniqueStandaloneModName(string requested)
    {
        var baseName = string.IsNullOrWhiteSpace(requested) ? "RavaFit Accessory" : requested.Trim();
        if (_penumbra.Mods.All(mod => !string.Equals(mod.Name, baseName, StringComparison.OrdinalIgnoreCase))) return baseName;
        for (var i = 2; ; i++)
        {
            var candidate = $"{baseName} ({i})";
            if (_penumbra.Mods.All(mod => !string.Equals(mod.Name, candidate, StringComparison.OrdinalIgnoreCase))) return candidate;
        }
    }

    private string MakeUniqueStandaloneDirectoryName(string displayName)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var safe = new string(displayName.Select(c => invalid.Contains(c) ? '_' : c).ToArray()).Trim().Trim('.');
        if (string.IsNullOrWhiteSpace(safe)) safe = "RavaFit Accessory";
        var candidate = safe;
        for (var i = 2; Directory.Exists(Path.Combine(_penumbra.ModDirectoryRoot, candidate)); i++) candidate = $"{safe} ({i})";
        return candidate;
    }


    private async Task<IReadOnlyList<string>> GetAccessorySupportMaterialReferencesAsync(string physicalMdl, IReadOnlyList<int> requestedPartIndices, CancellationToken cancellationToken)
    {
        using var reply = await _solver.CallAsync("inspect_mdl_parts", new { mdl = physicalMdl }, cancellationToken).ConfigureAwait(false);
        EnsureOk(reply.RootElement, "Could not inspect selected parts before the accessory split.");
        if (!reply.RootElement.TryGetProperty("parts", out var parts) || parts.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException("RavaFit could not read the selected model parts.");

        var requested = requestedPartIndices.Distinct().ToHashSet();
        var found = new HashSet<int>();
        var selectedMeshes = new HashSet<int>();
        var inspected = new List<(int PartIndex, int MeshIndex, string Material)>();
        foreach (var part in parts.EnumerateArray())
        {
            if (!part.TryGetProperty("part_index", out var partIndexValue) || !partIndexValue.TryGetInt32(out var partIndex)) continue;
            var meshIndex = part.TryGetProperty("mesh_index", out var meshIndexValue) && meshIndexValue.TryGetInt32(out var parsedMeshIndex) ? parsedMeshIndex : -1;
            var material = part.TryGetProperty("material", out var materialValue) && materialValue.ValueKind == JsonValueKind.String ? materialValue.GetString()?.Trim() ?? string.Empty : string.Empty;
            inspected.Add((partIndex, meshIndex, material));
            if (!requested.Contains(partIndex)) continue;
            found.Add(partIndex);
            selectedMeshes.Add(meshIndex);
            if (string.IsNullOrWhiteSpace(material) || !material.EndsWith(".mtrl", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("One of the selected parts has no usable material, so the accessory cannot be created safely.");
        }
        if (!requested.SetEquals(found))
            throw new InvalidDataException("One or more selected model parts are no longer valid. Refresh Customise and retry.");

        var materials = inspected.Where(part => selectedMeshes.Contains(part.MeshIndex))
            .Select(part => part.Material)
            .Where(material => !string.IsNullOrWhiteSpace(material) && material.EndsWith(".mtrl", StringComparison.OrdinalIgnoreCase))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(material => material, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (materials.Length == 0)
            throw new InvalidDataException("The selected garment parts do not reference any XIV materials.");
        return materials;
    }

    private IReadOnlyList<AccessoryMaterialPayload> ResolveAccessoryMaterialPayloads(PenumbraV4Document document, string groupKey, string optionKey, string sourceModelGamePath, IReadOnlyList<string> materialReferences)
    {
        var byFileName = new Dictionary<string, AccessoryMaterialPayload>(StringComparer.OrdinalIgnoreCase);
        foreach (var materialReference in materialReferences)
        {
            var gamePath = ResolveMaterialGamePath(document, groupKey, optionKey, sourceModelGamePath, materialReference)
                ?? throw new InvalidDataException($"Could not resolve source material '{materialReference}' for the selected garment parts.");
            var bytes = ReadResourceBytes(document, groupKey, optionKey, gamePath)
                ?? throw new FileNotFoundException($"Could not read source material '{gamePath}' for the selected garment parts.");
            if (bytes.Length == 0) throw new InvalidDataException($"Source material '{gamePath}' is empty.");
            var fileName = Path.GetFileName(gamePath.Replace('/', Path.DirectorySeparatorChar));
            if (string.IsNullOrWhiteSpace(fileName)) throw new InvalidDataException($"Source material path '{gamePath}' has no file name.");
            if (byFileName.TryGetValue(fileName, out var existing))
            {
                if (!existing.Bytes.AsSpan().SequenceEqual(bytes))
                    throw new InvalidDataException($"Two selected materials both use the file name '{fileName}'. Rename one in the source mod, then try again.");
                continue;
            }
            byFileName[fileName] = new AccessoryMaterialPayload(fileName, gamePath, bytes);
        }
        return byFileName.Values.OrderBy(value => value.FileName, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    private static IReadOnlyList<byte> ResolveTargetAccessoryMaterialIds(PenumbraV4Document document, string targetGamePath, ushort targetVariantId, byte? vanillaMaterialId)
    {
        var setId = ResolveAccessorySetId(targetGamePath);
        var slot = AccessoryModelSlots.FromGamePath(targetGamePath) ?? throw new InvalidDataException($"Could not identify accessory slot from '{targetGamePath}'.");
        var ids = new HashSet<byte>();
        if (vanillaMaterialId.HasValue) ids.Add(vanillaMaterialId.Value);
        CollectAccessoryMaterialIds(document.Root, setId, slot, targetVariantId, ids);
        if (ids.Count == 0)
            throw new InvalidDataException("Could not resolve the selected XIV accessory item's material variant. RavaFit will not create an accessory with guessed material paths.");
        return ids.OrderBy(id => id).ToArray();
    }

    private static void CollectAccessoryMaterialIds(JsonNode? node, ushort setId, string targetSlot, ushort targetVariantId, HashSet<byte> output)
    {
        if (node is JsonObject obj)
        {
            var type = ReadJsonString(obj, "Type");
            if (string.Equals(type, "Imc", StringComparison.OrdinalIgnoreCase))
            {
                if (FindJsonProperty(obj, "Manipulation") is JsonObject manipulation && MatchesAccessoryImcIdentity(manipulation, setId, targetSlot, targetVariantId, requireVariantMatch: false))
                    AddMaterialId(FindJsonProperty(manipulation, "Entry"), output);

                if (FindJsonProperty(obj, "Identifier") is JsonObject identifier && MatchesAccessoryImcIdentity(identifier, setId, targetSlot, targetVariantId, requireVariantMatch: false))
                {
                    AddMaterialId(FindJsonProperty(obj, "DefaultEntry"), output);
                    if (FindJsonProperty(obj, "Options") is JsonArray options)
                        foreach (var option in options)
                        {
                            AddMaterialId(option, output);
                            if (option is JsonObject optionObject) AddMaterialId(FindJsonProperty(optionObject, "Entry"), output);
                        }
                }
            }
            foreach (var child in obj) CollectAccessoryMaterialIds(child.Value, setId, targetSlot, targetVariantId, output);
        }
        else if (node is JsonArray array)
        {
            foreach (var child in array) CollectAccessoryMaterialIds(child, setId, targetSlot, targetVariantId, output);
        }
    }

    private static bool MatchesAccessoryImcIdentity(JsonObject identity, ushort setId, string targetSlot, ushort targetVariantId, bool requireVariantMatch)
    {
        if (!string.Equals(ReadJsonString(identity, "ObjectType"), "Accessory", StringComparison.OrdinalIgnoreCase)) return false;
        if (ReadJsonInt(identity, "PrimaryId") != setId) return false;
        if (!AccessorySlotMatches(ReadJsonString(identity, "EquipSlot"), targetSlot)) return false;
        if (!requireVariantMatch) return true;
        var variant = ReadJsonInt(identity, "Variant");
        return variant is null || variant == targetVariantId;
    }

    private static bool AccessorySlotMatches(string? authoredSlot, string targetSlot)
    {
        var value = (authoredSlot ?? string.Empty).Replace("_", string.Empty).Replace(" ", string.Empty).ToLowerInvariant();
        return targetSlot switch
        {
            AccessoryModelSlots.Earrings => value is "ear" or "ears" or "earring" or "earrings",
            AccessoryModelSlots.Necklace => value is "neck" or "necklace",
            AccessoryModelSlots.Wrists => value is "wrist" or "wrists",
            AccessoryModelSlots.RightRing => value is "rightring" or "rightfinger" or "fingerr" or "rfinger" or "ringr",
            AccessoryModelSlots.LeftRing => value is "leftring" or "leftfinger" or "fingerl" or "lfinger" or "ringl",
            _ => false,
        };
    }

    private static void AddMaterialId(JsonNode? node, HashSet<byte> output)
    {
        if (node is not JsonObject obj) return;
        var value = ReadJsonInt(obj, "MaterialId");
        if (value is >= byte.MinValue and <= byte.MaxValue) output.Add((byte)value.Value);
    }

    private static ushort ResolveAccessorySetId(string gamePath)
    {
        var normalized = NormalizeResourcePath(gamePath);
        var marker = normalized.LastIndexOf("/a", StringComparison.OrdinalIgnoreCase);
        if (marker < 0 || marker + 6 > normalized.Length || !ushort.TryParse(normalized.AsSpan(marker + 2, 4), out var setId))
            throw new InvalidDataException($"Could not identify accessory model set from '{gamePath}'.");
        return setId;
    }

    private static string BuildAccessoryMaterialGamePath(string targetModelGamePath, byte materialId, string fileName)
    {
        var model = NormalizeResourcePath(targetModelGamePath);
        var marker = model.IndexOf("/model/", StringComparison.OrdinalIgnoreCase);
        if (marker <= 0) throw new InvalidDataException($"Accessory model path '{targetModelGamePath}' has no model container root.");
        return $"{model[..marker]}/material/v{materialId:D4}/{fileName}";
    }

    private string? ResolveMaterialGamePath(PenumbraV4Document document, string groupKey, string optionKey, string modelGamePath, string materialReference)
    {
        var material = NormalizeResourcePath(materialReference);
        if (!material.EndsWith(".mtrl", StringComparison.OrdinalIgnoreCase)) return null;
        if (material.StartsWith("chara/", StringComparison.OrdinalIgnoreCase)) return material;
        var fileName = Path.GetFileName(material.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(fileName)) return null;

        var selected = GetSelectedOptionNode(document, groupKey, optionKey);
        var exact = FindGamePathByFileName(selected, fileName, ".mtrl")
            ?? FindGamePathByFileName(PenumbraV4Document.FindProperty(document.Root, "DefaultData"), fileName, ".mtrl")
            ?? FindGamePathByFileName(document.Root, fileName, ".mtrl");
        if (!string.IsNullOrWhiteSpace(exact)) return exact;

        var model = NormalizeResourcePath(modelGamePath);
        var marker = model.IndexOf("/model/", StringComparison.OrdinalIgnoreCase);
        if (marker <= 0) return null;
        return $"{model[..marker]}/material/{ExtractVersionFolder(material) ?? "v0001"}/{fileName}";
    }

    private byte[]? ReadResourceBytes(PenumbraV4Document document, string groupKey, string optionKey, string gamePath)
    {
        var selected = ResolveMappedPhysical(document, GetSelectedOptionNode(document, groupKey, optionKey), gamePath)
            ?? ResolveMappedPhysical(document, PenumbraV4Document.FindProperty(document.Root, "DefaultData"), gamePath);
        if (!string.IsNullOrWhiteSpace(selected) && File.Exists(selected)) return File.ReadAllBytes(selected);

        var resolved = _penumbra.ResolvePlayerPath(gamePath);
        if (Path.IsPathRooted(resolved) && File.Exists(resolved)) return File.ReadAllBytes(resolved);

        var anyMapped = FindMappedRelativeRecursive(document.Root, gamePath);
        if (!string.IsNullOrWhiteSpace(anyMapped))
        {
            var physical = document.ResolvePhysicalPath(new ModelRedirect(gamePath, anyMapped, false));
            if (File.Exists(physical)) return File.ReadAllBytes(physical);
        }

        var gameResource = _dataManager.GetFile(NormalizeResourcePath(resolved)) ?? _dataManager.GetFile(gamePath);
        return gameResource?.Data;
    }

    private static string? ResolveMappedPhysical(PenumbraV4Document document, JsonNode? node, string gamePath)
    {
        var relative = FindMappedRelative(node, gamePath);
        if (string.IsNullOrWhiteSpace(relative)) return null;
        return document.ResolvePhysicalPath(new ModelRedirect(gamePath, relative, false));
    }

    private static JsonNode? GetSelectedOptionNode(PenumbraV4Document document, string groupKey, string optionKey)
    {
        try { return document.GetOption(document.GetGroup(groupKey), optionKey).Node; }
        catch { return null; }
    }

    private static string? FindMappedRelative(JsonNode? node, string gamePath)
    {
        if (node is not JsonObject obj || PenumbraV4Document.FindProperty(obj, "Files") is not JsonObject files) return null;
        foreach (var pair in files)
            if (string.Equals(NormalizeResourcePath(pair.Key), NormalizeResourcePath(gamePath), StringComparison.OrdinalIgnoreCase)
                && pair.Value is JsonValue value && value.TryGetValue<string>(out var relative) && !string.IsNullOrWhiteSpace(relative))
                return relative;
        return null;
    }

    private static string? FindMappedRelativeRecursive(JsonNode? node, string gamePath)
    {
        if (node is JsonObject obj)
        {
            if (PenumbraV4Document.FindProperty(obj, "Files") is JsonObject files)
                foreach (var pair in files)
                    if (string.Equals(NormalizeResourcePath(pair.Key), NormalizeResourcePath(gamePath), StringComparison.OrdinalIgnoreCase)
                        && pair.Value is JsonValue value && value.TryGetValue<string>(out var relative) && !string.IsNullOrWhiteSpace(relative))
                        return relative;
            foreach (var child in obj)
            {
                var result = FindMappedRelativeRecursive(child.Value, gamePath);
                if (!string.IsNullOrWhiteSpace(result)) return result;
            }
        }
        else if (node is JsonArray array)
        {
            foreach (var child in array)
            {
                var result = FindMappedRelativeRecursive(child, gamePath);
                if (!string.IsNullOrWhiteSpace(result)) return result;
            }
        }
        return null;
    }

    private static string? FindGamePathByFileName(JsonNode? node, string fileName, string extension)
    {
        if (node is JsonObject obj)
        {
            foreach (var pair in obj)
            {
                if (pair.Key.EndsWith(extension, StringComparison.OrdinalIgnoreCase)
                    && string.Equals(Path.GetFileName(pair.Key.Replace('/', Path.DirectorySeparatorChar)), fileName, StringComparison.OrdinalIgnoreCase))
                    return NormalizeResourcePath(pair.Key);
                var nested = FindGamePathByFileName(pair.Value, fileName, extension);
                if (!string.IsNullOrWhiteSpace(nested)) return nested;
            }
        }
        else if (node is JsonArray array)
        {
            foreach (var child in array)
            {
                var nested = FindGamePathByFileName(child, fileName, extension);
                if (!string.IsNullOrWhiteSpace(nested)) return nested;
            }
        }
        return null;
    }

    private static string? ExtractVersionFolder(string value)
    {
        foreach (var segment in NormalizeResourcePath(value).Split('/', StringSplitOptions.RemoveEmptyEntries))
            if (segment.Length == 5 && segment[0] == 'v' && segment.AsSpan(1).ToString().All(char.IsDigit)) return segment;
        return null;
    }

    private static JsonNode? FindJsonProperty(JsonObject obj, string name)
    {
        foreach (var pair in obj)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase)) return pair.Value;
        return null;
    }

    private static string? ReadJsonString(JsonObject obj, string name)
        => FindJsonProperty(obj, name) is JsonValue value && value.TryGetValue<string>(out var result) ? result : null;

    private static int? ReadJsonInt(JsonObject obj, string name)
    {
        if (FindJsonProperty(obj, name) is not JsonValue value) return null;
        if (value.TryGetValue<int>(out var integer)) return integer;
        return value.TryGetValue<long>(out var large) && large is >= int.MinValue and <= int.MaxValue ? (int)large : null;
    }

    private static string NormalizeResourcePath(string value) => value.Trim().Trim('\0').Replace('\\', '/').TrimStart('/').ToLowerInvariant();

    private async Task<string> ExtractBodyAsync(BodyVariantInfo body, string raceCode, string destination, CancellationToken cancellationToken)
    {
        using var reply = await _solver.CallAsync("extract_payload", new
        {
            rbody = body.RBodyPath,
            body = body.BodyId,
            slot = body.Slot,
            variant = body.VariantId,
            race_code = raceCode,
            destination,
        }, cancellationToken).ConfigureAwait(false);
        EnsureOk(reply.RootElement, $"Could not extract {body.BodyName} / {body.VariantName}.");
        if (!reply.RootElement.TryGetProperty("target_path", out var path) || path.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(path.GetString()))
            throw new InvalidDataException($"{body.BodyName} / {body.VariantName} did not provide its XIV target model path.");
        return path.GetString()!;
    }

    private void RequireReady()
    {
        if (!_solver.Ready || !_solver.ConversionReady) throw new InvalidOperationException("RavaFit is still getting ready.");
        if (!_bridge.Status.Available) throw new InvalidOperationException("Model tools are unavailable.");
    }

    private static void EnsureOk(JsonElement root, string message)
    {
        if (!root.TryGetProperty("ok", out var ok) || !ok.GetBoolean()) throw new InvalidOperationException(message);
    }

    private static Vector3 ReadVector3(JsonElement parent, string propertyName, Vector3 fallback)
    {
        if (!parent.TryGetProperty(propertyName, out var value) || value.ValueKind != JsonValueKind.Array) return fallback;
        var components = value.EnumerateArray().Take(3).Select(component => component.GetSingle()).ToArray();
        return components.Length == 3 ? new Vector3(components[0], components[1], components[2]) : fallback;
    }

    private static string ResolveRaceCode(string gamePath)
    {
        var file = Path.GetFileName(gamePath.Replace('/', Path.DirectorySeparatorChar));
        if (file.Length >= 5 && char.ToLowerInvariant(file[0]) == 'c' && file.Substring(1, 4).All(char.IsDigit)) return file.Substring(1, 4);
        throw new InvalidDataException($"Could not identify human race code from '{gamePath}'.");
    }

    private static string ResolveModelSlot(string gamePath)
    {
        var file = Path.GetFileNameWithoutExtension(gamePath.Replace('/', Path.DirectorySeparatorChar));
        if (file.EndsWith("_top", StringComparison.OrdinalIgnoreCase)) return BodySlots.Chest;
        if (file.EndsWith("_dwn", StringComparison.OrdinalIgnoreCase)) return BodySlots.Legs;
        if (file.EndsWith("_glv", StringComparison.OrdinalIgnoreCase)) return BodySlots.Hands;
        if (file.EndsWith("_sho", StringComparison.OrdinalIgnoreCase)) return BodySlots.Feet;
        throw new InvalidDataException($"Unsupported equipment model path '{gamePath}'.");
    }

    private static string CreateWorkDirectory(string kind)
    {
        var directory = Path.Combine(Path.GetTempPath(), "RavaFit", kind, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }
}
