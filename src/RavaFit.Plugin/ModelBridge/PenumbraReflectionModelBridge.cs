using System.Collections;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text.RegularExpressions;
using Dalamud.Plugin.Services;
using RavaFit.Services;

namespace RavaFit.ModelBridge;

internal sealed partial class PenumbraReflectionModelBridge : IModelBridge
{
    private readonly IFramework _framework;
    private readonly IDataManager _dataManager;
    private readonly PenumbraService _penumbra;
    private readonly IPluginLog _log;
    private readonly SemaphoreSlim _gate = new(1, 1);

    private BridgeBindings? _bindings;
    private ModelBridgeStatus _status = new(false, false, "Penumbra Model I/O", "Not probed.");

    public PenumbraReflectionModelBridge(IFramework framework, IDataManager dataManager, PenumbraService penumbra, IPluginLog log)
    {
        _framework = framework;
        _dataManager = dataManager;
        _penumbra = penumbra;
        _log = log;
    }

    public ModelBridgeStatus Status => _status;

    public Task ProbeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            _bindings = BridgeBindings.Bind();
            var sharpGltf = _bindings.SceneType.Assembly.GetName();
            _status = new ModelBridgeStatus(true, false, "Penumbra Model I/O", $"Bound through Penumbra's live model-I/O signatures ({sharpGltf.Name} {sharpGltf.Version}). Optional round-trip diagnostics are available, but conversion does not require them.");
        }
        catch (Exception ex)
        {
            _bindings = null;
            _status = new ModelBridgeStatus(false, false, "Penumbra Model I/O", ex.Message);
            _log.Warning(ex, "RavaFit could not bind to Penumbra model I/O.");
        }

        return Task.CompletedTask;
    }

    public async Task ExportAsync(ModelExportRequest request, CancellationToken cancellationToken = default)
    {
        var bindings = RequireBindings();
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!File.Exists(request.PhysicalMdlPath))
                throw new FileNotFoundException("Source MDL does not exist.", request.PhysicalMdlPath);

            Directory.CreateDirectory(Path.GetDirectoryName(request.OutputGlbPath)!);
            var mdlBytes = await File.ReadAllBytesAsync(request.PhysicalMdlPath, cancellationToken).ConfigureAwait(false);
            var mdl = bindings.MdlConstructor.Invoke([mdlBytes]);

            var skeletonPaths = ResolveSkeletonPaths(request);
            var gltfSkeleton = await BuildGltfSkeletonAsync(bindings, skeletonPaths, cancellationToken).ConfigureAwait(false);

            var notifier = bindings.CreateNotifier();
            var config = Activator.CreateInstance(bindings.ExportConfigType)
                ?? throw new InvalidOperationException("Could not create Penumbra export configuration.");
            bindings.SetGenerateMissingBones(config, false);

            // Only carry the existing material names into glTF; do not touch XIV material data here.
            var materialNames = (string[]?)bindings.MdlMaterialsMember.GetValue(mdl) ?? [];
            var gltfMaterials = Array.CreateInstance(bindings.MaterialBuilderType, materialNames.Length);
            for (var i = 0; i < materialNames.Length; i++)
                gltfMaterials.SetValue(bindings.CreateNamedMaterial(materialNames[i]), i);

            var lods = (Array?)bindings.MdlLodsMember.GetValue(mdl)
                ?? throw new InvalidDataException("MDL contains no LOD data.");
            if (lods.Length == 0)
                throw new InvalidDataException("MDL contains no LOD0.");
            var lod0 = lods.GetValue(0)!;
            var meshStart = Convert.ToUInt16(bindings.LodMeshIndexMember.GetValue(lod0));
            var meshCount = Convert.ToUInt16(bindings.LodMeshCountMember.GetValue(lod0));
            if (meshCount == 0)
                throw new InvalidDataException("MDL LOD0 contains no meshes.");

            var scene = bindings.CreateScene();
            bindings.AddSkeletonRootToScene(scene, gltfSkeleton);

            for (var offset = 0; offset < meshCount; offset++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var meshIndex = checked((ushort)(meshStart + offset));
                var exportedMesh = InvokeUnwrapped(bindings.MeshExport, null,
                [
                    config, mdl, (byte)0, meshIndex, gltfMaterials, gltfSkeleton, notifier,
                ]) ?? throw new InvalidDataException($"Penumbra failed to export mesh {meshIndex}.");
                bindings.MeshAddToScene.Invoke(exportedMesh, [scene]);
            }

            var root = bindings.ToGltf2(scene);
            bindings.SaveModelRoot(root, request.OutputGlbPath);
            if (!File.Exists(request.OutputGlbPath) || new FileInfo(request.OutputGlbPath).Length == 0)
                throw new IOException("Penumbra model exporter produced no GLB output.");
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task<byte[]> ImportAsync(ModelImportRequest request, CancellationToken cancellationToken = default)
    {
        var bindings = RequireBindings();
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!File.Exists(request.InputGlbPath))
                throw new FileNotFoundException("Input GLB does not exist.", request.InputGlbPath);

            var root = InvokeUnwrapped(bindings.ModelRootLoad, null, BuildArguments(bindings.ModelRootLoad.GetParameters(), request.InputGlbPath))
                ?? throw new InvalidDataException("SharpGLTF could not load the generated GLB.");
            var notifier = bindings.CreateNotifier();
            var mdl = InvokeUnwrapped(bindings.ModelImport, null, [root, notifier])
                ?? throw new InvalidDataException("Penumbra model importer returned no MDL.");
            var bytes = (byte[]?)InvokeUnwrapped(bindings.MdlWrite, mdl, [])
                ?? throw new InvalidDataException("Penumbra MdlFile writer returned no data.");
            if (bytes.Length == 0)
                throw new InvalidDataException("Penumbra MdlFile writer returned an empty model.");
            return bytes;
        }
        finally
        {
            _gate.Release();
        }
    }

    private BridgeBindings RequireBindings()
        => _bindings ?? throw new InvalidOperationException($"Model bridge is unavailable: {_status.Detail}");

    public async Task<IReadOnlyList<string>> FindMissingBonesAsync(ModelExportRequest request, CancellationToken cancellationToken = default)
    {
        var bindings = RequireBindings();
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!File.Exists(request.PhysicalMdlPath))
                throw new FileNotFoundException("Source MDL does not exist.", request.PhysicalMdlPath);

            var mdlBytes = await File.ReadAllBytesAsync(request.PhysicalMdlPath, cancellationToken).ConfigureAwait(false);
            var mdl = bindings.MdlConstructor.Invoke([mdlBytes]);
            var requiredBones = bindings.GetLod0ReferencedBoneNames(mdl);
            if (requiredBones.Count == 0)
                return Array.Empty<string>();

            var skeletonPaths = ResolveSkeletonPaths(request);
            var gltfSkeleton = await BuildGltfSkeletonAsync(bindings, skeletonPaths, cancellationToken).ConfigureAwait(false);
            var names = bindings.SkeletonNamesMember.GetValue(gltfSkeleton) as IDictionary
                ?? throw new InvalidDataException("Penumbra glTF armature does not expose its bone-name map.");

            return requiredBones
                .Where(name => !names.Contains(name))
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task<object> BuildGltfSkeletonAsync(BridgeBindings bindings, IReadOnlyList<string> skeletonPaths, CancellationToken cancellationToken)
    {
        var skeletonList = Activator.CreateInstance(bindings.XivSkeletonListType)
            ?? throw new InvalidOperationException("Could not create the Penumbra skeleton list.");
        var addSkeleton = bindings.XivSkeletonListType.GetMethod("Add")
            ?? throw new MissingMethodException(bindings.XivSkeletonListType.FullName, "Add");

        for (var i = 0; i < skeletonPaths.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var sklbBytes = ReadResolvedGameFile(skeletonPaths[i]);
            var sklb = bindings.SklbConstructor.Invoke([sklbBytes]);
            var hkx = (byte[]?)bindings.SklbSkeletonMember.GetValue(sklb)
                ?? throw new InvalidDataException($"Could not read Havok payload from {skeletonPaths[i]}.");

            // Space Havok work across framework ticks; hammering it concurrently is unsafe.
            var xml = await _framework.RunOnTick(
                () => (string)InvokeUnwrapped(bindings.HkxToXml, null, [hkx])!,
                delayTicks: i,
                cancellationToken: cancellationToken).ConfigureAwait(false);
            var skeleton = InvokeUnwrapped(bindings.SkeletonFromXml, null, [xml])
                ?? throw new InvalidDataException($"Could not parse skeleton {skeletonPaths[i]}.");
            addSkeleton.Invoke(skeletonList, [skeleton]);
        }

        return InvokeUnwrapped(bindings.ConvertSkeleton, null, [skeletonList])
            ?? throw new InvalidDataException("Penumbra did not produce a glTF armature for this model.");
    }

    public IReadOnlyList<string> ResolveSkeletonPaths(ModelExportRequest request)
    {
        if (request.SkeletonPathsOverride is { Count: > 0 })
            return request.SkeletonPathsOverride.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();

        var normalised = request.GamePath.Replace('\\', '/').ToLowerInvariant();
        var match = RaceCodeRegex().Match(normalised);
        if (!match.Success)
            throw new NotSupportedException("RavaFit currently requires a human model path containing an XIV c#### race code.");

        var raceCode = match.Groups["Race"].Value;
        var paths = new List<string>
        {
            $"chara/human/c{raceCode}/skeleton/base/b0001/skl_c{raceCode}b0001.sklb",
        };

        // Mirror Penumbra's Body EST rule using the selected V4 option rather than guessing.
        var equipment = EquipmentModelRegex().Match(normalised);
        if (equipment.Success && string.Equals(equipment.Groups["Slot"].Value, "top", StringComparison.OrdinalIgnoreCase))
        {
            if (!int.TryParse(equipment.Groups["Set"].Value, out var setId))
                throw new InvalidDataException($"Could not parse equipment set id from {request.GamePath}.");

            var identity = RaceIdentityFromCode(raceCode)
                ?? throw new NotSupportedException($"RavaFit does not recognise human race code c{raceCode}.");
            var matching = (request.EstOverrides ?? [])
                .Where(x => x.SetId == setId
                         && string.Equals(x.Slot, "Body", StringComparison.OrdinalIgnoreCase)
                         && string.Equals(x.Gender, identity.Gender, StringComparison.OrdinalIgnoreCase)
                         && string.Equals(x.Race, identity.Race, StringComparison.OrdinalIgnoreCase))
                .Where(x => x.Entry > 0)
                .GroupBy(x => x.Entry)
                .Select(g => g.First())
                .ToArray();

            if (matching.Length > 1)
                throw new InvalidDataException($"Multiple Body EST skeleton entries apply to {request.GamePath}: {string.Join(", ", matching.Select(x => x.Entry))}.");
            if (matching.Length == 1)
            {
                var entry = matching[0].Entry;
                paths.Add($"chara/human/c{raceCode}/skeleton/body/b{entry:D4}/skl_c{raceCode}b{entry:D4}.sklb");
                _log.Information("RavaFit resolved supplemental Body EST skeleton {Skeleton} from {Source}.", paths[^1], matching[0].Source);
            }
        }

        return paths;
    }

    private static (string Gender, string Race)? RaceIdentityFromCode(string code)
        => code switch
        {
            "0101" => ("Male", "Midlander"),  "0201" => ("Female", "Midlander"),
            "0301" => ("Male", "Highlander"), "0401" => ("Female", "Highlander"),
            "0501" => ("Male", "Elezen"),     "0601" => ("Female", "Elezen"),
            "0701" => ("Male", "Miqote"),    "0801" => ("Female", "Miqote"),
            "0901" => ("Male", "Roegadyn"),  "1001" => ("Female", "Roegadyn"),
            "1101" => ("Male", "Lalafell"),  "1201" => ("Female", "Lalafell"),
            "1301" => ("Male", "AuRa"),      "1401" => ("Female", "AuRa"),
            "1501" => ("Male", "Hrothgar"),  "1601" => ("Female", "Hrothgar"),
            "1701" => ("Male", "Viera"),     "1801" => ("Female", "Viera"),
            _ => null,
        };

    private byte[] ReadResolvedGameFile(string gamePath)
    {
        var resolved = _penumbra.ResolvePlayerPath(gamePath);
        if (Path.IsPathRooted(resolved) && File.Exists(resolved))
            return File.ReadAllBytes(resolved);

        var resource = _dataManager.GetFile(resolved)
            ?? throw new FileNotFoundException($"Could not read game file {gamePath} (resolved as {resolved}).");
        return resource.Data;
    }

    private static object?[] BuildArguments(ParameterInfo[] parameters, string first)
    {
        var args = new object?[parameters.Length];
        args[0] = first;
        for (var i = 1; i < parameters.Length; i++)
            args[i] = parameters[i].HasDefaultValue ? parameters[i].DefaultValue : GetDefault(parameters[i].ParameterType);
        return args;
    }

    private static object? GetDefault(Type type)
        => type.IsValueType ? Activator.CreateInstance(type) : null;

    private static object? GetMemberValue(object instance, string name)
        => instance.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(instance)
        ?? instance.GetType().GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(instance);

    private static object? InvokeUnwrapped(MethodInfo method, object? target, object?[] args)
    {
        try
        {
            return method.Invoke(target, args);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    public void Dispose() => _gate.Dispose();

    [GeneratedRegex(@"(?:^|/)c(?<Race>[0-9]{4})(?:[a-z0-9_/.]|$)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex RaceCodeRegex();

    [GeneratedRegex(@"(?:^|/)c(?<Race>[0-9]{4})e(?<Set>[0-9]{4})_(?<Slot>top|dwn|glv|sho)\.mdl$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex EquipmentModelRegex();

    private sealed class BridgeBindings
    {
        public required ConstructorInfo MdlConstructor { get; init; }
        public required MethodInfo MdlWrite { get; init; }
        public required MemberAccessor MdlMaterialsMember { get; init; }
        public required MemberAccessor MdlLodsMember { get; init; }
        public required MemberAccessor MdlMeshesMember { get; init; }
        public required MemberAccessor MdlBoneTablesMember { get; init; }
        public required MemberAccessor MdlBonesMember { get; init; }
        public required MemberAccessor MeshBoneTableIndexMember { get; init; }
        public required MemberAccessor BoneTableBoneIndexMember { get; init; }
        public required MemberAccessor BoneTableBoneCountMember { get; init; }
        public required ConstructorInfo SklbConstructor { get; init; }
        public required MemberAccessor SklbSkeletonMember { get; init; }
        public required MethodInfo HkxToXml { get; init; }
        public required MethodInfo SkeletonFromXml { get; init; }
        public required MethodInfo ConvertSkeleton { get; init; }
        public required MethodInfo MeshExport { get; init; }
        public required MethodInfo MeshAddToScene { get; init; }
        public required Type ExportConfigType { get; init; }
        public required Type XivSkeletonListType { get; init; }
        public required Type MaterialBuilderType { get; init; }
        public required ConstructorInfo MaterialBuilderConstructor { get; init; }
        public required ConstructorInfo SceneConstructor { get; init; }
        public required Type SceneType { get; init; }
        public required MethodInfo ModelRootLoad { get; init; }
        public required MethodInfo ModelImport { get; init; }
        public required ConstructorInfo IoNotifierConstructor { get; init; }
        public required object PenumbraLogger { get; init; }
        public required MemberAccessor LodMeshIndexMember { get; init; }
        public required MemberAccessor LodMeshCountMember { get; init; }

        private MethodInfo? MaterialMetallicRoughness { get; init; }
        private MethodInfo? MaterialDoubleSide { get; init; }
        private MemberAccessor? GenerateMissingBonesMember { get; init; }
        private MethodInfo SceneAddNode { get; init; } = null!;
        private MemberAccessor SkeletonRootMember { get; init; } = null!;
        public MemberAccessor SkeletonNamesMember { get; init; } = null!;
        private MethodInfo ToGltf2Method { get; init; } = null!;
        private MethodInfo SaveMethod { get; init; } = null!;

        public static BridgeBindings Bind()
        {
            var penumbra = FindAssembly("Penumbra");
            var gameData = FindAssembly("Penumbra.GameData");

            var mdlType = RequireType(gameData, "Penumbra.GameData.Files.MdlFile");
            var sklbType = RequireType(gameData, "Penumbra.GameData.Files.SklbFile");
            var havokType = RequireType(penumbra, "Penumbra.Import.Models.HavokConverter");
            var skeletonConverterType = RequireType(penumbra, "Penumbra.Import.Models.SkeletonConverter");
            var modelExporterType = RequireType(penumbra, "Penumbra.Import.Models.Export.ModelExporter");
            var meshExporterType = RequireType(penumbra, "Penumbra.Import.Models.Export.MeshExporter");
            var modelImporterType = RequireType(penumbra, "Penumbra.Import.Models.Import.ModelImporter");
            var notifierType = RequireType(penumbra, "Penumbra.Import.Models.IoNotifier");
            var penumbraType = RequireType(penumbra, "Penumbra.Penumbra");

            var mdlCtor = mdlType.GetConstructor([typeof(byte[])])
                ?? throw new MissingMethodException(mdlType.FullName, ".ctor(byte[])");
            var mdlWrite = mdlType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "Write" && m.GetParameters().Length == 0)
                ?? throw new MissingMethodException(mdlType.FullName, "Write()");
            var sklbCtor = sklbType.GetConstructor([typeof(byte[])])
                ?? throw new MissingMethodException(sklbType.FullName, ".ctor(byte[])");
            var hkxToXml = havokType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(m => m.Name == "HkxToXml" && m.GetParameters() is [{ ParameterType: var t }] && t == typeof(byte[]))
                ?? throw new MissingMethodException(havokType.FullName, "HkxToXml(byte[])");
            var skeletonFromXml = skeletonConverterType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(m => m.Name == "FromXml" && m.GetParameters() is [{ ParameterType: var t }] && t == typeof(string))
                ?? throw new MissingMethodException(skeletonConverterType.FullName, "FromXml(string)");
            var convertSkeleton = modelExporterType.GetMethod("ConvertSkeleton", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new MissingMethodException(modelExporterType.FullName, "ConvertSkeleton");
            var meshExport = meshExporterType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .SingleOrDefault(m => m.Name == "Export" && m.GetParameters().Length == 7)
                ?? throw new MissingMethodException(meshExporterType.FullName, "Export");

            var meshExportParameters = meshExport.GetParameters();
            var exportConfigType = meshExportParameters[0].ParameterType.GetElementType()
                ?? meshExportParameters[0].ParameterType;
            var materialBuilderType = meshExportParameters
                .Select(p => p.ParameterType)
                .FirstOrDefault(t => t.IsArray && string.Equals(t.GetElementType()?.FullName, "SharpGLTF.Materials.MaterialBuilder", StringComparison.Ordinal))
                ?.GetElementType()
                ?? throw new TypeLoadException("Could not derive Penumbra's SharpGLTF.Materials.MaterialBuilder type from MeshExporter.Export.");

            var meshReturnType = meshExport.ReturnType;
            var meshAddToScene = meshReturnType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(m =>
                {
                    var p = m.GetParameters();
                    return m.Name == "AddToScene" && p.Length == 1
                        && string.Equals(p[0].ParameterType.FullName, "SharpGLTF.Scenes.SceneBuilder", StringComparison.Ordinal);
                })
                ?? throw new MissingMethodException(meshReturnType.FullName, "AddToScene(SharpGLTF.Scenes.SceneBuilder)");
            var sceneType = meshAddToScene.GetParameters()[0].ParameterType;

            var modelImport = modelImporterType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Import" && m.GetParameters().Length == 2)
                .FirstOrDefault(m => m.GetParameters()[1].ParameterType == notifierType
                    && string.Equals(m.GetParameters()[0].ParameterType.FullName, "SharpGLTF.Schema2.ModelRoot", StringComparison.Ordinal))
                ?? throw new MissingMethodException(modelImporterType.FullName, "Import(SharpGLTF.Schema2.ModelRoot, IoNotifier)");
            var modelRootType = modelImport.GetParameters()[0].ParameterType;

            var skeletonEnumerableType = convertSkeleton.GetParameters()[0].ParameterType;
            var xivSkeletonType = skeletonEnumerableType.GetGenericArguments().Single();
            var listType = typeof(List<>).MakeGenericType(xivSkeletonType);

            var materialCtor = materialBuilderType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .OrderBy(c => c.GetParameters().Length)
                .FirstOrDefault(c => c.GetParameters() is [{ ParameterType: var t }] && t == typeof(string))
                ?? throw new MissingMethodException(materialBuilderType.AssemblyQualifiedName, ".ctor(string)");

            // SceneBuilder has an optional name parameter, not necessarily a real parameterless constructor.
            var sceneCtor = sceneType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .OrderBy(c => c.GetParameters().Length)
                .FirstOrDefault(c => c.GetParameters().Length == 0)
                ?? sceneType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                    .OrderBy(c => c.GetParameters().Length)
                    .FirstOrDefault(c => c.GetParameters().All(x => x.HasDefaultValue || x.IsOptional))
                ?? throw new MissingMethodException(sceneType.AssemblyQualifiedName, ".ctor()");

            var modelRootLoad = modelRootType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Load")
                .FirstOrDefault(m =>
                {
                    var p = m.GetParameters();
                    return p.Length >= 1
                        && p[0].ParameterType == typeof(string)
                        && p.Skip(1).All(x => x.HasDefaultValue || x.IsOptional);
                })
                ?? throw new MissingMethodException(modelRootType.AssemblyQualifiedName, "Load(string, optional...)");

            var logger = penumbraType.GetField("Log", BindingFlags.Public | BindingFlags.Static)?.GetValue(null)
                ?? throw new MissingFieldException(penumbraType.FullName, "Log");
            var notifierCtor = notifierType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .SingleOrDefault(c => c.GetParameters().Length == 1)
                ?? throw new MissingMethodException(notifierType.FullName, ".ctor(logger)");

            var skeletonType = Nullable.GetUnderlyingType(meshExportParameters[5].ParameterType)
                ?? meshExportParameters[5].ParameterType;
            var skeletonRoot = RequireMember(skeletonType, "Root");
            var skeletonNames = RequireMember(skeletonType, "Names");
            var mdlMeshes = RequireMember(mdlType, "Meshes");
            var mdlBoneTables = RequireMember(mdlType, "BoneTables");
            var meshType = mdlMeshes.MemberType.GetElementType()
                ?? throw new TypeLoadException("Could not determine Penumbra mesh element type.");
            var boneTableType = mdlBoneTables.MemberType.GetElementType()
                ?? throw new TypeLoadException("Could not determine Penumbra bone-table element type.");
            var meshBoneTableIndex = RequireMember(meshType, "BoneTableIndex");
            var boneTableBoneIndex = RequireMember(boneTableType, "BoneIndex");
            var boneTableBoneCount = RequireMember(boneTableType, "BoneCount");
            var rootType = skeletonRoot.MemberType;
            var sceneAddNode = sceneType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "AddNode" && m.GetParameters().Length >= 1 && m.GetParameters()[0].ParameterType.IsAssignableFrom(rootType))
                ?? sceneType.GetMethods(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault(m => m.Name == "AddNode" && m.GetParameters().Length >= 1)
                ?? throw new MissingMethodException(sceneType.AssemblyQualifiedName, "AddNode");

            // Bind Penumbra's parameterless SceneBuilder.ToGltf2() instance method directly.
            var toGltf = sceneType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "ToGltf2"
                    && m.GetParameters().Length == 0
                    && m.ReturnType == modelRootType)
                ?? throw new MissingMethodException(sceneType.AssemblyQualifiedName, "ToGltf2()");
            var saveMethod = modelRootType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == "Save" && m.GetParameters() is [{ ParameterType: var p }, ..] && p == typeof(string))
                ?? throw new MissingMethodException(modelRootType.AssemblyQualifiedName, "Save(string)");

            return new BridgeBindings
            {
                MdlConstructor = mdlCtor,
                MdlWrite = mdlWrite,
                MdlMaterialsMember = RequireMember(mdlType, "Materials"),
                MdlLodsMember = RequireMember(mdlType, "Lods"),
                MdlMeshesMember = mdlMeshes,
                MdlBoneTablesMember = mdlBoneTables,
                MdlBonesMember = RequireMember(mdlType, "Bones"),
                MeshBoneTableIndexMember = meshBoneTableIndex,
                BoneTableBoneIndexMember = boneTableBoneIndex,
                BoneTableBoneCountMember = boneTableBoneCount,
                SklbConstructor = sklbCtor,
                SklbSkeletonMember = RequireMember(sklbType, "Skeleton"),
                HkxToXml = hkxToXml,
                SkeletonFromXml = skeletonFromXml,
                ConvertSkeleton = convertSkeleton,
                MeshExport = meshExport,
                MeshAddToScene = meshAddToScene,
                ExportConfigType = exportConfigType,
                XivSkeletonListType = listType,
                MaterialBuilderType = materialBuilderType,
                MaterialBuilderConstructor = materialCtor,
                SceneConstructor = sceneCtor,
                SceneType = sceneType,
                ModelRootLoad = modelRootLoad,
                ModelImport = modelImport,
                IoNotifierConstructor = notifierCtor,
                PenumbraLogger = logger,
                LodMeshIndexMember = RequireMember(lodsElementType(mdlType), "MeshIndex"),
                LodMeshCountMember = RequireMember(lodsElementType(mdlType), "MeshCount"),
                MaterialMetallicRoughness = materialBuilderType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .FirstOrDefault(m => m.Name == "WithMetallicRoughnessShader" && m.GetParameters().Length == 0),
                MaterialDoubleSide = materialBuilderType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .FirstOrDefault(m => m.Name == "WithDoubleSide" && m.GetParameters() is [{ ParameterType: var t }] && t == typeof(bool)),
                GenerateMissingBonesMember = TryMember(exportConfigType, "GenerateMissingBones"),
                SceneAddNode = sceneAddNode,
                SkeletonRootMember = skeletonRoot,
                SkeletonNamesMember = skeletonNames,
                ToGltf2Method = toGltf,
                SaveMethod = saveMethod,
            };

            static Type lodsElementType(Type mdl)
            {
                var member = RequireMember(mdl, "Lods");
                return member.MemberType.GetElementType()
                    ?? throw new TypeLoadException("Could not determine Penumbra LOD element type.");
            }
        }

        public IReadOnlyList<string> GetLod0ReferencedBoneNames(object mdl)
        {
            var bones = (string[]?)MdlBonesMember.GetValue(mdl) ?? [];
            var lods = (Array?)MdlLodsMember.GetValue(mdl)
                ?? throw new InvalidDataException("MDL contains no LOD data.");
            if (lods.Length == 0)
                throw new InvalidDataException("MDL contains no LOD0.");

            var lod0 = lods.GetValue(0)!;
            var meshStart = Convert.ToUInt16(LodMeshIndexMember.GetValue(lod0));
            var meshCount = Convert.ToUInt16(LodMeshCountMember.GetValue(lod0));
            var meshes = (Array?)MdlMeshesMember.GetValue(mdl)
                ?? throw new InvalidDataException("MDL contains no mesh table.");
            var boneTables = (Array?)MdlBoneTablesMember.GetValue(mdl)
                ?? throw new InvalidDataException("MDL contains no bone tables.");
            var required = new HashSet<string>(StringComparer.Ordinal);

            for (var offset = 0; offset < meshCount; offset++)
            {
                var meshIndex = checked(meshStart + offset);
                if (meshIndex >= meshes.Length)
                    throw new InvalidDataException($"LOD0 references missing mesh {meshIndex}.");
                var mesh = meshes.GetValue(meshIndex)!;
                var boneTableIndex = Convert.ToUInt16(MeshBoneTableIndexMember.GetValue(mesh));
                if (boneTableIndex == byte.MaxValue)
                    continue;
                if (boneTableIndex >= boneTables.Length)
                    throw new InvalidDataException($"Mesh {meshIndex} references missing bone table {boneTableIndex}.");

                var boneTable = boneTables.GetValue(boneTableIndex)!;
                var indices = (ushort[]?)BoneTableBoneIndexMember.GetValue(boneTable) ?? [];
                var boneCount = Math.Min(indices.Length, checked((int)Convert.ToUInt32(BoneTableBoneCountMember.GetValue(boneTable))));
                for (var i = 0; i < boneCount; i++)
                {
                    var boneIndex = indices[i];
                    if (boneIndex >= bones.Length)
                        throw new InvalidDataException($"Bone table {boneTableIndex} references missing model bone {boneIndex}.");
                    if (!string.IsNullOrWhiteSpace(bones[boneIndex]))
                        required.Add(bones[boneIndex]);
                }
            }

            return required.OrderBy(name => name, StringComparer.Ordinal).ToArray();
        }

        public object CreateNotifier() => IoNotifierConstructor.Invoke([PenumbraLogger]);

        public object CreateScene()
        {
            var p = SceneConstructor.GetParameters();
            var args = new object?[p.Length];
            for (var i = 0; i < p.Length; i++)
                args[i] = p[i].HasDefaultValue ? p[i].DefaultValue : GetDefault(p[i].ParameterType);
            return SceneConstructor.Invoke(args);
        }

        public object CreateNamedMaterial(string name)
        {
            var material = MaterialBuilderConstructor.Invoke([name]);
            MaterialMetallicRoughness?.Invoke(material, []);
            MaterialDoubleSide?.Invoke(material, [true]);
            return material;
        }

        public void SetGenerateMissingBones(object config, bool value) => GenerateMissingBonesMember?.SetValue(config, value);

        public void AddSkeletonRootToScene(object scene, object skeleton)
        {
            var root = SkeletonRootMember.GetValue(skeleton);
            if (root is null)
                throw new InvalidDataException("glTF skeleton has no root node.");
            var args = new object?[SceneAddNode.GetParameters().Length];
            args[0] = root;
            for (var i = 1; i < args.Length; i++)
                args[i] = SceneAddNode.GetParameters()[i].HasDefaultValue ? SceneAddNode.GetParameters()[i].DefaultValue : GetDefault(SceneAddNode.GetParameters()[i].ParameterType);
            InvokeUnwrapped(SceneAddNode, scene, args);
        }

        public object ToGltf2(object scene)
            => InvokeUnwrapped(ToGltf2Method, scene, [])
                ?? throw new InvalidDataException("SharpGLTF ToGltf2 returned null.");

        public void SaveModelRoot(object root, string path)
        {
            var p = SaveMethod.GetParameters();
            var args = new object?[p.Length];
            args[0] = path;
            for (var i = 1; i < p.Length; i++)
                args[i] = p[i].HasDefaultValue ? p[i].DefaultValue : GetDefault(p[i].ParameterType);
            InvokeUnwrapped(SaveMethod, root, args);
        }

        private static Assembly FindAssembly(string name)
            => AppDomain.CurrentDomain.GetAssemblies().FirstOrDefault(a => string.Equals(a.GetName().Name, name, StringComparison.OrdinalIgnoreCase))
            ?? throw new FileNotFoundException($"Required loaded assembly {name} was not found. Is Penumbra running?");

        private static Type RequireType(Assembly assembly, string name)
            => assembly.GetType(name, false) ?? throw new TypeLoadException($"{name} was not found in {assembly.GetName().Name}.");

        private static MemberAccessor RequireMember(Type type, string name)
            => TryMember(type, name) ?? throw new MissingMemberException(type.FullName, name);

        private static MemberAccessor? TryMember(Type type, string name)
        {
            var property = type.GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (property is not null)
                return new MemberAccessor(property);
            var field = type.GetField(name, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            return field is not null ? new MemberAccessor(field) : null;
        }
    }

    private sealed class MemberAccessor
    {
        private readonly PropertyInfo? _property;
        private readonly FieldInfo? _field;

        public MemberAccessor(PropertyInfo property) => _property = property;
        public MemberAccessor(FieldInfo field) => _field = field;

        public Type MemberType => _property?.PropertyType ?? _field!.FieldType;
        public object? GetValue(object target) => _property?.GetValue(target) ?? _field?.GetValue(target);
        public void SetValue(object target, object? value)
        {
            if (_property is not null)
                _property.SetValue(target, value);
            else
                _field!.SetValue(target, value);
        }
    }
}
