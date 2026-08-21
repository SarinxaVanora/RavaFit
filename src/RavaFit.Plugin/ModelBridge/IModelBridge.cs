namespace RavaFit.ModelBridge;

internal sealed record ModelBridgeStatus(bool Available, bool Validated, string Name, string Detail)
{
    public bool Ready => Available && Validated;
}

internal sealed record EstSkeletonOverride(string Slot, int Entry, string Gender, string Race, int SetId, string Source);
internal sealed record ModelExportRequest(
    string GamePath,
    string PhysicalMdlPath,
    string OutputGlbPath,
    IReadOnlyList<EstSkeletonOverride>? EstOverrides = null,
    IReadOnlyList<string>? SkeletonPathsOverride = null);
internal sealed record ModelImportRequest(string GamePath, string OriginalPhysicalMdlPath, string InputGlbPath);

internal interface IModelBridge : IDisposable
{
    ModelBridgeStatus Status { get; }
    Task ProbeAsync(CancellationToken cancellationToken = default);
    IReadOnlyList<string> ResolveSkeletonPaths(ModelExportRequest request);
    Task<IReadOnlyList<string>> FindMissingBonesAsync(ModelExportRequest request, CancellationToken cancellationToken = default);
    Task ExportAsync(ModelExportRequest request, CancellationToken cancellationToken = default);
    Task<byte[]> ImportAsync(ModelImportRequest request, CancellationToken cancellationToken = default);
}
