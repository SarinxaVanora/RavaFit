namespace RavaFit.ModelBridge;

internal sealed class UnavailableModelBridge : IModelBridge
{
    public ModelBridgeStatus Status { get; private set; } = new(false, false, "Model bridge", "Not built into this development binary.");

    public Task ProbeAsync(CancellationToken cancellationToken = default)
    {
        Status = new(false, false, "Model bridge", "The Penumbra model bridge is unavailable; conversion cannot run until the bridge can bind.");
        return Task.CompletedTask;
    }

    public IReadOnlyList<string> ResolveSkeletonPaths(ModelExportRequest request) => throw new InvalidOperationException(Status.Detail);

    public Task<IReadOnlyList<string>> FindMissingBonesAsync(ModelExportRequest request, CancellationToken cancellationToken = default)
        => Task.FromException<IReadOnlyList<string>>(new InvalidOperationException(Status.Detail));

    public Task ExportAsync(ModelExportRequest request, CancellationToken cancellationToken = default)
        => Task.FromException(new InvalidOperationException(Status.Detail));

    public Task<byte[]> ImportAsync(ModelImportRequest request, CancellationToken cancellationToken = default)
        => Task.FromException<byte[]>(new InvalidOperationException(Status.Detail));

    public void Dispose() { }
}
