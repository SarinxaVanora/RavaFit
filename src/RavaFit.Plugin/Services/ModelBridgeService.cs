using Dalamud.Plugin.Services;
using RavaFit.ModelBridge;

namespace RavaFit.Services;

internal sealed class ModelBridgeService : IDisposable
{
    private readonly IPluginLog _log;
    private readonly IModelBridge _bridge;
    private volatile bool _validatedThisSession;
    private string _validationDetail = string.Empty;

    public ModelBridgeService(IFramework framework, IDataManager dataManager, PenumbraService penumbra, IPluginLog log)
    {
        _log = log;
        _bridge = new PenumbraReflectionModelBridge(framework, dataManager, penumbra, log);
    }

    public ModelBridgeStatus Status
    {
        get
        {
            var provider = _bridge.Status;
            return provider with
            {
                Validated = provider.Available && _validatedThisSession,
                Detail = provider.Available && _validatedThisSession && !string.IsNullOrWhiteSpace(_validationDetail)
                    ? _validationDetail
                    : provider.Detail,
            };
        }
    }

    public Task ProbeAsync(CancellationToken cancellationToken = default) => _bridge.ProbeAsync(cancellationToken);

    public void MarkRoundTripValidated(string detail)
    {
        if (!_bridge.Status.Available)
            throw new InvalidOperationException("Cannot validate an unavailable model bridge.");
        _validationDetail = detail;
        _validatedThisSession = true;
    }

    public void ClearValidation()
    {
        _validationDetail = string.Empty;
        _validatedThisSession = false;
    }

    public IReadOnlyList<string> ResolveSkeletonPaths(ModelExportRequest request) => _bridge.ResolveSkeletonPaths(request);

    public Task<IReadOnlyList<string>> FindMissingBonesAsync(ModelExportRequest request, CancellationToken cancellationToken = default)
        => _bridge.FindMissingBonesAsync(request, cancellationToken);

    public async Task ExportAsync(ModelExportRequest request, CancellationToken cancellationToken = default)
    {
        _log.Debug("Exporting {GamePath} through {Bridge}", request.GamePath, Status.Name);
        await _bridge.ExportAsync(request, cancellationToken).ConfigureAwait(false);
    }

    public Task<byte[]> ImportAsync(ModelImportRequest request, CancellationToken cancellationToken = default)
        => _bridge.ImportAsync(request, cancellationToken);

    public void Dispose() => _bridge.Dispose();
}
