using Dalamud.Configuration;

namespace RavaFit;

[Serializable]
public sealed class Configuration : IPluginConfiguration
{
    public int Version { get; set; } = 3;
    public string BodyLibraryDirectory { get; set; } = string.Empty;
    public string RuntimeDirectory { get; set; } = string.Empty;
    public string RuntimeProductionRevision { get; set; } = string.Empty;
    public string DeveloperPythonPath { get; set; } = string.Empty;
    public bool OpenLogOnFailure { get; set; }
}
