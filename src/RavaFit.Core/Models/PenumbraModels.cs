using System.Text.Json.Nodes;

namespace RavaFit.Core.Models;

public sealed record PenumbraModInfo(string Directory, string Name, string ModRoot);

public sealed record V4OptionInfo(Guid? Id, string Name, JsonObject Node)
{
    public string StableKey => Id?.ToString("D") ?? Name;
}

public sealed record V4GroupInfo(Guid? Id, string Name, string Type, JsonObject Node, IReadOnlyList<V4OptionInfo> Options)
{
    public string StableKey => Id?.ToString("D") ?? Name;
    public bool SupportsAppend => Type is "Single" or "Multi";
}

public sealed record ModelRedirect(string GamePath, string RelativePath, bool FromDefaultData)
{
    public string FileName => Path.GetFileName(GamePath.Replace('/', Path.DirectorySeparatorChar));
}

public sealed record V4AppendRequest(
    string MetaPath,
    string GroupKey,
    string SourceOptionKey,
    string NewOptionName,
    IReadOnlyDictionary<string, string> ModelRedirections,
    IReadOnlyDictionary<string, string>? AdditionalFileRedirections = null);

public sealed record V4AppendResult(Guid NewOptionId, string BackupPath, string MetaPath, int NewOptionCount);

public sealed record GeneratedModelFile(string GamePath, string RelativePath, string AbsolutePath);

public sealed record EstSkeletonOverrideInfo(string Slot, int Entry, string Gender, string Race, int SetId, string Source);

public sealed record V4TargetBodyGroupRequest(
    string ParentGroupKey,
    string ParentOutputOptionName,
    string BodyName,
    string Slot,
    string TargetGamePath,
    string SourceKey,
    JsonObject Group);

public sealed record V4CustomisationGroupRequest(
    string BodyName,
    string Slot,
    string TargetGamePath,
    string SourceKey,
    JsonObject Group);

public sealed record V4CustomisationResult(string BackupPath, string MetaPath, int RemovedPiercingGroups, int AddedGroups);

[Flags]
public enum ModOptionKind
{
    None = 0,
    Gear = 1,
    Sound = 2,
    Vfx = 4,
    Animation = 8,
    RavaFitGenerated = 16,
    Other = 32,
}

public sealed record ModOptionCleanupCandidate(
    string GroupKey,
    string GroupName,
    string GroupType,
    string OptionKey,
    string OptionName,
    ModOptionKind Kinds,
    int LocalFileReferenceCount);

public sealed record ModOptionCleanupSelection(string GroupKey, string OptionKey);
public sealed record ModOptionCleanupResult(int RemovedOptions, int RemovedGroups, int RemovedFiles, string? Warning = null);
