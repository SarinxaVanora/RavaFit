using FFXIVClientStructs.Havok.Common.Base.Math.QsTransform;
using System.Collections.ObjectModel;
using System.Text.RegularExpressions;

namespace RavaFit.Services.Animation;

internal static partial class XivAnimationSkeletonIdentity
{
    [GeneratedRegex(@"^c\d{4}$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex HumanSkeletonCodeRegex();

    [GeneratedRegex(@"^skl_c\d{4}b\d{4}$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex HumanBaseSkeletonRegex();

    [GeneratedRegex(@"^c\d{4}(?:f\d{4})?_\d+:mdl:(?<partial>[a-z0-9_]+)$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex HumanModelPartialRegex();

    public static string NormalizeSkeletonKey(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return string.Empty;
        var normalized = value.Replace('\\', '/').Trim();
        var slash = normalized.LastIndexOf('/');
        if (slash >= 0 && slash < normalized.Length - 1) normalized = normalized[(slash + 1)..];
        if (normalized.EndsWith(".sklb", StringComparison.OrdinalIgnoreCase)) normalized = normalized[..^5];
        var partialMatch = HumanModelPartialRegex().Match(normalized);
        if (partialMatch.Success)
        {
            var partialName = partialMatch.Groups["partial"].Value;
            if (!string.IsNullOrWhiteSpace(partialName)) normalized = partialName;
        }
        return normalized.ToLowerInvariant();
    }

    public static string NormalizeHumanAnimationFamilyKey(string? value)
    {
        var normalized = NormalizeSkeletonKey(value);
        if (string.IsNullOrWhiteSpace(normalized)) return string.Empty;
        if (HumanSkeletonCodeRegex().IsMatch(normalized) || HumanBaseSkeletonRegex().IsMatch(normalized)) return "human-base";
        if (string.Equals(normalized, "n_root", StringComparison.Ordinal)) return "human-partial:n_root";
        if (normalized.StartsWith("j_", StringComparison.Ordinal)) return "human-partial:" + normalized;
        return string.Empty;
    }
    public static bool IsHumanPlayerAnimationPapGamePath(string? gamePath)
    {
        if (string.IsNullOrWhiteSpace(gamePath)) return false;
        var normalized = gamePath.Trim().Replace('\\', '/').TrimStart('/').ToLowerInvariant();

        return HumanBodyPapGamePathRegex().IsMatch(normalized)
            && !normalized.Contains("/mt_", StringComparison.Ordinal);
    }

    [GeneratedRegex(@"^chara/human/c\d{4}/animation/a0001/.+\.pap$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex HumanBodyPapGamePathRegex();

}

internal sealed class AnimationTargetSkeletonSnapshot
{
    public AnimationTargetSkeletonSnapshot(string resourcePath, string skeletonName, IReadOnlyList<string?> boneNamesByIndex, IReadOnlyDictionary<string, short> boneNameToIndex, IReadOnlyList<hkQsTransformf>? referencePoseByIndex = null)
    {
        ResourcePath = resourcePath ?? string.Empty;
        SkeletonName = skeletonName ?? string.Empty;
        BoneNamesByIndex = boneNamesByIndex ?? Array.Empty<string?>();
        BoneNameToIndex = boneNameToIndex ?? new ReadOnlyDictionary<string, short>(new Dictionary<string, short>(StringComparer.OrdinalIgnoreCase));
        ReferencePoseByIndex = referencePoseByIndex ?? Array.Empty<hkQsTransformf>();
        NormalizedSkeletonName = XivAnimationSkeletonIdentity.NormalizeSkeletonKey(SkeletonName);
        NormalizedResourceName = XivAnimationSkeletonIdentity.NormalizeSkeletonKey(ResourcePath);
        HumanAnimationFamilyKey = string.IsNullOrWhiteSpace(XivAnimationSkeletonIdentity.NormalizeHumanAnimationFamilyKey(SkeletonName))
            ? XivAnimationSkeletonIdentity.NormalizeHumanAnimationFamilyKey(ResourcePath)
            : XivAnimationSkeletonIdentity.NormalizeHumanAnimationFamilyKey(SkeletonName);
    }

    public string ResourcePath { get; }
    public string SkeletonName { get; }
    public string NormalizedSkeletonName { get; }
    public string NormalizedResourceName { get; }
    public string HumanAnimationFamilyKey { get; }
    public bool IsHumanAnimationSkeleton => !string.IsNullOrWhiteSpace(HumanAnimationFamilyKey);
    public IReadOnlyList<string?> BoneNamesByIndex { get; }
    public IReadOnlyDictionary<string, short> BoneNameToIndex { get; }
    public IReadOnlyList<hkQsTransformf> ReferencePoseByIndex { get; }
    public int BoneCount => BoneNamesByIndex.Count;
}

internal sealed record AnimationSkeletonChoice(string Id, string DisplayName, string Description, bool IsLive);

internal enum AnimationPapRewriteStatus
{
    OriginalSafe,
    Converted,
    Blocked,
}

internal sealed record AnimationPapRewriteResult(AnimationPapRewriteStatus Status, int KeptTrackCount, int DroppedTrackCount, int BindingCount, string Reason);

internal enum AnimationPortStage
{
    Idle,
    Copying,
    ReadingSkeleton,
    Converting,
    Writing,
    Complete,
    Failed,
}

internal sealed record AnimationPortProgress(AnimationPortStage Stage, float Progress, string Detail = "");
internal sealed record AnimationPortResult(string ModName, string ModRoot, int ConvertedPaps, int UnchangedPaps, int DroppedTracks, int BindingCount);
