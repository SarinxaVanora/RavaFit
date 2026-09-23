namespace RavaFit.Core.Models;

public enum ConversionStage
{
    Idle,
    Preparing,
    Exporting,
    Fitting,
    Building,
    UpdatingPenumbra,
    Complete,
    Failed,
}

public sealed record ConversionProgress(ConversionStage Stage, float Progress, string Detail = "");

public sealed record RoundTripRequest(
    string ModDirectory,
    string ModName,
    string ModRoot,
    string GroupKey,
    string SourceOptionKey,
    string GamePath,
    string SourceMdlPath,
    string OutputOptionName);

public sealed record ConversionRequest(
    string ModDirectory,
    string ModName,
    string ModRoot,
    string GroupKey,
    string SourceOptionKey,
    IReadOnlyList<ModelRedirect> Models,
    IReadOnlyList<SlotConversionSelection> Slots,
    string OutputOptionName,
    string? PrimaryBodySlot = null,
    bool? SourceContainsBody = null,
    bool? TransplantTargetBody = null);

public sealed record GarmentCoverageSlot(
    string Slot,
    bool Recommended,
    bool Primary,
    float Confidence,
    string Reason);

public sealed record SourceBodyMatch(
    string Slot,
    BodyVariantInfo Variant,
    float Confidence,
    string Reason);

public sealed record GarmentCoverageResult(
    string PrimarySlot,
    IReadOnlyDictionary<string, GarmentCoverageSlot> Slots,
    IReadOnlyDictionary<string, SourceBodyMatch> SourceMatches,
    bool SourceContainsBody = true);

public sealed record RaceSwapConversionRequest(
    string ModDirectory,
    string ModName,
    string ModRoot,
    string GroupKey,
    string SourceOptionKey,
    IReadOnlyList<ModelRedirect> Models,
    IReadOnlyList<SlotConversionSelection> Slots,
    string OutputOptionName,
    string TargetRaceCode,
    string? PrimaryBodySlot = null,
    bool? SourceContainsBody = null,
    bool? TransplantTargetBody = null);
