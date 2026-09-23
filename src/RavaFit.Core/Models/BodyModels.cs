namespace RavaFit.Core.Models;

public static class BodySlots
{
    public const string Chest = "Chest";
    public const string Legs = "Legs";
    public const string Hands = "Hands";
    public const string Feet = "Feet";

    public static readonly string[] All = [Chest, Legs, Hands, Feet];
}


public static class AccessoryModelSlots
{
    public const string Earrings = "Earrings";
    public const string Necklace = "Necklace";
    public const string Wrists = "Wrists";
    public const string RightRing = "Right Ring";
    public const string LeftRing = "Left Ring";

    public static readonly string[] All = [Earrings, Necklace, Wrists, RightRing, LeftRing];

    public static string? FromGamePath(string gamePath)
    {
        var path = (gamePath ?? string.Empty).Replace('\\', '/');
        if (path.EndsWith("_ear.mdl", StringComparison.OrdinalIgnoreCase)) return Earrings;
        if (path.EndsWith("_nek.mdl", StringComparison.OrdinalIgnoreCase)) return Necklace;
        if (path.EndsWith("_wrs.mdl", StringComparison.OrdinalIgnoreCase)) return Wrists;
        if (path.EndsWith("_rir.mdl", StringComparison.OrdinalIgnoreCase)) return RightRing;
        if (path.EndsWith("_ril.mdl", StringComparison.OrdinalIgnoreCase)) return LeftRing;
        return null;
    }

    public static string Suffix(string slot) => slot switch
    {
        Earrings => "ear",
        Necklace => "nek",
        Wrists => "wrs",
        RightRing => "rir",
        LeftRing => "ril",
        _ => throw new ArgumentOutOfRangeException(nameof(slot), slot, "Unknown accessory model slot."),
    };
}

public sealed record BodyVariantInfo(
    string Collection,
    string BodyId,
    string BodyName,
    string Slot,
    string VariantId,
    string VariantName,
    string RBodyPath,
    string? CanonicalPayloadId,
    IReadOnlyCollection<string> RaceCodes,
    string? TargetOptionProfileId = null,
    string? CanonicalRaceCode = null,
    IReadOnlyCollection<string>? Sexes = null,
    string? SupportSurface = null,
    IReadOnlyCollection<string>? PiercingRaceCodes = null)
{
    public bool SupportsGender(string? gender)
    {
        if (string.IsNullOrWhiteSpace(gender)) return true;
        // Lalafell feet are a shared race asset: female Lalafell use the c1101 feet model too.
        if (string.Equals(gender, "Female", StringComparison.OrdinalIgnoreCase)
            && string.Equals(Slot, BodySlots.Feet, StringComparison.OrdinalIgnoreCase)
            && RaceCodes.Contains("1101", StringComparer.OrdinalIgnoreCase))
            return true;
        if (Sexes is not null && Sexes.Count > 0) return Sexes.Contains(gender, StringComparer.OrdinalIgnoreCase);
        var inferred = RaceCodes.Select(CharacterRaceCatalog.FromCode).Where(identity => identity is not null).Select(identity => identity!.Gender).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        return inferred.Length == 0 || inferred.Contains(gender, StringComparer.OrdinalIgnoreCase);
    }

    public bool IsSmallclothesSupport
        => string.Equals(SupportSurface, "smallclothes", StringComparison.OrdinalIgnoreCase);
}

public sealed record BodyLibraryInfo(
    string Collection,
    string Path,
    int Version,
    IReadOnlyList<BodyVariantInfo> Variants);

public sealed record SlotConversionSelection(
    string Slot,
    BodyVariantInfo? Source,
    BodyVariantInfo Target,
    string? RaceCode = null);


public sealed record TargetBodyOptionGroupInfo(
    string SourcePackage,
    string SourceGroupFile,
    string Kind,
    IReadOnlyCollection<string> Slots,
    IReadOnlyCollection<string> RaceCodes,
    System.Text.Json.Nodes.JsonObject Group,
    IReadOnlyDictionary<string, string> FileAssets);

public sealed record TargetBodyOptionProfileInfo(
    string Id,
    string BodyId,
    string BodyName,
    string Collection,
    string RBodyPath,
    IReadOnlyList<TargetBodyOptionGroupInfo> Groups);
