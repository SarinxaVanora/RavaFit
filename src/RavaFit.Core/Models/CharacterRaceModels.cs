namespace RavaFit.Core.Models;

public sealed record CharacterRaceIdentity(string Code, string Gender, string Race)
{
    public string DisplayName => $"{Gender} {Race}";
    public bool IsMale => string.Equals(Gender, "Male", StringComparison.OrdinalIgnoreCase);
    public string MidlanderFallbackCode => IsMale ? "0101" : "0201";
}

public static class CharacterRaceCatalog
{
    public static readonly CharacterRaceIdentity[] All =
    [
        new("0101", "Male", "Midlander"),
        new("0201", "Female", "Midlander"),
        new("0301", "Male", "Highlander"),
        new("0401", "Female", "Highlander"),
        new("0501", "Male", "Elezen"),
        new("0601", "Female", "Elezen"),
        new("0701", "Male", "Miqo'te"),
        new("0801", "Female", "Miqo'te"),
        new("0901", "Male", "Roegadyn"),
        new("1001", "Female", "Roegadyn"),
        new("1101", "Male", "Lalafell"),
        new("1201", "Female", "Lalafell"),
        new("1301", "Male", "Au Ra"),
        new("1401", "Female", "Au Ra"),
        new("1501", "Male", "Hrothgar"),
        new("1601", "Female", "Hrothgar"),
        new("1701", "Male", "Viera"),
        new("1801", "Female", "Viera"),
    ];

    public static CharacterRaceIdentity? FromCode(string? code)
        => string.IsNullOrWhiteSpace(code) ? null : All.FirstOrDefault(x => string.Equals(x.Code, code, StringComparison.OrdinalIgnoreCase));

    public static CharacterRaceIdentity? FromCustomize(byte race, byte gender, byte tribe)
    {
        var female = gender != 0;
        var code = race switch
        {
            1 when tribe == 2 => female ? "0401" : "0301", // Hyur Highlander
            1 => female ? "0201" : "0101",                 // Hyur Midlander
            2 => female ? "0601" : "0501",                 // Elezen
            3 => female ? "1201" : "1101",                 // Lalafell
            4 => female ? "0801" : "0701",                 // Miqo'te
            5 => female ? "1001" : "0901",                 // Roegadyn
            6 => female ? "1401" : "1301",                 // Au Ra
            7 => female ? "1601" : "1501",                 // Hrothgar
            8 => female ? "1801" : "1701",                 // Viera
            _ => null,
        };
        return FromCode(code);
    }

    public static string RewriteHumanRaceCode(string gamePath, string targetRaceCode)
    {
        if (FromCode(targetRaceCode) is null)
            throw new ArgumentOutOfRangeException(nameof(targetRaceCode), $"Unknown XIV human race code c{targetRaceCode}.");
        var normalised = gamePath.Replace('\\', '/');
        var fileStart = normalised.LastIndexOf('/') + 1;
        if (normalised.Length < fileStart + 5 || char.ToLowerInvariant(normalised[fileStart]) != 'c')
            throw new InvalidDataException($"Model path '{gamePath}' does not expose a c#### human race code.");
        var existing = normalised.AsSpan(fileStart + 1, 4);
        if (!existing.ToString().All(char.IsDigit))
            throw new InvalidDataException($"Model path '{gamePath}' does not expose a c#### human race code.");
        return normalised[..(fileStart + 1)] + targetRaceCode + normalised[(fileStart + 5)..];
    }

    public static string? ResolveBodyPayloadRace(BodyVariantInfo variant, string targetRaceCode)
    {
        var target = FromCode(targetRaceCode);
        if (target is null)
            return null;
        if (variant.RaceCodes.Contains(target.Code, StringComparer.OrdinalIgnoreCase))
            return target.Code;
        if (string.Equals(target.Code, "1201", StringComparison.OrdinalIgnoreCase)
            && string.Equals(variant.Slot, BodySlots.Feet, StringComparison.OrdinalIgnoreCase)
            && variant.RaceCodes.Contains("1101", StringComparer.OrdinalIgnoreCase))
            return "1101";
        if (string.Equals(target.Race, "Lalafell", StringComparison.OrdinalIgnoreCase))
            return null;
        return variant.RaceCodes.Contains(target.MidlanderFallbackCode, StringComparer.OrdinalIgnoreCase)
            ? target.MidlanderFallbackCode
            : null;
    }

    public static bool BodySupports(BodyVariantInfo variant, string targetRaceCode)
        => ResolveBodyPayloadRace(variant, targetRaceCode) is not null;

    public static bool RequiresSmallclothesTarget(string? sourceGender, CharacterRaceIdentity target, string slot)
        => string.Equals(sourceGender, "Female", StringComparison.OrdinalIgnoreCase)
           && target.IsMale
           && string.Equals(slot, BodySlots.Legs, StringComparison.OrdinalIgnoreCase);

    public static bool RequiresSmallclothesSource(CharacterRaceIdentity source, string? targetGender, string slot)
        => source.IsMale
           && string.Equals(targetGender, "Female", StringComparison.OrdinalIgnoreCase)
           && string.Equals(slot, BodySlots.Legs, StringComparison.OrdinalIgnoreCase);

    public static bool RequiresCrossSexSmallclothesBridge(CharacterRaceIdentity source, CharacterRaceIdentity target, string slot)
        => !string.Equals(source.Gender, target.Gender, StringComparison.OrdinalIgnoreCase)
           && string.Equals(slot, BodySlots.Legs, StringComparison.OrdinalIgnoreCase);

    public static bool RequiresSkeletonProportionRetarget(string? sourceRaceCode, string? targetRaceCode)
        => FromCode(sourceRaceCode) is not null
           && FromCode(targetRaceCode) is not null
           && !string.Equals(sourceRaceCode, targetRaceCode, StringComparison.OrdinalIgnoreCase);
}
