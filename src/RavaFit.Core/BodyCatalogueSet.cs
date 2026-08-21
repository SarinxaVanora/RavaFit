using RavaFit.Core.Models;

namespace RavaFit.Core;

public sealed class BodyCatalogueSet
{
    private readonly List<BodyLibraryInfo> _libraries = [];
    private readonly List<string> _errors = [];

    public IReadOnlyList<BodyLibraryInfo> Libraries => _libraries;
    public IReadOnlyList<string> Errors => _errors;
    public IReadOnlyList<BodyVariantInfo> Variants => _libraries.SelectMany(l => l.Variants).ToArray();

    public void Scan(string directory, string? additionalDirectory = null)
    {
        _libraries.Clear();
        _errors.Clear();
        var discovered = new List<string>();
        discovered.AddRange(EnumerateLibraries(directory, preferUnified: true));
        if (!string.IsNullOrWhiteSpace(additionalDirectory) && !SameDirectory(directory, additionalDirectory))
            discovered.AddRange(EnumerateLibraries(additionalDirectory, preferUnified: false));
        var paths = discovered.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        foreach (var path in paths)
        {
            try
            {
                _libraries.Add(BodyCatalogueReader.Read(path));
            }
            catch (Exception ex)
            {
                _errors.Add($"{Path.GetFileName(path)}: {ex.Message}");
            }
        }
    }

    private static IReadOnlyList<string> EnumerateLibraries(string directory, bool preferUnified)
    {
        Directory.CreateDirectory(directory);
        var unified = Path.Combine(directory, "Bodies.rbody");
        if (preferUnified && File.Exists(unified))
            return [unified];
        return Directory.EnumerateFiles(directory, "*.rbody", SearchOption.TopDirectoryOnly)
            .OrderBy(x => x, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static bool SameDirectory(string left, string right)
        => string.Equals(
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(left)),
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(right)),
            StringComparison.OrdinalIgnoreCase);

    public IReadOnlyList<BodyVariantInfo> ForSlot(string slot, string? raceCode = null)
        => Variants.Where(v => string.Equals(v.Slot, slot, StringComparison.OrdinalIgnoreCase))
                   .Where(v => string.IsNullOrWhiteSpace(raceCode) || v.RaceCodes.Contains(raceCode, StringComparer.OrdinalIgnoreCase))
                   .OrderBy(v => v.BodyName, StringComparer.OrdinalIgnoreCase)
                   .ThenBy(v => v.VariantName, StringComparer.OrdinalIgnoreCase)
                   .ToArray();
}
