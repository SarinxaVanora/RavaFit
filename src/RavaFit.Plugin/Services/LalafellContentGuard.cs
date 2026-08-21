using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using RavaFit.Core.Models;
using RavaFit.Core.Penumbra;

namespace RavaFit.Services;

internal sealed record LalafellGuardResult(bool Blocked, IReadOnlyList<string> Reasons)
{
    public static readonly LalafellGuardResult Allowed = new(false, []);
}

internal static partial class LalafellContentGuard
{
    private static readonly string[] ExplicitTerms =
    [
        "nsfw", "18+", "adult only", "explicit", "nude", "nudity", "naked", "genital", "genitals",
        "penis", "penile", "vagina", "vaginal", "vulva", "labia", "clitoris", "futanari", "futa",
        "erect", "erection", "masturbat", "orgasm", "blowjob", "handjob", "intercourse", "penetration",
        "semen", "cum", "cumshot", "porn", "pornographic", "topless", "nipple", "nipples", "anus",
        "dildo", "buttplug", "butt plug", "sex toy", "strap on", "strap-on", "bdsm", "bondage", "erotic",
        "sex pose", "sexual animation", "sexual pose", "anal sex", "oral sex", "vaginal sex", "cock", "dick", "pussy"
    ];

    public static bool IsLalafell(CharacterRaceIdentity? identity)
        => identity is not null && string.Equals(identity.Race, "Lalafell", StringComparison.OrdinalIgnoreCase);

    public static LalafellGuardResult InspectOutfit(PenumbraModInfo mod, IReadOnlyList<(string GroupKey, string OptionKey)> selections, IEnumerable<BodyVariantInfo>? selectedBodies = null)
    {
        var document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
        var evidence = new List<string>();
        CheckText($"{document.ModName} {document.Description}", "mod metadata", evidence);
        if (PenumbraV4Document.FindProperty(document.Root, "Tags") is JsonNode tags) CheckNode(tags, "mod tags", evidence);
        if (PenumbraV4Document.FindProperty(document.Root, "DefaultData") is JsonNode defaults) CheckNode(defaults, "required/default assets", evidence);

        foreach (var selection in selections.Distinct())
        {
            var group = document.GetGroup(selection.GroupKey);
            var option = document.GetOption(group, selection.OptionKey);
            CheckText($"{group.Name} {ReadString(group.Node, "Description")}", $"group '{group.Name}'", evidence);
            CheckNode(option.Node, $"option '{group.Name} / {option.Name}'", evidence);
        }

        if (selectedBodies is not null)
        {
            foreach (var body in selectedBodies.DistinctBy(body => $"{body.RBodyPath}|{body.BodyId}|{body.VariantId}"))
                CheckText($"{body.Collection} {body.BodyName} {body.VariantName}", $"body '{body.BodyName} / {body.VariantName}'", evidence);
        }
        return BuildResult(evidence);
    }

    public static LalafellGuardResult InspectAnimation(PenumbraModInfo mod)
    {
        var document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
        var evidence = new List<string>();
        CheckText($"{document.ModName} {document.Description}", "mod metadata", evidence);
        if (PenumbraV4Document.FindProperty(document.Root, "Tags") is JsonNode tags) CheckNode(tags, "mod tags", evidence);
        if (PenumbraV4Document.FindProperty(document.Root, "DefaultData") is JsonNode defaults) CheckNode(defaults, "required/default assets", evidence);
        if (PenumbraV4Document.FindProperty(document.Root, "Groups") is JsonNode groups) CheckNode(groups, "animation groups/options", evidence);
        return BuildResult(evidence);
    }

    private static LalafellGuardResult BuildResult(List<string> evidence)
    {
        var unique = evidence.Distinct(StringComparer.OrdinalIgnoreCase).Take(8).ToArray();
        return unique.Length == 0 ? LalafellGuardResult.Allowed : new LalafellGuardResult(true, unique);
    }

    private static void CheckNode(JsonNode? node, string source, List<string> evidence)
    {
        if (node is null) return;
        if (node is JsonValue value && value.TryGetValue<string>(out var text))
        {
            CheckText(text, source, evidence);
            return;
        }
        if (node is JsonArray array)
        {
            foreach (var child in array) CheckNode(child, source, evidence);
            return;
        }
        if (node is not JsonObject obj) return;
        foreach (var pair in obj)
        {
            CheckText(pair.Key, source, evidence);
            CheckNode(pair.Value, source, evidence);
        }
    }

    private static void CheckText(string? text, string source, List<string> evidence)
    {
        if (string.IsNullOrWhiteSpace(text) || evidence.Count >= 12) return;
        var normalised = text.Replace('_', ' ').Replace('-', ' ').Replace('/', ' ').Replace('\\', ' ');
        foreach (var term in ExplicitTerms)
        {
            if (!ContainsTerm(normalised, term)) continue;
            evidence.Add($"{source}: '{term}'");
            if (evidence.Count >= 12) return;
        }
    }

    private static bool ContainsTerm(string text, string term)
    {
        if (term.Any(char.IsWhiteSpace) || term.Contains('+'))
            return text.Contains(term, StringComparison.OrdinalIgnoreCase);
        return Regex.IsMatch(text, $@"(?<![A-Za-z0-9]){Regex.Escape(term)}(?![A-Za-z0-9])", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    }

    private static string? ReadString(JsonObject obj, string name)
    {
        var node = PenumbraV4Document.FindProperty(obj, name);
        return node is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
    }
}
