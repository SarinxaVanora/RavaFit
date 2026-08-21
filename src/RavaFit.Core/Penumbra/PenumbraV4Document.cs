using System.Text.Json;
using System.Text.Json.Nodes;
using RavaFit.Core.Models;

namespace RavaFit.Core.Penumbra;

public sealed class PenumbraV4Document
{
    private static readonly JsonDocumentOptions DocumentOptions = new()
    {
        AllowTrailingCommas = true,
        CommentHandling = JsonCommentHandling.Skip,
    };

    private PenumbraV4Document(string metaPath, JsonObject root, IReadOnlyList<V4GroupInfo> groups)
    {
        MetaPath = metaPath;
        Root = root;
        Groups = groups;
    }

    public string MetaPath { get; }
    public string ModRoot => Path.GetDirectoryName(MetaPath) ?? throw new InvalidOperationException("meta.json has no parent directory.");
    public JsonObject Root { get; }
    public IReadOnlyList<V4GroupInfo> Groups { get; }
    public string ModName => ReadString(Root, "Name") ?? Path.GetFileName(ModRoot);
    public string Author => ReadString(Root, "Author") ?? string.Empty;
    public string Description => ReadString(Root, "Description") ?? string.Empty;
    public string Website => ReadString(Root, "Website") ?? string.Empty;

    public static PenumbraV4Document Load(string metaPath)
    {
        var full = Path.GetFullPath(metaPath);
        if (!File.Exists(full))
            throw new FileNotFoundException("Penumbra meta.json was not found.", full);

        var root = JsonNode.Parse(File.ReadAllText(full), documentOptions: DocumentOptions)?.AsObject()
            ?? throw new InvalidDataException("meta.json does not contain a JSON object.");

        var version = ReadInt(root, "FileVersion");
        if (version != 4)
            throw new InvalidDataException($"RavaFit only supports Penumbra V4 mods. Found FileVersion={version?.ToString() ?? "<missing>"}.");

        var groups = new List<V4GroupInfo>();
        if (FindProperty(root, "Groups") is JsonArray groupArray)
        {
            foreach (var groupNode in groupArray.OfType<JsonObject>())
            {
                var name = ReadString(groupNode, "Name") ?? "Unnamed Group";
                var type = ReadString(groupNode, "Type") ?? "Unknown";
                var id = ReadGuid(groupNode, "Id");
                var options = new List<V4OptionInfo>();
                if (FindProperty(groupNode, "Options") is JsonArray optionArray)
                {
                    foreach (var optionNode in optionArray.OfType<JsonObject>())
                    {
                        var optionName = ReadString(optionNode, "Name") ?? "Unnamed Option";
                        options.Add(new V4OptionInfo(ReadGuid(optionNode, "Id"), optionName, optionNode));
                    }
                }
                groups.Add(new V4GroupInfo(id, name, type, groupNode, options));
            }
        }

        return new PenumbraV4Document(full, root, groups);
    }

    public V4GroupInfo GetGroup(string key)
        => Groups.FirstOrDefault(g => string.Equals(g.StableKey, key, StringComparison.OrdinalIgnoreCase)
                                   || string.Equals(g.Name, key, StringComparison.OrdinalIgnoreCase))
            ?? throw new KeyNotFoundException($"Could not find Penumbra group '{key}'.");

    public V4OptionInfo GetOption(V4GroupInfo group, string key)
        => group.Options.FirstOrDefault(o => string.Equals(o.StableKey, key, StringComparison.OrdinalIgnoreCase)
                                          || string.Equals(o.Name, key, StringComparison.OrdinalIgnoreCase))
            ?? throw new KeyNotFoundException($"Could not find option '{key}' in group '{group.Name}'.");

    public IReadOnlyList<ModelRedirect> GetModelRedirects(string groupKey, string optionKey)
    {
        var group = GetGroup(groupKey);
        var option = GetOption(group, optionKey);
        var map = new Dictionary<string, ModelRedirect>(StringComparer.OrdinalIgnoreCase);

        if (FindProperty(Root, "DefaultData") is JsonObject defaults)
            AddModelRedirects(map, defaults, true);
        AddModelRedirects(map, option.Node, false);

        return map.Values.OrderBy(m => m.GamePath, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    public IReadOnlyList<ModelRedirect> GetOptionModelRedirects(string groupKey, string optionKey)
    {
        var group = GetGroup(groupKey);
        var option = GetOption(group, optionKey);
        var map = new Dictionary<string, ModelRedirect>(StringComparer.OrdinalIgnoreCase);
        AddModelRedirects(map, option.Node, false);
        return map.Values.OrderBy(model => model.GamePath, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    public IReadOnlyList<ModelRedirect> GetDefaultModelRedirects()
    {
        var map = new Dictionary<string, ModelRedirect>(StringComparer.OrdinalIgnoreCase);
        if (FindProperty(Root, "DefaultData") is JsonObject defaults)
            AddModelRedirects(map, defaults, true);
        return map.Values.OrderBy(model => model.GamePath, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    public IReadOnlyList<EstSkeletonOverrideInfo> GetEstSkeletonOverrides(
        string groupKey,
        string optionKey,
        string gamePath,
        IReadOnlyDictionary<string, IReadOnlyCollection<string>>? activeSettings = null)
    {
        var group = GetGroup(groupKey);
        var option = GetOption(group, optionKey);
        var target = ParseEquipmentEstTarget(gamePath);
        if (target is null)
            return [];

        var selected = new List<EstSkeletonOverrideInfo>();
        AddEstManipulations(selected, option.Node, $"{group.Name} / {option.Name}");

        if (activeSettings is not null)
        {
            foreach (var (activeGroupName, selectedOptions) in activeSettings)
            {
                var activeGroup = Groups.FirstOrDefault(g => string.Equals(g.Name, activeGroupName, StringComparison.OrdinalIgnoreCase));
                if (activeGroup is null)
                    continue;

                foreach (var selectedName in selectedOptions)
                {
                    var activeOption = activeGroup.Options.FirstOrDefault(o => string.Equals(o.Name, selectedName, StringComparison.OrdinalIgnoreCase)
                        || string.Equals(o.StableKey, selectedName, StringComparison.OrdinalIgnoreCase));
                    if (activeOption is null)
                        continue;

                    if (string.Equals(activeGroup.Name, group.Name, StringComparison.OrdinalIgnoreCase))
                    {
                        // Single replaces the live choice; Multi keeps the other selected options.
                        if (string.Equals(group.Type, "Single", StringComparison.OrdinalIgnoreCase))
                            continue;
                        if (string.Equals(activeOption.StableKey, option.StableKey, StringComparison.OrdinalIgnoreCase)
                            || string.Equals(activeOption.Name, option.Name, StringComparison.OrdinalIgnoreCase))
                            continue;
                    }

                    AddEstManipulations(selected, activeOption.Node, $"active: {activeGroup.Name} / {activeOption.Name}");
                }
            }

            var selectedWinners = FindTargetManipulations(selected, target.Value);
            if (selectedWinners.Count > 1)
                throw new InvalidDataException(
                    $"The selected Penumbra option context contains multiple different {target.Value.Slot} EST entries for {target.Value.Race} {target.Value.Gender} set {target.Value.SetId:D4}: " +
                    string.Join(", ", selectedWinners.Select(x => x.Entry)) + ". RavaFit will not guess which armature is correct.");
            if (selectedWinners.Count == 1)
                return [selectedWinners[0]];

            if (FindProperty(Root, "DefaultData") is JsonObject defaults)
            {
                var defaultManipulations = new List<EstSkeletonOverrideInfo>();
                AddEstManipulations(defaultManipulations, defaults, "DefaultData");
                var defaultWinner = FindWinningTargetManipulation(defaultManipulations, target.Value);
                if (defaultWinner is not null)
                    return [defaultWinner];
            }

            // Live settings were available and gave us no EST, so use the base skeleton.
            return [];
        }

        // If the live collection is unavailable, an EST on the chosen option still wins.
        var directWinner = FindWinningTargetManipulation(selected, target.Value);
        if (directWinner is not null)
            return [directWinner];

        if (FindProperty(Root, "DefaultData") is JsonObject fallbackDefaults)
        {
            var defaultManipulations = new List<EstSkeletonOverrideInfo>();
            AddEstManipulations(defaultManipulations, fallbackDefaults, "DefaultData");
            var defaultWinner = FindWinningTargetManipulation(defaultManipulations, target.Value);
            if (defaultWinner is not null)
                return [defaultWinner];
        }

        var possible = new List<EstSkeletonOverrideInfo>();
        foreach (var g in Groups)
            foreach (var o in g.Options)
                AddEstManipulations(possible, o.Node, $"{g.Name} / {o.Name}");
        if (FindTargetManipulations(possible, target.Value).Any(x => x.Entry > 0))
            throw new InvalidOperationException(
                "RavaFit could not read the active Penumbra mod settings, and this mod contains optional supplemental Body skeletons. " +
                "The bridge will not guess whether IVCS/EST is enabled.");

        return [];
    }

    private static EstSkeletonOverrideInfo? FindWinningTargetManipulation(
        IReadOnlyList<EstSkeletonOverrideInfo> manipulations,
        (string Gender, string Race, int SetId, string Slot) target)
    {
        for (var i = manipulations.Count - 1; i >= 0; i--)
            if (MatchesTarget(manipulations[i], target))
                return manipulations[i];
        return null;
    }

    private static IReadOnlyList<EstSkeletonOverrideInfo> FindTargetManipulations(
        IEnumerable<EstSkeletonOverrideInfo> manipulations,
        (string Gender, string Race, int SetId, string Slot) target)
        => manipulations
            .Where(x => MatchesTarget(x, target))
            .GroupBy(x => x.Entry)
            .Select(g => g.Last())
            .ToArray();

    private static void AddEstManipulations(List<EstSkeletonOverrideInfo> output, JsonObject container, string source)
    {
        if (FindProperty(container, "Manipulations") is not JsonArray manipulations)
            return;
        foreach (var node in manipulations.OfType<JsonObject>())
        {
            if (!string.Equals(ReadString(node, "Type"), "Est", StringComparison.OrdinalIgnoreCase))
                continue;
            if (FindProperty(node, "Manipulation") is not JsonObject est)
                continue;
            var slot = ReadString(est, "Slot");
            var gender = ReadString(est, "Gender");
            var race = ReadString(est, "Race");
            var entry = ReadFlexibleInt(est, "Entry");
            var setId = ReadFlexibleInt(est, "SetId");
            if (slot is null || gender is null || race is null || entry is null || setId is null)
                continue;
            output.Add(new EstSkeletonOverrideInfo(slot, entry.Value, gender, race, setId.Value, source));
        }
    }

    private static int? ReadFlexibleInt(JsonObject obj, string name)
    {
        var node = FindProperty(obj, name);
        if (node is null)
            return null;
        if (node is JsonValue value)
        {
            if (value.TryGetValue<int>(out var i))
                return i;
            if (value.TryGetValue<string>(out var text) && int.TryParse(text, out i))
                return i;
        }
        return null;
    }

    private static (string Gender, string Race, int SetId, string Slot)? ParseEquipmentEstTarget(string gamePath)
    {
        var path = gamePath.Replace('\\', '/').ToLowerInvariant();
        var file = Path.GetFileName(path);
        if (!file.EndsWith("_top.mdl", StringComparison.OrdinalIgnoreCase))
            return null; // Penumbra only resolves equipment EST here for Body/Head.
        var c = System.Text.RegularExpressions.Regex.Match(file, @"^c(?<race>[0-9]{4})e(?<set>[0-9]{4})_top\.mdl$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (!c.Success || !int.TryParse(c.Groups["set"].Value, out var setId))
            return null;
        var identity = RaceIdentityFromCode(c.Groups["race"].Value);
        return identity is null ? null : (identity.Value.Gender, identity.Value.Race, setId, "Body");
    }

    private static bool MatchesTarget(EstSkeletonOverrideInfo x, (string Gender, string Race, int SetId, string Slot) target)
        => x.SetId == target.SetId
        && string.Equals(x.Slot, target.Slot, StringComparison.OrdinalIgnoreCase)
        && string.Equals(x.Gender, target.Gender, StringComparison.OrdinalIgnoreCase)
        && string.Equals(x.Race, target.Race, StringComparison.OrdinalIgnoreCase);

    private static (string Gender, string Race)? RaceIdentityFromCode(string code)
        => code switch
        {
            "0101" => ("Male", "Midlander"),  "0201" => ("Female", "Midlander"),
            "0301" => ("Male", "Highlander"), "0401" => ("Female", "Highlander"),
            "0501" => ("Male", "Elezen"),     "0601" => ("Female", "Elezen"),
            "0701" => ("Male", "Miqote"),    "0801" => ("Female", "Miqote"),
            "0901" => ("Male", "Roegadyn"),  "1001" => ("Female", "Roegadyn"),
            "1101" => ("Male", "Lalafell"),  "1201" => ("Female", "Lalafell"),
            "1301" => ("Male", "AuRa"),      "1401" => ("Female", "AuRa"),
            "1501" => ("Male", "Hrothgar"),  "1601" => ("Female", "Hrothgar"),
            "1701" => ("Male", "Viera"),     "1801" => ("Female", "Viera"),
            _ => null,
        };

    public string ResolvePhysicalPath(ModelRedirect redirect)
    {
        var relative = redirect.RelativePath.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        var full = Path.GetFullPath(Path.Combine(ModRoot, relative));
        var rootWithSeparator = Path.TrimEndingDirectorySeparator(Path.GetFullPath(ModRoot)) + Path.DirectorySeparatorChar;
        if (!full.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Redirection escapes the mod directory: {redirect.RelativePath}");
        return full;
    }

    public static JsonNode? FindProperty(JsonObject obj, string name)
    {
        foreach (var pair in obj)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase))
                return pair.Value;
        return null;
    }

    public static string? FindPropertyName(JsonObject obj, string name)
    {
        foreach (var pair in obj)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase))
                return pair.Key;
        return null;
    }

    public static string? ReadString(JsonObject obj, string name) => FindProperty(obj, name)?.GetValue<string?>();
    public static int? ReadInt(JsonObject obj, string name) => FindProperty(obj, name)?.GetValue<int?>();

    public static Guid? ReadGuid(JsonObject obj, string name)
    {
        var value = ReadString(obj, name);
        return Guid.TryParse(value, out var id) ? id : null;
    }

    private static void AddModelRedirects(Dictionary<string, ModelRedirect> map, JsonObject container, bool fromDefault)
    {
        if (FindProperty(container, "Files") is not JsonObject files)
            return;

        foreach (var pair in files)
        {
            if (!pair.Key.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
                continue;
            if (pair.Value is null)
                continue;
            var relative = pair.Value.GetValue<string>();
            map[pair.Key] = new ModelRedirect(pair.Key, relative, fromDefault);
        }
    }
}
