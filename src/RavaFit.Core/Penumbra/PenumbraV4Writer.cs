using System.Text.Json;
using System.Text.Json.Nodes;
using RavaFit.Core.Models;

namespace RavaFit.Core.Penumbra;

public sealed class PenumbraV4Writer
{
    private static readonly JsonSerializerOptions WriteOptions = new()
    {
        WriteIndented = true,
    };

    public async Task<V4CustomisationResult> ReplaceModelAndPiercingControlsAsync(
        string metaPathRaw,
        string groupKey,
        string optionKey,
        string gamePathRaw,
        string modelRelativePathRaw,
        IReadOnlyList<V4CustomisationGroupRequest> piercingGroups,
        IReadOnlyList<string>? sourcePiercingResourcePaths = null,
        CancellationToken cancellationToken = default)
    {
        var metaPath = Path.GetFullPath(metaPathRaw);
        var originalMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
        var document = PenumbraV4Document.Load(metaPath);
        var parentGroup = document.GetGroup(groupKey);
        var parentOption = document.GetOption(parentGroup, optionKey);
        var parentOptionId = parentOption.Id ?? throw new InvalidDataException($"Penumbra option '{parentGroup.Name} / {parentOption.Name}' has no Id, so RavaFit cannot scope replacement piercing controls to it safely.");

        var filesName = PenumbraV4Document.FindPropertyName(parentOption.Node, "Files") ?? "Files";
        var files = PenumbraV4Document.FindProperty(parentOption.Node, "Files") as JsonObject;
        if (files is null)
        {
            files = new JsonObject();
            parentOption.Node[filesName] = files;
        }
        files[NormalizeGamePath(gamePathRaw)] = NormalizeRelativePath(modelRelativePathRaw);

        var groups = PenumbraV4Document.FindProperty(document.Root, "Groups") as JsonArray
            ?? throw new InvalidDataException("Penumbra V4 meta.json has no Groups array.");
        var removed = 0;
        for (var i = groups.Count - 1; i >= 0; i--)
        {
            if (groups[i] is not JsonObject groupNode || ReferenceEquals(groupNode, parentGroup.Node))
                continue;
            if (PenumbraV4Document.FindProperty(groupNode, "Condition") is JsonObject condition
                && ReadString(condition, "Type") is { } conditionType
                && string.Equals(conditionType, "Setting", StringComparison.OrdinalIgnoreCase)
                && ReadString(condition, "Setting") is { } setting
                && Guid.TryParse(setting, out var settingId)
                && settingId != parentOptionId)
                continue;

            var piercingKind = ClassifyPiercingGroup(groupNode, sourcePiercingResourcePaths);
            if (piercingKind == PiercingGroupKind.None)
                continue;
            if (piercingKind == PiercingGroupKind.Mixed)
                throw new NotSupportedException($"Piercing controls in group '{ReadString(groupNode, "Name") ?? "Unnamed group"}' are mixed with unrelated controls. RavaFit refused to remove the whole group; split the piercing controls into their own Penumbra group first.");

            groups.RemoveAt(i);
            removed++;
        }

        var pageNames = PenumbraV4Document.FindProperty(document.Root, "PageNames") as JsonObject;
        if (pageNames is null)
        {
            pageNames = new JsonObject();
            SetProperty(document.Root, "PageNames", pageNames);
        }
        var existingCustomPage = pageNames.FirstOrDefault(pair => pair.Value is JsonValue value
            && value.TryGetValue<string>(out var text)
            && string.Equals(text, "Customise", StringComparison.OrdinalIgnoreCase));
        var customPage = int.TryParse(existingCustomPage.Key, out var parsedCustomPage)
            ? parsedCustomPage
            : Math.Max(0, document.Groups.Select(g => ReadInt(g.Node, "Page") ?? -1).Concat(pageNames.Select(pair => int.TryParse(pair.Key, out var page) ? page : -1)).DefaultIfEmpty(-1).Max() + 1);
        pageNames[customPage.ToString()] = "Customise";

        var added = 0;
        foreach (var request in piercingGroups)
        {
            var clone = request.Group.DeepClone().AsObject();
            if (PenumbraV4Document.FindProperty(clone, "Condition") is not null)
                throw new NotSupportedException($"Captured piercing group '{ReadString(clone, "Name") ?? request.SourceKey}' already has a dependency RavaFit cannot safely replace.");
            RetargetTargetBodyGroup(clone, request.TargetGamePath);
            if (TryFoldAlwaysOnTargetGroup(parentOption.Node, clone))
                continue;

            var originalName = ReadString(clone, "Name") ?? "Piercings";
            SetProperty(clone, "Id", JsonValue.Create(Guid.NewGuid().ToString("D")));
            SetProperty(clone, "Name", JsonValue.Create(BuildTargetGroupDisplayName(request.BodyName, originalName, clone)));
            SetProperty(clone, "Description", JsonValue.Create(string.Empty));
            SetProperty(clone, "Page", JsonValue.Create(customPage));
            SetProperty(clone, "Condition", new JsonObject
            {
                ["Type"] = "Setting",
                ["Setting"] = parentOptionId.ToString("D"),
            });
            if (PenumbraV4Document.FindProperty(clone, "Options") is JsonArray options)
            {
                foreach (var option in options.OfType<JsonObject>())
                {
                    SetProperty(option, "Id", JsonValue.Create(Guid.NewGuid().ToString("D")));
                    var optionName = ReadString(option, "Name");
                    if (!string.IsNullOrWhiteSpace(optionName))
                        SetProperty(option, "Name", JsonValue.Create(CleanTargetOptionName(optionName)));
                }
            }
            groups.Add(clone);
            added++;
        }

        SetProperty(document.Root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));
        var backupPath = await CommitCustomisationAsync(metaPath, originalMetaBytes, document, cancellationToken).ConfigureAwait(false);
        return new V4CustomisationResult(backupPath, metaPath, removed, added);
    }

    public Task<V4CustomisationResult> AddAttributeVisibilityToggleAsync(
        string metaPathRaw,
        string groupKey,
        string optionKey,
        string toggleName,
        string gamePathRaw,
        string taggedRelativePathRaw,
        string attributeNameRaw,
        CancellationToken cancellationToken = default)
        => AddAttributeVisibilityTogglesAsync(metaPathRaw, groupKey, optionKey, gamePathRaw, taggedRelativePathRaw, new[] { (ToggleName: toggleName, AttributeName: attributeNameRaw) }, cancellationToken);

    public async Task<V4CustomisationResult> AddAttributeVisibilityTogglesAsync(
        string metaPathRaw,
        string groupKey,
        string optionKey,
        string gamePathRaw,
        string taggedRelativePathRaw,
        IReadOnlyList<(string ToggleName, string AttributeName)> toggles,
        CancellationToken cancellationToken = default)
    {
        if (toggles.Count == 0) throw new ArgumentException("No visibility toggles were supplied.", nameof(toggles));
        var normalized = toggles.Select(toggle => (ToggleName: toggle.ToggleName.Trim(), AttributeName: toggle.AttributeName.Trim())).ToArray();
        if (normalized.Any(toggle => string.IsNullOrWhiteSpace(toggle.ToggleName))) throw new ArgumentException("A visibility toggle name is empty.", nameof(toggles));
        if (normalized.Any(toggle => string.IsNullOrWhiteSpace(toggle.AttributeName))) throw new ArgumentException("A visibility attribute is empty.", nameof(toggles));
        if (normalized.Select(toggle => toggle.ToggleName).Distinct(StringComparer.OrdinalIgnoreCase).Count() != normalized.Length)
            throw new InvalidOperationException("Visibility toggle names must be unique within the batch.");
        if (normalized.Select(toggle => toggle.AttributeName).Distinct(StringComparer.OrdinalIgnoreCase).Count() != normalized.Length)
            throw new InvalidOperationException("Visibility attributes must be unique within the batch.");

        var metaPath = Path.GetFullPath(metaPathRaw);
        var originalMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
        var document = PenumbraV4Document.Load(metaPath);
        var parentGroup = document.GetGroup(groupKey);
        var parentOption = document.GetOption(parentGroup, optionKey);
        var parentOptionId = parentOption.Id ?? throw new InvalidDataException($"Penumbra option '{parentGroup.Name} / {parentOption.Name}' has no Id, so RavaFit cannot scope visibility controls safely.");
        var targetGamePath = NormalizeGamePath(gamePathRaw);
        var targetRelative = NormalizeRelativePath(taggedRelativePathRaw);

        var filesName = PenumbraV4Document.FindPropertyName(parentOption.Node, "Files") ?? "Files";
        var files = PenumbraV4Document.FindProperty(parentOption.Node, "Files") as JsonObject;
        if (files is null)
        {
            files = new JsonObject();
            parentOption.Node[filesName] = files;
        }
        files[targetGamePath] = targetRelative;

        var parentManipulationsName = PenumbraV4Document.FindPropertyName(parentOption.Node, "Manipulations") ?? "Manipulations";
        var parentManipulations = PenumbraV4Document.FindProperty(parentOption.Node, "Manipulations") as JsonArray;
        if (parentManipulations is null)
        {
            parentManipulations = new JsonArray();
            parentOption.Node[parentManipulationsName] = parentManipulations;
        }
        foreach (var toggle in normalized)
        {
            if (parentManipulations.OfType<JsonObject>().Any(manipulation => string.Equals(ReadString(manipulation, "Type"), "Atr", StringComparison.OrdinalIgnoreCase)
                    && PenumbraV4Document.FindProperty(manipulation, "Manipulation") is JsonObject payload
                    && string.Equals(ReadString(payload, "Attribute"), toggle.AttributeName, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidOperationException($"Model attribute '{toggle.AttributeName}' already has a baseline manipulation in the owning option.");
            parentManipulations.Add(new JsonObject
            {
                ["Type"] = "Atr",
                ["Manipulation"] = new JsonObject { ["Attribute"] = toggle.AttributeName, ["Entry"] = false },
            });
        }

        var groups = PenumbraV4Document.FindProperty(document.Root, "Groups") as JsonArray;
        if (groups is null)
        {
            groups = new JsonArray();
            SetProperty(document.Root, "Groups", groups);
        }
        var pageNames = PenumbraV4Document.FindProperty(document.Root, "PageNames") as JsonObject;
        if (pageNames is null)
        {
            pageNames = new JsonObject();
            SetProperty(document.Root, "PageNames", pageNames);
        }
        var existingCustomPage = pageNames.FirstOrDefault(pair => pair.Value is JsonValue value
            && value.TryGetValue<string>(out var text)
            && string.Equals(text, "Customise", StringComparison.OrdinalIgnoreCase));
        var customPage = int.TryParse(existingCustomPage.Key, out var parsedCustomPage)
            ? parsedCustomPage
            : Math.Max(0, document.Groups.Select(g => ReadInt(g.Node, "Page") ?? -1).Concat(pageNames.Select(pair => int.TryParse(pair.Key, out var page) ? page : -1)).DefaultIfEmpty(-1).Max() + 1);
        pageNames[customPage.ToString()] = "Customise";

        var marker = $"RavaFit attribute visibility: {parentOptionId:D}";
        var groupNode = groups.OfType<JsonObject>().FirstOrDefault(group => string.Equals(ReadString(group, "Description"), marker, StringComparison.OrdinalIgnoreCase));
        var addedGroup = false;
        if (groupNode is null)
        {
            var maxExistingPriority = document.Groups.Select(group => ReadInt(group.Node, "Priority").GetValueOrDefault()).DefaultIfEmpty(0).Max();
            var priority = maxExistingPriority == int.MaxValue ? int.MaxValue : maxExistingPriority + 1;
            groupNode = new JsonObject
            {
                ["Type"] = "Multi",
                ["Id"] = Guid.NewGuid().ToString("D"),
                ["Name"] = $"Visibility · {parentOption.Name}",
                ["Description"] = marker,
                ["Page"] = customPage,
                ["Priority"] = priority,
                ["DefaultSettings"] = 0UL,
                ["Condition"] = new JsonObject { ["Type"] = "Setting", ["Setting"] = parentOptionId.ToString("D") },
                ["Options"] = new JsonArray(),
            };
            groups.Add(groupNode);
            addedGroup = true;
        }
        else if (!string.Equals(ReadString(groupNode, "Type"), "Multi", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The existing RavaFit visibility group is not a Penumbra Multi group.");
        }

        var options = PenumbraV4Document.FindProperty(groupNode, "Options") as JsonArray;
        if (options is null)
        {
            options = new JsonArray();
            SetProperty(groupNode, "Options", options);
        }
        if (options.Count + normalized.Length > 32)
            throw new InvalidOperationException($"This model option already has {options.Count} independent visibility controls; adding {normalized.Length} would exceed the 32-attribute limit.");

        var existingNames = options.OfType<JsonObject>().Select(option => ReadString(option, "Name")).Where(name => !string.IsNullOrWhiteSpace(name)).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var toggle in normalized)
        {
            if (existingNames.Contains(toggle.ToggleName))
                throw new InvalidOperationException($"A visibility toggle named '{toggle.ToggleName}' already exists for this model option.");
        }
        var existingAttributes = options.OfType<JsonObject>()
            .SelectMany(option => (PenumbraV4Document.FindProperty(option, "Manipulations") as JsonArray)?.OfType<JsonObject>() ?? [])
            .Where(manipulation => string.Equals(ReadString(manipulation, "Type"), "Atr", StringComparison.OrdinalIgnoreCase))
            .Select(manipulation => PenumbraV4Document.FindProperty(manipulation, "Manipulation") as JsonObject)
            .Where(payload => payload is not null)
            .Select(payload => ReadString(payload!, "Attribute"))
            .Where(attribute => !string.IsNullOrWhiteSpace(attribute))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var toggle in normalized)
        {
            if (existingAttributes.Contains(toggle.AttributeName))
                throw new InvalidOperationException($"Model attribute '{toggle.AttributeName}' already has a RavaFit visibility control.");
        }

        var defaults = ReadUInt64(groupNode, "DefaultSettings");
        foreach (var toggle in normalized)
        {
            var optionIndex = options.Count;
            options.Add(new JsonObject
            {
                ["Id"] = Guid.NewGuid().ToString("D"),
                ["Name"] = toggle.ToggleName,
                ["Description"] = $"Show or hide {toggle.ToggleName}",
                ["Files"] = new JsonObject(),
                ["FileSwaps"] = new JsonObject(),
                ["Manipulations"] = new JsonArray
                {
                    new JsonObject
                    {
                        ["Type"] = "Atr",
                        ["Manipulation"] = new JsonObject { ["Attribute"] = toggle.AttributeName, ["Entry"] = true },
                    },
                },
            });
            defaults |= 1UL << optionIndex;
        }
        SetProperty(groupNode, "DefaultSettings", JsonValue.Create(defaults));
        SetProperty(document.Root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));
        var backupPath = await CommitCustomisationAsync(metaPath, originalMetaBytes, document, cancellationToken).ConfigureAwait(false);
        return new V4CustomisationResult(backupPath, metaPath, 0, addedGroup ? 1 : 0);
    }

    public async Task<V4CustomisationResult> AddVisibilityToggleAsync(
        string metaPathRaw,
        string groupKey,
        string optionKey,
        string toggleName,
        string gamePathRaw,
        string hiddenRelativePathRaw,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(toggleName))
            throw new ArgumentException("Visibility toggle name is empty.", nameof(toggleName));
        var metaPath = Path.GetFullPath(metaPathRaw);
        var originalMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
        var document = PenumbraV4Document.Load(metaPath);
        var parentGroup = document.GetGroup(groupKey);
        var parentOption = document.GetOption(parentGroup, optionKey);
        var parentOptionId = parentOption.Id ?? throw new InvalidDataException($"Penumbra option '{parentGroup.Name} / {parentOption.Name}' has no Id, so RavaFit cannot scope a visibility toggle safely.");
        if (document.Groups.Any(group => string.Equals(group.Name, toggleName.Trim(), StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException($"A Penumbra group named '{toggleName.Trim()}' already exists.");
        var targetGamePath = NormalizeGamePath(gamePathRaw);
        var visibilityMarker = BuildVisibilityMarker(targetGamePath);
        if (document.Groups.Any(group => IsVisibilityControlFor(group.Node, parentOptionId, visibilityMarker)))
            throw new InvalidOperationException("This model option already has a RavaFit visibility control. Select every mesh that should hide together in that control; independent model-file visibility toggles cannot be safely stacked.");

        var groups = PenumbraV4Document.FindProperty(document.Root, "Groups") as JsonArray;
        if (groups is null)
        {
            groups = new JsonArray();
            SetProperty(document.Root, "Groups", groups);
        }
        var pageNames = PenumbraV4Document.FindProperty(document.Root, "PageNames") as JsonObject;
        if (pageNames is null)
        {
            pageNames = new JsonObject();
            SetProperty(document.Root, "PageNames", pageNames);
        }
        var existingCustomPage = pageNames.FirstOrDefault(pair => pair.Value is JsonValue value
            && value.TryGetValue<string>(out var text)
            && string.Equals(text, "Customise", StringComparison.OrdinalIgnoreCase));
        var customPage = int.TryParse(existingCustomPage.Key, out var parsedCustomPage)
            ? parsedCustomPage
            : Math.Max(0, document.Groups.Select(g => ReadInt(g.Node, "Page") ?? -1).Concat(pageNames.Select(pair => int.TryParse(pair.Key, out var page) ? page : -1)).DefaultIfEmpty(-1).Max() + 1);
        pageNames[customPage.ToString()] = "Customise";

        var showId = Guid.NewGuid();
        var hideId = Guid.NewGuid();
        var groupId = Guid.NewGuid();
        var targetRelative = NormalizeRelativePath(hiddenRelativePathRaw);
        var maxExistingPriority = document.Groups.Select(group => ReadInt(group.Node, "Priority").GetValueOrDefault()).DefaultIfEmpty(0).Max();
        var visibilityPriority = maxExistingPriority == int.MaxValue ? int.MaxValue : maxExistingPriority + 1;
        groups.Add(new JsonObject
        {
            ["Type"] = "Single",
            ["Id"] = groupId.ToString("D"),
            ["Name"] = toggleName.Trim(),
            ["Description"] = visibilityMarker,
            ["Page"] = customPage,
            ["Priority"] = visibilityPriority,
            ["DefaultSettings"] = 0,
            ["Condition"] = new JsonObject { ["Type"] = "Setting", ["Setting"] = parentOptionId.ToString("D") },
            ["Options"] = new JsonArray
            {
                new JsonObject
                {
                    ["Id"] = showId.ToString("D"), ["Name"] = "Show", ["Description"] = string.Empty,
                    ["Files"] = new JsonObject(), ["FileSwaps"] = new JsonObject(), ["Manipulations"] = new JsonArray(),
                },
                new JsonObject
                {
                    ["Id"] = hideId.ToString("D"), ["Name"] = "Hide", ["Description"] = string.Empty,
                    ["Files"] = new JsonObject { [targetGamePath] = targetRelative }, ["FileSwaps"] = new JsonObject(), ["Manipulations"] = new JsonArray(),
                },
            },
        });
        SetProperty(document.Root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));
        var backupPath = await CommitCustomisationAsync(metaPath, originalMetaBytes, document, cancellationToken).ConfigureAwait(false);
        return new V4CustomisationResult(backupPath, metaPath, 0, 1);
    }

    private static string BuildGeneratedOptionDescription(string? existing)
    {
        const string notice = "Made with RavaFit. Check the original creator's permissions before sharing.";
        return string.IsNullOrWhiteSpace(existing) ? notice : $"{existing.Trim()}\n\n{notice}";
    }

    private static string BuildVisibilityMarker(string targetGamePath) => $"RavaFit visibility control: {targetGamePath}";

    private static bool IsVisibilityControlFor(JsonObject group, Guid parentOptionId, string marker)
    {
        if (!string.Equals(ReadString(group, "Description"), marker, StringComparison.OrdinalIgnoreCase))
            return false;
        if (PenumbraV4Document.FindProperty(group, "Condition") is not JsonObject condition)
            return false;
        return string.Equals(ReadString(condition, "Type"), "Setting", StringComparison.OrdinalIgnoreCase)
            && Guid.TryParse(ReadString(condition, "Setting"), out var settingId)
            && settingId == parentOptionId;
    }

    private enum PiercingGroupKind
    {
        None,
        Piercing,
        Mixed,
    }

    private static PiercingGroupKind ClassifyPiercingGroup(JsonObject group, IReadOnlyList<string>? sourcePiercingResourcePaths = null)
    {
        var groupLabel = $"{ReadString(group, "Name")} {ReadString(group, "Description")}";
        if (LooksLikePiercingText(groupLabel))
            return PiercingGroupKind.Piercing;

        var options = PenumbraV4Document.FindProperty(group, "Options") as JsonArray;
        if (options is null || options.Count == 0)
            return ContainsPiercingHint(group) || ContainsPiercingResource(group, sourcePiercingResourcePaths) ? PiercingGroupKind.Piercing : PiercingGroupKind.None;

        var recognised = 0;
        var optionCount = 0;
        foreach (var option in options.OfType<JsonObject>())
        {
            optionCount++;
            var optionLabel = $"{ReadString(option, "Name")} {ReadString(option, "Description")}";
            if (LooksLikePiercingText(optionLabel) || ContainsPiercingHint(option) || ContainsPiercingResource(option, sourcePiercingResourcePaths))
                recognised++;
        }

        if (optionCount == 0)
            return ContainsPiercingHint(group) || ContainsPiercingResource(group, sourcePiercingResourcePaths) ? PiercingGroupKind.Piercing : PiercingGroupKind.None;
        if (recognised == optionCount)
            return PiercingGroupKind.Piercing;
        if (recognised > 0)
            return PiercingGroupKind.Mixed;

        return ContainsPiercingHint(group) || ContainsPiercingResource(group, sourcePiercingResourcePaths) ? PiercingGroupKind.Piercing : PiercingGroupKind.None;
    }

    private static bool ContainsPiercingHint(JsonObject node)
    {
        var text = node.ToJsonString().ToLowerInvariant();
        return text.Contains("pierc", StringComparison.Ordinal)
            || text.Contains("jewel", StringComparison.Ordinal)
            || text.Contains("dermal", StringComparison.Ordinal)
            || text.Contains("barbell", StringComparison.Ordinal)
            || text.Contains("bellyring", StringComparison.Ordinal)
            || text.Contains("nipple", StringComparison.Ordinal);
    }

    private static bool LooksLikePiercingText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        var normalized = text.ToLowerInvariant();
        return normalized.Contains("pierc", StringComparison.Ordinal)
            || normalized.Contains("jewel", StringComparison.Ordinal)
            || normalized.Contains("dermal", StringComparison.Ordinal)
            || normalized.Contains("barbell", StringComparison.Ordinal)
            || normalized.Contains("bellyring", StringComparison.Ordinal)
            || normalized.Contains("nipple", StringComparison.Ordinal);
    }

    private static bool ContainsPiercingResource(JsonObject node, IReadOnlyList<string>? sourcePiercingResourcePaths)
    {
        if (sourcePiercingResourcePaths is null || sourcePiercingResourcePaths.Count == 0)
            return false;
        var text = node.ToJsonString().Replace('\\', '/').ToLowerInvariant();
        foreach (var resource in sourcePiercingResourcePaths)
        {
            if (string.IsNullOrWhiteSpace(resource)) continue;
            var normalized = NormalizeGamePath(resource).ToLowerInvariant();
            if (normalized.Length > 0 && text.Contains(normalized, StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    private static void ApplyGeneratedRedirections(JsonObject files, V4AppendRequest request)
    {
        foreach (var (gamePathRaw, relativePathRaw) in request.ModelRedirections)
        {
            var gamePath = NormalizeGamePath(gamePathRaw);
            var relativePath = NormalizeRelativePath(relativePathRaw);
            if (!gamePath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"RavaFit only accepts .mdl paths in model redirections. '{gamePath}' is not an .mdl path.");
            files[gamePath] = relativePath;
        }

        if (request.AdditionalFileRedirections is null) return;
        foreach (var (gamePathRaw, relativePathRaw) in request.AdditionalFileRedirections)
        {
            var gamePath = NormalizeGamePath(gamePathRaw);
            var relativePath = NormalizeRelativePath(relativePathRaw);
            if (!gamePath.EndsWith(".mtrl", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"RavaFit generated supporting redirections currently only accept .mtrl paths. '{gamePath}' is not an .mtrl path.");
            if (request.ModelRedirections.Keys.Any(path => string.Equals(NormalizeGamePath(path), gamePath, StringComparison.OrdinalIgnoreCase)))
                throw new InvalidDataException($"Generated supporting redirection '{gamePath}' conflicts with a model redirection.");
            files[gamePath] = relativePath;
        }
    }

    private static async Task<string> CommitCustomisationAsync(string metaPath, byte[] originalMetaBytes, PenumbraV4Document document, CancellationToken cancellationToken)
    {
        var tempPath = metaPath + $".ravafit-{Guid.NewGuid():N}.tmp";
        var backupPath = metaPath + ".ravafit-last.bak";
        try
        {
            await WriteDurableAsync(tempPath, document.Root.ToJsonString(WriteOptions), cancellationToken).ConfigureAwait(false);
            _ = PenumbraV4Document.Load(tempPath);
            var liveMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
            if (!originalMetaBytes.AsSpan().SequenceEqual(liveMetaBytes))
                throw new InvalidDataException("meta.json changed on disk while RavaFit was preparing the customisation. Nothing was written; retry.");
            if (File.Exists(backupPath)) File.Delete(backupPath);
            File.Replace(tempPath, metaPath, backupPath, true);
            return backupPath;
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    public async Task<V4AppendResult> AppendClonedOptionAsync(V4AppendRequest request, CancellationToken cancellationToken = default)
    {
        var metaPath = Path.GetFullPath(request.MetaPath);
        var originalMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
        var document = PenumbraV4Document.Load(metaPath);
        var group = document.GetGroup(request.GroupKey);
        if (!group.SupportsAppend)
            throw new NotSupportedException($"Group '{group.Name}' is type '{group.Type}'. RavaFit v0.1 only appends to Single and Multi groups.");
        var source = document.GetOption(group, request.SourceOptionKey);

        if (string.IsNullOrWhiteSpace(request.NewOptionName))
            throw new ArgumentException("The new option name is empty.", nameof(request));
        if (group.Options.Any(o => string.Equals(o.Name, request.NewOptionName, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException($"An option named '{request.NewOptionName}' already exists in '{group.Name}'.");
        if (request.ModelRedirections.Count == 0)
            throw new InvalidOperationException("No converted model redirections were supplied.");

        var sourceBefore = source.Node.ToJsonString(WriteOptions);
        var clone = source.Node.DeepClone().AsObject();
        var newId = Guid.NewGuid();
        SetProperty(clone, "Id", JsonValue.Create(newId.ToString("D")));
        SetProperty(clone, "Name", JsonValue.Create(request.NewOptionName));
        SetProperty(clone, "Description", JsonValue.Create(BuildGeneratedOptionDescription(ReadString(clone, "Description"))));

        var filesName = PenumbraV4Document.FindPropertyName(clone, "Files") ?? "Files";
        var files = PenumbraV4Document.FindProperty(clone, "Files") as JsonObject;
        if (files is null)
        {
            files = new JsonObject();
            clone[filesName] = files;
        }

        ApplyGeneratedRedirections(files, request);

        var optionArray = PenumbraV4Document.FindProperty(group.Node, "Options") as JsonArray
            ?? throw new InvalidDataException($"Group '{group.Name}' has no Options array.");
        optionArray.Add(clone);
        SetProperty(document.Root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));

        var sourceAfter = source.Node.ToJsonString(WriteOptions);
        if (!string.Equals(sourceBefore, sourceAfter, StringComparison.Ordinal))
            throw new InvalidOperationException("Source option changed while preparing the non-destructive append. Transaction aborted.");

        var tempPath = metaPath + $".ravafit-{Guid.NewGuid():N}.tmp";
        var backupPath = metaPath + ".ravafit-last.bak";

        try
        {
            var json = document.Root.ToJsonString(WriteOptions);
            await WriteDurableAsync(tempPath, json, cancellationToken).ConfigureAwait(false);

            // Validate the temp file before replacing the original.
            var verify = PenumbraV4Document.Load(tempPath);
            var verifiedGroup = verify.GetGroup(group.StableKey);
            if (verifiedGroup.Options.Count != group.Options.Count + 1)
                throw new InvalidDataException("V4 transaction validation failed: option count did not increase by exactly one.");
            var verifiedNew = verifiedGroup.Options.SingleOrDefault(o => o.Id == newId)
                ?? throw new InvalidDataException("V4 transaction validation failed: appended option ID was not found.");
            if (!string.Equals(verifiedNew.Name, request.NewOptionName, StringComparison.Ordinal))
                throw new InvalidDataException("V4 transaction validation failed: appended option name changed.");

            // Bail if meta.json changed under us.
            var liveMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
            if (!originalMetaBytes.AsSpan().SequenceEqual(liveMetaBytes))
                throw new InvalidDataException("meta.json changed on disk while RavaFit was preparing the transaction. Nothing was written; retry the conversion.");

            if (File.Exists(backupPath))
                File.Delete(backupPath);
            File.Replace(tempPath, metaPath, backupPath, true);
            return new V4AppendResult(newId, backupPath, metaPath, verifiedGroup.Options.Count);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }


    public Task<IReadOnlyList<V4AppendResult>> AppendClonedOptionsAsync(IReadOnlyList<V4AppendRequest> requests, CancellationToken cancellationToken = default)
        => AppendClonedOptionsCoreAsync(requests, [], cancellationToken);

    public Task<IReadOnlyList<V4AppendResult>> AppendClonedOptionsWithTargetBodyGroupsAsync(IReadOnlyList<V4AppendRequest> requests, IReadOnlyList<V4TargetBodyGroupRequest> targetBodyGroups, CancellationToken cancellationToken = default)
        => AppendClonedOptionsCoreAsync(requests, targetBodyGroups, cancellationToken);

    private async Task<IReadOnlyList<V4AppendResult>> AppendClonedOptionsCoreAsync(IReadOnlyList<V4AppendRequest> requests, IReadOnlyList<V4TargetBodyGroupRequest> targetBodyGroups, CancellationToken cancellationToken)
    {
        if (requests.Count == 0)
            throw new InvalidOperationException("No Penumbra options were supplied for the outfit transaction.");

        var metaPath = Path.GetFullPath(requests[0].MetaPath);
        if (requests.Any(r => !string.Equals(Path.GetFullPath(r.MetaPath), metaPath, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException("All outfit rows must belong to the same Penumbra mod/meta.json transaction.");

        var originalMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
        var document = PenumbraV4Document.Load(metaPath);
        var originalCounts = document.Groups.ToDictionary(g => g.StableKey, g => g.Options.Count, StringComparer.OrdinalIgnoreCase);
        var pendingNames = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
        var sourceSnapshots = new List<(V4OptionInfo Source, string Json)>();
        var appended = new List<(string GroupKey, Guid Id, string Name, JsonObject Node)>();

        foreach (var request in requests)
        {
            var group = document.GetGroup(request.GroupKey);
            if (!group.SupportsAppend)
                throw new NotSupportedException($"Group '{group.Name}' is type '{group.Type}'. RavaFit only appends to Single and Multi groups.");
            var source = document.GetOption(group, request.SourceOptionKey);
            if (string.IsNullOrWhiteSpace(request.NewOptionName))
                throw new ArgumentException("A generated outfit option name is empty.", nameof(requests));
            if (request.ModelRedirections.Count == 0)
                throw new InvalidOperationException($"No converted model redirections were supplied for '{group.Name}'.");

            if (!pendingNames.TryGetValue(group.StableKey, out var names))
            {
                names = new HashSet<string>(group.Options.Select(o => o.Name), StringComparer.OrdinalIgnoreCase);
                pendingNames[group.StableKey] = names;
            }
            if (!names.Add(request.NewOptionName))
                throw new InvalidOperationException($"An option named '{request.NewOptionName}' already exists or is already being added to '{group.Name}'.");

            sourceSnapshots.Add((source, source.Node.ToJsonString(WriteOptions)));
            var clone = source.Node.DeepClone().AsObject();
            var newId = Guid.NewGuid();
            SetProperty(clone, "Id", JsonValue.Create(newId.ToString("D")));
            SetProperty(clone, "Name", JsonValue.Create(request.NewOptionName));
            SetProperty(clone, "Description", JsonValue.Create(BuildGeneratedOptionDescription(ReadString(clone, "Description"))));

            var filesName = PenumbraV4Document.FindPropertyName(clone, "Files") ?? "Files";
            var files = PenumbraV4Document.FindProperty(clone, "Files") as JsonObject;
            if (files is null)
            {
                files = new JsonObject();
                clone[filesName] = files;
            }

            ApplyGeneratedRedirections(files, request);

            var optionArray = PenumbraV4Document.FindProperty(group.Node, "Options") as JsonArray
                ?? throw new InvalidDataException($"Group '{group.Name}' has no Options array.");
            optionArray.Add(clone);
            appended.Add((group.StableKey, newId, request.NewOptionName, clone));
        }

        foreach (var (source, json) in sourceSnapshots)
            if (!string.Equals(json, source.Node.ToJsonString(WriteOptions), StringComparison.Ordinal))
                throw new InvalidOperationException("A source option changed while preparing the non-destructive outfit transaction. Transaction aborted.");

        var appendedTargetGroups = new List<(Guid Id, string Name)>();
        if (targetBodyGroups.Count > 0)
        {
            var groupArray = PenumbraV4Document.FindProperty(document.Root, "Groups") as JsonArray;
            if (groupArray is null)
            {
                groupArray = new JsonArray();
                SetProperty(document.Root, "Groups", groupArray);
            }

            var seenTargetGroups = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var candidates = new List<(V4TargetBodyGroupRequest Request, V4GroupInfo ParentGroup, JsonObject Group, int Order)>();
            var order = 0;
            foreach (var request in targetBodyGroups)
            {
                var dedupeKey = string.Join("|", request.ParentGroupKey, request.ParentOutputOptionName, request.SourceKey);
                if (!seenTargetGroups.Add(dedupeKey))
                    continue;

                var parentGroup = document.GetGroup(request.ParentGroupKey);
                var parentAddition = appended.FirstOrDefault(x => string.Equals(x.GroupKey, parentGroup.StableKey, StringComparison.OrdinalIgnoreCase)
                    && string.Equals(x.Name, request.ParentOutputOptionName, StringComparison.OrdinalIgnoreCase));
                if (parentAddition == default)
                {
                    parentAddition = appended.FirstOrDefault(x => string.Equals(x.GroupKey, request.ParentGroupKey, StringComparison.OrdinalIgnoreCase)
                        && string.Equals(x.Name, request.ParentOutputOptionName, StringComparison.OrdinalIgnoreCase));
                }
                if (parentAddition == default)
                    throw new InvalidOperationException($"Target-body option group '{request.SourceKey}' could not find generated parent option '{request.ParentOutputOptionName}'.");

                _ = parentGroup.Id
                    ?? throw new InvalidDataException($"Penumbra V4 group '{parentGroup.Name}' has no Id, so RavaFit cannot safely scope target-body controls to the generated option without changing the source group identity.");

                var clone = request.Group.DeepClone().AsObject();
                if (PenumbraV4Document.FindProperty(clone, "Condition") is not null)
                    throw new NotSupportedException($"Captured target-body group '{ReadString(clone, "Name") ?? request.SourceKey}' already has its own Penumbra condition. RavaFit will not silently replace that dependency.");

                // A one-choice Single group is package plumbing, so fold it into the generated option.
                if (TryFoldAlwaysOnTargetGroup(parentAddition.Node, clone))
                    continue;

                RetargetTargetBodyGroup(clone, request.TargetGamePath);
                if (IsPiercingLikeGroup(clone))
                {
                    FilterAlreadyExistingPiercingOptions(document, clone);
                    if (PenumbraV4Document.FindProperty(clone, "Options") is JsonArray remaining && remaining.Count == 0)
                        continue;
                }
                candidates.Add((request, parentGroup, clone, order++));
            }

            var effective = new List<(V4TargetBodyGroupRequest Request, V4GroupInfo ParentGroup, JsonObject Group, int Order)>();
            var imcWinners = new Dictionary<string, (V4TargetBodyGroupRequest Request, V4GroupInfo ParentGroup, JsonObject Group, int Order)>(StringComparer.OrdinalIgnoreCase);
            foreach (var candidate in candidates)
            {
                if (!string.Equals(ReadString(candidate.Group, "Type"), "Imc", StringComparison.OrdinalIgnoreCase))
                {
                    effective.Add(candidate);
                    continue;
                }

                var authorityKey = $"{candidate.Request.ParentGroupKey}|{candidate.Request.ParentOutputOptionName}|{BuildImcAuthorityKey(candidate.Group)}";
                if (!imcWinners.TryGetValue(authorityKey, out var current)
                    || ReadInt(candidate.Group, "Priority").GetValueOrDefault() > ReadInt(current.Group, "Priority").GetValueOrDefault()
                    || (ReadInt(candidate.Group, "Priority").GetValueOrDefault() == ReadInt(current.Group, "Priority").GetValueOrDefault() && candidate.Order > current.Order))
                    imcWinners[authorityKey] = candidate;
            }
            effective.AddRange(imcWinners.Values);

            if (effective.Count > 0)
            {
                var pageNames = PenumbraV4Document.FindProperty(document.Root, "PageNames") as JsonObject;
                if (pageNames is null)
                {
                    pageNames = new JsonObject();
                    SetProperty(document.Root, "PageNames", pageNames);
                }
                var existingTargetPage = pageNames.FirstOrDefault(pair => pair.Value is JsonValue value
                    && value.TryGetValue<string>(out var text)
                    && (string.Equals(text, "Body Options", StringComparison.OrdinalIgnoreCase)
                        || string.Equals(text, "Target Body Options", StringComparison.OrdinalIgnoreCase)));
                var targetPage = int.TryParse(existingTargetPage.Key, out var parsedTargetPage)
                    ? parsedTargetPage
                    : Math.Max(0, document.Groups.Select(g => ReadInt(g.Node, "Page") ?? -1).Concat(pageNames.Select(pair => int.TryParse(pair.Key, out var page) ? page : -1)).DefaultIfEmpty(-1).Max() + 1);
                pageNames[targetPage.ToString()] = "Body Options";

                foreach (var candidate in effective
                    .OrderBy(x => GetTargetGroupSortOrder(x.Group))
                    .ThenBy(x => x.Order))
                {
                    var request = candidate.Request;
                    var parentGroup = candidate.ParentGroup;
                    var clone = candidate.Group;
                    var parentAddition = appended.First(x =>
                        (string.Equals(x.GroupKey, parentGroup.StableKey, StringComparison.OrdinalIgnoreCase)
                         || string.Equals(x.GroupKey, request.ParentGroupKey, StringComparison.OrdinalIgnoreCase))
                        && string.Equals(x.Name, request.ParentOutputOptionName, StringComparison.OrdinalIgnoreCase));

                    var originalName = ReadString(clone, "Name") ?? "Piercings";
                    var displayName = BuildTargetGroupDisplayName(request.BodyName, originalName, clone);
                    var newGroupId = Guid.NewGuid();
                    SetProperty(clone, "Id", JsonValue.Create(newGroupId.ToString("D")));
                    SetProperty(clone, "Name", JsonValue.Create(displayName));
                    SetProperty(clone, "Description", JsonValue.Create(string.Empty));
                    SetProperty(clone, "Page", JsonValue.Create(targetPage));
                    SetProperty(clone, "Condition", new JsonObject
                    {
                        ["Type"] = "Setting",
                        ["Setting"] = parentAddition.Id.ToString("D"),
                    });

                    if (PenumbraV4Document.FindProperty(clone, "Options") is JsonArray options)
                    {
                        foreach (var option in options.OfType<JsonObject>())
                        {
                            SetProperty(option, "Id", JsonValue.Create(Guid.NewGuid().ToString("D")));
                            var optionName = ReadString(option, "Name");
                            if (!string.IsNullOrWhiteSpace(optionName))
                                SetProperty(option, "Name", JsonValue.Create(CleanTargetOptionName(optionName)));
                        }
                    }

                    ValidateNoConflictingImcAuthority(document, requests, request, parentGroup, clone);
                    groupArray.Add(clone);
                    appendedTargetGroups.Add((newGroupId, displayName));
                }
            }
        }

        SetProperty(document.Root, "LastWrite", JsonValue.Create(DateTimeOffset.UtcNow.ToString("O")));
        var tempPath = metaPath + $".ravafit-{Guid.NewGuid():N}.tmp";
        var backupPath = metaPath + ".ravafit-last.bak";

        try
        {
            var json = document.Root.ToJsonString(WriteOptions);
            await WriteDurableAsync(tempPath, json, cancellationToken).ConfigureAwait(false);

            var verify = PenumbraV4Document.Load(tempPath);
            foreach (var groupAdditions in appended.GroupBy(x => x.GroupKey, StringComparer.OrdinalIgnoreCase))
            {
                var verifiedGroup = verify.GetGroup(groupAdditions.Key);
                var expectedCount = originalCounts[groupAdditions.Key] + groupAdditions.Count();
                if (verifiedGroup.Options.Count != expectedCount)
                    throw new InvalidDataException($"V4 outfit transaction validation failed for '{verifiedGroup.Name}': expected {expectedCount} options, found {verifiedGroup.Options.Count}.");
                foreach (var addition in groupAdditions)
                {
                    var verifiedNew = verifiedGroup.Options.SingleOrDefault(o => o.Id == addition.Id)
                        ?? throw new InvalidDataException($"V4 outfit transaction validation failed: appended option '{addition.Name}' was not found.");
                    if (!string.Equals(verifiedNew.Name, addition.Name, StringComparison.Ordinal))
                        throw new InvalidDataException("V4 outfit transaction validation failed: appended option name changed.");
                }
            }

            foreach (var targetGroup in appendedTargetGroups)
            {
                var verifiedTarget = verify.Groups.SingleOrDefault(g => g.Id == targetGroup.Id)
                    ?? throw new InvalidDataException($"V4 target-body transaction validation failed: group '{targetGroup.Name}' was not found.");
                if (!string.Equals(verifiedTarget.Name, targetGroup.Name, StringComparison.Ordinal))
                    throw new InvalidDataException("V4 target-body transaction validation failed: generated group name changed.");
            }

            var liveMetaBytes = await File.ReadAllBytesAsync(metaPath, cancellationToken).ConfigureAwait(false);
            if (!originalMetaBytes.AsSpan().SequenceEqual(liveMetaBytes))
                throw new InvalidDataException("meta.json changed on disk while RavaFit was preparing the outfit transaction. Nothing was written; retry the conversion.");

            if (File.Exists(backupPath))
                File.Delete(backupPath);
            File.Replace(tempPath, metaPath, backupPath, true);

            var finalDocument = PenumbraV4Document.Load(metaPath);
            return appended.Select(addition =>
            {
                var finalGroup = finalDocument.GetGroup(addition.GroupKey);
                return new V4AppendResult(addition.Id, backupPath, metaPath, finalGroup.Options.Count);
            }).ToArray();
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private static bool IsPiercingLikeGroup(JsonObject group)
    {
        static bool HasPiercingWord(string? value)
        {
            if (string.IsNullOrWhiteSpace(value)) return false;
            var text = value.ToLowerInvariant();
            return text.Contains("pierc", StringComparison.Ordinal)
                || text.Contains("dermal", StringComparison.Ordinal)
                || text.Contains("nipple", StringComparison.Ordinal)
                || text.Contains("jewel", StringComparison.Ordinal)
                || text.Contains("barbell", StringComparison.Ordinal)
                || text.Contains("bellyring", StringComparison.Ordinal);
        }

        if (HasPiercingWord(ReadString(group, "Name")) || HasPiercingWord(ReadString(group, "Description")))
            return true;
        if (PenumbraV4Document.FindProperty(group, "Options") is not JsonArray options)
            return false;
        return options.OfType<JsonObject>().Any(option =>
            HasPiercingWord(ReadString(option, "Name"))
            || HasPiercingWord(option.ToJsonString()));
    }

    private static string CanonicalPiercingOptionKey(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return string.Empty;
        var clean = value.ToLowerInvariant();
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"\bcollar\s*bones?\b", "collarbone");
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"\bpiercings?\b", " ");
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"\bnsfw\b|\bonly\b|\btoggle\b|\btoggles\b|\boption\b|\boptions\b", " ");
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"[^a-z0-9]+", " ");
        var tokens = clean.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return string.Join(' ', tokens);
    }

    private static void FilterAlreadyExistingPiercingOptions(PenumbraV4Document document, JsonObject candidateGroup)
    {
        if (PenumbraV4Document.FindProperty(candidateGroup, "Options") is not JsonArray candidateOptions || candidateOptions.Count == 0)
            return;

        var existing = document.Groups
            .Where(group => IsPiercingLikeGroup(group.Node))
            .SelectMany(group => group.Options)
            .Select(option => CanonicalPiercingOptionKey(option.Name))
            .Where(key => !string.IsNullOrWhiteSpace(key))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (existing.Count == 0)
            return;

        var oldOptions = candidateOptions.OfType<JsonObject>().ToArray();
        if (oldOptions.Length != candidateOptions.Count)
            return;
        var keep = new List<int>(oldOptions.Length);
        for (var i = 0; i < oldOptions.Length; i++)
        {
            var key = CanonicalPiercingOptionKey(ReadString(oldOptions[i], "Name"));
            if (string.IsNullOrWhiteSpace(key) || !existing.Contains(key))
                keep.Add(i);
        }
        if (keep.Count == oldOptions.Length)
            return;

        var type = ReadString(candidateGroup, "Type") ?? string.Empty;
        var oldDefault = ReadUInt64(candidateGroup, "DefaultSettings");
        candidateOptions.Clear();
        foreach (var index in keep)
            candidateOptions.Add(oldOptions[index]);

        if (string.Equals(type, "Multi", StringComparison.OrdinalIgnoreCase))
        {
            ulong remapped = 0;
            for (var newIndex = 0; newIndex < keep.Count && newIndex < 64; newIndex++)
            {
                var oldIndex = keep[newIndex];
                if (oldIndex < 64 && (oldDefault & (1UL << oldIndex)) != 0)
                    remapped |= 1UL << newIndex;
            }
            SetProperty(candidateGroup, "DefaultSettings", JsonValue.Create(remapped));
        }
        else
        {
            var selectedOldIndex = oldDefault <= int.MaxValue ? (int)oldDefault : 0;
            var selectedNewIndex = keep.IndexOf(selectedOldIndex);
            SetProperty(candidateGroup, "DefaultSettings", JsonValue.Create(selectedNewIndex >= 0 ? selectedNewIndex : 0));
        }
    }

    private static bool TryFoldAlwaysOnTargetGroup(JsonObject parentOption, JsonObject group)
    {
        if (!string.Equals(ReadString(group, "Type"), "Single", StringComparison.OrdinalIgnoreCase)
            || PenumbraV4Document.FindProperty(group, "Options") is not JsonArray options
            || options.Count != 1
            || options[0] is not JsonObject onlyOption
            || ContainsModelEffect(onlyOption))
            return false;

        MergeObjectMap(parentOption, onlyOption, "Files");
        MergeObjectMap(parentOption, onlyOption, "FileSwaps");
        MergeManipulations(parentOption, onlyOption);
        return true;
    }

    private static bool ContainsModelEffect(JsonObject option)
    {
        if (PenumbraV4Document.FindProperty(option, "Files") is JsonObject files
            && files.Any(pair => pair.Key.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase)))
            return true;
        if (PenumbraV4Document.FindProperty(option, "FileSwaps") is JsonObject swaps
            && swaps.Any(pair => pair.Key.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase)
                || (pair.Value is JsonValue value && value.TryGetValue<string>(out var target) && target.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))))
            return true;
        return false;
    }

    private static void MergeObjectMap(JsonObject destinationOption, JsonObject sourceOption, string propertyName)
    {
        if (PenumbraV4Document.FindProperty(sourceOption, propertyName) is not JsonObject source || source.Count == 0)
            return;
        var destinationName = PenumbraV4Document.FindPropertyName(destinationOption, propertyName) ?? propertyName;
        var destination = PenumbraV4Document.FindProperty(destinationOption, propertyName) as JsonObject;
        if (destination is null)
        {
            destination = new JsonObject();
            destinationOption[destinationName] = destination;
        }
        foreach (var pair in source)
            destination[pair.Key] = pair.Value?.DeepClone();
    }

    private static void MergeManipulations(JsonObject destinationOption, JsonObject sourceOption)
    {
        if (PenumbraV4Document.FindProperty(sourceOption, "Manipulations") is not JsonArray source || source.Count == 0)
            return;
        var destinationName = PenumbraV4Document.FindPropertyName(destinationOption, "Manipulations") ?? "Manipulations";
        var destination = PenumbraV4Document.FindProperty(destinationOption, "Manipulations") as JsonArray;
        if (destination is null)
        {
            destination = new JsonArray();
            destinationOption[destinationName] = destination;
        }
        var existing = destination.Select(node => node?.ToJsonString() ?? "null").ToHashSet(StringComparer.Ordinal);
        foreach (var manipulation in source)
        {
            var json = manipulation?.ToJsonString() ?? "null";
            if (existing.Add(json))
                destination.Add(manipulation?.DeepClone());
        }
    }

    private static string BuildImcAuthorityKey(JsonObject group)
    {
        if (PenumbraV4Document.FindProperty(group, "Identifier") is not JsonObject identifier)
            throw new InvalidDataException($"Target-body IMC group '{ReadString(group, "Name")}' has no Identifier.");
        return string.Join("|", new[] { "PrimaryId", "SecondaryId", "Variant", "ObjectType", "EquipSlot", "BodySlot" }
            .Select(name => CanonicalJsonProperty(identifier, name)));
    }

    private static string CanonicalJsonProperty(JsonObject obj, string name)
    {
        var node = PenumbraV4Document.FindProperty(obj, name);
        if (node is null)
            return "<null>";
        if (node is JsonValue value)
        {
            if (value.TryGetValue<string>(out var text))
                return text.Trim().ToLowerInvariant();
            if (value.TryGetValue<int>(out var integer))
                return integer.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (value.TryGetValue<uint>(out var unsigned))
                return unsigned.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        return node.ToJsonString();
    }

    private static int GetTargetGroupSortOrder(JsonObject group)
    {
        if (string.Equals(ReadString(group, "Type"), "Imc", StringComparison.OrdinalIgnoreCase))
            return 0;
        var name = ReadString(group, "Name") ?? string.Empty;
        return name.Contains("color", StringComparison.OrdinalIgnoreCase) || name.Contains("colour", StringComparison.OrdinalIgnoreCase) ? 1 : 2;
    }

    private static string BuildTargetGroupDisplayName(string bodyName, string originalName, JsonObject group)
    {
        if (string.Equals(ReadString(group, "Type"), "Imc", StringComparison.OrdinalIgnoreCase))
            return $"{bodyName} - Piercings";

        var clean = originalName.Trim();
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"(?i)^\s*(?:\[?required\]?|reguired|required|base install)\s*[:|\-]*\s*", string.Empty).Trim();
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"(?i)\bcolor\b", "Colour").Trim();
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"(?i)\bpiercings?\s+colour\b", "Piercing Colour").Trim();
        if (string.Equals(clean, "Colour", StringComparison.OrdinalIgnoreCase))
            clean = "Piercing Colour";
        if (string.IsNullOrWhiteSpace(clean))
            clean = "Piercings";
        return $"{bodyName} - {clean}";
    }

    private static string CleanTargetOptionName(string value)
    {
        var clean = System.Text.RegularExpressions.Regex.Replace(value.Trim(), @"(?i)^\s*NSFW\s+ONLY\s*:\s*", string.Empty);
        clean = System.Text.RegularExpressions.Regex.Replace(clean, @"(?i)\s*\(\s*use only one!?\s*\)\s*$", string.Empty).Trim();
        return clean;
    }

    private static void ValidateNoConflictingImcAuthority(PenumbraV4Document document, IReadOnlyList<V4AppendRequest> appendRequests, V4TargetBodyGroupRequest request, V4GroupInfo parentGroup, JsonObject targetGroup)
    {
        if (!string.Equals(ReadString(targetGroup, "Type"), "Imc", StringComparison.OrdinalIgnoreCase))
            return;
        var targetIdentifier = PenumbraV4Document.FindProperty(targetGroup, "Identifier") as JsonObject
            ?? throw new InvalidDataException($"Target-body IMC group '{ReadString(targetGroup, "Name") ?? request.SourceKey}' has no Identifier.");

        var parentAppend = appendRequests.FirstOrDefault(x =>
            (string.Equals(x.GroupKey, parentGroup.StableKey, StringComparison.OrdinalIgnoreCase)
             || string.Equals(x.GroupKey, parentGroup.Name, StringComparison.OrdinalIgnoreCase))
            && string.Equals(x.NewOptionName, request.ParentOutputOptionName, StringComparison.OrdinalIgnoreCase));
        if (parentAppend is null)
            throw new InvalidOperationException($"Could not resolve the source option for generated parent '{request.ParentOutputOptionName}'.");
        var sourceOption = document.GetOption(parentGroup, parentAppend.SourceOptionKey);

        foreach (var existing in document.Groups.Where(g => string.Equals(g.Type, "Imc", StringComparison.OrdinalIgnoreCase)))
        {
            if (PenumbraV4Document.FindProperty(existing.Node, "Identifier") is not JsonObject existingIdentifier
                || !ImcIdentifiersEqual(existingIdentifier, targetIdentifier))
                continue;

            if (IsInactiveWhenGeneratedSingleOptionSelected(existing, parentGroup, sourceOption))
                continue;

            throw new InvalidDataException(
                $"Target-body IMC group '{ReadString(targetGroup, "Name") ?? request.SourceKey}' conflicts with existing IMC group '{existing.Name}' for the generated model. " +
                "RavaFit will not rely on Penumbra group priority to choose a winner. Disable/remove the conflicting IMC authority or use an outfit option whose existing IMC is conditioned only on the original Single-group option.");
        }
    }

    private static bool IsInactiveWhenGeneratedSingleOptionSelected(V4GroupInfo existing, V4GroupInfo parentGroup, V4OptionInfo sourceOption)
    {
        if (!string.Equals(parentGroup.Type, "Single", StringComparison.OrdinalIgnoreCase) || parentGroup.Id is null || sourceOption.Id is null)
            return false;
        if (PenumbraV4Document.FindProperty(existing.Node, "Condition") is not JsonObject condition)
            return false;

        // Current Penumbra uses Setting dependencies; the original Single option cannot also be selected.
        if (string.Equals(ReadString(condition, "Type"), "Setting", StringComparison.OrdinalIgnoreCase))
            return ReadGuid(condition, "Setting") == sourceOption.Id;

        if (!string.Equals(ReadString(condition, "Type"), "AnySetting", StringComparison.OrdinalIgnoreCase)
            || ReadGuid(condition, "Group") != parentGroup.Id
            || PenumbraV4Document.FindProperty(condition, "Options") is not JsonArray options)
            return false;
        return options.OfType<JsonValue>().Any(value => value.TryGetValue<string>(out var text)
            && Guid.TryParse(text, out var id) && id == sourceOption.Id.Value);
    }

    private static bool ImcIdentifiersEqual(JsonObject left, JsonObject right)
    {
        static string Canonical(JsonObject obj, string name)
        {
            var node = PenumbraV4Document.FindProperty(obj, name);
            if (node is null)
                return "<null>";
            if (node is JsonValue value)
            {
                if (value.TryGetValue<string>(out var text))
                    return text.Trim().ToLowerInvariant();
                if (value.TryGetValue<int>(out var integer))
                    return integer.ToString(System.Globalization.CultureInfo.InvariantCulture);
                if (value.TryGetValue<uint>(out var unsigned))
                    return unsigned.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
            return node.ToJsonString();
        }

        return new[] { "PrimaryId", "SecondaryId", "Variant", "ObjectType", "EquipSlot", "BodySlot" }
            .All(name => string.Equals(Canonical(left, name), Canonical(right, name), StringComparison.OrdinalIgnoreCase));
    }

    private static void RetargetTargetBodyGroup(JsonObject group, string targetGamePath)
    {
        if (!string.Equals(ReadString(group, "Type"), "Imc", StringComparison.OrdinalIgnoreCase))
            return;
        var identifier = PenumbraV4Document.FindProperty(group, "Identifier") as JsonObject
            ?? throw new InvalidDataException($"Target-body IMC group '{ReadString(group, "Name")}' has no Identifier.");
        var match = System.Text.RegularExpressions.Regex.Match(NormalizeGamePath(targetGamePath), @"/e(?<set>[0-9]{4})/model/c[0-9]{4}e[0-9]{4}_(?<suffix>top|dwn|glv|sho)\.mdl$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (!match.Success || !int.TryParse(match.Groups["set"].Value, out var setId))
            throw new InvalidDataException($"Could not retarget target-body IMC group to equipment path '{targetGamePath}'.");
        var equipSlot = match.Groups["suffix"].Value.ToLowerInvariant() switch
        {
            "top" => "Body",
            "dwn" => "Legs",
            "glv" => "Hands",
            "sho" => "Feet",
            _ => throw new InvalidDataException($"Unsupported equipment model path '{targetGamePath}'."),
        };
        SetProperty(identifier, "ObjectType", JsonValue.Create("Equipment"));
        SetProperty(identifier, "PrimaryId", JsonValue.Create(setId));
        SetProperty(identifier, "SecondaryId", JsonValue.Create(0));
        SetProperty(identifier, "EquipSlot", JsonValue.Create(equipSlot));
    }

    private static string? ReadString(JsonObject obj, string name)
    {
        var node = PenumbraV4Document.FindProperty(obj, name);
        return node is JsonValue value && value.TryGetValue<string>(out var result) ? result : null;
    }

    private static ulong ReadUInt64(JsonObject obj, string name)
    {
        if (PenumbraV4Document.FindProperty(obj, name) is not JsonValue value) return 0;
        if (value.TryGetValue<ulong>(out var unsigned)) return unsigned;
        if (value.TryGetValue<long>(out var signed) && signed >= 0) return (ulong)signed;
        if (value.TryGetValue<int>(out var integer) && integer >= 0) return (ulong)integer;
        return 0;
    }

    private static int? ReadInt(JsonObject obj, string name)
    {
        var node = PenumbraV4Document.FindProperty(obj, name);
        return node is JsonValue value && value.TryGetValue<int>(out var result) ? result : null;
    }

    private static Guid? ReadGuid(JsonObject obj, string name)
    {
        var value = ReadString(obj, name);
        return Guid.TryParse(value, out var result) ? result : null;
    }

    public static string NormalizeGamePath(string value)
        => value.Trim().Replace('\\', '/').TrimStart('/');

    public static string NormalizeRelativePath(string value)
    {
        var normalized = value.Trim().Replace('\\', '/').TrimStart('/');
        if (normalized.Contains(':', StringComparison.Ordinal))
            throw new InvalidDataException("Penumbra V4 Files values must be relative paths, not absolute paths.");
        if (normalized.Split('/').Any(part => part == ".."))
            throw new InvalidDataException("Relative output path may not escape the mod directory.");
        return normalized;
    }

    private static void SetProperty(JsonObject obj, string canonicalName, JsonNode? value)
    {
        var actualName = PenumbraV4Document.FindPropertyName(obj, canonicalName) ?? canonicalName;
        obj[actualName] = value;
    }

    private static async Task WriteDurableAsync(string path, string content, CancellationToken cancellationToken)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(content);
        await using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.Asynchronous | FileOptions.WriteThrough);
        await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
        await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
        stream.Flush(true);
    }
}
