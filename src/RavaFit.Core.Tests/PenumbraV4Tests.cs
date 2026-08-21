using Xunit;
using System.Text.Json.Nodes;
using RavaFit.Core.Penumbra;
using RavaFit.Core.Models;

namespace RavaFit.Core.Tests;

public sealed class PenumbraV4Tests
{
    [Fact]
    public void ReadsDefaultAndOptionModelsWithOptionWinning()
    {
        using var fixture = Fixture.Create();
        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var models = doc.GetModelRedirects("Body Size", "YAB Medium Buff");
        Assert.Equal(2, models.Count);
        Assert.Contains(models, m => m.GamePath == "chara/equipment/e0001/model/c0101e0001_top.mdl" && !m.FromDefaultData && m.RelativePath == "files/yab/top.mdl");
        Assert.Contains(models, m => m.GamePath == "chara/equipment/e0001/model/c0101e0001_dwn.mdl" && m.FromDefaultData);
    }

    [Fact]
    public void OutfitRowsReadOnlyModelsOwnedByTheChosenOption()
    {
        using var fixture = Fixture.Create();
        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var models = doc.GetOptionModelRedirects("Body Size", "YAB Medium Buff");

        var model = Assert.Single(models);
        Assert.Equal("chara/equipment/e0001/model/c0101e0001_top.mdl", model.GamePath);
        Assert.False(model.FromDefaultData);
        Assert.DoesNotContain(models, candidate => candidate.GamePath.EndsWith("_dwn.mdl", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task AppendsNewOptionWithoutChangingSource()
    {
        using var fixture = Fixture.Create();
        var before = PenumbraV4Document.Load(fixture.MetaPath);
        var group = before.GetGroup("Body Size");
        var source = before.GetOption(group, "YAB Medium Buff");
        var sourceJson = source.Node.ToJsonString();

        var writer = new PenumbraV4Writer();
        var result = await writer.AppendClonedOptionAsync(new V4AppendRequest(
            fixture.MetaPath,
            group.StableKey,
            source.StableKey,
            "Neolithe Hazelnut M",
            new Dictionary<string, string>
            {
                ["chara/equipment/e0001/model/c0101e0001_top.mdl"] = "RavaFit/Neolithe/top.mdl",
            }));

        Assert.True(File.Exists(result.BackupPath));
        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var afterGroup = after.GetGroup("Body Size");
        Assert.Equal(3, afterGroup.Options.Count);
        var afterSource = after.GetOption(afterGroup, source.StableKey);
        Assert.Equal(sourceJson, afterSource.Node.ToJsonString());
        var added = afterGroup.Options.Single(o => o.Id == result.NewOptionId);
        Assert.Equal("Neolithe Hazelnut M", added.Name);
        Assert.Equal("files/shared.tex", added.Node["Files"]!["chara/equipment/e0001/texture/shared.tex"]!.GetValue<string>());
        Assert.Equal("RavaFit/Neolithe/top.mdl", added.Node["Files"]!["chara/equipment/e0001/model/c0101e0001_top.mdl"]!.GetValue<string>());
    }

    [Fact]
    public async Task AppendedOptionCanCarryGeneratedMaterialRedirectionsWithoutTouchingSource()
    {
        using var fixture = Fixture.Create();
        var before = PenumbraV4Document.Load(fixture.MetaPath);
        var group = before.GetGroup("Body Size");
        var source = before.GetOption(group, "YAB Medium Buff");
        var sourceJson = source.Node.ToJsonString();

        var writer = new PenumbraV4Writer();
        var result = await writer.AppendClonedOptionAsync(new V4AppendRequest(
            fixture.MetaPath, group.StableKey, source.StableKey, "Accessory split",
            new Dictionary<string, string>
            {
                ["chara/accessory/a0135/model/c0101a0135_wrs.mdl"] = "RavaFit/Split/wrs.mdl",
            },
            new Dictionary<string, string>
            {
                ["chara/accessory/a0135/material/v0004/mt_source_top.mtrl"] = "RavaFit/Split/mt_source_top.mtrl",
            }));

        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var afterGroup = after.GetGroup("Body Size");
        Assert.Equal(sourceJson, after.GetOption(afterGroup, source.StableKey).Node.ToJsonString());
        var added = afterGroup.Options.Single(option => option.Id == result.NewOptionId);
        Assert.Equal("RavaFit/Split/wrs.mdl", added.Node["Files"]!["chara/accessory/a0135/model/c0101a0135_wrs.mdl"]!.GetValue<string>());
        Assert.Equal("RavaFit/Split/mt_source_top.mtrl", added.Node["Files"]!["chara/accessory/a0135/material/v0004/mt_source_top.mtrl"]!.GetValue<string>());
    }

    [Fact]
    public async Task BatchAppendWritesAllGeneratedOptionsInOneMetaTransaction()
    {
        using var fixture = Fixture.Create();
        var writer = new PenumbraV4Writer();
        var results = await writer.AppendClonedOptionsAsync(new[]
        {
            new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Medium Buff", "Neolithe M",
                new Dictionary<string, string> { ["chara/equipment/e0001/model/c0101e0001_top.mdl"] = "RavaFit/Neolithe/top.mdl" }),
            new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Large", "TBSE-F",
                new Dictionary<string, string> { ["chara/equipment/e0001/model/c0101e0001_dwn.mdl"] = "RavaFit/TBSE/dwn.mdl" }),
        });

        Assert.Equal(2, results.Count);
        Assert.True(File.Exists(results[0].BackupPath));
        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var group = after.GetGroup("Body Size");
        Assert.Equal(4, group.Options.Count);
        Assert.Contains(group.Options, option => option.Name == "Neolithe M");
        Assert.Contains(group.Options, option => option.Name == "TBSE-F");
    }

    [Fact]
    public async Task TargetBodyGroupsArePlacedOnDedicatedPageConditionedOnGeneratedOptionAndImcRetargeted()
    {
        using var fixture = Fixture.Create();
        var before = PenumbraV4Document.Load(fixture.MetaPath);
        var parent = before.GetGroup("Body Size");
        Assert.NotNull(parent.Id);

        var targetGroup = new JsonObject
        {
            ["Name"] = "CHEST OPTIONS: SmallClothes",
            ["Type"] = "Imc",
            ["Page"] = 1,
            ["Priority"] = 1,
            ["DefaultSettings"] = 0,
            ["Identifier"] = new JsonObject
            {
                ["ObjectType"] = "Equipment",
                ["PrimaryId"] = 0,
                ["SecondaryId"] = 0,
                ["Variant"] = 1,
                ["EquipSlot"] = "Body",
                ["BodySlot"] = "Unknown",
            },
            ["DefaultEntry"] = new JsonObject { ["AttributeMask"] = 0 },
            ["AllVariants"] = true,
            ["OnlyAttributes"] = false,
            ["Options"] = new JsonArray(new JsonObject { ["Name"] = "Nipple Piercings", ["AttributeMask"] = 4 }),
        };

        var writer = new PenumbraV4Writer();
        var results = await writer.AppendClonedOptionsWithTargetBodyGroupsAsync(
            [new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Medium Buff", "Neolithe M",
                new Dictionary<string, string> { ["chara/equipment/e0042/model/c0201e0042_top.mdl"] = "RavaFit/Neolithe/top.mdl" })],
            [new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:chest", targetGroup)]);

        var appended = Assert.Single(results);
        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var generated = after.GetOption(after.GetGroup("Body Size"), appended.NewOptionId.ToString("D"));
        var bodyGroup = Assert.Single(after.Groups.Where(group => group.Name == "Neolithe - Piercings"));
        var condition = Assert.IsType<JsonObject>(bodyGroup.Node["Condition"]);
        Assert.Equal("Setting", condition["Type"]!.GetValue<string>());
        Assert.Equal(generated.Id!.Value.ToString("D"), condition["Setting"]!.GetValue<string>());
        Assert.Null(condition["Group"]);
        Assert.Null(condition["Options"]);
        var identifier = bodyGroup.Node["Identifier"]!.AsObject();
        Assert.Equal(42, identifier["PrimaryId"]!.GetValue<int>());
        Assert.Equal("Body", identifier["EquipSlot"]!.GetValue<string>());
        var page = bodyGroup.Node["Page"]!.GetValue<int>();
        Assert.Equal("Body Options", after.Root["PageNames"]![page.ToString()]!.GetValue<string>());
    }

    [Fact]
    public async Task AlwaysOnTargetSupportIsFoldedAndDuplicateRetargetedImcShowsOnlyEffectivePiercingGroup()
    {
        using var fixture = Fixture.Create();
        var supportGroup = new JsonObject
        {
            ["Name"] = "BASE INSTALL: Piercings",
            ["Type"] = "Single",
            ["Options"] = new JsonArray(new JsonObject
            {
                ["Name"] = "Required",
                ["Files"] = new JsonObject { ["chara/test/piercing.mtrl"] = "RavaFit/TargetBodyOptions/test/piercing.mtrl" },
            }),
        };
        var smallclothes = CreateTargetImcGroup(0);
        smallclothes["Name"] = "CHEST OPTIONS: SmallClothes";
        smallclothes["Options"] = new JsonArray(new JsonObject { ["Name"] = "NSFW ONLY: Nipple piercings", ["AttributeMask"] = 4 });
        var emperor = CreateTargetImcGroup(0);
        emperor["Name"] = "CHEST OPTIONS: The Emperor's New Robe";
        emperor["Options"] = new JsonArray(new JsonObject { ["Name"] = "NSFW ONLY: Nipple heart piercings", ["AttributeMask"] = 4 });

        var writer = new PenumbraV4Writer();
        var results = await writer.AppendClonedOptionsWithTargetBodyGroupsAsync(
            [new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Medium Buff", "Neolithe M",
                new Dictionary<string, string> { ["chara/equipment/e0042/model/c0201e0042_top.mdl"] = "RavaFit/Neolithe/top.mdl" })],
            [
                new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                    "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:support", supportGroup),
                new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                    "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:small", smallclothes),
                new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                    "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:emperor", emperor),
            ]);

        var appended = Assert.Single(results);
        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var generated = after.GetOption(after.GetGroup("Body Size"), appended.NewOptionId.ToString("D"));
        Assert.Equal("RavaFit/TargetBodyOptions/test/piercing.mtrl", generated.Node["Files"]!["chara/test/piercing.mtrl"]!.GetValue<string>());

        var bodyGroups = after.Groups.Where(group => group.Name.StartsWith("Neolithe -", StringComparison.Ordinal)).ToArray();
        var bodyGroup = Assert.Single(bodyGroups);
        Assert.Equal("Neolithe - Piercings", bodyGroup.Name);
        var option = Assert.Single(bodyGroup.Options);
        Assert.Equal("Nipple heart piercings", option.Name);
    }

    [Fact]
    public async Task ConflictingExistingImcAuthorityIsRejected()
    {
        using var fixture = Fixture.Create();
        AddExistingImcGroup(fixture.MetaPath, CreateTargetImcGroup(42), null);
        var writer = new PenumbraV4Writer();

        var error = await Assert.ThrowsAsync<InvalidDataException>(() => writer.AppendClonedOptionsWithTargetBodyGroupsAsync(
            [new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Medium Buff", "Neolithe M",
                new Dictionary<string, string> { ["chara/equipment/e0042/model/c0201e0042_top.mdl"] = "RavaFit/Neolithe/top.mdl" })],
            [new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:chest", CreateTargetImcGroup(0))]));

        Assert.Contains("conflicts with existing IMC group", error.Message);
    }

    [Fact]
    public async Task ExistingImcConditionedOnOriginalSingleOptionIsSafe()
    {
        using var fixture = Fixture.Create();
        var before = PenumbraV4Document.Load(fixture.MetaPath);
        var parent = before.GetGroup("Body Size");
        var source = before.GetOption(parent, "YAB Medium Buff");
        Assert.NotNull(parent.Id);
        Assert.NotNull(source.Id);
        var safeCondition = new JsonObject
        {
            ["Type"] = "Setting",
            ["Setting"] = source.Id!.Value.ToString("D"),
        };
        AddExistingImcGroup(fixture.MetaPath, CreateTargetImcGroup(42), safeCondition);

        var writer = new PenumbraV4Writer();
        var results = await writer.AppendClonedOptionsWithTargetBodyGroupsAsync(
            [new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Medium Buff", "Neolithe M",
                new Dictionary<string, string> { ["chara/equipment/e0042/model/c0201e0042_top.mdl"] = "RavaFit/Neolithe/top.mdl" })],
            [new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:chest", CreateTargetImcGroup(0))]);

        Assert.Single(results);
    }

    [Fact]
    public async Task ExistingLegacyAnySettingImcConditionedOnOriginalSingleOptionIsStillRecognisedAsSafe()
    {
        using var fixture = Fixture.Create();
        var before = PenumbraV4Document.Load(fixture.MetaPath);
        var parent = before.GetGroup("Body Size");
        var source = before.GetOption(parent, "YAB Medium Buff");
        Assert.NotNull(parent.Id);
        Assert.NotNull(source.Id);
        var safeCondition = new JsonObject
        {
            ["Type"] = "AnySetting",
            ["Group"] = parent.Id!.Value.ToString("D"),
            ["Options"] = new JsonArray(source.Id!.Value.ToString("D")),
        };
        AddExistingImcGroup(fixture.MetaPath, CreateTargetImcGroup(42), safeCondition);

        var writer = new PenumbraV4Writer();
        var results = await writer.AppendClonedOptionsWithTargetBodyGroupsAsync(
            [new V4AppendRequest(fixture.MetaPath, "Body Size", "YAB Medium Buff", "Neolithe M",
                new Dictionary<string, string> { ["chara/equipment/e0042/model/c0201e0042_top.mdl"] = "RavaFit/Neolithe/top.mdl" })],
            [new V4TargetBodyGroupRequest("Body Size", "Neolithe M", "Neolithe", BodySlots.Chest,
                "chara/equipment/e0042/model/c0201e0042_top.mdl", "neolithe:chest", CreateTargetImcGroup(0))]);

        Assert.Single(results);
    }

    [Fact]
    public async Task DuplicateOptionNameIsRejected()
    {
        using var fixture = Fixture.Create();
        var writer = new PenumbraV4Writer();
        await Assert.ThrowsAsync<InvalidOperationException>(() => writer.AppendClonedOptionAsync(new V4AppendRequest(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", "YAB Large",
            new Dictionary<string, string> { ["x/model.mdl"] = "RavaFit/model.mdl" })));
    }


    [Fact]
    public async Task PiercingReplacementRemovesDedicatedPiercingGroups()
    {
        using var fixture = Fixture.Create();
        var root = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        root["Groups"]!.AsArray().Add(new JsonObject
        {
            ["Name"] = "Piercing Colour",
            ["Type"] = "Single",
            ["Id"] = Guid.NewGuid().ToString("D"),
            ["Options"] = new JsonArray(
                new JsonObject { ["Id"] = Guid.NewGuid().ToString("D"), ["Name"] = "Gold", ["Files"] = new JsonObject() },
                new JsonObject { ["Id"] = Guid.NewGuid().ToString("D"), ["Name"] = "Silver", ["Files"] = new JsonObject() }),
        });
        File.WriteAllText(fixture.MetaPath, root.ToJsonString());

        var writer = new PenumbraV4Writer();
        var result = await writer.ReplaceModelAndPiercingControlsAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff",
            "chara/equipment/e0001/model/c0101e0001_top.mdl", "RavaFit/Customise/replaced.mdl", []);

        Assert.Equal(1, result.RemovedPiercingGroups);
        var after = PenumbraV4Document.Load(fixture.MetaPath);
        Assert.DoesNotContain(after.Groups, group => group.Name == "Piercing Colour");
        Assert.Equal("RavaFit/Customise/replaced.mdl", after.GetOption(after.GetGroup("Body Size"), "YAB Medium Buff").Node["Files"]!["chara/equipment/e0001/model/c0101e0001_top.mdl"]!.GetValue<string>());
    }

    [Fact]
    public async Task PiercingReplacementRemovesGenericSupportGroupWithPiercingResources()
    {
        using var fixture = Fixture.Create();
        var root = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        root["Groups"]!.AsArray().Add(new JsonObject
        {
            ["Name"] = "Required Files",
            ["Type"] = "Single",
            ["Id"] = Guid.NewGuid().ToString("D"),
            ["Options"] = new JsonArray(new JsonObject
            {
                ["Name"] = "---",
                ["Files"] = new JsonObject { ["chara/example/piercings_n.tex"] = "files/piercings_n.tex" },
            }),
        });
        File.WriteAllText(fixture.MetaPath, root.ToJsonString());

        var writer = new PenumbraV4Writer();
        var result = await writer.ReplaceModelAndPiercingControlsAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff",
            "chara/equipment/e0001/model/c0101e0001_top.mdl", "RavaFit/Customise/replaced.mdl", []);

        Assert.Equal(1, result.RemovedPiercingGroups);
        var after = PenumbraV4Document.Load(fixture.MetaPath);
        Assert.DoesNotContain(after.Groups, group => group.Name == "Required Files");
    }

    [Fact]
    public async Task PiercingReplacementFailsClosedForMixedControlGroup()
    {
        using var fixture = Fixture.Create();
        var original = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        original["Groups"]!.AsArray().Add(new JsonObject
        {
            ["Name"] = "toggles | smallclothes body",
            ["Type"] = "Imc",
            ["Id"] = Guid.NewGuid().ToString("D"),
            ["Options"] = new JsonArray(
                new JsonObject { ["Name"] = "lumme piercings", ["AttributeMask"] = 3 },
                new JsonObject { ["Name"] = "sfw bra", ["Description"] = "body coverage", ["AttributeMask"] = 240 },
                new JsonObject { ["Name"] = "better pits", ["AttributeMask"] = 256 }),
        });
        File.WriteAllText(fixture.MetaPath, original.ToJsonString());
        var beforeBytes = File.ReadAllBytes(fixture.MetaPath);

        var writer = new PenumbraV4Writer();
        var error = await Assert.ThrowsAsync<NotSupportedException>(() => writer.ReplaceModelAndPiercingControlsAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff",
            "chara/equipment/e0001/model/c0101e0001_top.mdl", "RavaFit/Customise/replaced.mdl", []));

        Assert.Contains("mixed with unrelated controls", error.Message);
        Assert.Equal(beforeBytes, File.ReadAllBytes(fixture.MetaPath));
    }

    [Fact]
    public async Task VisibilityControlGetsPriorityAboveExistingGroups()
    {
        using var fixture = Fixture.Create();
        var root = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        root["Groups"]![0]!["Priority"] = 7;
        File.WriteAllText(fixture.MetaPath, root.ToJsonString());

        var writer = new PenumbraV4Writer();
        await writer.AddVisibilityToggleAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", "Hide Straps",
            "chara/equipment/e0001/model/c0101e0001_top.mdl", "RavaFit/Customise/hide-straps.mdl");

        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var visibility = Assert.Single(after.Groups.Where(group => group.Name == "Hide Straps"));
        Assert.Equal(8, visibility.Node["Priority"]!.GetValue<int>());
    }

    [Fact]
    public async Task DuplicateVisibilityControlForSameModelIsRejected()
    {
        using var fixture = Fixture.Create();
        var writer = new PenumbraV4Writer();
        await writer.AddVisibilityToggleAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", "Hide Straps",
            "chara/equipment/e0001/model/c0101e0001_top.mdl", "RavaFit/Customise/hide-straps.mdl");
        var beforeBytes = File.ReadAllBytes(fixture.MetaPath);

        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => writer.AddVisibilityToggleAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", "Hide Buckles",
            "chara/equipment/e0001/model/c0101e0001_top.mdl", "RavaFit/Customise/hide-buckles.mdl"));

        Assert.Contains("already has a RavaFit visibility control", error.Message);
        Assert.Equal(beforeBytes, File.ReadAllBytes(fixture.MetaPath));
    }

    [Fact]
    public async Task AttributeVisibilityControlsComposeInOneMultiGroup()
    {
        using var fixture = Fixture.Create();
        var writer = new PenumbraV4Writer();
        const string gamePath = "chara/equipment/e0001/model/c0101e0001_top.mdl";

        await writer.AddAttributeVisibilityToggleAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", "Armband", gamePath,
            "RavaFit/Customise/tagged-1.mdl", "atrx_ravafit_armband");
        await writer.AddAttributeVisibilityToggleAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", "Belly Ring", gamePath,
            "RavaFit/Customise/tagged-2.mdl", "atrx_ravafit_bellyring");

        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var source = after.GetOption(after.GetGroup("Body Size"), "YAB Medium Buff");
        Assert.Equal("RavaFit/Customise/tagged-2.mdl", source.Node["Files"]![gamePath]!.GetValue<string>());
        var visibility = Assert.Single(after.Groups.Where(group => group.Name == "Visibility · YAB Medium Buff"));
        Assert.Equal("Multi", visibility.Node["Type"]!.GetValue<string>());
        Assert.Equal(2, visibility.Options.Count);
        Assert.Equal(3UL, visibility.Node["DefaultSettings"]!.GetValue<ulong>());
        Assert.Contains(visibility.Options, option => option.Name == "Armband");
        Assert.Contains(visibility.Options, option => option.Name == "Belly Ring");
        var attrs = visibility.Options
            .SelectMany(option => option.Node["Manipulations"]!.AsArray().OfType<JsonObject>())
            .Select(manipulation => manipulation["Manipulation"]!["Attribute"]!.GetValue<string>())
            .ToArray();
        Assert.Contains("atrx_ravafit_armband", attrs);
        Assert.Contains("atrx_ravafit_bellyring", attrs);
        var baselineAttrs = source.Node["Manipulations"]!.AsArray().OfType<JsonObject>()
            .Where(manipulation => manipulation["Type"]?.GetValue<string>() == "Atr")
            .Select(manipulation => manipulation["Manipulation"]!.AsObject())
            .ToDictionary(payload => payload["Attribute"]!.GetValue<string>(), payload => payload["Entry"]!.GetValue<bool>(), StringComparer.OrdinalIgnoreCase);
        Assert.False(baselineAttrs["atrx_ravafit_armband"]);
        Assert.False(baselineAttrs["atrx_ravafit_bellyring"]);
    }

    [Fact]
    public async Task AttributeVisibilityBatchAddsMultipleControlsInOneWrite()
    {
        using var fixture = Fixture.Create();
        var writer = new PenumbraV4Writer();
        const string gamePath = "chara/equipment/e0001/model/c0101e0001_top.mdl";

        await writer.AddAttributeVisibilityTogglesAsync(
            fixture.MetaPath, "Body Size", "YAB Medium Buff", gamePath,
            "RavaFit/Customise/tagged-batch.mdl",
            new[]
            {
                (ToggleName: "Tanktop", AttributeName: "atrx_ravafit_tanktop"),
                (ToggleName: "Bra", AttributeName: "atrx_ravafit_bra"),
                (ToggleName: "Belly Ring", AttributeName: "atrx_ravafit_bellyring"),
            });

        var after = PenumbraV4Document.Load(fixture.MetaPath);
        var source = after.GetOption(after.GetGroup("Body Size"), "YAB Medium Buff");
        Assert.Equal("RavaFit/Customise/tagged-batch.mdl", source.Node["Files"]![gamePath]!.GetValue<string>());
        var visibility = Assert.Single(after.Groups.Where(group => group.Name == "Visibility · YAB Medium Buff"));
        Assert.Equal(3, visibility.Options.Count);
        Assert.Equal(7UL, visibility.Node["DefaultSettings"]!.GetValue<ulong>());
        Assert.Equal(new[] { "Tanktop", "Bra", "Belly Ring" }, visibility.Options.Select(option => option.Name).ToArray());
        var baselines = source.Node["Manipulations"]!.AsArray().OfType<JsonObject>()
            .Where(manipulation => manipulation["Type"]?.GetValue<string>() == "Atr")
            .Select(manipulation => manipulation["Manipulation"]!.AsObject())
            .ToDictionary(payload => payload["Attribute"]!.GetValue<string>(), payload => payload["Entry"]!.GetValue<bool>(), StringComparer.OrdinalIgnoreCase);
        Assert.False(baselines["atrx_ravafit_tanktop"]);
        Assert.False(baselines["atrx_ravafit_bra"]);
        Assert.False(baselines["atrx_ravafit_bellyring"]);
    }

    [Fact]
    public void ReadsDirectBodyEstWithNumericStrings()
    {
        using var fixture = Fixture.Create();
        var root = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        var source = root["Groups"]![0]!["Options"]![0]!.AsObject();
        source["Manipulations"] = new JsonArray(CreateEst("7", "1"));
        File.WriteAllText(fixture.MetaPath, root.ToJsonString());

        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var est = doc.GetEstSkeletonOverrides("Body Size", "YAB Medium Buff", "chara/equipment/e0001/model/c0101e0001_top.mdl");

        var value = Assert.Single(est);
        Assert.Equal(7, value.Entry);
        Assert.Equal("Body", value.Slot);
        Assert.Equal("Male", value.Gender);
        Assert.Equal("Midlander", value.Race);
    }

    [Fact]
    public void SourceOptionEntryZeroOverridesDefaultEst()
    {
        using var fixture = Fixture.Create();
        var root = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        root["DefaultData"]!["Manipulations"] = new JsonArray(CreateEst(7, 1));
        root["Groups"]![0]!["Options"]![0]!["Manipulations"] = new JsonArray(CreateEst(0, 1));
        File.WriteAllText(fixture.MetaPath, root.ToJsonString());

        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var est = doc.GetEstSkeletonOverrides("Body Size", "YAB Medium Buff", "chara/equipment/e0001/model/c0101e0001_top.mdl");

        var value = Assert.Single(est);
        Assert.Equal(0, value.Entry);
        Assert.Contains("YAB Medium Buff", value.Source);
    }

    [Fact]
    public void UsesActuallySelectedCompanionEstOption()
    {
        using var fixture = Fixture.Create();
        AddSkeletonGroup(fixture.MetaPath, ("Vanilla", null), ("IVCS", CreateEst(42, 1)));
        var active = new Dictionary<string, IReadOnlyCollection<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["Body Size"] = new[] { "YAB Medium Buff" },
            ["Skeleton"] = new[] { "IVCS" },
        };

        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var est = doc.GetEstSkeletonOverrides("Body Size", "YAB Medium Buff", "chara/equipment/e0001/model/c0101e0001_top.mdl", active);

        var value = Assert.Single(est);
        Assert.Equal(42, value.Entry);
        Assert.StartsWith("active:", value.Source);
    }

    [Fact]
    public void DoesNotEnableInactiveCompanionEstOption()
    {
        using var fixture = Fixture.Create();
        AddSkeletonGroup(fixture.MetaPath, ("Vanilla", null), ("IVCS", CreateEst(42, 1)));
        var active = new Dictionary<string, IReadOnlyCollection<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["Body Size"] = new[] { "YAB Medium Buff" },
            ["Skeleton"] = new[] { "Vanilla" },
        };

        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var est = doc.GetEstSkeletonOverrides("Body Size", "YAB Medium Buff", "chara/equipment/e0001/model/c0101e0001_top.mdl", active);

        Assert.Empty(est);
    }

    [Fact]
    public void MultipleActiveCompanionEstValuesAreRejected()
    {
        using var fixture = Fixture.Create();
        AddSkeletonGroup(fixture.MetaPath, ("IVCS A", CreateEst(42, 1)), ("IVCS B", CreateEst(43, 1)));
        var active = new Dictionary<string, IReadOnlyCollection<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["Skeleton"] = new[] { "IVCS A", "IVCS B" },
        };

        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        Assert.Throws<InvalidDataException>(() => doc.GetEstSkeletonOverrides(
            "Body Size", "YAB Medium Buff", "chara/equipment/e0001/model/c0101e0001_top.mdl", active));
    }

    [Fact]
    public void LegsDoNotRequestSupplementalBodyEst()
    {
        using var fixture = Fixture.Create();
        var root = JsonNode.Parse(File.ReadAllText(fixture.MetaPath))!.AsObject();
        root["Groups"]![0]!["Options"]![0]!["Manipulations"] = new JsonArray(CreateEst(42, 1));
        File.WriteAllText(fixture.MetaPath, root.ToJsonString());

        var doc = PenumbraV4Document.Load(fixture.MetaPath);
        var est = doc.GetEstSkeletonOverrides("Body Size", "YAB Medium Buff", "chara/equipment/e0001/model/c0101e0001_dwn.mdl");

        Assert.Empty(est);
    }

    private static JsonObject CreateTargetImcGroup(int setId)
        => new()
        {
            ["Name"] = "Piercings",
            ["Type"] = "Imc",
            ["Priority"] = 1,
            ["DefaultSettings"] = 0,
            ["Identifier"] = new JsonObject
            {
                ["ObjectType"] = "Equipment",
                ["PrimaryId"] = setId,
                ["SecondaryId"] = 0,
                ["Variant"] = 1,
                ["EquipSlot"] = "Body",
                ["BodySlot"] = "Unknown",
            },
            ["DefaultEntry"] = new JsonObject { ["AttributeMask"] = 0 },
            ["AllVariants"] = true,
            ["OnlyAttributes"] = false,
            ["Options"] = new JsonArray(new JsonObject { ["Name"] = "Nipple Piercings", ["AttributeMask"] = 4 }),
        };

    private static void AddExistingImcGroup(string metaPath, JsonObject group, JsonObject? condition)
    {
        var root = JsonNode.Parse(File.ReadAllText(metaPath))!.AsObject();
        group["Id"] = Guid.NewGuid().ToString("D");
        if (condition is not null)
            group["Condition"] = condition;
        root["Groups"]!.AsArray().Add(group);
        File.WriteAllText(metaPath, root.ToJsonString());
    }

    private static JsonObject CreateEst(object entry, object setId)
        => new()
        {
            ["Type"] = "Est",
            ["Manipulation"] = new JsonObject
            {
                ["Entry"] = ToJsonValue(entry),
                ["Gender"] = "Male",
                ["Race"] = "Midlander",
                ["SetId"] = ToJsonValue(setId),
                ["Slot"] = "Body",
            },
        };

    private static JsonNode ToJsonValue(object value)
        => value switch
        {
            int i => JsonValue.Create(i)!,
            string text => JsonValue.Create(text)!,
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

    private static void AddSkeletonGroup(string metaPath, params (string Name, JsonObject? Manipulation)[] options)
    {
        var root = JsonNode.Parse(File.ReadAllText(metaPath))!.AsObject();
        var groupOptions = new JsonArray();
        foreach (var (name, manipulation) in options)
        {
            var option = new JsonObject
            {
                ["Name"] = name,
                ["Id"] = Guid.NewGuid().ToString("D"),
                ["Files"] = new JsonObject(),
            };
            if (manipulation is not null)
                option["Manipulations"] = new JsonArray(manipulation);
            groupOptions.Add(option);
        }

        root["Groups"]!.AsArray().Add(new JsonObject
        {
            ["Name"] = "Skeleton",
            ["Type"] = "Multi",
            ["Id"] = Guid.NewGuid().ToString("D"),
            ["DefaultSettings"] = 0,
            ["Options"] = groupOptions,
        });
        File.WriteAllText(metaPath, root.ToJsonString());
    }

    [Fact]
    public void V3IsRejected()
    {
        using var fixture = Fixture.Create(fileVersion: 3);
        Assert.Throws<InvalidDataException>(() => PenumbraV4Document.Load(fixture.MetaPath));
    }

    private sealed class Fixture : IDisposable
    {
        private Fixture(string root, string metaPath) { Root = root; MetaPath = metaPath; }
        public string Root { get; }
        public string MetaPath { get; }

        public static Fixture Create(int fileVersion = 4)
        {
            var root = Path.Combine(Path.GetTempPath(), "RavaFitTests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            var meta = Path.Combine(root, "meta.json");
            var groupId = Guid.NewGuid();
            var sourceId = Guid.NewGuid();
            var largeId = Guid.NewGuid();
            var json = $$"""
            {
              "FileVersion": {{fileVersion}},
              "Name": "Fixture Outfit",
              "LastWrite": "2026-08-10T00:00:00Z",
              "DefaultData": {
                "Files": {
                  "chara/equipment/e0001/model/c0101e0001_top.mdl": "files/default/top.mdl",
                  "chara/equipment/e0001/model/c0101e0001_dwn.mdl": "files/default/dwn.mdl"
                }
              },
              "Groups": [
                {
                  "Name": "Body Size",
                  "Type": "Single",
                  "Id": "{{groupId:D}}",
                  "DefaultSettings": 0,
                  "Options": [
                    {
                      "Name": "YAB Medium Buff",
                      "Id": "{{sourceId:D}}",
                      "Files": {
                        "chara/equipment/e0001/model/c0101e0001_top.mdl": "files/yab/top.mdl",
                        "chara/equipment/e0001/texture/shared.tex": "files/shared.tex"
                      },
                      "Manipulations": [{"Type":"UnknownFixtureValue"}]
                    },
                    {
                      "Name": "YAB Large",
                      "Id": "{{largeId:D}}",
                      "Files": {"chara/equipment/e0001/model/c0101e0001_top.mdl":"files/yab-large/top.mdl"}
                    }
                  ]
                }
              ],
              "FutureUnknownField": {"KeepMe": true}
            }
            """;
            File.WriteAllText(meta, json);
            return new Fixture(root, meta);
        }

        public void Dispose()
        {
            try { Directory.Delete(Root, true); } catch { }
        }
    }
}
