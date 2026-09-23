using System.IO.Compression;
using RavaFit.Core;
using RavaFit.Core.Models;
using Xunit;

namespace RavaFit.Core.Tests;

public sealed class BodyCatalogueTests
{
    [Fact]
    public void ReadsRacePayloadsAndFiltersBySelectedModelRace()
    {
        var root = Path.Combine(Path.GetTempPath(), "RavaFitBodyTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var path = Path.Combine(root, "Fixture.rbody");
            using (var archive = ZipFile.Open(path, ZipArchiveMode.Create))
            {
                Write(archive, "manifest.json", """
                { "format": "RBODY", "version": 3, "collection": "Fixture" }
                """);
                Write(archive, "catalogue.json", """
                {
                  "collection": "Fixture",
                  "bodies": [
                    {
                      "id": "body-a",
                      "display_name": "Body A",
                      "slots": {
                        "Chest": [
                          {
                            "id": "variant-a",
                            "display_name": "Variant A",
                            "sexes": ["Female"],
                            "support_surface": "smallclothes",
                            "canonical_solver_payload_id": "canonical",
                            "race_payloads": [
                              { "race_code": "0201", "solver_payload_id": "payload-0201" },
                              { "race_code": "1801", "solver_payload_id": "payload-1801" }
                            ]
                          }
                        ]
                      }
                    },
                    {
                      "id": "body-b",
                      "display_name": "Body B",
                      "slots": {
                        "Chest": [
                          {
                            "id": "variant-b",
                            "display_name": "Variant B",
                            "canonical_solver_payload_id": "other",
                            "race_payloads": [
                              { "race_code": "0101", "solver_payload_id": "payload-0101" }
                            ]
                          }
                        ]
                      }
                    }
                  ]
                }
                """);
            }

            var library = BodyCatalogueReader.Read(path);
            var a = Assert.Single(library.Variants.Where(v => v.BodyName == "Body A"));
            Assert.Contains("0201", a.RaceCodes);
            Assert.Contains("1801", a.RaceCodes);
            Assert.Equal("canonical", a.CanonicalPayloadId);
            Assert.Null(a.CanonicalRaceCode);
            Assert.True(a.SupportsGender("Female"));
            Assert.False(a.SupportsGender("Male"));
            Assert.True(a.IsSmallclothesSupport);

            var set = new BodyCatalogueSet();
            set.Scan(root);
            var midlanderFemale = Assert.Single(set.ForSlot(BodySlots.Chest, "0201"));
            Assert.Equal("Body A", midlanderFemale.BodyName);
            Assert.Empty(set.ForSlot(BodySlots.Chest, "9999"));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    [Fact]
    public void CatalogueMarksOnlyRacePayloadsThatContainPiercingGeometry()
    {
        var root = Path.Combine(Path.GetTempPath(), "RavaFitBodyTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var path = Path.Combine(root, "PiercingFixture.rbody");
            using (var archive = ZipFile.Open(path, ZipArchiveMode.Create))
            {
                Write(archive, "manifest.json", """{ "format": "RBODY", "version": 4, "collection": "Fixture" }""");
                Write(archive, "catalogue.json", """
                {
                  "collection":"Fixture",
                  "bodies":[{
                    "id":"body", "display_name":"Body", "target_option_profile_id":"piercing-profile",
                    "slots":{"Chest":[{
                      "id":"variant", "display_name":"Variant", "canonical_solver_payload_id":"pierced",
                      "race_payloads":[
                        {"race_code":"0201","solver_payload_id":"pierced"},
                        {"race_code":"0401","solver_payload_id":"plain"},
                        {"race_code":"0601","solver_payload_id":"attribute-pierced"},
                        {"race_code":"0801","solver_payload_id":"dermal-pierced"},
                        {"race_code":"1001","solver_payload_id":"neolithe-permuted"}
                      ]
                    }]}
                  }]
                }
                """);
                Write(archive, "payload_index.json", """
                {
                  "pierced": {
                    "materials":["/body.mtrl","/piercings.mtrl"],
                    "mesh_records":[
                      {"material_index":0,"vertex_count":100,"index_count":300},
                      {"material_index":1,"vertex_count":20,"index_count":60}
                    ]
                  },
                  "plain": {
                    "materials":["/body.mtrl"],
                    "mesh_records":[{"material_index":0,"vertex_count":100,"index_count":300}]
                  },
                  "attribute-pierced": {
                    "materials":["/body.mtrl"],
                    "mesh_records":[{
                      "material_index":0,"vertex_count":100,"index_count":300,
                      "submeshes":[{"index_count":60,"attributes":["bellyring"]}]
                    }]
                  },
                  "dermal-pierced": {
                    "materials":["/body.mtrl","/dermal_gold.mtrl"],
                    "mesh_records":[
                      {"material_index":0,"vertex_count":100,"index_count":300},
                      {"material_index":1,"vertex_count":20,"index_count":60}
                    ]
                  },
                  "neolithe-permuted": {
                    "materials":["/body.mtrl","/underwear.mtrl","/mt_c0201b0001_neolithe_piercings.mtrl"],
                    "mesh_records":[
                      {"mesh_index":0,"material_index":0,"vertex_count":7164,"index_count":39450},
                      {"mesh_index":1,"material_index":2,"vertex_count":0,"index_count":0},
                      {"mesh_index":2,"material_index":1,"vertex_count":3733,"index_count":17556}
                    ]
                  }
                }
                """);
            }

            var variant = Assert.Single(BodyCatalogueReader.Read(path).Variants);
            Assert.Contains("0201", variant.PiercingRaceCodes!);
            Assert.DoesNotContain("0401", variant.PiercingRaceCodes!);
            Assert.Contains("0601", variant.PiercingRaceCodes!);
            Assert.Contains("0801", variant.PiercingRaceCodes!);
            Assert.Contains("1001", variant.PiercingRaceCodes!);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    [Fact]
    public void UnifiedBodiesLibraryIsPreferredOverLegacySiblingsAndPreservesCollectionAndTargetProfile()
    {
        var root = Path.Combine(Path.GetTempPath(), "RavaFitBodyTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            WriteLibrary(Path.Combine(root, "Bodies.rbody"), "Unified", "Bibo", "body-unified", "profile-unified");
            WriteLibrary(Path.Combine(root, "Female.rbody"), "Legacy", "Female", "body-legacy", null);

            var set = new BodyCatalogueSet();
            set.Scan(root);

            var library = Assert.Single(set.Libraries);
            Assert.Equal("Bodies.rbody", Path.GetFileName(library.Path));
            var variant = Assert.Single(set.Variants);
            Assert.Equal("Bibo", variant.Collection);
            Assert.Equal("body-unified", variant.BodyId);
            Assert.Equal("profile-unified", variant.TargetOptionProfileId);
            Assert.Equal("0201", variant.CanonicalRaceCode);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    [Fact]
    public void UnifiedCatalogueAndPersistentUserCatalogueLoadTogether()
    {
        var root = Path.Combine(Path.GetTempPath(), "RavaFitBodyTests", Guid.NewGuid().ToString("N"));
        var bundled = Path.Combine(root, "Bundled");
        var user = Path.Combine(root, "UserBodies");
        Directory.CreateDirectory(bundled);
        Directory.CreateDirectory(user);
        try
        {
            WriteLibrary(Path.Combine(bundled, "Bodies.rbody"), "Unified", "Bibo", "body-unified", null);
            WriteLibrary(Path.Combine(bundled, "Female.rbody"), "Legacy", "Female", "body-legacy", null);
            WriteLibrary(Path.Combine(user, "UserBodies.rbody"), "Custom", "Custom", "body-custom", null);

            var set = new BodyCatalogueSet();
            set.Scan(bundled, user);

            Assert.Equal(2, set.Libraries.Count);
            Assert.Contains(set.Variants, variant => variant.BodyId == "body-unified");
            Assert.Contains(set.Variants, variant => variant.BodyId == "body-custom");
            Assert.DoesNotContain(set.Variants, variant => variant.BodyId == "body-legacy");
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    [Fact]
    public void CharacterRaceCatalogueUsesExactRaceThenSameGenderMidlanderFallback()
    {
        Assert.Equal("0401", CharacterRaceCatalog.FromCustomize(1, 1, 2)!.Code);
        Assert.Equal("0301", CharacterRaceCatalog.FromCustomize(1, 0, 2)!.Code);
        Assert.Equal("1601", CharacterRaceCatalog.FromCustomize(7, 1, 1)!.Code);
        Assert.Equal("1701", CharacterRaceCatalog.FromCustomize(8, 0, 2)!.Code);

        var femaleMidlanderOnly = new BodyVariantInfo("Test", "body", "Body", BodySlots.Chest, "v", "V", "test.rbody", "payload", ["0201"], null, "0201");
        Assert.Equal("0201", CharacterRaceCatalog.ResolveBodyPayloadRace(femaleMidlanderOnly, "0401"));
        Assert.Null(CharacterRaceCatalog.ResolveBodyPayloadRace(femaleMidlanderOnly, "0301"));
        Assert.Null(CharacterRaceCatalog.ResolveBodyPayloadRace(femaleMidlanderOnly, "1201"));
        var femaleLalafell = femaleMidlanderOnly with { RaceCodes = ["1201"], CanonicalRaceCode = "1201" };
        Assert.Equal("1201", CharacterRaceCatalog.ResolveBodyPayloadRace(femaleLalafell, "1201"));
        var sharedLalafellFeet = femaleMidlanderOnly with { Slot = BodySlots.Feet, RaceCodes = ["1101"], CanonicalRaceCode = "1101", Sexes = ["Male"] };
        Assert.True(sharedLalafellFeet.SupportsGender("Female"));
        Assert.Equal("1101", CharacterRaceCatalog.ResolveBodyPayloadRace(sharedLalafellFeet, "1201"));
        Assert.True(CharacterRaceCatalog.RequiresSkeletonProportionRetarget("0201", "1201"));
        Assert.True(CharacterRaceCatalog.RequiresSkeletonProportionRetarget("0201", "0401"));
        Assert.False(CharacterRaceCatalog.RequiresSkeletonProportionRetarget("0201", "0201"));
        Assert.Equal("chara/equipment/e0001/model/c0401e0001_top.mdl", CharacterRaceCatalog.RewriteHumanRaceCode("chara/equipment/e0001/model/c0201e0001_top.mdl", "0401"));
    }


    [Fact]
    public void SmallclothesBridgeIsSymmetricAcrossGenderSwaps()
    {
        var maleHighlander = CharacterRaceCatalog.FromCode("0301")!;
        var femaleHighlander = CharacterRaceCatalog.FromCode("0401")!;

        Assert.True(CharacterRaceCatalog.RequiresSmallclothesTarget("Female", maleHighlander, BodySlots.Legs));
        Assert.False(CharacterRaceCatalog.RequiresSmallclothesTarget("Male", maleHighlander, BodySlots.Legs));
        Assert.False(CharacterRaceCatalog.RequiresSmallclothesTarget("Female", maleHighlander, BodySlots.Chest));
        Assert.False(CharacterRaceCatalog.RequiresSmallclothesTarget("Male", femaleHighlander, BodySlots.Legs));

        Assert.True(CharacterRaceCatalog.RequiresSmallclothesSource(maleHighlander, "Female", BodySlots.Legs));
        Assert.False(CharacterRaceCatalog.RequiresSmallclothesSource(maleHighlander, "Male", BodySlots.Legs));
        Assert.False(CharacterRaceCatalog.RequiresSmallclothesSource(maleHighlander, "Female", BodySlots.Chest));
        Assert.False(CharacterRaceCatalog.RequiresSmallclothesSource(femaleHighlander, "Male", BodySlots.Legs));

        Assert.True(CharacterRaceCatalog.RequiresCrossSexSmallclothesBridge(femaleHighlander, maleHighlander, BodySlots.Legs));
        Assert.True(CharacterRaceCatalog.RequiresCrossSexSmallclothesBridge(maleHighlander, femaleHighlander, BodySlots.Legs));
        Assert.False(CharacterRaceCatalog.RequiresCrossSexSmallclothesBridge(maleHighlander, maleHighlander, BodySlots.Legs));
        Assert.False(CharacterRaceCatalog.RequiresCrossSexSmallclothesBridge(femaleHighlander, maleHighlander, BodySlots.Chest));
    }

    private static void WriteLibrary(string path, string collection, string bodyCollection, string bodyId, string? targetProfile)
    {
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        Write(archive, "manifest.json", $$"""
        { "format": "RBODY", "version": 4, "collection": "{{collection}}" }
        """);
        var profile = targetProfile is null ? "" : $$""", "target_option_profile_id": "{{targetProfile}}""";
        Write(archive, "catalogue.json", $$"""
        {
          "collection": "{{collection}}",
          "bodies": [{
            "id": "{{bodyId}}",
            "display_name": "{{bodyId}}",
            "collection": "{{bodyCollection}}"{{profile}},
            "slots": { "Chest": [{
              "id": "default", "display_name": "Default", "canonical_solver_payload_id": "payload",
              "race_payloads": [{ "race_code": "0201", "solver_payload_id": "payload" }]
            }] }
          }]
        }
        """);
    }

    private static void Write(ZipArchive archive, string name, string content)
    {
        var entry = archive.CreateEntry(name);
        using var writer = new StreamWriter(entry.Open());
        writer.Write(content);
    }
}
