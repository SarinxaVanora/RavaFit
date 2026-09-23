using System.IO.Compression;
using System.Text.Json;
using RavaFit.Core.Models;

namespace RavaFit.Core;

public static class BodyCatalogueReader
{
    public static BodyLibraryInfo Read(string path)
    {
        using var archive = ZipFile.OpenRead(path);
        using var manifest = ReadJson(archive, "manifest.json");
        using var catalogue = ReadJson(archive, "catalogue.json");
        using var payloadIndex = ReadOptionalJson(archive, "payload_index.json");
        var piercingPayloadIds = ReadPiercingPayloadIds(payloadIndex?.RootElement);
        var format = GetString(manifest.RootElement, "format");
        var version = GetInt(manifest.RootElement, "version");
        if (!string.Equals(format, "RBODY", StringComparison.OrdinalIgnoreCase) || version < 3)
            throw new InvalidDataException($"'{path}' is not an RBODY V3+ library.");

        var collection = GetString(catalogue.RootElement, "collection")
                      ?? GetString(manifest.RootElement, "collection")
                      ?? Path.GetFileNameWithoutExtension(path);
        var variants = new List<BodyVariantInfo>();
        if (!TryGetProperty(catalogue.RootElement, "bodies", out var bodies) || bodies.ValueKind != JsonValueKind.Array)
            return new BodyLibraryInfo(collection, Path.GetFullPath(path), version, variants);

        foreach (var body in bodies.EnumerateArray())
        {
            var bodyId = GetString(body, "id") ?? "unknown";
            var bodyName = GetString(body, "display_name") ?? bodyId;
            var bodyCollection = GetString(body, "collection") ?? collection;
            var targetOptionProfileId = GetString(body, "target_option_profile_id");
            if (!TryGetProperty(body, "slots", out var slots) || slots.ValueKind != JsonValueKind.Object)
                continue;

            foreach (var slot in slots.EnumerateObject())
            {
                if (!BodySlots.All.Contains(slot.Name, StringComparer.OrdinalIgnoreCase) || slot.Value.ValueKind != JsonValueKind.Array)
                    continue;
                foreach (var variant in slot.Value.EnumerateArray())
                {
                    var variantId = GetString(variant, "id") ?? "unknown";
                    var variantName = GetString(variant, "display_name") ?? variantId;
                    var payloadId = GetString(variant, "canonical_solver_payload_id");
                    var sexes = GetStringArray(variant, "sexes");
                    var supportSurface = GetString(variant, "support_surface");
                    var raceCodes = new List<string>();
                    string? canonicalRaceCode = null;
                    var piercingRaceCodes = new List<string>();
                    if (TryGetProperty(variant, "race_payloads", out var racePayloads) && racePayloads.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var racePayload in racePayloads.EnumerateArray())
                        {
                            var raceCode = GetString(racePayload, "race_code");
                            var racePayloadId = GetString(racePayload, "solver_payload_id");
                            if (!string.IsNullOrWhiteSpace(raceCode) && !raceCodes.Contains(raceCode, StringComparer.OrdinalIgnoreCase))
                                raceCodes.Add(raceCode);
                            if (canonicalRaceCode is null && !string.IsNullOrWhiteSpace(raceCode) && !string.IsNullOrWhiteSpace(payloadId)
                                && string.Equals(racePayloadId, payloadId, StringComparison.OrdinalIgnoreCase))
                                canonicalRaceCode = raceCode;
                            if (!string.IsNullOrWhiteSpace(raceCode) && !string.IsNullOrWhiteSpace(racePayloadId) && piercingPayloadIds.Contains(racePayloadId))
                                piercingRaceCodes.Add(raceCode);
                        }
                    }
                    variants.Add(new BodyVariantInfo(bodyCollection, bodyId, bodyName, CanonicalSlot(slot.Name), variantId, variantName, Path.GetFullPath(path), payloadId, raceCodes, targetOptionProfileId, canonicalRaceCode, sexes, supportSurface, piercingRaceCodes.Distinct(StringComparer.OrdinalIgnoreCase).ToArray()));
                }
            }
        }

        return new BodyLibraryInfo(collection, Path.GetFullPath(path), version, variants);
    }


    private static JsonDocument? ReadOptionalJson(ZipArchive archive, string name)
    {
        var entry = archive.GetEntry(name);
        if (entry is null) return null;
        using var stream = entry.Open();
        return JsonDocument.Parse(stream);
    }

    private static HashSet<string> ReadPiercingPayloadIds(JsonElement? payloadIndex)
    {
        var result = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (payloadIndex is null || payloadIndex.Value.ValueKind != JsonValueKind.Object) return result;
        foreach (var property in payloadIndex.Value.EnumerateObject())
        {
            var payload = property.Value;
            if (payload.ValueKind != JsonValueKind.Object) continue;
            var materialNames = TryGetProperty(payload, "materials", out var materials) && materials.ValueKind == JsonValueKind.Array
                ? materials.EnumerateArray().Select(material => material.ValueKind == JsonValueKind.String ? material.GetString() : null).ToArray()
                : Array.Empty<string?>();

            // RBODY materials are stored in mesh-record order, not by XIV material index.
            if (!TryGetProperty(payload, "mesh_records", out var records) || records.ValueKind != JsonValueKind.Array)
                continue;
            var recordArray = records.EnumerateArray().ToArray();
            for (var ordinal = 0; ordinal < recordArray.Length; ordinal++)
            {
                var record = recordArray[ordinal];
                var material = ResolveMeshMaterial(record, ordinal, materialNames, recordArray.Length);
                var materialBacked = IsPiercingMaterial(material);
                var attributeBacked = HasPiercingSubmeshAttribute(record);
                if (!materialBacked && !attributeBacked) continue;
                var vertices = TryGetProperty(record, "vertex_count", out var vertexCount) && vertexCount.TryGetInt32(out var v) ? v : 0;
                var indices = TryGetProperty(record, "index_count", out var indexCount) && indexCount.TryGetInt32(out var i) ? i : 0;
                if (vertices <= 0 || indices <= 0) continue;
                result.Add(property.Name);
                break;
            }
        }
        return result;
    }

    private static string? ResolveMeshMaterial(JsonElement record, int ordinal, IReadOnlyList<string?> materialNames, int recordCount)
    {
        // New/reprocessed libraries may carry the resolved material directly on the record.
        var direct = GetString(record, "material");
        if (!string.IsNullOrWhiteSpace(direct)) return direct;

        // Current RBODY catalogues store one resolved material per mesh record.
        if (materialNames.Count == recordCount && ordinal >= 0 && ordinal < materialNames.Count)
            return materialNames[ordinal];

        // Fall back to material-table indexing only for compatible legacy catalogues.
        if (TryGetProperty(record, "material_index", out var materialIndex) && materialIndex.TryGetInt32(out var index) && index >= 0 && index < materialNames.Count)
            return materialNames[index];
        return ordinal >= 0 && ordinal < materialNames.Count ? materialNames[ordinal] : null;
    }

    private static bool HasPiercingSubmeshAttribute(JsonElement record)
    {
        if (!TryGetProperty(record, "submeshes", out var submeshes) || submeshes.ValueKind != JsonValueKind.Array) return false;
        foreach (var submesh in submeshes.EnumerateArray())
        {
            var indexCount = TryGetProperty(submesh, "index_count", out var count) && count.TryGetInt32(out var value) ? value : 0;
            if (indexCount <= 0 || !TryGetProperty(submesh, "attributes", out var attributes) || attributes.ValueKind != JsonValueKind.Array) continue;
            foreach (var attribute in attributes.EnumerateArray())
            {
                if (attribute.ValueKind != JsonValueKind.String) continue;
                var name = attribute.GetString();
                if (IsPiercingMaterial(name)) return true;
            }
        }
        return false;
    }

    private static bool IsPiercingMaterial(string? material)
    {
        if (string.IsNullOrWhiteSpace(material)) return false;
        return material.Contains("pierc", StringComparison.OrdinalIgnoreCase)
            || material.Contains("jewel", StringComparison.OrdinalIgnoreCase)
            || material.Contains("dermal", StringComparison.OrdinalIgnoreCase)
            || material.Contains("barbell", StringComparison.OrdinalIgnoreCase)
            || material.Contains("bellyring", StringComparison.OrdinalIgnoreCase);
    }

    private static JsonDocument ReadJson(ZipArchive archive, string name)
    {
        var entry = archive.GetEntry(name) ?? throw new InvalidDataException($"RBODY is missing {name}.");
        using var stream = entry.Open();
        return JsonDocument.Parse(stream);
    }

    private static string CanonicalSlot(string slot)
        => BodySlots.All.FirstOrDefault(s => string.Equals(s, slot, StringComparison.OrdinalIgnoreCase)) ?? slot;

    private static bool TryGetProperty(JsonElement element, string name, out JsonElement value)
    {
        foreach (var property in element.EnumerateObject())
        {
            if (!string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase))
                continue;
            value = property.Value;
            return true;
        }
        value = default;
        return false;
    }

    private static string? GetString(JsonElement element, string name)
        => TryGetProperty(element, name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static int GetInt(JsonElement element, string name)
        => TryGetProperty(element, name, out var value) && value.TryGetInt32(out var result) ? result : 0;

    private static IReadOnlyCollection<string> GetStringArray(JsonElement element, string name)
    {
        if (!TryGetProperty(element, name, out var value) || value.ValueKind != JsonValueKind.Array)
            return Array.Empty<string>();
        return value.EnumerateArray()
            .Where(item => item.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(item.GetString()))
            .Select(item => item.GetString()!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }
}
