using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using RavaFit.Core.Models;

namespace RavaFit.Core;

public static class TargetBodyOptionReader
{
    public static TargetBodyOptionProfileInfo? ReadProfile(BodyVariantInfo variant)
    {
        if (string.IsNullOrWhiteSpace(variant.TargetOptionProfileId))
            return null;
        var profiles = ReadProfiles(variant.RBodyPath);
        return profiles.TryGetValue(variant.TargetOptionProfileId, out var profile) ? profile : null;
    }

    public static IReadOnlyDictionary<string, TargetBodyOptionProfileInfo> ReadProfiles(string rbodyPath)
    {
        using var archive = ZipFile.OpenRead(rbodyPath);
        var entry = archive.GetEntry("target_options.json");
        if (entry is null)
            return new Dictionary<string, TargetBodyOptionProfileInfo>(StringComparer.OrdinalIgnoreCase);
        using var stream = entry.Open();
        var root = JsonNode.Parse(stream)?.AsObject() ?? throw new InvalidDataException("RBODY target_options.json is not a JSON object.");
        var profilesNode = root["profiles"] as JsonObject;
        if (profilesNode is null)
            return new Dictionary<string, TargetBodyOptionProfileInfo>(StringComparer.OrdinalIgnoreCase);

        var output = new Dictionary<string, TargetBodyOptionProfileInfo>(StringComparer.OrdinalIgnoreCase);
        foreach (var pair in profilesNode)
        {
            if (pair.Value is not JsonObject profile) continue;
            var profileId = profile["id"]?.GetValue<string>() ?? pair.Key;
            var groups = new List<TargetBodyOptionGroupInfo>();
            if (profile["groups"] is JsonArray groupArray)
            {
                foreach (var node in groupArray.OfType<JsonObject>())
                {
                    var group = node["group"]?.DeepClone().AsObject() ?? throw new InvalidDataException($"RBODY target option profile '{profileId}' contains a group without JSON data.");
                    var slots = (node["slots"] as JsonArray)?.Select(x => x?.GetValue<string>()).Where(x => !string.IsNullOrWhiteSpace(x)).Cast<string>().ToArray() ?? [];
                    var raceCodes = (node["race_codes"] as JsonArray)?.Select(x => x?.GetValue<string>()).Where(x => !string.IsNullOrWhiteSpace(x)).Cast<string>().ToArray() ?? [];
                    var fileAssets = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                    if (node["file_assets"] is JsonObject fileAssetNode)
                        foreach (var asset in fileAssetNode)
                            if (asset.Value is not null)
                                fileAssets[asset.Key.Replace('\\', '/')] = asset.Value.GetValue<string>();
                    groups.Add(new TargetBodyOptionGroupInfo(
                        node["source_package"]?.GetValue<string>() ?? string.Empty,
                        node["source_group_file"]?.GetValue<string>() ?? string.Empty,
                        node["kind"]?.GetValue<string>() ?? "control",
                        slots,
                        raceCodes,
                        group,
                        fileAssets));
                }
            }

            output[profileId] = new TargetBodyOptionProfileInfo(
                profileId,
                profile["body_id"]?.GetValue<string>() ?? string.Empty,
                profile["body_name"]?.GetValue<string>() ?? string.Empty,
                profile["collection"]?.GetValue<string>() ?? string.Empty,
                Path.GetFullPath(rbodyPath),
                groups);
        }
        return output;
    }

    public static async Task ExtractAssetAsync(string rbodyPath, string assetId, string destination, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(assetId) || assetId.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || assetId.Contains('/') || assetId.Contains('\\'))
            throw new InvalidDataException($"Invalid RBODY target-option asset id '{assetId}'.");
        using var archive = ZipFile.OpenRead(rbodyPath);
        var entry = archive.GetEntry($"target_option_assets/{assetId}") ?? throw new InvalidDataException($"RBODY is missing target-option asset '{assetId}'.");
        Directory.CreateDirectory(Path.GetDirectoryName(destination) ?? throw new InvalidOperationException("Target option asset destination has no parent directory."));
        var temp = destination + $".ravafit-{Guid.NewGuid():N}.tmp";
        try
        {
            await using (var source = entry.Open())
            await using (var target = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.Asynchronous | FileOptions.WriteThrough))
            {
                await source.CopyToAsync(target, cancellationToken).ConfigureAwait(false);
                await target.FlushAsync(cancellationToken).ConfigureAwait(false);
                target.Flush(true);
            }
            var expectedSha = Path.GetFileNameWithoutExtension(assetId);
            string actualSha;
            await using (var tempRead = File.OpenRead(temp))
                actualSha = Convert.ToHexString(await SHA256.HashDataAsync(tempRead, cancellationToken).ConfigureAwait(false)).ToLowerInvariant();
            if (!string.Equals(expectedSha, actualSha, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"RBODY target-option asset '{assetId}' failed SHA-256 verification.");
            if (File.Exists(destination))
            {
                string existingSha;
                await using (var existingRead = File.OpenRead(destination))
                    existingSha = Convert.ToHexString(await SHA256.HashDataAsync(existingRead, cancellationToken).ConfigureAwait(false)).ToLowerInvariant();
                if (!string.Equals(existingSha, actualSha, StringComparison.OrdinalIgnoreCase))
                    throw new IOException($"RavaFit target-option asset path already exists with different bytes: {destination}");
                File.Delete(temp);
                return;
            }
            File.Move(temp, destination);
        }
        finally
        {
            if (File.Exists(temp)) File.Delete(temp);
        }
    }
}
