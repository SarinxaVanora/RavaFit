using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Dalamud.Interface.Textures;
using Dalamud.Plugin.Services;
using RavaFit.Core.Models;
using RavaFit.Core.Penumbra;

namespace RavaFit.Services;

internal sealed partial class ModelPreviewTextureService
{
    [GeneratedRegex(@"[a-z0-9_./\\-]{3,}?\.tex", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex TexturePathRegex();

    private readonly IDataManager _dataManager;
    private readonly ITextureProvider _textures;
    private readonly PenumbraService _penumbra;
    private readonly IPluginLog _log;
    private readonly Dictionary<string, string?> _materialGamePaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string?> _baseTexturePaths = new(StringComparer.OrdinalIgnoreCase);

    public ModelPreviewTextureService(IDataManager dataManager, ITextureProvider textures, PenumbraService penumbra, IPluginLog log)
    {
        _dataManager = dataManager;
        _textures = textures;
        _penumbra = penumbra;
        _log = log;
    }

    public ISharedImmediateTexture? GetBaseTexture(PenumbraV4Document? document, string? groupKey, string? optionKey, ModelRedirect model, string materialReference)
    {
        try
        {
            var context = $"{document?.MetaPath}|{groupKey}|{optionKey}|{model.GamePath}";
            var materialKey = $"{context}|{materialReference}";
            if (!_materialGamePaths.TryGetValue(materialKey, out var materialPath))
            {
                materialPath = ResolveMaterialGamePath(document, groupKey, optionKey, model.GamePath, materialReference);
                _materialGamePaths[materialKey] = materialPath;
            }
            if (string.IsNullOrWhiteSpace(materialPath)) return null;

            var textureKey = $"{context}|{materialPath}";
            if (!_baseTexturePaths.TryGetValue(textureKey, out var texturePath))
            {
                var raw = ReadResourceBytes(document, groupKey, optionKey, materialPath);
                texturePath = raw is null ? null : ResolveBaseTexturePath(materialPath, raw);
                _baseTexturePaths[textureKey] = texturePath;
            }
            if (string.IsNullOrWhiteSpace(texturePath)) return null;

            var mappedTexture = ResolveSelectedPhysical(document, groupKey, optionKey, texturePath);
            return !string.IsNullOrWhiteSpace(mappedTexture) && File.Exists(mappedTexture)
                ? _textures.GetFromFile(mappedTexture)
                : _textures.GetFromGame(texturePath);
        }
        catch (Exception ex)
        {
            _log.Debug(ex, "RavaFit preview could not resolve texture for {Material}", materialReference);
            return null;
        }
    }

    public void Clear()
    {
        _materialGamePaths.Clear();
        _baseTexturePaths.Clear();
    }

    private string? ResolveMaterialGamePath(PenumbraV4Document? document, string? groupKey, string? optionKey, string modelGamePath, string materialReference)
    {
        var material = Normalize(materialReference);
        if (!material.EndsWith(".mtrl", StringComparison.OrdinalIgnoreCase)) return null;
        if (material.StartsWith("chara/", StringComparison.OrdinalIgnoreCase)) return material;

        var file = Path.GetFileName(material.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(file)) return null;
        if (document is not null)
        {
            var exact = FindGamePathByFileName(GetSelectedOptionNode(document, groupKey, optionKey), file, ".mtrl")
                ?? FindGamePathByFileName(PenumbraV4Document.FindProperty(document.Root, "DefaultData"), file, ".mtrl")
                ?? FindGamePathByFileName(document.Root, file, ".mtrl");
            if (!string.IsNullOrWhiteSpace(exact)) return exact;
        }

        var model = Normalize(modelGamePath);
        var marker = model.IndexOf("/model/", StringComparison.OrdinalIgnoreCase);
        if (marker <= 0) return null;
        var basePath = model[..marker];
        var version = ExtractVersionFolder(material) ?? "v0001";
        return $"{basePath}/material/{version}/{file}";
    }

    private string? ResolveBaseTexturePath(string materialGamePath, byte[] raw)
    {
        var ascii = Encoding.ASCII.GetString(raw);
        var candidates = TexturePathRegex().Matches(ascii).Select(match => ResolveTextureCandidate(materialGamePath, match.Value))
            .Where(path => !string.IsNullOrWhiteSpace(path)).Cast<string>().Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        if (candidates.Length == 0) return null;
        return candidates.OrderBy(TexturePreference).ThenBy(path => path, StringComparer.OrdinalIgnoreCase).FirstOrDefault(path => ResourceExists(path))
            ?? candidates.OrderBy(TexturePreference).First();
    }

    private string? ResolveTextureCandidate(string materialGamePath, string raw)
    {
        var value = Normalize(raw);
        var chara = value.IndexOf("chara/", StringComparison.OrdinalIgnoreCase);
        if (chara >= 0) return value[chara..];
        if (value.StartsWith("common/", StringComparison.OrdinalIgnoreCase)) return value;

        var file = Path.GetFileName(value.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(file)) return null;
        var marker = materialGamePath.IndexOf("/material/", StringComparison.OrdinalIgnoreCase);
        if (marker <= 0) return null;
        var root = materialGamePath[..marker];
        var version = ExtractVersionFolder(materialGamePath) ?? "v0001";
        return $"{root}/texture/{version}/{file}";
    }

    private byte[]? ReadResourceBytes(PenumbraV4Document? document, string? groupKey, string? optionKey, string gamePath)
    {
        var selected = ResolveSelectedPhysical(document, groupKey, optionKey, gamePath);
        if (!string.IsNullOrWhiteSpace(selected) && File.Exists(selected)) return File.ReadAllBytes(selected);
        var resolved = _penumbra.ResolvePlayerPath(gamePath);
        if (Path.IsPathRooted(resolved) && File.Exists(resolved)) return File.ReadAllBytes(resolved);
        return _dataManager.GetFile(Normalize(resolved))?.Data ?? _dataManager.GetFile(gamePath)?.Data;
    }

    private string? ResolveSelectedPhysical(PenumbraV4Document? document, string? groupKey, string? optionKey, string gamePath)
    {
        if (document is null) return null;
        var relative = FindMappedRelative(GetSelectedOptionNode(document, groupKey, optionKey), gamePath)
            ?? FindMappedRelative(PenumbraV4Document.FindProperty(document.Root, "DefaultData"), gamePath);
        if (string.IsNullOrWhiteSpace(relative)) return null;
        return document.ResolvePhysicalPath(new ModelRedirect(gamePath, relative, false));
    }

    private static JsonNode? GetSelectedOptionNode(PenumbraV4Document document, string? groupKey, string? optionKey)
    {
        if (string.IsNullOrWhiteSpace(groupKey) || string.IsNullOrWhiteSpace(optionKey)) return null;
        try
        {
            var group = document.GetGroup(groupKey);
            return document.GetOption(group, optionKey).Node;
        }
        catch { return null; }
    }

    private static string? FindMappedRelative(JsonNode? node, string gamePath)
    {
        if (node is not JsonObject obj) return null;
        if (PenumbraV4Document.FindProperty(obj, "Files") is not JsonObject files) return null;
        foreach (var pair in files)
            if (string.Equals(Normalize(pair.Key), Normalize(gamePath), StringComparison.OrdinalIgnoreCase)
                && pair.Value is JsonValue value && value.TryGetValue<string>(out var relative) && !string.IsNullOrWhiteSpace(relative))
                return relative;
        return null;
    }

    private bool ResourceExists(string gamePath)
    {
        var resolved = _penumbra.ResolvePlayerPath(gamePath);
        return (Path.IsPathRooted(resolved) && File.Exists(resolved)) || _dataManager.FileExists(Normalize(resolved)) || _dataManager.FileExists(gamePath);
    }

    private static string? FindGamePathByFileName(JsonNode? node, string fileName, string extension)
    {
        if (node is JsonObject obj)
        {
            foreach (var pair in obj)
            {
                if (pair.Key.EndsWith(extension, StringComparison.OrdinalIgnoreCase)
                    && string.Equals(Path.GetFileName(pair.Key.Replace('/', Path.DirectorySeparatorChar)), fileName, StringComparison.OrdinalIgnoreCase))
                    return Normalize(pair.Key);
                var nested = FindGamePathByFileName(pair.Value, fileName, extension);
                if (!string.IsNullOrWhiteSpace(nested)) return nested;
            }
        }
        else if (node is JsonArray array)
        {
            foreach (var child in array)
            {
                var nested = FindGamePathByFileName(child, fileName, extension);
                if (!string.IsNullOrWhiteSpace(nested)) return nested;
            }
        }
        return null;
    }

    private static int TexturePreference(string path)
    {
        var file = Path.GetFileName(path).ToLowerInvariant();
        if (file.Contains("_base.tex") || file.Contains("_d.tex") || file.Contains("_c.tex")) return 0;
        if (file.Contains("diff") || file.Contains("color") || file.Contains("colour")) return 1;
        if (file.Contains("_n.tex") || file.Contains("normal")) return 8;
        if (file.Contains("_s.tex") || file.Contains("spec")) return 7;
        if (file.Contains("_m.tex") || file.Contains("mask")) return 6;
        return 3;
    }

    private static string? ExtractVersionFolder(string value)
    {
        var match = Regex.Match(value, @"(?:^|/)(v\d{4})(?:/|$)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        return match.Success ? match.Groups[1].Value.ToLowerInvariant() : null;
    }

    private static string Normalize(string value) => value.Trim().Trim('\0').Replace('\\', '/').TrimStart('/').ToLowerInvariant();
}
