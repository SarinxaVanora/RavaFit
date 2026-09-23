using RavaFit.Core.Models;

namespace RavaFit.Core.Penumbra;

public sealed class GeneratedModelStore
{
    public Task<GeneratedModelFile> WriteAsync(string modRoot, string gamePath, string optionName, ReadOnlyMemory<byte> mdlBytes, CancellationToken cancellationToken = default)
    {
        if (!gamePath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"Invalid model game path '{gamePath}'.");
        return WriteResourceAsync(modRoot, gamePath, optionName, mdlBytes, cancellationToken);
    }

    public async Task<GeneratedModelFile> WriteResourceAsync(string modRoot, string gamePath, string optionName, ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken = default)
    {
        if (bytes.IsEmpty)
            throw new InvalidDataException("Generated resource output is empty.");

        var normalizedGamePath = PenumbraV4Writer.NormalizeGamePath(gamePath);
        var extension = Path.GetExtension(normalizedGamePath);
        if (!string.Equals(extension, ".mdl", StringComparison.OrdinalIgnoreCase) && !string.Equals(extension, ".mtrl", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"RavaFit generated resources only support MDL/MTRL paths. '{gamePath}' is not supported.");

        var safeName = MakeSafePathSegment(optionName);
        var id = Guid.NewGuid().ToString("N")[..12];
        var fileName = Path.GetFileName(normalizedGamePath.Replace('/', Path.DirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(fileName))
            throw new InvalidDataException($"Invalid generated resource game path '{gamePath}'.");

        var relative = Path.Combine("RavaFit", safeName, id, fileName).Replace('\\', '/');
        var absolute = Path.GetFullPath(Path.Combine(modRoot, relative.Replace('/', Path.DirectorySeparatorChar)));
        var parent = Path.GetDirectoryName(absolute) ?? throw new InvalidOperationException("Generated resource has no parent directory.");
        Directory.CreateDirectory(parent);

        var temp = absolute + ".tmp";
        await File.WriteAllBytesAsync(temp, bytes.ToArray(), cancellationToken).ConfigureAwait(false);
        File.Move(temp, absolute, true);
        return new GeneratedModelFile(normalizedGamePath, relative, absolute);
    }

    private static string MakeSafePathSegment(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        var result = new string(value.Trim().Select(c => invalid.Contains(c) ? '_' : c).ToArray());
        result = result.Trim().Trim('.');
        return string.IsNullOrWhiteSpace(result) ? "Converted" : result;
    }
}
