namespace RavaFit.Services;

internal static class RavaFitDistribution
{
    internal const string AssetManifestUrl = "https://raw.githubusercontent.com/SarinxaVanora/RavaFit/master/distribution/ravafit-assets.json";

    internal static bool IsConfigured
        => Uri.TryCreate(AssetManifestUrl, UriKind.Absolute, out var uri)
        && string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase);
}
