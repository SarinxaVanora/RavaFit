using Dalamud.Game.Command;
using Dalamud.Interface.Windowing;
using Dalamud.IoC;
using Dalamud.Plugin;
using Dalamud.Plugin.Services;
using RavaFit.Services;
using RavaFit.Services.Animation;
using RavaFit.Windows;

namespace RavaFit;

public sealed class Plugin : IDalamudPlugin
{
    private const string CommandName = "/ravafit";

    [PluginService] internal static IDalamudPluginInterface PluginInterface { get; private set; } = null!;
    [PluginService] internal static ICommandManager CommandManager { get; private set; } = null!;
    [PluginService] internal static IPluginLog Log { get; private set; } = null!;
    [PluginService] internal static IFramework Framework { get; private set; } = null!;
    [PluginService] internal static IDataManager DataManager { get; private set; } = null!;
    [PluginService] internal static IObjectTable ObjectTable { get; private set; } = null!;
    [PluginService] internal static ISigScanner SigScanner { get; private set; } = null!;
    [PluginService] internal static ITextureProvider TextureProvider { get; private set; } = null!;

    internal Configuration Configuration { get; }
    internal WindowSystem WindowSystem { get; } = new("RavaFit");
    internal RavaFitAssetService Assets { get; }
    internal PenumbraService Penumbra { get; }
    internal BodyLibraryService Bodies { get; }
    internal SolverHostService Solver { get; }
    internal ModelBridgeService ModelBridge { get; }
    internal ConversionService Conversion { get; }
    internal CustomiseModService Customise { get; }
    internal ModCleanupService Cleanup { get; }
    internal VanillaAssetService VanillaAssets { get; }
    internal ModelPreviewTextureService PreviewTextures { get; }
    internal CharacterRaceService CharacterRace { get; }
    internal AnimationSkeletonService AnimationSkeletons { get; }
    internal AnimationPortService AnimationPort { get; }

    private MainWindow MainWindow { get; }

    public Plugin()
    {
        Configuration = PluginInterface.GetPluginConfig() as Configuration ?? new Configuration();
        var pluginRoot = PluginInterface.AssemblyLocation.DirectoryName
            ?? throw new InvalidOperationException("Dalamud did not provide the RavaFit plugin assembly directory.");
        var localRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "RavaFit");
        var legacyUserBodies = Path.Combine(localRoot, "UserBodies");
        var userBodies = PluginInterface.ConfigDirectory.FullName;

        // Runtime and body assets are installed separately from GitHub.
        Assets = new RavaFitAssetService(localRoot, Log);
        Assets.ApplyInstalledState(Configuration);
        Assets.ActivationReady += OnAssetActivationReady;
        Configuration.Version = Math.Max(Configuration.Version, 3);

        // Keep developer runtime overrides outside the live plugin directory.
        if (!string.IsNullOrWhiteSpace(Configuration.DeveloperPythonPath) && IsWithinDirectory(Configuration.DeveloperPythonPath, pluginRoot))
        {
            Log.Warning("RavaFit cleared DeveloperPythonPath because it points inside the live plugin directory: {Path}", Configuration.DeveloperPythonPath);
            Configuration.DeveloperPythonPath = string.Empty;
        }

        TryMigrateLegacyUserBodies(legacyUserBodies, userBodies);

        Solver = new SolverHostService(Configuration, Log);
        Penumbra = new PenumbraService(PluginInterface, Framework, Log);
        Bodies = new BodyLibraryService(Configuration.BodyLibraryDirectory, userBodies, Solver, Log);
        ModelBridge = new ModelBridgeService(Framework, DataManager, Penumbra, Log);
        Conversion = new ConversionService(Penumbra, DataManager, Solver, ModelBridge, Bodies, Log);
        Customise = new CustomiseModService(Penumbra, DataManager, Solver, ModelBridge, Bodies, Conversion, Log);
        Cleanup = new ModCleanupService(Penumbra, Framework, Log);
        VanillaAssets = new VanillaAssetService(DataManager, Penumbra);
        PreviewTextures = new ModelPreviewTextureService(DataManager, TextureProvider, Penumbra, Log);
        CharacterRace = new CharacterRaceService(ObjectTable);
        AnimationSkeletons = new AnimationSkeletonService(Framework, DataManager, ObjectTable, CharacterRace, Log);
        var animationRetargeter = new AnimationPapRetargeter(Framework, Log, SigScanner);
        AnimationPort = new AnimationPortService(Penumbra, AnimationSkeletons, animationRetargeter, Log);

        Bodies.Refresh();
        Penumbra.Refresh();
        _ = Solver.ProbeAsync();
        _ = ModelBridge.ProbeAsync();
        _ = Assets.InitialiseAsync();

        MainWindow = new MainWindow(this);
        WindowSystem.AddWindow(MainWindow);

        CommandManager.AddHandler(CommandName, new CommandInfo((_, _) => MainWindow.Toggle())
        {
            HelpMessage = "Open RavaFit.",
        });
        PluginInterface.UiBuilder.Draw += WindowSystem.Draw;
        PluginInterface.UiBuilder.OpenMainUi += ToggleMainUi;
        PluginInterface.UiBuilder.OpenConfigUi += ToggleMainUi;
        PluginInterface.SavePluginConfig(Configuration);
    }

    private void OnAssetActivationReady(RavaFitAssetActivation activation)
    {
        _ = Framework.RunOnFrameworkThread(() =>
        {
            ApplyAssetActivation(activation);
            return true;
        });
    }

    private void ApplyAssetActivation(RavaFitAssetActivation activation)
    {
        Configuration.RuntimeDirectory = activation.RuntimeDirectory;
        Configuration.RuntimeProductionRevision = activation.RuntimeProductionRevision;
        Configuration.BodyLibraryDirectory = activation.BodyLibraryDirectory;
        Bodies.SetDirectory(activation.BodyLibraryDirectory);
        PluginInterface.SavePluginConfig(Configuration);
        _ = Solver.RestartAsync();
        Log.Information("RavaFit activated GitHub assets. Runtime={RuntimeVersion}, Bodies={BodiesVersion}", activation.RuntimeVersion, activation.BodiesVersion);
    }

    public void Dispose()
    {
        PluginInterface.UiBuilder.Draw -= WindowSystem.Draw;
        PluginInterface.UiBuilder.OpenMainUi -= ToggleMainUi;
        PluginInterface.UiBuilder.OpenConfigUi -= ToggleMainUi;
        CommandManager.RemoveHandler(CommandName);
        Assets.ActivationReady -= OnAssetActivationReady;
        Solver.Dispose();
        Assets.Dispose();
        WindowSystem.RemoveAllWindows();
        MainWindow.Dispose();
        ModelBridge.Dispose();
        Penumbra.Dispose();
    }

    private static void TryMigrateLegacyUserBodies(string legacyDirectory, string configDirectory)
    {
        try
        {
            var legacyPath = Path.Combine(legacyDirectory, "UserBodies.rbody");
            var configPath = Path.Combine(configDirectory, "UserBodies.rbody");
            if (!File.Exists(legacyPath) || File.Exists(configPath))
                return;

            Directory.CreateDirectory(configDirectory);
            File.Copy(legacyPath, configPath, overwrite: false);
            Log.Information("RavaFit migrated the custom body catalogue into Dalamud PluginConfigs: {Path}", configPath);
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "RavaFit could not migrate the legacy custom body catalogue into Dalamud PluginConfigs. The old catalogue was left untouched.");
        }
    }

    private static bool IsWithinDirectory(string candidate, string directory)
    {
        try
        {
            var root = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            var path = Path.GetFullPath(candidate);
            return path.StartsWith(root, StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }

    internal bool AssetUpdateSafe
        => !Conversion.Busy && !Customise.Busy && !Cleanup.Busy && !AnimationPort.Busy;

    internal void SaveConfiguration() => PluginInterface.SavePluginConfig(Configuration);
    internal void ToggleMainUi() => MainWindow.Toggle();
}
