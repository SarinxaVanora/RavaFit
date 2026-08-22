using System.Collections.Concurrent;
using System.Numerics;
using System.Text.Json.Nodes;
using Dalamud.Bindings.ImGui;
using Dalamud.Interface.Utility;
using Dalamud.Interface.Textures.TextureWraps;
using Dalamud.Interface.Utility.Raii;
using Dalamud.Interface.Windowing;
using RavaFit.Core.Models;
using RavaFit.Core.Penumbra;
using RavaFit.Services;
using RavaFit.Services.Animation;

namespace RavaFit.Windows;

internal sealed class MainWindow : Window, IDisposable
{
    private enum MainTab
    {
        Convert,
        Customise,
        RaceSwap,
        AnimationPort,
        BodyCatalogue,
    }

    private enum CustomiseSubTab
    {
        Piercings,
        Visibility,
        SplitAccessory,
        RemoveOptions,
    }

    private enum AnimationSubTab
    {
        Port,
        Cleanup,
    }

    private enum VisibilityCreationMode
    {
        Separate,
        Combined,
    }

    private enum PreviewDisplayMode
    {
        Textured,
        Shaded,
        Flat,
        Wireframe,
    }

    private sealed record OutfitModelChoice(V4GroupInfo Group, V4OptionInfo Option, ModelRedirect Model, string Label)
    {
        public string Identity => $"{Group.StableKey}|{Option.StableKey}|{Model.GamePath}";
    }

    private sealed record BodyImportModelChoice(ModelRedirect Model, string Slot, string RaceCode, string PhysicalPath);
    private sealed record BodyImportChoice(V4GroupInfo Group, V4OptionInfo Option, IReadOnlyList<BodyImportModelChoice> Models, string Label)
    {
        public string Identity => $"{Group.StableKey}|{Option.StableKey}";
    }

    private sealed class OutfitRowState
    {
        public OutfitRowState(string slot) => Slot = slot;

        public string Slot { get; }
        public bool Enabled { get; set; }
        public OutfitModelChoice? Selection { get; set; }
        public BodyVariantInfo? Source { get; set; }
        public BodyVariantInfo? Target { get; set; }
        public GarmentCoverageResult? Analysis { get; set; }
        public bool Analysing { get; set; }
        public bool SourceUserOverride { get; set; }
        public bool SourceInferred { get; set; }
        public int SourceAutoPriority { get; set; }
        public int Revision { get; set; }
        public CancellationTokenSource? Cancellation { get; set; }
        public string Status { get; set; } = string.Empty;
    }

    private sealed class AccessoryOutfitRowState
    {
        public AccessoryOutfitRowState(string slot)
        {
            Slot = slot;
            Sources = BodySlots.All.ToDictionary(bodySlot => bodySlot, _ => (BodyVariantInfo?)null, StringComparer.OrdinalIgnoreCase);
            Targets = BodySlots.All.ToDictionary(bodySlot => bodySlot, _ => (BodyVariantInfo?)null, StringComparer.OrdinalIgnoreCase);
        }

        public string Slot { get; }
        public bool Enabled { get; set; }
        public OutfitModelChoice? Selection { get; set; }
        public GarmentCoverageResult? Analysis { get; set; }
        public bool Analysing { get; set; }
        public int Revision { get; set; }
        public CancellationTokenSource? Cancellation { get; set; }
        public string Status { get; set; } = string.Empty;
        public Dictionary<string, BodyVariantInfo?> Sources { get; }
        public Dictionary<string, BodyVariantInfo?> Targets { get; }
        public HashSet<string> SourceUserOverrides { get; } = new(StringComparer.OrdinalIgnoreCase);
    }

    private static readonly Vector4 SuccessColour = new(0.32f, 0.88f, 0.56f, 1f);
    private static readonly Vector4 WarningColour = new(0.96f, 0.67f, 0.30f, 1f);
    private static readonly Vector4 ErrorColour = new(1.00f, 0.38f, 0.38f, 1f);

    private readonly Plugin _plugin;
    private PenumbraModInfo? _selectedMod;
    private PenumbraV4Document? _document;
    private MainTab _currentTab = MainTab.Convert;
    private string _modFilter = string.Empty;
    private string _bodyFilter = string.Empty;
    private string _bodyFilterOwner = string.Empty;
    private string _uiMessage = string.Empty;
    private bool _uiMessageError;
    private readonly ConcurrentQueue<Action> _uiActions = new();
    private readonly Dictionary<string, OutfitRowState> _outfitRows = BodySlots.All.ToDictionary(slot => slot, slot => new OutfitRowState(slot), StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, OutfitRowState> _swapRows = BodySlots.All.ToDictionary(slot => slot, slot => new OutfitRowState(slot), StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, AccessoryOutfitRowState> _accessoryRows = AccessoryModelSlots.All.ToDictionary(slot => slot, slot => new AccessoryOutfitRowState(slot), StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, AccessoryOutfitRowState> _swapAccessoryRows = AccessoryModelSlots.All.ToDictionary(slot => slot, slot => new AccessoryOutfitRowState(slot), StringComparer.OrdinalIgnoreCase);
    private string? _swapTargetRaceOverride;
    private string? _animationSourceRaceOverride;
    private string? _animationTargetRaceOverride;
    private string _animationSkeletonChoiceId = AnimationSkeletonService.StandardChoiceId;
    private readonly Dictionary<string, List<OutfitModelChoice>> _outfitChoices = BodySlots.All.ToDictionary(slot => slot, _ => new List<OutfitModelChoice>(), StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<OutfitModelChoice>> _accessoryChoices = AccessoryModelSlots.All.ToDictionary(slot => slot, _ => new List<OutfitModelChoice>(), StringComparer.OrdinalIgnoreCase);
    private string _selectionFilter = string.Empty;
    private PenumbraModInfo? _bodyImportMod;
    private PenumbraV4Document? _bodyImportDocument;
    private readonly List<BodyImportChoice> _bodyImportChoices = [];
    private BodyImportChoice? _bodyImportChoice;
    private string _bodyImportModFilter = string.Empty;
    private string _bodyImportChoiceFilter = string.Empty;
    private OutfitModelChoice? _customiseSelection;
    private BodyVariantInfo? _customisePiercingBody;
    private IReadOnlyList<BodyVariantInfo> _customisePiercingBodies = [];
    private bool _customisePiercingBodiesLoading;
    private string _customisePiercingBodiesKey = string.Empty;
    private IReadOnlyList<CustomiseModelPart> _customiseParts = [];
    private IReadOnlyList<CustomisePreviewTriangle> _customisePreviewTriangles = [];
    private IReadOnlyList<CustomisePreviewEdge> _customisePreviewEdges = [];
    private Vector3 _customisePreviewMin = Vector3.Zero;
    private Vector3 _customisePreviewMax = Vector3.One;
    private readonly HashSet<int> _customiseSelectedParts = [];
    private readonly Dictionary<int, string> _customisePartToggleNames = [];
    private readonly Dictionary<int, int> _customisePartColourSlots = [];
    private int _customiseNextColourSlot;
    private int _customiseHoveredPart = -1;
    private float _customisePreviewYaw = 0.35f;
    private float _customisePreviewPitch = -0.08f;
    private float _customisePreviewZoom = 1f;
    private Vector2 _customisePreviewPan = Vector2.Zero;
    private Vector3? _customisePreviewFocusCentre;
    private float? _customisePreviewFocusRadius;
    private bool _customisePreviewDimOthers = true;
    private bool _customisePreviewOnlySelected;
    private PreviewDisplayMode _customisePreviewMode = PreviewDisplayMode.Shaded;
    private string _customiseFilter = string.Empty;
    private string _customisePiercingFilter = string.Empty;
    private string _customiseCombinedToggleName = string.Empty;
    private VanillaAccessoryAsset? _customiseAccessoryTarget;
    private IReadOnlyList<VanillaAccessoryAsset> _customiseAccessoryAssets = [];
    private string _customiseAccessoryFilter = string.Empty;
    private string _customiseAccessoryOptionName = string.Empty;
    private CustomiseSubTab _customiseSubTab = CustomiseSubTab.Piercings;
    private AnimationSubTab _animationSubTab = AnimationSubTab.Port;
    private VisibilityCreationMode _customiseVisibilityMode = VisibilityCreationMode.Separate;
    private IReadOnlyList<ModOptionCleanupCandidate> _cleanupCandidates = [];
    private readonly HashSet<string> _customiseCleanupSelected = [];
    private readonly HashSet<string> _animationCleanupSelected = [];
    private string _customiseCleanupFilter = string.Empty;
    private string _animationCleanupFilter = string.Empty;
    private bool _convertUseVanilla;
    private VanillaOutfitAsset? _vanillaOutfit;
    private IReadOnlyList<VanillaOutfitAsset> _vanillaOutfitAssets = [];
    private bool _vanillaOutfitLoading;
    private bool _vanillaOutfitLoadStarted;
    private string _vanillaOutfitFilter = string.Empty;
    private string? _vanillaOutfitRaceOverride;
    private bool _convertVanillaReady;
    private string _vanillaSourceDisplay = "Vanilla body (from XIV)";
    private VanillaAnimationAsset? _vanillaAnimation;
    private IReadOnlyList<VanillaAnimationAsset> _vanillaAnimationAssets = [];
    private string _vanillaAnimationFilter = string.Empty;
    private bool _vanillaAnimationLoading;
    private bool _vanillaAnimationLoadStarted;
    private bool _animationUseVanilla;
    private bool _customiseInspecting;
    private bool _customiseInspectionComplete;
    private string _bodyImportBodyName = string.Empty;
    private string _bodyImportVariantName = string.Empty;
    private bool _bodyImportBusy;
    private bool _resetScrollNextFrame = true;
    private bool _resetScrollThisFrame;
    private double _conversionBusyStartedAt = -1d;

    public MainWindow(Plugin plugin)
        : base("RavaFit##MainWindow")
    {
        _plugin = plugin;
        SizeConstraints = new WindowSizeConstraints
        {
            MinimumSize = new Vector2(720, 640),
            MaximumSize = new Vector2(float.MaxValue, float.MaxValue),
        };
    }

    public void Dispose()
    {
        foreach (var row in _outfitRows.Values.Concat(_swapRows.Values))
        {
            row.Cancellation?.Cancel(); row.Cancellation?.Dispose(); row.Cancellation = null;
        }
        foreach (var row in _accessoryRows.Values.Concat(_swapAccessoryRows.Values))
        {
            row.Cancellation?.Cancel(); row.Cancellation?.Dispose(); row.Cancellation = null;
        }
    }

    public override void Draw()
    {
        while (_uiActions.TryDequeue(out var uiAction))
            uiAction();

        ReconcileSelectedPenumbraMod();
        _resetScrollThisFrame = _resetScrollNextFrame;
        _resetScrollNextFrame = false;
        if (_plugin.Conversion.Busy)
        {
            if (_conversionBusyStartedAt < 0d) _conversionBusyStartedAt = ImGui.GetTime();
        }
        else
        {
            _conversionBusyStartedAt = -1d;
        }

        using var chrome = RavaFitUiChrome.BeginScope();
        DrawHeader();
        DrawNavigation();
        DrawAssetPanel();
        if (_plugin.Assets.GetStatus().Busy)
            return;

        if (ImGui.BeginChild("##RavaFitMainContent", new Vector2(0f, 0f), false, ImGuiWindowFlags.None))
        {
            if (_resetScrollThisFrame) ImGui.SetScrollY(0f);
            switch (_currentTab)
            {
                case MainTab.Convert:
                    DrawConvertTab();
                    break;
                case MainTab.Customise:
                    DrawCustomiseTab();
                    break;
                case MainTab.RaceSwap:
                    DrawRaceSwapTab();
                    break;
                case MainTab.AnimationPort:
                    DrawAnimationPortTab();
                    break;
                case MainTab.BodyCatalogue:
                    DrawBodyCatalogueTab();
                    break;
            }
        }
        ImGui.EndChild();
    }

    private void RequestScrollTop() => _resetScrollNextFrame = true;

    private void ActivateMainTab(MainTab tab)
    {
        if (_currentTab == tab) return;
        _currentTab = tab;
        RequestScrollTop();
    }

    private void ActivateCustomiseSubTab(CustomiseSubTab tab)
    {
        if (_customiseSubTab == tab) return;
        _customiseSubTab = tab;
        RequestScrollTop();
        if (tab is CustomiseSubTab.Visibility or CustomiseSubTab.SplitAccessory) EnsureCustomiseInspection();
    }

    private void ActivateAnimationSubTab(AnimationSubTab tab)
    {
        if (_animationSubTab == tab) return;
        _animationSubTab = tab;
        RequestScrollTop();
    }

    private void DrawHeader()
    {
        var (status, colour) = GetOverallStatus();
        RavaFitUiChrome.DrawHero(
            "RavaFit",
            _currentTab switch
            {
                MainTab.AnimationPort => "Port animations",
                MainTab.Customise => "Edit a mod",
                MainTab.BodyCatalogue => "Add a body",
                _ => "Port outfits",
            },
            status,
            colour);
    }

    private (string Status, Vector4 Colour) GetOverallStatus()
    {
        var assets = _plugin.Assets.GetStatus();
        if (assets.Busy)
            return ("Preparing assets", WarningColour);
        if (!assets.Ready)
            return ("Setup required", WarningColour);
        if (_currentTab == MainTab.AnimationPort)
            return _plugin.Penumbra.Available ? ("Ready", SuccessColour) : ("Penumbra unavailable", ErrorColour);
        if (_currentTab == MainTab.BodyCatalogue)
        {
            if (!_plugin.Penumbra.Available)
                return ("Penumbra unavailable", ErrorColour);
            if (!_plugin.Solver.Ready)
                return ("Setup required", WarningColour);
            return ("Ready", SuccessColour);
        }
        if (!_plugin.Penumbra.Available || !_plugin.Penumbra.ModsEnabled)
            return ("Penumbra unavailable", ErrorColour);
        if (!_plugin.Bodies.Ready || !_plugin.Solver.Ready || !_plugin.Solver.ConversionReady)
            return ("Setup required", WarningColour);
        if (!_plugin.ModelBridge.Status.Available)
            return ("Bridge unavailable", ErrorColour);
        return ("Ready", SuccessColour);
    }

    private void DrawAssetPanel()
    {
        var assets = _plugin.Assets.GetStatus();
        if (assets.Ready && !assets.Busy && !assets.UpdateAvailable && string.IsNullOrWhiteSpace(assets.LastError))
            return;

        using (RavaFitUiChrome.BeginCard("##RavaFitAssetSetupCard", 170f * ImGuiHelpers.GlobalScale, allowScroll: false))
        {
            ImGui.TextUnformatted(assets.Stage);
            if (!string.IsNullOrWhiteSpace(assets.Detail))
                RavaFitUiChrome.DrawMutedWrappedText(assets.Detail);

            if (assets.Busy)
            {
                ImGuiHelpers.ScaledDummy(6f);
                ImGui.ProgressBar((float)Math.Clamp(assets.Progress, 0d, 1d), new Vector2(-1f, 18f * ImGuiHelpers.GlobalScale));
            }

            if (!string.IsNullOrWhiteSpace(assets.RuntimeVersion) || !string.IsNullOrWhiteSpace(assets.BodiesVersion))
            {
                ImGuiHelpers.ScaledDummy(5f);
                if (!string.IsNullOrWhiteSpace(assets.RuntimeVersion)) RavaFitUiChrome.DrawMutedText($"Runtime: {assets.RuntimeVersion}");
                if (!string.IsNullOrWhiteSpace(assets.BodiesVersion)) RavaFitUiChrome.DrawMutedText($"Bodies: {assets.BodiesVersion}");
            }

            if (!assets.Busy)
            {
                ImGuiHelpers.ScaledDummy(7f);
                var label = !assets.Ready ? "Install RavaFit assets" : assets.UpdateAvailable ? "Update RavaFit assets" : "Retry asset check";
                using (ImRaii.Disabled(!_plugin.AssetUpdateSafe || !RavaFitDistribution.IsConfigured))
                {
                    if (ImGui.Button(label, new Vector2(-1f, 32f * ImGuiHelpers.GlobalScale)))
                    {
                        if (assets.UpdateAvailable || !assets.Ready)
                            _ = _plugin.Assets.InstallAvailableAsync();
                        else
                            _ = _plugin.Assets.RefreshManifestAsync();
                    }
                }
                if (!_plugin.AssetUpdateSafe)
                    RavaFitUiChrome.DrawMutedWrappedText("Finish the current RavaFit operation before changing runtime assets.");
                else if (!RavaFitDistribution.IsConfigured)
                    RavaFitUiChrome.DrawMutedWrappedText("Set the GitHub manifest URL in RavaFitDistribution.cs, then rebuild the plugin.");
            }
        }
        ImGuiHelpers.ScaledDummy(7f);
    }

    private static float GetWorkCardHeight(float minimum = 220f)
    {
        var minimumHeight = minimum * ImGuiHelpers.GlobalScale;
        return MathF.Max(minimumHeight, ImGui.GetContentRegionAvail().Y);
    }

    private float GetConvertSourceCardHeight()
    {
        var scale = ImGuiHelpers.GlobalScale;
        if (!_plugin.Penumbra.Available) return 138f * scale;
        if (!_convertUseVanilla) return 112f * scale;

        return 196f * scale;
    }

    private static float GetFieldColumnWidth()
    {
        var scale = ImGuiHelpers.GlobalScale;
        var available = MathF.Max(1f, ImGui.GetContentRegionAvail().X);
        return Math.Clamp(available * 0.28f, 78f * scale, 116f * scale);
    }

    private void DrawNavigation()
    {
        DrawResponsiveSegmentTabs([
            ("Convert", _currentTab == MainTab.Convert, (Action)(() => ActivateMainTab(MainTab.Convert))),
            ("Customise mod", _currentTab == MainTab.Customise, (Action)(() => ActivateMainTab(MainTab.Customise))),
            ("Gender / Race Swap", _currentTab == MainTab.RaceSwap, (Action)(() => ActivateMainTab(MainTab.RaceSwap))),
            ("Animation Porting", _currentTab == MainTab.AnimationPort, (Action)(() => ActivateMainTab(MainTab.AnimationPort))),
            ("Import Body", _currentTab == MainTab.BodyCatalogue, (Action)(() => ActivateMainTab(MainTab.BodyCatalogue))),
        ], 104f);
        ImGuiHelpers.ScaledDummy(7f);
    }

    private static void DrawResponsiveSegmentTabs(IReadOnlyList<(string Label, bool Selected, Action Activate)> tabs, float minimumButtonWidth)
    {
        if (tabs.Count == 0) return;
        var scale = ImGuiHelpers.GlobalScale;
        var spacing = 4f * scale;
        var available = MathF.Max(1f, ImGui.GetContentRegionAvail().X);
        var desired = MathF.Max(minimumButtonWidth * scale, tabs.Max(tab => ImGui.CalcTextSize(tab.Label).X + (28f * scale)));
        var columns = Math.Clamp((int)MathF.Floor((available + spacing) / (desired + spacing)), 1, tabs.Count);
        var width = MathF.Max(1f, (available - (spacing * (columns - 1))) / columns);
        for (var i = 0; i < tabs.Count; i++)
        {
            var tab = tabs[i];
            if (RavaFitUiChrome.DrawSegmentTab(tab.Label, tab.Selected, width)) tab.Activate();
            if ((i + 1) % columns != 0 && i + 1 < tabs.Count) ImGui.SameLine(0f, spacing);
        }
    }

    private void DrawConvertTab()
    {
        RavaFitUiChrome.DrawSectionTitle("Outfit", "Choose a mod or vanilla item");
        using (RavaFitUiChrome.BeginCard("##RavaFitSourceCard", GetConvertSourceCardHeight(), allowScroll: false))
        {
            if (!_plugin.Penumbra.Available)
            {
                ImGui.TextUnformatted("Penumbra is not available.");
                RavaFitUiChrome.DrawMutedWrappedText(_plugin.Penumbra.LastError);
                ImGuiHelpers.ScaledDummy(5f);
                if (ImGui.Button("Refresh Penumbra")) _plugin.Penumbra.Refresh();
            }
            else
            {
                var vanilla = _convertUseVanilla;
                if (ImGui.RadioButton("Mod", !vanilla)) { _convertUseVanilla = false; _convertVanillaReady = false; RequestScrollTop(); }
                ImGui.SameLine();
                if (ImGui.RadioButton("Vanilla", vanilla)) { _convertUseVanilla = true; _convertVanillaReady = false; RequestScrollTop(); StartLoadVanillaOutfitCatalogue(); }
                ImGuiHelpers.ScaledDummy(6f);
                if (_convertUseVanilla)
                {
                    if (ImGui.BeginTable("##RavaFitVanillaOutfitFields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
                    {
                        ImGui.TableSetupColumn("##VanillaOutfitLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                        ImGui.TableSetupColumn("##VanillaOutfitValue", ImGuiTableColumnFlags.WidthStretch, 1f);
                        DrawFieldRow("Outfit", DrawVanillaOutfitPicker);
                        DrawFieldRow("Character", DrawVanillaOutfitRacePicker);
                        ImGui.EndTable();
                    }
                    ImGuiHelpers.ScaledDummy(5f);
                    using (ImRaii.Disabled(_vanillaOutfit is null))
                    {
                        if (ImGui.Button(_convertVanillaReady ? "Vanilla outfit loaded" : "Load vanilla outfit", new Vector2(-1, 32f * ImGuiHelpers.GlobalScale)))
                            StartVanillaModelSource();
                    }
                }
                else
                {
                    DrawSourceFields();
                }
            }
        }

        if (_document is null || (_convertUseVanilla && !_convertVanillaReady))
        {
            DrawMessage();
            return;
        }

        RavaFitUiChrome.DrawSectionTitle("Outfit mapping", "Pick the pieces and target bodies.");
        using (RavaFitUiChrome.BeginCard("##RavaFitOutfitRowsCard", GetWorkCardHeight(), allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            if (!_convertUseVanilla) DrawCreatorPermissionsNotice();
            DrawOutfitRows();
            DrawAccessoryOutfitRows(swap: false);
            DrawConversionControls();
        }
    }

    private CharacterRaceIdentity GetVanillaOutfitRace()
        => CharacterRaceCatalog.FromCode(_vanillaOutfitRaceOverride) ?? _plugin.CharacterRace.Current ?? CharacterRaceCatalog.FromCode("0201")!;

    private void DrawVanillaOutfitRacePicker()
    {
        var race = GetVanillaOutfitRace();
        var current = _plugin.CharacterRace.Current;
        var preview = _vanillaOutfitRaceOverride is null && current is not null ? $"Current character · {current.DisplayName}" : race.DisplayName;
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##VanillaOutfitRace", preview);
        if (!combo.Success) return;
        if (current is not null && ImGui.Selectable($"Current character · {current.DisplayName}##vanilla-current", _vanillaOutfitRaceOverride is null))
        {
            _vanillaOutfitRaceOverride = null;
            _convertVanillaReady = false;
        }
        foreach (var identity in CharacterRaceCatalog.All)
        {
            if (!ImGui.Selectable($"{identity.DisplayName}##vanilla-race-{identity.Code}", string.Equals(_vanillaOutfitRaceOverride, identity.Code, StringComparison.OrdinalIgnoreCase))) continue;
            _vanillaOutfitRaceOverride = identity.Code;
            _convertVanillaReady = false;
        }
    }

    private void StartLoadVanillaOutfitCatalogue()
    {
        if (_vanillaOutfitLoadStarted || _vanillaOutfitAssets.Count > 0) return;
        _vanillaOutfitLoadStarted = true;
        _vanillaOutfitLoading = true;
        _ = RunUiTask(async () =>
        {
            try
            {
                var assets = await Task.Run(_plugin.VanillaAssets.GetOutfitAssets).ConfigureAwait(false);
                _uiActions.Enqueue(() =>
                {
                    _vanillaOutfitAssets = assets;
                    _vanillaOutfitLoading = false;
                });
            }
            catch
            {
                _uiActions.Enqueue(() => { _vanillaOutfitLoading = false; _vanillaOutfitLoadStarted = false; });
                throw;
            }
        });
    }

    private void DrawPinnedSearchResults(string id, int visibleCount, Action drawResults)
    {
        var scale = ImGuiHelpers.GlobalScale;
        var rowHeight = ImGui.GetFrameHeightWithSpacing();
        var height = Math.Clamp((visibleCount * rowHeight) + (6f * scale), 72f * scale, 300f * scale);
        if (ImGui.BeginChild(id, new Vector2(-1f, height), false, ImGuiWindowFlags.None))
        {
            if (_resetScrollThisFrame) ImGui.SetScrollY(0f);
            drawResults();
        }
        ImGui.EndChild();
    }

    private void DrawVanillaOutfitPicker()
    {
        StartLoadVanillaOutfitCatalogue();
        if (_vanillaOutfitLoading)
        {
            ImGui.TextUnformatted("Loading outfits...");
            return;
        }
        var assets = _vanillaOutfitAssets;
        if (_vanillaOutfit is null && assets.Count == 1) { _vanillaOutfit = assets[0]; _convertVanillaReady = false; }
        if (assets.Count == 0)
        {
            RavaFitUiChrome.DrawMutedText("No vanilla outfits found.");
            return;
        }
        var preview = _vanillaOutfit?.DisplayName ?? "Select an in-game outfit";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##VanillaOutfit", preview);
        if (!combo.Success) return;
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##VanillaOutfitFilter", "Search outfit or piece name...", ref _vanillaOutfitFilter, 128);
        ImGui.Separator();
        var visibleAssets = assets.Where(asset => string.IsNullOrWhiteSpace(_vanillaOutfitFilter) || asset.Name.Contains(_vanillaOutfitFilter, StringComparison.CurrentCultureIgnoreCase) || asset.SearchText.Contains(_vanillaOutfitFilter, StringComparison.CurrentCultureIgnoreCase)).Take(250).ToArray();
        DrawPinnedSearchResults("##VanillaOutfitResults", visibleAssets.Length, () =>
        {
            foreach (var asset in visibleAssets)
            {
                if (!ImGui.Selectable($"{asset.DisplayName}##vanilla-outfit-{asset.ModelSetId}", Equals(asset, _vanillaOutfit))) continue;
                _vanillaOutfit = asset;
                _vanillaOutfitFilter = string.Empty;
                _convertVanillaReady = false;
            }
        });
    }

    private void StartVanillaModelSource()
    {
        if (_vanillaOutfit is null) return;
        var asset = _vanillaOutfit;
        var sourceRace = GetVanillaOutfitRace();
        _ = RunUiTask(async () =>
        {
            var mod = await _plugin.VanillaAssets.CreateModelSourceAsync(asset, sourceRace.Code).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                SelectMod(mod);
                _convertUseVanilla = true;
                _vanillaOutfit = asset;
                _vanillaOutfitRaceOverride = sourceRace.Code;
                _convertVanillaReady = true;
                RequestScrollTop();
                var pieceCount = _outfitChoices.Count(pair => pair.Value.Count > 0);
                SetMessage($"Loaded {asset.Name} for {sourceRace.DisplayName} · {pieceCount} piece{(pieceCount == 1 ? string.Empty : "s")}.", false);
            });
        });
    }

    private void DrawCustomiseTab()
    {
        DrawCustomiseSubTabs();
        ImGuiHelpers.ScaledDummy(8f);
        RavaFitUiChrome.DrawSectionTitle("Mod", _customiseSubTab == CustomiseSubTab.RemoveOptions ? "Select the mod you want to edit" : "Select the model you want to customise");
        using (RavaFitUiChrome.BeginCard("##RavaFitCustomiseSource", 112f * ImGuiHelpers.GlobalScale, allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            if (!_plugin.Penumbra.Available)
            {
                ImGui.TextUnformatted("Penumbra is not available.");
                if (ImGui.Button("Refresh Penumbra")) _plugin.Penumbra.Refresh();
                return;
            }
            if (ImGui.BeginTable("##RavaFitCustomiseSourceFields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##CustomiseLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##CustomiseValue", ImGuiTableColumnFlags.WidthStretch, 1f);
                DrawFieldRow("Mod", DrawModPicker);
                if (_customiseSubTab != CustomiseSubTab.RemoveOptions) DrawFieldRow("Model", DrawCustomiseModelPicker);
                ImGui.EndTable();
            }
        }

        if (_selectedMod is null || _document is null || (_customiseSubTab != CustomiseSubTab.RemoveOptions && _customiseSelection is null))
        {
            DrawMessage();
            return;
        }

        DrawCreatorPermissionsNotice();
        var cardHeight = GetWorkCardHeight(440f);
        using (RavaFitUiChrome.BeginCard("##RavaFitCustomiseWork", cardHeight, allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            switch (_customiseSubTab)
            {
                case CustomiseSubTab.Piercings:
                    DrawCustomisePiercingTab();
                    break;
                case CustomiseSubTab.Visibility:
                    DrawCustomiseVisibilityTab();
                    break;
                case CustomiseSubTab.SplitAccessory:
                    DrawCustomiseSplitAccessoryTab();
                    break;
                case CustomiseSubTab.RemoveOptions:
                    DrawCustomiseRemoveOptionsTab();
                    break;
            }

            if (_plugin.Customise.Busy && !string.IsNullOrWhiteSpace(_plugin.Customise.Status))
            {
                ImGuiHelpers.ScaledDummy(7f);
                RavaFitUiChrome.DrawMutedText(_plugin.Customise.Status);
            }
            if (_plugin.Cleanup.Busy && !string.IsNullOrWhiteSpace(_plugin.Cleanup.Status))
            {
                ImGuiHelpers.ScaledDummy(7f);
                RavaFitUiChrome.DrawMutedText(_plugin.Cleanup.Status);
            }
            DrawMessage();
        }
    }

    private void DrawCustomiseSubTabs()
    {
        DrawResponsiveSegmentTabs([
            ("Piercing assignment", _customiseSubTab == CustomiseSubTab.Piercings, (Action)(() => ActivateCustomiseSubTab(CustomiseSubTab.Piercings))),
            ("Visibility toggles", _customiseSubTab == CustomiseSubTab.Visibility, (Action)(() => ActivateCustomiseSubTab(CustomiseSubTab.Visibility))),
            ("Split to accessory", _customiseSubTab == CustomiseSubTab.SplitAccessory, (Action)(() => ActivateCustomiseSubTab(CustomiseSubTab.SplitAccessory))),
            ("Remove options", _customiseSubTab == CustomiseSubTab.RemoveOptions, (Action)(() => ActivateCustomiseSubTab(CustomiseSubTab.RemoveOptions))),
        ], 120f);
    }

    private void DrawCustomisePiercingTab()
    {
        RavaFitUiChrome.DrawSectionTitle("Piercing assignment");
        RavaFitUiChrome.DrawMutedWrappedText("Swap this model's piercings for a body from your catalogue.");
        ImGuiHelpers.ScaledDummy(6f);
        DrawCustomisePiercingPicker();
        ImGuiHelpers.ScaledDummy(6f);
        using (ImRaii.Disabled(_plugin.Customise.Busy || _customisePiercingBody is null))
        {
            if (ImGui.Button(_plugin.Customise.Busy ? "Working..." : "Assign piercings", new Vector2(-1, 34f * ImGuiHelpers.GlobalScale)))
                StartPiercingAssignment();
        }
    }

    private void DrawCustomiseVisibilityTab()
    {
        EnsureCustomiseInspection();
        RavaFitUiChrome.DrawSectionTitle("Visibility toggles");
        RavaFitUiChrome.DrawMutedWrappedText("Pick the regions you want to toggle. Colours match the list and preview.");
        ImGuiHelpers.ScaledDummy(6f);
        DrawCustomiseMeshParts();
        ImGuiHelpers.ScaledDummy(8f);

        if (ImGui.RadioButton("Separate toggles", _customiseVisibilityMode == VisibilityCreationMode.Separate))
            _customiseVisibilityMode = VisibilityCreationMode.Separate;
        ImGui.SameLine();
        if (ImGui.RadioButton("All selected as one", _customiseVisibilityMode == VisibilityCreationMode.Combined))
            _customiseVisibilityMode = VisibilityCreationMode.Combined;

        ImGuiHelpers.ScaledDummy(6f);
        if (_customiseVisibilityMode == VisibilityCreationMode.Separate)
        {
            if (_customiseSelectedParts.Count == 0)
                RavaFitUiChrome.DrawMutedText("Pick one or more regions, then name them.");
            else
                RavaFitUiChrome.DrawMutedText("Each checked region gets its own toggle.");
        }
        else
        {
            ImGui.SetNextItemWidth(-1);
            ImGui.InputTextWithHint("##CustomiseCombinedToggleName", "Name for the combined toggle", ref _customiseCombinedToggleName, 96);
            RavaFitUiChrome.DrawMutedText("All checked regions will use this one toggle.");
        }

        ImGuiHelpers.ScaledDummy(7f);
        var visibilityReady = CanCreateVisibilityToggles();
        using (ImRaii.Disabled(!visibilityReady))
        {
            var count = _customiseVisibilityMode == VisibilityCreationMode.Separate ? _customiseSelectedParts.Count : 1;
            var label = _plugin.Customise.Busy
                ? "Working..."
                : _customiseVisibilityMode == VisibilityCreationMode.Separate
                    ? count == 1 ? "Create toggle" : $"Create {count} toggles"
                    : "Create combined toggle";
            if (ImGui.Button(label, new Vector2(-1, 34f * ImGuiHelpers.GlobalScale)))
                StartVisibilityToggles();
        }
    }

    private void DrawCustomiseSplitAccessoryTab()
    {
        EnsureCustomiseInspection();
        RavaFitUiChrome.DrawSectionTitle("Split to accessory");
        RavaFitUiChrome.DrawMutedWrappedText("Select authored model parts and move them together onto an XIV accessory. A new option is created; the original option stays untouched.");
        ImGuiHelpers.ScaledDummy(6f);
        DrawCustomiseMeshParts();
        ImGuiHelpers.ScaledDummy(8f);
        DrawCustomiseAccessoryPicker();
        ImGuiHelpers.ScaledDummy(6f);
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##CustomiseAccessoryOptionName", "New option name", ref _customiseAccessoryOptionName, 128);
        RavaFitUiChrome.DrawMutedText("The accessory contains only the selected visible garment parts. No body model is transplanted into it.");
        ImGuiHelpers.ScaledDummy(7f);
        var ready = !_plugin.Customise.Busy && !_customiseInspecting && _customiseSelectedParts.Count > 0 && _customiseAccessoryTarget is not null && !string.IsNullOrWhiteSpace(_customiseAccessoryOptionName);
        using (ImRaii.Disabled(!ready))
        {
            var label = _plugin.Customise.Busy ? "Working..." : "Split selected to accessory";
            if (ImGui.Button(label, new Vector2(-1, 34f * ImGuiHelpers.GlobalScale))) StartAccessorySplit();
        }
    }

    private void DrawCustomiseAccessoryPicker()
    {
        if (_customiseAccessoryAssets.Count == 0)
        {
            try { _customiseAccessoryAssets = _plugin.VanillaAssets.GetAccessoryAssets(); }
            catch (Exception ex) { RavaFitUiChrome.DrawMutedWrappedText("Accessory catalogue unavailable: " + ex.Message); return; }
        }
        var preview = _customiseAccessoryTarget?.DisplayName ?? "Choose accessory target";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##CustomiseAccessoryTarget", preview);
        if (!combo.Success) return;
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##CustomiseAccessoryFilter", "Search accessories...", ref _customiseAccessoryFilter, 128);
        ImGui.Separator();
        var visible = _customiseAccessoryAssets.Where(asset => string.IsNullOrWhiteSpace(_customiseAccessoryFilter) || asset.SearchText.Contains(_customiseAccessoryFilter, StringComparison.OrdinalIgnoreCase)).Take(400).ToArray();
        DrawPinnedSearchResults("##CustomiseAccessoryResults", visible.Length, () =>
        {
            foreach (var asset in visible)
            {
                if (!ImGui.Selectable($"{asset.DisplayName}##accessory-{asset.Slot}-{asset.ModelSetId}-{asset.VariantId}-{asset.ItemId}", _customiseAccessoryTarget == asset)) continue;
                _customiseAccessoryTarget = asset;
                if (string.IsNullOrWhiteSpace(_customiseAccessoryOptionName)) _customiseAccessoryOptionName = $"{asset.Name} split";
                _customiseAccessoryFilter = string.Empty;
            }
        });
    }

    private void StartAccessorySplit()
    {
        if (_selectedMod is null || _customiseSelection is null || _customiseAccessoryTarget is null || _customiseSelectedParts.Count == 0) return;
        var sourceRaceCode = GetModelRaceCode(_customiseSelection.Model.GamePath);
        if (string.IsNullOrWhiteSpace(sourceRaceCode)) { SetMessage("Could not determine the selected model's race/gender.", true); return; }
        string targetGamePath;
        try { targetGamePath = _plugin.VanillaAssets.ResolveAccessoryModelPath(_customiseAccessoryTarget, sourceRaceCode); }
        catch (Exception ex) { SetMessage(ex.Message, true); return; }
        var targetMaterialId = _plugin.VanillaAssets.ResolveAccessoryMaterialId(_customiseAccessoryTarget);
        var request = new AccessorySplitRequest(_selectedMod, _customiseSelection.Group.StableKey, _customiseSelection.Option.StableKey, _customiseSelection.Model,
            _customiseSelectedParts.OrderBy(x => x).ToArray(), targetGamePath, _customiseAccessoryTarget.VariantId, targetMaterialId, _customiseAccessoryTarget.DisplayName, _customiseAccessoryOptionName.Trim());
        _ = RunUiTask(async () =>
        {
            await _plugin.Customise.SplitToAccessoryAsync(request).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                if (_selectedMod is not null)
                {
                    _document = PenumbraV4Document.Load(Path.Combine(_selectedMod.ModRoot, "meta.json"));
                    BuildOutfitChoices();
                }
                SetMessage($"Created {_customiseAccessoryOptionName} with selected parts on {_customiseAccessoryTarget?.DisplayName}.", false);
            });
        });
    }

    private void DrawCustomiseRemoveOptionsTab()
    {
        RavaFitUiChrome.DrawSectionTitle("Remove options");
        ImGui.TextColored(ErrorColour, "This is permanent.");
        RavaFitUiChrome.DrawMutedWrappedText("Pick what you want gone. Shared files stay put.");
        ImGuiHelpers.ScaledDummy(6f);
        DrawCleanupCandidateList(_cleanupCandidates, _customiseCleanupSelected, ref _customiseCleanupFilter, candidate => true, "##CustomiseCleanup");
        ImGuiHelpers.ScaledDummy(7f);
        using (ImRaii.Disabled(_plugin.Cleanup.Busy || _selectedMod is null || _customiseCleanupSelected.Count == 0))
        {
            var label = _plugin.Cleanup.Busy ? "Working..." : $"Permanently remove {_customiseCleanupSelected.Count} option{(_customiseCleanupSelected.Count == 1 ? string.Empty : "s")}";
            if (ImGui.Button(label, new Vector2(-1, 34f * ImGuiHelpers.GlobalScale)))
                StartCleanup(_customiseCleanupSelected);
        }
    }

    private bool CanCreateVisibilityToggles()
    {
        if (_plugin.Customise.Busy || _customiseInspecting || _customiseSelectedParts.Count == 0) return false;
        if (_customiseVisibilityMode == VisibilityCreationMode.Combined) return !string.IsNullOrWhiteSpace(_customiseCombinedToggleName);
        return _customiseSelectedParts.All(partIndex => _customisePartToggleNames.TryGetValue(partIndex, out var name) && !string.IsNullOrWhiteSpace(name));
    }

    private void DrawCustomiseModelPicker()
    {
        var choices = _outfitChoices.Values.SelectMany(x => x).Concat(_accessoryChoices.Values.SelectMany(x => x))
            .OrderBy(x => GetModelContainerSlot(x.Model.GamePath), StringComparer.OrdinalIgnoreCase)
            .ThenBy(x => x.Label, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (_customiseSelection is null && choices.Length == 1) SelectCustomiseModel(choices[0]);
        var preview = _customiseSelection is null
            ? (choices.Length == 0 ? "No model options" : "Select model")
            : $"{GetModelContainerSlot(_customiseSelection.Model.GamePath) ?? "Model"} · {_customiseSelection.Label}";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##CustomiseModel", preview);
        if (!combo.Success) return;
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##CustomiseModelFilter", "Search model options...", ref _customiseFilter, 128);
        ImGui.Separator();
        var visibleChoices = choices.Where(choice => string.IsNullOrWhiteSpace(_customiseFilter)
            || choice.Label.Contains(_customiseFilter, StringComparison.OrdinalIgnoreCase)
            || choice.Model.FileName.Contains(_customiseFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults("##CustomiseModelResults", visibleChoices.Length, () =>
        {
            foreach (var choice in visibleChoices)
            {
                var slot = GetModelContainerSlot(choice.Model.GamePath) ?? "Model";
                if (!ImGui.Selectable($"{slot} · {choice.Label}##custom-{choice.Identity}", string.Equals(_customiseSelection?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase))) continue;
                SelectCustomiseModel(choice);
                _customiseFilter = string.Empty;
            }
        });
    }

    private void SelectCustomiseModel(OutfitModelChoice choice)
    {
        RequestScrollTop();
        _customiseSelection = choice;
        _customisePiercingBody = null;
        _customisePiercingBodies = [];
        _customisePiercingBodiesLoading = false;
        _customisePiercingBodiesKey = string.Empty;
        _customiseParts = [];
        _customisePreviewTriangles = [];
        _customisePreviewEdges = [];
        _customisePreviewMin = Vector3.Zero;
        _customisePreviewMax = Vector3.One;
        _customiseSelectedParts.Clear();
        _customisePartToggleNames.Clear();
        _customisePartColourSlots.Clear();
        _customiseNextColourSlot = 0;
        _customiseCombinedToggleName = string.Empty;
        _customiseAccessoryTarget = null;
        _customiseAccessoryFilter = string.Empty;
        _customiseAccessoryOptionName = string.Empty;
        _customiseHoveredPart = -1;
        _customisePreviewPan = Vector2.Zero;
        _customisePreviewFocusCentre = null;
        _customisePreviewFocusRadius = null;
        _customisePreviewZoom = 1f;
        _customiseInspecting = false;
        _customiseInspectionComplete = false;
        if (_customiseSubTab is CustomiseSubTab.Visibility or CustomiseSubTab.SplitAccessory)
            EnsureCustomiseInspection();
    }

    private void EnsureCustomiseInspection()
    {
        if (_customiseSelection is null || _selectedMod is null || _customiseInspecting || _customiseInspectionComplete) return;
        var choice = _customiseSelection;
        var mod = _selectedMod;
        _customiseInspecting = true;
        _ = RunUiTask(async () =>
        {
            try
            {
                var inspection = await _plugin.Customise.InspectPartsAsync(mod, choice.Group.StableKey, choice.Option.StableKey, choice.Model).ConfigureAwait(false);
                _uiActions.Enqueue(() =>
                {
                    if (!string.Equals(_customiseSelection?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase)) return;
                    _customiseParts = inspection.Parts;
                    _customisePreviewTriangles = inspection.Triangles;
                    _customisePreviewEdges = inspection.Edges;
                    _customisePreviewMin = inspection.BoundsMin;
                    _customisePreviewMax = inspection.BoundsMax;
                    _customiseInspecting = false;
                    _customiseInspectionComplete = true;
                });
            }
            catch
            {
                _uiActions.Enqueue(() =>
                {
                    if (!string.Equals(_customiseSelection?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase)) return;
                    _customiseInspecting = false;
                    _customiseInspectionComplete = true;
                });
                throw;
            }
        });
    }

    private void DrawCustomisePiercingPicker()
    {
        if (_customiseSelection is null) return;
        EnsurePiercingBodyCatalogue();
        if (_customisePiercingBodiesLoading)
        {
            RavaFitUiChrome.DrawMutedText("Loading piercing bodies...");
            return;
        }
        var available = _customisePiercingBodies;
        if (_customisePiercingBody is null && available.Count == 1) _customisePiercingBody = available[0];
        var preview = _customisePiercingBody is null ? (available.Count == 0 ? "No donor piercing geometry for this slot" : "Select piercing body") : $"{_customisePiercingBody.BodyName} / {_customisePiercingBody.VariantName}";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##CustomisePiercingBody", preview);
        if (!combo.Success) return;
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##CustomisePiercingFilter", "Search piercing bodies...", ref _customisePiercingFilter, 128);
        ImGui.Separator();
        var visibleBodies = available.Where(body => string.IsNullOrWhiteSpace(_customisePiercingFilter)
            || body.BodyName.Contains(_customisePiercingFilter, StringComparison.OrdinalIgnoreCase)
            || body.VariantName.Contains(_customisePiercingFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults("##CustomisePiercingResults", visibleBodies.Length, () =>
        {
            foreach (var body in visibleBodies)
            {
                if (!ImGui.Selectable($"{body.BodyName} / {body.VariantName}##piercing-{body.BodyId}-{body.VariantId}", body == _customisePiercingBody)) continue;
                _customisePiercingBody = body;
                _customisePiercingFilter = string.Empty;
            }
        });
    }

    private void EnsurePiercingBodyCatalogue()
    {
        if (_customiseSelection is null) return;
        var slot = GetModelSlot(_customiseSelection.Model.GamePath);
        var race = GetModelRaceCode(_customiseSelection.Model.GamePath);
        if (slot is null) return;
        var key = $"{slot}|{race ?? "*"}";
        if (_customisePiercingBodiesLoading || string.Equals(_customisePiercingBodiesKey, key, StringComparison.OrdinalIgnoreCase)) return;
        _customisePiercingBodiesLoading = true;
        _customisePiercingBodiesKey = key;
        _ = RunUiTask(async () =>
        {
            try
            {
                var bodies = await Task.Run(() => _plugin.Customise.GetPiercingBodies(slot, race).ToArray()).ConfigureAwait(false);
                _uiActions.Enqueue(() =>
                {
                    if (!string.Equals(_customisePiercingBodiesKey, key, StringComparison.OrdinalIgnoreCase)) return;
                    _customisePiercingBodies = bodies;
                    _customisePiercingBodiesLoading = false;
                });
            }
            catch
            {
                _uiActions.Enqueue(() =>
                {
                    if (!string.Equals(_customisePiercingBodiesKey, key, StringComparison.OrdinalIgnoreCase)) return;
                    _customisePiercingBodiesLoading = false;
                    _customisePiercingBodiesKey = string.Empty;
                });
                throw;
            }
        });
    }

    private void DrawCustomiseMeshParts()
    {
        if (_customiseInspecting)
        {
            RavaFitUiChrome.DrawMutedText("Preparing model preview...");
            return;
        }
        if (_customiseParts.Count == 0)
        {
            RavaFitUiChrome.DrawMutedWrappedText(_customiseInspectionComplete ? "This model does not have separate regions to toggle." : "Choose a model to inspect its parts.");
            return;
        }

        _customiseHoveredPart = -1;
        var scale = ImGuiHelpers.GlobalScale;
        var availableWidth = ImGui.GetContentRegionAvail().X;
        var vertical = availableWidth < 860f * scale;
        var listHeight = vertical ? 230f * scale : Math.Clamp(ImGui.GetContentRegionAvail().Y * 0.58f, 330f * scale, 520f * scale);
        var previewHeight = vertical ? 360f * scale : listHeight;

        if (vertical)
        {
            DrawCustomisePartList(listHeight);
            ImGuiHelpers.ScaledDummy(7f);
            DrawCustomisePartPreview(previewHeight);
        }
        else if (ImGui.BeginTable("##RavaFitPartPicker", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings | ImGuiTableFlags.Resizable))
        {
            ImGui.TableSetupColumn("##PartList", ImGuiTableColumnFlags.WidthStretch, 0.40f);
            ImGui.TableSetupColumn("##PartPreview", ImGuiTableColumnFlags.WidthStretch, 0.60f);
            ImGui.TableNextRow();
            ImGui.TableSetColumnIndex(0);
            DrawCustomisePartList(listHeight);
            ImGui.TableSetColumnIndex(1);
            DrawCustomisePartPreview(previewHeight);
            ImGui.EndTable();
        }
        RavaFitUiChrome.DrawMutedText("Pick any regions you like. Colours match the list and preview. Drag to rotate, middle-drag to pan, wheel to zoom.");
    }

    private void DrawCustomisePartList(float height)
    {
        if (!ImGui.BeginChild("##CustomisePartList", new Vector2(-1, height), true))
        {
            ImGui.EndChild();
            return;
        }
        if (_resetScrollThisFrame) ImGui.SetScrollY(0f);
        foreach (var part in _customiseParts)
        {
            var selected = _customiseSelectedParts.Contains(part.PartIndex);
            var frameHeight = ImGui.GetFrameHeight();
            var rowHeight = selected && _customiseSubTab == CustomiseSubTab.Visibility && _customiseVisibilityMode == VisibilityCreationMode.Separate
                ? (frameHeight * 2f) + ImGui.GetStyle().ItemSpacing.Y + (4f * ImGuiHelpers.GlobalScale)
                : frameHeight + (4f * ImGuiHelpers.GlobalScale);
            var rowMin = ImGui.GetCursorScreenPos();
            var rowMax = new Vector2(rowMin.X + ImGui.GetContentRegionAvail().X, rowMin.Y + rowHeight);
            var mouse = ImGui.GetMousePos();
            var rowHovered = mouse.X >= rowMin.X && mouse.X <= rowMax.X && mouse.Y >= rowMin.Y && mouse.Y <= rowMax.Y;
            var draw = ImGui.GetWindowDrawList();

            if (selected)
            {
                var colour = GetCustomisePartColour(part.PartIndex, rowHovered ? 0.32f : 0.24f);
                draw.AddRectFilled(rowMin, rowMax, ImGui.GetColorU32(colour), 4f * ImGuiHelpers.GlobalScale);
                draw.AddRectFilled(rowMin, new Vector2(rowMin.X + (5f * ImGuiHelpers.GlobalScale), rowMax.Y), ImGui.GetColorU32(GetCustomisePartColour(part.PartIndex, 1f)), 4f * ImGuiHelpers.GlobalScale);
            }
            else if (rowHovered)
            {
                draw.AddRectFilled(rowMin, rowMax, ImGui.GetColorU32(new Vector4(0.35f, 0.35f, 0.42f, 0.18f)), 4f * ImGuiHelpers.GlobalScale);
            }

            ImGui.SetCursorScreenPos(new Vector2(rowMin.X + (9f * ImGuiHelpers.GlobalScale), rowMin.Y + (2f * ImGuiHelpers.GlobalScale)));
            var checkbox = selected;
            using (ImRaii.Disabled(!part.AttributeCapable))
            {
                if (ImGui.Checkbox($"##meshpart-{part.PartIndex}", ref checkbox))
                {
                    if (checkbox)
                    {
                        _customiseSelectedParts.Add(part.PartIndex);
                        if (!_customisePartColourSlots.ContainsKey(part.PartIndex)) _customisePartColourSlots[part.PartIndex] = _customiseNextColourSlot++;
                        _customisePartToggleNames.TryAdd(part.PartIndex, string.Empty);
                        selected = true;
                    }
                    else
                    {
                        _customiseSelectedParts.Remove(part.PartIndex);
                        selected = false;
                    }
                }
            }
            ImGui.SameLine();
            ImGui.TextUnformatted($"Mesh {part.MeshIndex + 1} · Part {part.SubmeshIndex + 1}");
            if (part.PiercingLike)
            {
                ImGui.SameLine();
                RavaFitUiChrome.DrawMutedText("piercing");
            }

            if (selected && _customiseSubTab == CustomiseSubTab.Visibility && _customiseVisibilityMode == VisibilityCreationMode.Separate)
            {
                ImGui.SetCursorScreenPos(new Vector2(rowMin.X + (9f * ImGuiHelpers.GlobalScale), rowMin.Y + frameHeight + ImGui.GetStyle().ItemSpacing.Y));
                var name = _customisePartToggleNames.TryGetValue(part.PartIndex, out var existing) ? existing : string.Empty;
                ImGui.SetNextItemWidth(MathF.Max(80f * ImGuiHelpers.GlobalScale, rowMax.X - rowMin.X - (18f * ImGuiHelpers.GlobalScale)));
                if (ImGui.InputTextWithHint($"##CustomisePartToggleName-{part.PartIndex}", "Name this toggle", ref name, 96)) _customisePartToggleNames[part.PartIndex] = name;
            }

            if (rowHovered)
            {
                _customiseHoveredPart = part.PartIndex;
                using (ImRaii.Tooltip())
                {
                    ImGui.TextUnformatted($"Mesh {part.MeshIndex + 1} · part {part.SubmeshIndex + 1}");
                    if (!string.IsNullOrWhiteSpace(part.Material)) ImGui.TextWrapped(part.Material);
                    if (part.Attributes.Count > 0) ImGui.TextWrapped("Existing attributes: " + string.Join(", ", part.Attributes));
                    ImGui.TextUnformatted($"{part.VertexCount:N0} vertices · {part.IndexCount / 3:N0} triangles");
                    if (!part.AttributeCapable) ImGui.TextWrapped("Preview only — this old mesh cannot have its own toggle.");
                }
            }

            ImGui.SetCursorScreenPos(new Vector2(rowMin.X, rowMax.Y));
            ImGui.Dummy(new Vector2(1f, 2f * ImGuiHelpers.GlobalScale));
        }
        ImGui.EndChild();
    }

    private void DrawCustomisePartPreview(float totalHeight)
    {
        var scaleUi = ImGuiHelpers.GlobalScale;
        var toolbarStartY = ImGui.GetCursorPosY();
        var compactToolbar = ImGui.GetContentRegionAvail().X < 430f * scaleUi;
        var modePreview = _customisePreviewMode.ToString();
        ImGui.SetNextItemWidth(compactToolbar ? -1f : MathF.Min(130f * scaleUi, MathF.Max(90f * scaleUi, ImGui.GetContentRegionAvail().X * 0.28f)));
        using (var combo = ImRaii.Combo("##PreviewDisplayMode", modePreview))
        {
            if (combo.Success)
            {
                foreach (var mode in Enum.GetValues<PreviewDisplayMode>())
                    if (ImGui.Selectable(mode.ToString(), mode == _customisePreviewMode)) _customisePreviewMode = mode;
            }
        }
        if (!compactToolbar) ImGui.SameLine();
        ImGui.Checkbox("Dim others", ref _customisePreviewDimOthers);
        ImGui.SameLine();
        ImGui.Checkbox("Only selected", ref _customisePreviewOnlySelected);

        var buttons = new (string Label, Action Action)[]
        {
            ("Front", () => SetCustomisePreviewView(0f, 0f)),
            ("Back", () => SetCustomisePreviewView(MathF.PI, 0f)),
            ("Left", () => SetCustomisePreviewView(-MathF.PI / 2f, 0f)),
            ("Right", () => SetCustomisePreviewView(MathF.PI / 2f, 0f)),
            ("Top", () => SetCustomisePreviewView(0f, -MathF.PI / 2f)),
            ("Bottom", () => SetCustomisePreviewView(0f, MathF.PI / 2f)),
            ("Focus", FocusCustomisePreviewSelection),
            ("Reset", ResetCustomisePreview),
        };
        var xStart = ImGui.GetCursorPosX();
        var right = xStart + ImGui.GetContentRegionAvail().X;
        foreach (var button in buttons)
        {
            var width = ImGui.CalcTextSize(button.Label).X + (16f * scaleUi);
            if (ImGui.GetCursorPosX() + width > right && ImGui.GetCursorPosX() > xStart + 1f) ImGui.NewLine();
            if (ImGui.SmallButton(button.Label)) button.Action();
            ImGui.SameLine(0f, 4f * scaleUi);
        }
        ImGui.NewLine();

        var toolbarHeight = MathF.Max(0f, ImGui.GetCursorPosY() - toolbarStartY);
        var height = MathF.Max(240f * scaleUi, totalHeight - toolbarHeight - (8f * scaleUi));
        var size = new Vector2(MathF.Max(180f * scaleUi, ImGui.GetContentRegionAvail().X), height);
        ImGui.InvisibleButton("##CustomisePartPreview", size);
        var min = ImGui.GetItemRectMin();
        var max = ImGui.GetItemRectMax();
        var hovered = ImGui.IsItemHovered();
        if (ImGui.IsItemActive() && ImGui.IsMouseDragging(ImGuiMouseButton.Left))
        {
            var delta = ImGui.GetIO().MouseDelta;
            _customisePreviewYaw += delta.X * 0.012f;
            _customisePreviewPitch = Math.Clamp(_customisePreviewPitch + delta.Y * 0.012f, -1.56f, 1.56f);
        }
        if (hovered && ImGui.IsMouseDragging(ImGuiMouseButton.Middle)) _customisePreviewPan += ImGui.GetIO().MouseDelta;
        if (hovered && MathF.Abs(ImGui.GetIO().MouseWheel) > 0.001f) _customisePreviewZoom = Math.Clamp(_customisePreviewZoom * MathF.Pow(1.12f, ImGui.GetIO().MouseWheel), 0.20f, 8f);

        var draw = ImGui.GetWindowDrawList();
        draw.AddRectFilled(min, max, ImGui.GetColorU32(new Vector4(0.045f, 0.045f, 0.055f, 1f)), 7f * scaleUi);
        draw.AddRect(min, max, ImGui.GetColorU32(new Vector4(0.28f, 0.25f, 0.34f, 0.9f)), 7f * scaleUi);
        if (_customisePreviewTriangles.Count == 0)
        {
            var text = "Preview unavailable";
            var textSize = ImGui.CalcTextSize(text);
            draw.AddText((min + max - textSize) * 0.5f, ImGui.GetColorU32(new Vector4(0.65f, 0.65f, 0.7f, 1f)), text);
            return;
        }

        var centre3 = _customisePreviewFocusCentre ?? ((_customisePreviewMin + _customisePreviewMax) * 0.5f);
        var extent = _customisePreviewMax - _customisePreviewMin;
        var defaultRadius = MathF.Max(0.0001f, MathF.Max(extent.X, MathF.Max(extent.Y, extent.Z)) * 0.5f);
        var radius = MathF.Max(0.0001f, _customisePreviewFocusRadius ?? defaultRadius);
        var screenCentre = ((min + max) * 0.5f) + _customisePreviewPan;
        var screenScale = MathF.Min(max.X - min.X, max.Y - min.Y) * 0.43f * _customisePreviewZoom / radius;
        var cy = MathF.Cos(_customisePreviewYaw); var sy = MathF.Sin(_customisePreviewYaw);
        var cp = MathF.Cos(_customisePreviewPitch); var sp = MathF.Sin(_customisePreviewPitch);

        (Vector2 Screen, float Depth) Project(Vector3 point)
        {
            var p = point - centre3;
            var x = (cy * p.X) + (sy * p.Z);
            var z = (-sy * p.X) + (cy * p.Z);
            var y = (cp * p.Y) - (sp * z);
            var z2 = (sp * p.Y) + (cp * z);
            return (new Vector2(screenCentre.X + (x * screenScale), screenCentre.Y - (y * screenScale)), z2);
        }

        Vector3 RotateNormal(Vector3 normal)
        {
            var x = (cy * normal.X) + (sy * normal.Z);
            var z = (-sy * normal.X) + (cy * normal.Z);
            var y = (cp * normal.Y) - (sp * z);
            var z2 = (sp * normal.Y) + (cp * z);
            var result = new Vector3(x, y, z2);
            return result.LengthSquared() > 1e-8f ? Vector3.Normalize(result) : Vector3.UnitZ;
        }

        var partById = _customiseParts.ToDictionary(part => part.PartIndex);
        var textureWraps = new Dictionary<int, IDalamudTextureWrap>();
        if (_customisePreviewMode == PreviewDisplayMode.Textured && _customiseSelection is not null)
        {
            foreach (var part in _customiseParts)
            {
                var shared = _plugin.PreviewTextures.GetBaseTexture(_document, _customiseSelection.Group.StableKey, _customiseSelection.Option.StableKey, _customiseSelection.Model, part.Material);
                var wrap = shared?.GetWrapOrDefault();
                if (wrap is not null) textureWraps[part.PartIndex] = wrap;
            }
        }

        var projected = _customisePreviewTriangles.Select(triangle =>
        {
            var a = Project(triangle.A); var b = Project(triangle.B); var c = Project(triangle.C);
            return (Triangle: triangle, A: a.Screen, B: b.Screen, C: c.Screen, Depth: (a.Depth + b.Depth + c.Depth) / 3f, Normal: RotateNormal(triangle.Normal));
        }).Where(item => !_customisePreviewOnlySelected || _customiseSelectedParts.Contains(item.Triangle.PartIndex) || item.Triangle.PartIndex == _customiseHoveredPart)
          .OrderBy(item => item.Depth).ToArray();

        draw.PushClipRect(min, max, true);
        var light = Vector3.Normalize(new Vector3(-0.35f, 0.45f, 0.82f));
        foreach (var item in projected)
        {
            var selected = _customiseSelectedParts.Contains(item.Triangle.PartIndex);
            var hot = item.Triangle.PartIndex == _customiseHoveredPart;
            var diffuse = Math.Clamp(0.34f + (0.66f * MathF.Abs(Vector3.Dot(item.Normal, light))), 0.25f, 1f);
            var baseAlpha = selected || hot ? 1f : (_customisePreviewDimOthers ? 0.28f : 0.82f);
            var shaded = new Vector4(0.72f * diffuse, 0.74f * diffuse, 0.80f * diffuse, baseAlpha);

            if (_customisePreviewMode == PreviewDisplayMode.Wireframe)
            {
                draw.AddTriangle(item.A, item.B, item.C, ImGui.GetColorU32(shaded), 1f * scaleUi);
            }
            else
            {
                var textured = false;
                if (_customisePreviewMode == PreviewDisplayMode.Textured && textureWraps.TryGetValue(item.Triangle.PartIndex, out var textureWrap))
                {
                    var tint = new Vector4(diffuse, diffuse, diffuse, baseAlpha);
                    draw.AddImageQuad(textureWrap.Handle, item.A, item.B, item.C, item.C, item.Triangle.UvA, item.Triangle.UvB, item.Triangle.UvC, item.Triangle.UvC, ImGui.GetColorU32(tint));
                    textured = true;
                }
                if (!textured)
                {
                    var colour = _customisePreviewMode == PreviewDisplayMode.Flat && partById.TryGetValue(item.Triangle.PartIndex, out var flatPart)
                        ? GetPreviewMaterialColour(flatPart.Material, baseAlpha)
                        : shaded;
                    draw.AddTriangleFilled(item.A, item.B, item.C, ImGui.GetColorU32(colour));
                }

                if (selected || hot)
                {
                    var overlay = selected ? GetCustomisePartColour(item.Triangle.PartIndex, hot ? 0.52f : 0.38f) : new Vector4(1f, 0.88f, 0.36f, 0.42f);
                    draw.AddTriangleFilled(item.A, item.B, item.C, ImGui.GetColorU32(overlay));
                }
            }
        }

        draw.PopClipRect();
    }

    private void SetCustomisePreviewView(float yaw, float pitch)
    {
        _customisePreviewYaw = yaw;
        _customisePreviewPitch = pitch;
        _customisePreviewPan = Vector2.Zero;
    }

    private void ResetCustomisePreview()
    {
        _customisePreviewYaw = 0.35f;
        _customisePreviewPitch = -0.08f;
        _customisePreviewZoom = 1f;
        _customisePreviewPan = Vector2.Zero;
        _customisePreviewFocusCentre = null;
        _customisePreviewFocusRadius = null;
    }

    private void FocusCustomisePreviewSelection()
    {
        var selected = _customisePreviewTriangles.Where(triangle => _customiseSelectedParts.Contains(triangle.PartIndex)).ToArray();
        if (selected.Length == 0)
        {
            _customisePreviewFocusCentre = null;
            _customisePreviewFocusRadius = null;
            _customisePreviewPan = Vector2.Zero;
            return;
        }
        var points = selected.SelectMany(triangle => new[] { triangle.A, triangle.B, triangle.C }).ToArray();
        var min = new Vector3(points.Min(point => point.X), points.Min(point => point.Y), points.Min(point => point.Z));
        var max = new Vector3(points.Max(point => point.X), points.Max(point => point.Y), points.Max(point => point.Z));
        _customisePreviewFocusCentre = (min + max) * 0.5f;
        var extent = max - min;
        _customisePreviewFocusRadius = MathF.Max(0.001f, MathF.Max(extent.X, MathF.Max(extent.Y, extent.Z)) * 0.58f);
        _customisePreviewPan = Vector2.Zero;
        _customisePreviewZoom = 1f;
    }

    private static Vector4 GetPreviewMaterialColour(string material, float alpha)
    {
        var hash = StringComparer.OrdinalIgnoreCase.GetHashCode(material ?? string.Empty);
        var r = 0.42f + (((hash >> 0) & 0xFF) / 255f * 0.34f);
        var g = 0.42f + (((hash >> 8) & 0xFF) / 255f * 0.34f);
        var b = 0.42f + (((hash >> 16) & 0xFF) / 255f * 0.34f);
        return new Vector4(r, g, b, alpha);
    }

    private Vector4 GetCustomisePartColour(int partIndex, float alpha)
    {
        if (!_customisePartColourSlots.TryGetValue(partIndex, out var slot)) slot = Math.Abs(partIndex);
        var hue = (0.07f + (slot * 0.61803398875f)) % 1f;
        var rgb = HsvToRgb(hue, 0.68f, 0.96f);
        return new Vector4(rgb, alpha);
    }

    private static Vector3 HsvToRgb(float hue, float saturation, float value)
    {
        var h = ((hue % 1f) + 1f) % 1f * 6f;
        var sector = (int)MathF.Floor(h);
        var fraction = h - sector;
        var p = value * (1f - saturation);
        var q = value * (1f - (saturation * fraction));
        var t = value * (1f - (saturation * (1f - fraction)));
        return sector switch
        {
            0 => new Vector3(value, t, p),
            1 => new Vector3(q, value, p),
            2 => new Vector3(p, value, t),
            3 => new Vector3(p, q, value),
            4 => new Vector3(t, p, value),
            _ => new Vector3(value, p, q),
        };
    }

    private void DrawCleanupCandidateList(IReadOnlyList<ModOptionCleanupCandidate> candidates, HashSet<string> selected, ref string filter, Func<ModOptionCleanupCandidate, bool> predicate, string id)
    {
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint(id + "Filter", "Search group or option...", ref filter, 128);
        var activeFilter = filter;
        var visible = candidates.Where(predicate).Where(candidate => string.IsNullOrWhiteSpace(activeFilter)
            || candidate.GroupName.Contains(activeFilter, StringComparison.OrdinalIgnoreCase)
            || candidate.OptionName.Contains(activeFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        if (visible.Length == 0)
        {
            RavaFitUiChrome.DrawMutedText(_selectedMod is null ? "Select a mod." : "No matching options found.");
            return;
        }

        var height = MathF.Min(290f * ImGuiHelpers.GlobalScale, MathF.Max(120f * ImGuiHelpers.GlobalScale, visible.Length * 28f * ImGuiHelpers.GlobalScale));
        var childVisible = ImGui.BeginChild(id + "List", new Vector2(-1, height), true);
        if (_resetScrollThisFrame) ImGui.SetScrollY(0f);
        if (childVisible)
        {
            foreach (var candidate in visible)
            {
                var key = CleanupKey(candidate.GroupKey, candidate.OptionKey);
                var value = selected.Contains(key);
                if (ImGui.Checkbox($"##cleanup-{id}-{key}", ref value))
                {
                    if (value) selected.Add(key); else selected.Remove(key);
                }
                ImGui.SameLine();
                ImGui.TextUnformatted($"{candidate.GroupName}  /  {candidate.OptionName}");
                ImGui.SameLine();
                RavaFitUiChrome.DrawMutedText(BuildCleanupKindLabel(candidate.Kinds));
            }
        }
        ImGui.EndChild();
    }

    private static string BuildCleanupKindLabel(ModOptionKind kinds)
    {
        var labels = new List<string>();
        if ((kinds & ModOptionKind.Gear) != 0) labels.Add("Gear");
        if ((kinds & ModOptionKind.Sound) != 0) labels.Add("Sound");
        if ((kinds & ModOptionKind.Vfx) != 0) labels.Add("VFX");
        if ((kinds & ModOptionKind.Animation) != 0) labels.Add("Animation");
        if ((kinds & ModOptionKind.RavaFitGenerated) != 0) labels.Add("RavaFit");
        if (labels.Count == 0) labels.Add("Option");
        return string.Join(" · ", labels);
    }

    private static string CleanupKey(string groupKey, string optionKey) => $"{groupKey}\u001f{optionKey}";

    private void RefreshCleanupCandidates()
    {
        _cleanupCandidates = [];
        _customiseCleanupSelected.Clear();
        _animationCleanupSelected.Clear();
        if (_selectedMod is null) return;
        try { _cleanupCandidates = _plugin.Cleanup.Inspect(_selectedMod); }
        catch (Exception ex) { SetMessage(ex.Message, true); }
    }

    private void StartCleanup(HashSet<string> selected)
    {
        if (_selectedMod is null || selected.Count == 0) return;
        var mod = _selectedMod;
        var chosen = _cleanupCandidates.Where(candidate => selected.Contains(CleanupKey(candidate.GroupKey, candidate.OptionKey)))
            .Select(candidate => new ModOptionCleanupSelection(candidate.GroupKey, candidate.OptionKey)).ToArray();
        if (chosen.Length == 0) return;
        _ = RunUiTask(async () =>
        {
            var result = await _plugin.Cleanup.RemoveOptionsAsync(mod, chosen).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                _plugin.Penumbra.Refresh();
                var refreshed = _plugin.Penumbra.Mods.FirstOrDefault(candidate => candidate.Directory == mod.Directory) ?? mod;
                SelectMod(refreshed);
                var summary = $"Removed {result.RemovedOptions} option(s), {result.RemovedGroups} dependent group(s), and {result.RemovedFiles} orphaned file(s).";
                if (!string.IsNullOrWhiteSpace(result.Warning)) summary += " " + result.Warning;
                SetMessage(summary, false);
            });
        });
    }

    private void StartPiercingAssignment()
    {
        if (_selectedMod is null || _customiseSelection is null || _customisePiercingBody is null) return;
        var request = new PiercingAssignmentRequest(_selectedMod, _customiseSelection.Group.StableKey, _customiseSelection.Option.StableKey, _customiseSelection.Model, _customisePiercingBody);
        _ = RunUiTask(async () =>
        {
            await _plugin.Customise.AssignPiercingsAsync(request).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                var selected = _selectedMod;
                _plugin.Penumbra.Refresh();
                if (selected is not null) SelectMod(_plugin.Penumbra.Mods.FirstOrDefault(m => m.Directory == selected.Directory) ?? selected);
                SetMessage($"Assigned {request.PiercingBody.BodyName} piercings.", false);
            });
        });
    }

    private void StartVisibilityToggles()
    {
        if (_selectedMod is null || _customiseSelection is null || _customiseSelectedParts.Count == 0) return;
        VisibilityToggleDefinition[] toggles;
        if (_customiseVisibilityMode == VisibilityCreationMode.Separate)
        {
            toggles = _customiseSelectedParts
                .OrderBy(partIndex => _customisePartColourSlots.TryGetValue(partIndex, out var slot) ? slot : int.MaxValue)
                .Select(partIndex => new VisibilityToggleDefinition(_customisePartToggleNames.TryGetValue(partIndex, out var name) ? name.Trim() : string.Empty, new[] { partIndex }))
                .ToArray();
        }
        else
        {
            toggles = new[] { new VisibilityToggleDefinition(_customiseCombinedToggleName.Trim(), _customiseSelectedParts.OrderBy(x => x).ToArray()) };
        }
        if (toggles.Length == 0 || toggles.Any(toggle => string.IsNullOrWhiteSpace(toggle.ToggleName))) return;

        var request = new VisibilityToggleBatchRequest(_selectedMod, _customiseSelection.Group.StableKey, _customiseSelection.Option.StableKey, _customiseSelection.Model, toggles);
        _ = RunUiTask(async () =>
        {
            await _plugin.Customise.AddVisibilityTogglesAsync(request).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                var selected = _selectedMod;
                _plugin.Penumbra.Refresh();
                if (selected is not null)
                {
                    var refreshed = _plugin.Penumbra.Mods.FirstOrDefault(mod => mod.Directory == selected.Directory) ?? selected;
                    SelectMod(refreshed);
                    var refreshedChoice = _outfitChoices.Values.SelectMany(choices => choices).FirstOrDefault(choice =>
                        string.Equals(choice.Group.StableKey, request.GroupKey, StringComparison.OrdinalIgnoreCase)
                        && string.Equals(choice.Option.StableKey, request.OptionKey, StringComparison.OrdinalIgnoreCase)
                        && string.Equals(choice.Model.GamePath, request.Model.GamePath, StringComparison.OrdinalIgnoreCase));
                    if (refreshedChoice is not null) SelectCustomiseModel(refreshedChoice);
                }
                SetMessage(request.Toggles.Count == 1
                    ? $"Added '{request.Toggles[0].ToggleName}' visibility toggle."
                    : $"Added {request.Toggles.Count} visibility toggles.", false);
            });
        });
    }

    private void DrawRaceSwapTab()
    {
        RavaFitUiChrome.DrawSectionTitle("Outfit", "Select a mod");
        using (RavaFitUiChrome.BeginCard("##RavaFitRaceSwapSourceCard", 108f * ImGuiHelpers.GlobalScale, allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            if (!_plugin.Penumbra.Available)
            {
                ImGui.TextUnformatted("Penumbra is not available.");
                if (ImGui.Button("Refresh Penumbra"))
                    _plugin.Penumbra.Refresh();
            }
            else if (ImGui.BeginTable("##RavaFitRaceSwapSourceFields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##RaceSwapLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##RaceSwapValue", ImGuiTableColumnFlags.WidthStretch, 1f);
                DrawFieldRow("Outfit", DrawModPicker);
                DrawFieldRow("Target character", DrawSwapTargetCharacterPicker);
                ImGui.EndTable();
            }
        }

        if (_document is null)
        {
            DrawMessage();
            return;
        }

        RavaFitUiChrome.DrawSectionTitle("Outfit mapping");
        using (RavaFitUiChrome.BeginCard("##RavaFitRaceSwapRowsCard", GetWorkCardHeight(), allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            DrawCreatorPermissionsNotice();
            if (LalafellContentGuard.IsLalafell(GetSwapTargetIdentity()))
            {
                RavaFitUiChrome.DrawMutedWrappedText("Lalafell ports are SFW only.");
                ImGuiHelpers.ScaledDummy(5f);
            }
            DrawSwapOutfitRows();
            DrawAccessoryOutfitRows(swap: true);
            DrawRaceSwapConversionControls();
        }
    }

    private void DrawAnimationPortTab()
    {
        DrawResponsiveSegmentTabs([
            ("Port animations", _animationSubTab == AnimationSubTab.Port, (Action)(() => ActivateAnimationSubTab(AnimationSubTab.Port))),
            ("Remove sounds / VFX", _animationSubTab == AnimationSubTab.Cleanup, (Action)(() => ActivateAnimationSubTab(AnimationSubTab.Cleanup))),
        ], 150f);
        ImGuiHelpers.ScaledDummy(8f);
        if (_animationSubTab == AnimationSubTab.Cleanup) DrawAnimationCleanupTab();
        else DrawAnimationPortContent();
    }

    private void DrawAnimationCleanupTab()
    {
        RavaFitUiChrome.DrawSectionTitle("Remove sounds / VFX", "Remove the extras you do not want.");
        using (RavaFitUiChrome.BeginCard("##RavaFitAnimationCleanupCard", MathF.Max(300f * ImGuiHelpers.GlobalScale, ImGui.GetContentRegionAvail().Y), allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            if (ImGui.BeginTable("##AnimationCleanupSource", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##AnimationCleanupLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##AnimationCleanupValue", ImGuiTableColumnFlags.WidthStretch, 1f);
                DrawFieldRow("Mod", DrawModPicker);
                ImGui.EndTable();
            }
            ImGuiHelpers.ScaledDummy(7f);
            ImGui.TextColored(ErrorColour, "This is permanent.");
            RavaFitUiChrome.DrawMutedWrappedText("Pick the Sound/VFX options you want gone. Anything tied to the animation itself is left alone.");
            ImGuiHelpers.ScaledDummy(7f);
            DrawCleanupCandidateList(_cleanupCandidates, _animationCleanupSelected, ref _animationCleanupFilter, candidate => (candidate.Kinds & (ModOptionKind.Sound | ModOptionKind.Vfx)) != 0 && (candidate.Kinds & ModOptionKind.Animation) == 0, "##AnimationCleanup");
            ImGuiHelpers.ScaledDummy(7f);
            using (ImRaii.Disabled(_plugin.Cleanup.Busy || _selectedMod is null || _animationCleanupSelected.Count == 0))
            {
                var label = _plugin.Cleanup.Busy ? "Working..." : $"Permanently remove {_animationCleanupSelected.Count} selected option{(_animationCleanupSelected.Count == 1 ? string.Empty : "s")}";
                if (ImGui.Button(label, new Vector2(-1, 34f * ImGuiHelpers.GlobalScale)))
                    StartCleanup(_animationCleanupSelected);
            }
            if (_plugin.Cleanup.Busy && !string.IsNullOrWhiteSpace(_plugin.Cleanup.Status))
            {
                ImGuiHelpers.ScaledDummy(5f);
                RavaFitUiChrome.DrawMutedText(_plugin.Cleanup.Status);
            }
            DrawMessage();
        }
    }

    private void DrawAnimationPortContent()
    {
        RavaFitUiChrome.DrawSectionTitle("Animation", "Choose a mod or vanilla player animation");
        var sourceCharacters = _animationUseVanilla ? Array.Empty<CharacterRaceIdentity>() : _plugin.AnimationPort.GetSourceCharacters(_selectedMod).ToArray();
        var cardHeight = MathF.Max(238f * ImGuiHelpers.GlobalScale, ImGui.GetContentRegionAvail().Y);
        using (RavaFitUiChrome.BeginCard("##RavaFitAnimationPortCard", cardHeight, allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            if (!_plugin.Penumbra.Available)
            {
                ImGui.TextUnformatted("Penumbra is not available.");
                if (ImGui.Button("Refresh Penumbra")) _plugin.Penumbra.Refresh();
                return;
            }

            var useVanilla = _animationUseVanilla;
            if (ImGui.RadioButton("Mod", !useVanilla)) { _animationUseVanilla = false; RequestScrollTop(); }
            ImGui.SameLine();
            if (ImGui.RadioButton("Vanilla", useVanilla))
            {
                _animationUseVanilla = true;
                RequestScrollTop();
                StartLoadVanillaAnimationCatalogue();
            }
            ImGuiHelpers.ScaledDummy(7f);

            if (ImGui.BeginTable("##RavaFitAnimationFields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##AnimationLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##AnimationValue", ImGuiTableColumnFlags.WidthStretch, 1f);
                if (_animationUseVanilla)
                {
                    StartLoadVanillaAnimationCatalogue();
                    DrawFieldRow("Animation", DrawVanillaAnimationPicker);
                    DrawFieldRow("Base race", DrawVanillaAnimationSourcePicker);
                }
                else
                {
                    DrawFieldRow("Mod", DrawModPicker);
                    if (sourceCharacters.Length > 1) DrawFieldRow("Source", DrawAnimationSourcePicker);
                }
                DrawFieldRow("Target race / gender", DrawAnimationTargetPicker);
                DrawFieldRow("Skeleton", DrawAnimationSkeletonPicker);
                ImGui.EndTable();
            }
            if (!_animationUseVanilla) DrawCreatorPermissionsNotice();
            if (LalafellContentGuard.IsLalafell(GetAnimationTargetIdentity()))
            {
                RavaFitUiChrome.DrawMutedWrappedText("Lalafell animation ports are SFW only. Vanilla is fine.");
                ImGuiHelpers.ScaledDummy(5f);
            }

            if (_animationUseVanilla && _vanillaAnimationLoading)
            {
                ImGuiHelpers.ScaledDummy(5f);
                RavaFitUiChrome.DrawMutedText("Loading emotes, idles, poses and movement...");
            }
            else if (!_animationUseVanilla && _selectedMod is not null && sourceCharacters.Length == 0)
            {
                ImGuiHelpers.ScaledDummy(5f);
                RavaFitUiChrome.DrawMutedText(_plugin.AnimationPort.GetPortabilityIssue(_selectedMod) ?? "No player animations found in this mod.");
            }

            ImGuiHelpers.ScaledDummy(7f);
            ImGui.Separator();
            ImGuiHelpers.ScaledDummy(7f);
            DrawAnimationPortControls();
        }
    }

    private void StartLoadVanillaAnimationCatalogue()
    {
        if (_vanillaAnimationLoadStarted || _vanillaAnimationAssets.Count > 0) return;
        _vanillaAnimationLoadStarted = true;
        _vanillaAnimationLoading = true;
        _ = RunUiTask(async () =>
        {
            try
            {
                var assets = await Task.Run(_plugin.VanillaAssets.GetAnimationAssets).ConfigureAwait(false);
                _uiActions.Enqueue(() =>
                {
                    _vanillaAnimationAssets = assets;
                    _vanillaAnimationLoading = false;
                });
            }
            catch
            {
                _uiActions.Enqueue(() => { _vanillaAnimationLoading = false; _vanillaAnimationLoadStarted = false; });
                throw;
            }
        });
    }

    private void DrawVanillaAnimationPicker()
    {
        if (_vanillaAnimationLoading)
        {
            ImGui.TextUnformatted("Loading emotes and non-combat animations...");
            return;
        }
        if (_vanillaAnimationAssets.Count == 0)
        {
            RavaFitUiChrome.DrawMutedWrappedText("No emotes, idles, poses or movement found. " + _plugin.VanillaAssets.AnimationCatalogueStatus);
            return;
        }

        if (_vanillaAnimation is null && _vanillaAnimationAssets.Count == 1)
        {
            _vanillaAnimation = _vanillaAnimationAssets[0];
            _animationSourceRaceOverride = PickDefaultVanillaAnimationSourceRace(_vanillaAnimation)?.Code;
        }
        var preview = _vanillaAnimation?.DisplayName ?? "Select emote / idle / movement...";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##VanillaAnimation", preview);
        if (!combo.Success) return;

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##VanillaAnimationFilter", "Search emote, walk, idle or pose...", ref _vanillaAnimationFilter, 128);
        ImGui.Separator();

        var visibleAnimations = _vanillaAnimationAssets.Where(asset => string.IsNullOrWhiteSpace(_vanillaAnimationFilter)
            || asset.Name.Contains(_vanillaAnimationFilter, StringComparison.CurrentCultureIgnoreCase)
            || asset.Category.Contains(_vanillaAnimationFilter, StringComparison.CurrentCultureIgnoreCase)
            || asset.SearchText.Contains(_vanillaAnimationFilter, StringComparison.CurrentCultureIgnoreCase)
            || asset.Variants.Any(variant => variant.SourceRace?.DisplayName.Contains(_vanillaAnimationFilter, StringComparison.CurrentCultureIgnoreCase) ?? false)).ToArray();
        DrawPinnedSearchResults("##VanillaAnimationResults", visibleAnimations.Length, () =>
        {
            foreach (var asset in visibleAnimations)
            {
                if (!ImGui.Selectable($"{asset.DisplayName}##vanilla-animation-{asset.Id}", string.Equals(asset.Id, _vanillaAnimation?.Id, StringComparison.OrdinalIgnoreCase))) continue;
                _vanillaAnimation = asset;
                _animationSourceRaceOverride = PickDefaultVanillaAnimationSourceRace(asset)?.Code;
                _vanillaAnimationFilter = string.Empty;
            }
        });
    }

    private VanillaAnimationVariant? GetSelectedVanillaAnimationVariant()
    {
        if (_vanillaAnimation is null) return null;
        if (!string.IsNullOrWhiteSpace(_animationSourceRaceOverride))
        {
            var selected = _vanillaAnimation.Variants.FirstOrDefault(variant => string.Equals(variant.SourceRaceCode, _animationSourceRaceOverride, StringComparison.OrdinalIgnoreCase));
            if (selected is not null) return selected;
        }

        var fallback = PickDefaultVanillaAnimationSourceRace(_vanillaAnimation);
        if (fallback is null) return null;
        _animationSourceRaceOverride = fallback.Code;
        return _vanillaAnimation.Variants.FirstOrDefault(variant => string.Equals(variant.SourceRaceCode, fallback.Code, StringComparison.OrdinalIgnoreCase));
    }

    private CharacterRaceIdentity? PickDefaultVanillaAnimationSourceRace(VanillaAnimationAsset asset)
    {
        if (_plugin.CharacterRace.Current is { } current
            && asset.Variants.Any(variant => string.Equals(variant.SourceRaceCode, current.Code, StringComparison.OrdinalIgnoreCase)))
            return current;

        if (asset.Variants.FirstOrDefault(variant => string.Equals(variant.SourceRaceCode, "0101", StringComparison.OrdinalIgnoreCase))?.SourceRace is { } midlanderMale)
            return midlanderMale;

        return asset.Variants.Select(variant => variant.SourceRace).FirstOrDefault(identity => identity is not null);
    }

    private void DrawVanillaAnimationSourcePicker()
    {
        if (_vanillaAnimation is null)
        {
            ImGui.TextDisabled("Select an animation");
            return;
        }
        if (_vanillaAnimation.Variants.Count == 1) _animationSourceRaceOverride = _vanillaAnimation.Variants[0].SourceRaceCode;

        var selected = GetSelectedVanillaAnimationVariant();
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##VanillaAnimationSourceRace", selected?.SourceRace?.DisplayName ?? "Select base race");
        if (!combo.Success) return;

        foreach (var variant in _vanillaAnimation.Variants)
        {
            var identity = variant.SourceRace;
            if (identity is null) continue;
            var label = $"{identity.DisplayName} ({variant.GamePaths.Count} PAP{(variant.GamePaths.Count == 1 ? string.Empty : "s")})";
            if (!ImGui.Selectable(label, string.Equals(selected?.SourceRaceCode, variant.SourceRaceCode, StringComparison.OrdinalIgnoreCase))) continue;
            _animationSourceRaceOverride = variant.SourceRaceCode;
        }
    }

    private CharacterRaceIdentity? GetAnimationSourceIdentity()
    {
        if (_animationUseVanilla) return GetSelectedVanillaAnimationVariant()?.SourceRace;
        var available = _plugin.AnimationPort.GetSourceCharacters(_selectedMod);
        if (CharacterRaceCatalog.FromCode(_animationSourceRaceOverride) is { } selected && available.Any(x => string.Equals(x.Code, selected.Code, StringComparison.OrdinalIgnoreCase)))
            return selected;
        return available.Count == 1 ? available[0] : null;
    }

    private CharacterRaceIdentity GetAnimationTargetIdentity()
    {
        if (CharacterRaceCatalog.FromCode(_animationTargetRaceOverride) is { } overridden) return overridden;
        if (_plugin.CharacterRace.Current is { } current) return current;
        var source = GetAnimationSourceIdentity();
        return CharacterRaceCatalog.FromCode(source?.MidlanderFallbackCode ?? "0201")!;
    }

    private void DrawAnimationSourcePicker()
    {
        var available = _plugin.AnimationPort.GetSourceCharacters(_selectedMod);
        var current = GetAnimationSourceIdentity();
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##AnimationSource", current?.DisplayName ?? "Select source");
        if (!combo.Success) return;
        foreach (var identity in available)
        {
            if (!ImGui.Selectable(identity.DisplayName, string.Equals(current?.Code, identity.Code, StringComparison.OrdinalIgnoreCase))) continue;
            _animationSourceRaceOverride = identity.Code;
        }
    }

    private void DrawAnimationTargetPicker()
    {
        var current = _plugin.CharacterRace.Current;
        var target = GetAnimationTargetIdentity();
        var preview = _animationTargetRaceOverride is null
            ? current is null ? $"{target.DisplayName} (fallback)" : $"Current character - {current.DisplayName}"
            : target.DisplayName;

        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##AnimationTarget", preview);
        if (!combo.Success) return;

        if (current is not null)
        {
            if (ImGui.Selectable($"Current character - {current.DisplayName}", _animationTargetRaceOverride is null))
            {
                _animationTargetRaceOverride = null;
                _animationSkeletonChoiceId = AnimationSkeletonService.StandardChoiceId;
            }
            ImGui.Separator();
        }

        foreach (var identity in CharacterRaceCatalog.All)
        {
            if (!ImGui.Selectable(identity.DisplayName, string.Equals(_animationTargetRaceOverride, identity.Code, StringComparison.OrdinalIgnoreCase))) continue;
            _animationTargetRaceOverride = identity.Code;
            _animationSkeletonChoiceId = AnimationSkeletonService.StandardChoiceId;
        }
    }

    private void DrawAnimationSkeletonPicker()
    {
        var target = GetAnimationTargetIdentity();
        var choices = _plugin.AnimationSkeletons.GetChoices(target);
        var current = choices.FirstOrDefault(x => string.Equals(x.Id, _animationSkeletonChoiceId, StringComparison.OrdinalIgnoreCase)) ?? choices[0];
        _animationSkeletonChoiceId = current.Id;
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##AnimationSkeleton", current.DisplayName);
        if (!combo.Success) return;
        foreach (var choice in choices)
        {
            if (!ImGui.Selectable(choice.DisplayName, string.Equals(current.Id, choice.Id, StringComparison.OrdinalIgnoreCase))) continue;
            _animationSkeletonChoiceId = choice.Id;
        }
    }

    private void DrawAnimationPortControls()
    {
        var ready = CanPortAnimation(out var reason);
        var progress = _plugin.AnimationPort.Progress;
        if (_plugin.AnimationPort.Busy || progress.Stage is not AnimationPortStage.Idle)
        {
            if (!string.IsNullOrWhiteSpace(progress.Detail)) RavaFitUiChrome.DrawMutedText(progress.Detail);
            ImGui.ProgressBar(progress.Progress, new Vector2(-1, 0), string.Empty);
            ImGuiHelpers.ScaledDummy(5f);
        }
        else if (!ready && !string.IsNullOrWhiteSpace(reason))
        {
            RavaFitUiChrome.DrawMutedText(reason);
            ImGuiHelpers.ScaledDummy(5f);
        }

        using (ImRaii.Disabled(!ready || _plugin.AnimationPort.Busy))
        {
            if (ImGui.Button(_plugin.AnimationPort.Busy ? "Working..." : "Port animations", new Vector2(-1, 36f * ImGuiHelpers.GlobalScale)))
                StartAnimationPort();
        }
        DrawMessage();
    }

    private bool CanPortAnimation(out string reason)
    {
        if (_animationUseVanilla)
        {
            if (_vanillaAnimation is null)
            {
                reason = _vanillaAnimationLoading ? "Building the vanilla player-animation catalogue." : "Select a vanilla player animation.";
                return false;
            }
            var variant = GetSelectedVanillaAnimationVariant();
            if (variant?.SourceRace is null || variant.GamePaths.Count == 0
                || variant.GamePaths.Any(path => !XivAnimationSkeletonIdentity.IsHumanPlayerAnimationPapGamePath(path)))
            {
                reason = "Choose one of the base-race variants available for this animation.";
                return false;
            }
        }
        else
        {
            if (_selectedMod is null)
            {
                reason = "Select a mod.";
                return false;
            }
            if (_plugin.AnimationPort.GetPortabilityIssue(_selectedMod) is { } portabilityIssue)
            {
                reason = portabilityIssue;
                return false;
            }
        }
        if (GetAnimationSourceIdentity() is null)
        {
            reason = _animationUseVanilla ? "Choose the base race for this vanilla animation." : (_plugin.AnimationPort.GetSourceCharacters(_selectedMod).Count > 1 ? "Choose the source character." : "No human animations found.");
            return false;
        }
        var target = GetAnimationTargetIdentity();
        if (!_plugin.AnimationSkeletons.GetChoices(target).Any(x => string.Equals(x.Id, _animationSkeletonChoiceId, StringComparison.OrdinalIgnoreCase)))
        {
            reason = "Choose a target skeleton.";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    private void StartAnimationPort()
    {
        if (GetAnimationSourceIdentity() is not { } source) return;
        var target = GetAnimationTargetIdentity();
        var skeleton = _animationSkeletonChoiceId;
        if (_animationUseVanilla)
        {
            if (_vanillaAnimation is null || GetSelectedVanillaAnimationVariant() is not { } variant) return;
            var gamePaths = variant.GamePaths.ToArray();
            var familyName = _vanillaAnimation.Name;
            _ = RunUiTask(async () =>
            {
                var result = await _plugin.AnimationPort.PortVanillaAsync(gamePaths, source, target, skeleton, familyName).ConfigureAwait(false);
                _uiActions.Enqueue(() => SetMessage($"Created '{result.ModName}'.", false));
            });
            return;
        }
        if (_selectedMod is null) return;
        var mod = _selectedMod;
        _ = RunUiTask(async () =>
        {
            var result = await _plugin.AnimationPort.PortAsync(mod, source, target, skeleton).ConfigureAwait(false);
            _uiActions.Enqueue(() => SetMessage($"Created '{result.ModName}'.", false));
        });
    }

    private CharacterRaceIdentity GetSwapTargetIdentity()
    {
        if (CharacterRaceCatalog.FromCode(_swapTargetRaceOverride) is { } overridden)
            return overridden;
        if (_plugin.CharacterRace.Current is { } current)
            return current;

        var sourceCode = _swapRows.Values
            .Select(row => row.Selection is null ? null : GetModelRaceCode(row.Selection.Model.GamePath))
            .FirstOrDefault(code => !string.IsNullOrWhiteSpace(code));
        var source = CharacterRaceCatalog.FromCode(sourceCode);
        return CharacterRaceCatalog.FromCode(source?.MidlanderFallbackCode ?? "0201")!;
    }

    private void DrawSwapTargetCharacterPicker()
    {
        var current = _plugin.CharacterRace.Current;
        var target = GetSwapTargetIdentity();
        var preview = _swapTargetRaceOverride is null
            ? current is null ? $"{target.DisplayName} (fallback)" : $"Current character - {current.DisplayName}"
            : target.DisplayName;

        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##RaceSwapTargetCharacter", preview);
        if (!combo.Success)
            return;

        if (current is not null)
        {
            if (ImGui.Selectable($"Current character - {current.DisplayName}", _swapTargetRaceOverride is null))
            {
                _swapTargetRaceOverride = null;
                OnSwapTargetCharacterChanged();
            }
            ImGui.Separator();
        }

        foreach (var identity in CharacterRaceCatalog.All)
        {
            if (!ImGui.Selectable(identity.DisplayName, string.Equals(_swapTargetRaceOverride, identity.Code, StringComparison.OrdinalIgnoreCase)))
                continue;
            _swapTargetRaceOverride = identity.Code;
            OnSwapTargetCharacterChanged();
        }
    }

    private void OnSwapTargetCharacterChanged()
    {
        var target = GetSwapTargetIdentity();
        foreach (var row in _swapRows.Values)
            if (row.Target is not null && !IsValidSwapTargetVariant(row.Target, row.Slot, target, GetModelGender(row)))
                row.Target = null;
    }

    private void DrawSwapOutfitRows()
    {
        if (ImGui.GetContentRegionAvail().X < 920f * ImGuiHelpers.GlobalScale)
        {
            DrawSwapOutfitRowsNarrow();
            return;
        }

        const ImGuiTableFlags flags = ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.RowBg | ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.NoSavedSettings;
        if (!ImGui.BeginTable("##RavaFitRaceSwapMapping", 5, flags))
            return;

        ImGui.TableSetupColumn("Refit", ImGuiTableColumnFlags.WidthFixed, 54f * ImGuiHelpers.GlobalScale);
        ImGui.TableSetupColumn("Slot", ImGuiTableColumnFlags.WidthFixed, 72f * ImGuiHelpers.GlobalScale);
        ImGui.TableSetupColumn("Selection", ImGuiTableColumnFlags.WidthStretch, 1.55f);
        ImGui.TableSetupColumn("Source", ImGuiTableColumnFlags.WidthStretch, 1f);
        ImGui.TableSetupColumn("Target", ImGuiTableColumnFlags.WidthStretch, 1f);
        ImGui.TableHeadersRow();

        foreach (var slot in BodySlots.All)
        {
            var row = _swapRows[slot];
            ImGui.TableNextRow();
            ImGui.TableNextColumn();
            var enabled = row.Enabled;
            using (ImRaii.Disabled(row.Selection is null))
            {
                if (ImGui.Checkbox($"##SwapRefit_{slot}", ref enabled))
                    row.Enabled = enabled;
            }
            ImGui.TableNextColumn();
            ImGui.AlignTextToFramePadding();
            ImGui.TextUnformatted(slot);
            ImGui.TableNextColumn();
            DrawSwapSelectionCell(row);
            ImGui.TableNextColumn();
            DrawSwapBodyCell(row, isTarget: false);
            ImGui.TableNextColumn();
            DrawSwapBodyCell(row, isTarget: true);
        }

        ImGui.EndTable();
    }

    private void DrawSwapOutfitRowsNarrow()
    {
        foreach (var slot in BodySlots.All)
        {
            var row = _swapRows[slot];
            ImGui.PushID($"swap-narrow-{slot}");
            ImGui.Separator();
            var enabled = row.Enabled;
            using (ImRaii.Disabled(row.Selection is null))
            {
                if (ImGui.Checkbox("Refit", ref enabled)) row.Enabled = enabled;
            }
            ImGui.SameLine();
            ImGui.TextUnformatted(slot);
            if (ImGui.BeginTable("##fields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##label", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##value", ImGuiTableColumnFlags.WidthStretch, 1f);
                DrawFieldRow("Selection", () => DrawSwapSelectionCell(row));
                DrawFieldRow("Source", () => DrawSwapBodyCell(row, false));
                DrawFieldRow("Target", () => DrawSwapBodyCell(row, true));
                ImGui.EndTable();
            }
            ImGui.PopID();
            ImGuiHelpers.ScaledDummy(4f);
        }
        ImGui.Separator();
    }

    private void DrawSwapSelectionCell(OutfitRowState row)
    {
        var choices = _outfitChoices[row.Slot];
        var preview = row.Selection?.Label ?? (choices.Count == 0 ? "No model options" : "Select option");
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo($"##SwapSelection_{row.Slot}", preview);
        if (!combo.Success)
            return;
        if (choices.Count == 0)
        {
            RavaFitUiChrome.DrawMutedText("No options found for this slot.");
            return;
        }

        foreach (var choice in choices)
        {
            if (!ImGui.Selectable(choice.Label + "##Swap" + choice.Identity, string.Equals(row.Selection?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase)))
                continue;
            SetSwapSelection(row, choice);
        }
    }

    private void DrawSwapBodyCell(OutfitRowState row, bool isTarget)
    {
        var target = GetSwapTargetIdentity();
        var sourceRaceCode = row.Selection is null ? null : GetModelRaceCode(row.Selection.Model.GamePath);
        var sourceGender = CharacterRaceCatalog.FromCode(sourceRaceCode)?.Gender;
        var sourceIdentity = CharacterRaceCatalog.FromCode(sourceRaceCode);
        var available = _plugin.Bodies.ForSlot(row.Slot)
            .Where(item => isTarget
                ? IsValidSwapTargetVariant(item, row.Slot, target, sourceGender)
                : IsValidSwapSourceVariant(item, row.Slot, sourceIdentity, target))
            .ToArray();
        var current = isTarget ? row.Target : row.Source;
        var preview = current is null ? (!isTarget && row.Analysing ? "Checking..." : "Select body") : $"{current.BodyName} / {current.VariantName}";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo($"##Swap{(isTarget ? "Target" : "Source")}_{row.Slot}", preview);
        if (!combo.Success)
            return;

        var bodyFilterOwner = $"Swap:{(isTarget ? "Target" : "Source")}:{row.Slot}";
        if (!string.Equals(_bodyFilterOwner, bodyFilterOwner, StringComparison.Ordinal))
        {
            _bodyFilterOwner = bodyFilterOwner;
            _bodyFilter = string.Empty;
        }

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint($"##SwapBodyFilter_{(isTarget ? "Target" : "Source")}_{row.Slot}", "Search bodies...", ref _bodyFilter, 128);
        ImGui.Separator();
        var visibleBodies = available.Where(v => string.IsNullOrWhiteSpace(_bodyFilter)
            || v.BodyName.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)
            || v.VariantName.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)
            || v.Collection.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults($"##SwapBodyResults_{(isTarget ? "Target" : "Source")}_{row.Slot}", visibleBodies.Length, () =>
        {
            foreach (var item in visibleBodies)
            {
                var text = $"{item.BodyName} / {item.VariantName}";
                if (!ImGui.Selectable(text, current == item))
                    continue;
                if (isTarget)
                {
                    row.Target = item;
                }
                else
                {
                    row.Source = item;
                    row.SourceUserOverride = true;
                    row.SourceInferred = false;
                    row.SourceAutoPriority = 100;
                }
                _bodyFilter = string.Empty;
            }
        });

        if (available.Length == 0)
        {
            var needsTargetSmallclothes = isTarget && CharacterRaceCatalog.RequiresSmallclothesTarget(sourceGender, target, row.Slot);
            var needsSourceSmallclothes = !isTarget && sourceIdentity is not null && CharacterRaceCatalog.RequiresSmallclothesSource(sourceIdentity, target.Gender, row.Slot);
            RavaFitUiChrome.DrawMutedText(needsTargetSmallclothes
                ? "No SFW male legs are loaded for this target."
                : needsSourceSmallclothes
                    ? "No SFW male legs are loaded for this source."
                    : "No bodies found.");
        }
    }

    private void DrawCreatorPermissionsNotice()
    {
        if (_document is null) return;
        var creator = string.IsNullOrWhiteSpace(_document.Author) ? "the original creator" : _document.Author;
        RavaFitUiChrome.DrawMutedWrappedText($"Credit stays with {creator}. Check their permissions before sharing this port.");
        ImGuiHelpers.ScaledDummy(5f);
    }

    private void DrawSourceFields()
    {
        if (!ImGui.BeginTable("##RavaFitSourceFields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            return;

        ImGui.TableSetupColumn("##SourceLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
        ImGui.TableSetupColumn("##SourceValue", ImGuiTableColumnFlags.WidthStretch, 1f);
        DrawFieldRow("Outfit", DrawModPicker);
        ImGui.EndTable();
    }

    private static void DrawFieldRow(string label, Action drawControl)
    {
        ImGui.TableNextRow();
        ImGui.TableNextColumn();
        ImGui.AlignTextToFramePadding();
        ImGui.TextUnformatted(label);
        ImGui.TableNextColumn();
        drawControl();
    }

    private void DrawModPicker()
    {
        if (_selectedMod is null && _plugin.Penumbra.Mods.Count == 1) SelectMod(_plugin.Penumbra.Mods[0]);
        ImGui.SetNextItemWidth(-1);
        var preview = _selectedMod?.Name ?? "Select a mod";
        using var combo = ImRaii.Combo("##Mod", preview);
        if (!combo.Success)
            return;

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##ModFilter", "Search mods...", ref _modFilter, 128);
        ImGui.Separator();

        var visibleMods = _plugin.Penumbra.Mods.Where(m => string.IsNullOrWhiteSpace(_modFilter) || m.Name.Contains(_modFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults("##ModResults", visibleMods.Length, () =>
        {
            foreach (var mod in visibleMods)
            {
                var selected = _selectedMod?.Directory == mod.Directory;
                if (ImGui.Selectable(mod.Name, selected))
                    SelectMod(mod);
            }
        });
    }

    private void DrawAccessoryOutfitRows(bool swap)
    {
        var choicesMap = _accessoryChoices;
        if (!choicesMap.Values.Any(choices => choices.Count > 0)) return;
        var rows = swap ? _swapAccessoryRows : _accessoryRows;
        ImGuiHelpers.ScaledDummy(8f);
        RavaFitUiChrome.DrawSectionTitle("Accessory garments", "Accessories are fitted as garments only; their body support comes from coverage analysis.");
        foreach (var slot in AccessoryModelSlots.All.Where(slot => choicesMap[slot].Count > 0))
        {
            var row = rows[slot];
            ImGui.PushID($"{(swap ? "swap" : "convert")}-accessory-{slot}");
            if (ImGui.BeginTable("##AccessoryRow", 3, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##enable", ImGuiTableColumnFlags.WidthFixed, 58f * ImGuiHelpers.GlobalScale);
                ImGui.TableSetupColumn("##slot", ImGuiTableColumnFlags.WidthFixed, 90f * ImGuiHelpers.GlobalScale);
                ImGui.TableSetupColumn("##selection", ImGuiTableColumnFlags.WidthStretch, 1f);
                ImGui.TableNextRow(); ImGui.TableNextColumn();
                var enabled = row.Enabled;
                using (ImRaii.Disabled(row.Selection is null)) if (ImGui.Checkbox("Refit", ref enabled)) row.Enabled = enabled;
                ImGui.TableNextColumn(); ImGui.AlignTextToFramePadding(); ImGui.TextUnformatted(slot);
                ImGui.TableNextColumn(); DrawAccessorySelectionCell(row, swap);
                ImGui.EndTable();
            }
            if (row.Analysing) RavaFitUiChrome.DrawMutedText("Checking garment coverage...");
            else if (row.Analysis is not null)
            {
                var required = string.Join(" + ", row.Analysis.Slots.Values.Where(e => e.Primary || e.Recommended).OrderByDescending(e => e.Primary).Select(e => e.Slot).Distinct(StringComparer.OrdinalIgnoreCase));
                RavaFitUiChrome.DrawMutedText($"Fits as {row.Analysis.PrimarySlot}{(string.IsNullOrWhiteSpace(required) ? string.Empty : $" · support {required}")} · garment only");
                DrawAccessorySupportBodies(row, swap);
            }
            else if (!string.IsNullOrWhiteSpace(row.Status)) RavaFitUiChrome.DrawMutedWrappedText(row.Status);
            ImGui.PopID();
        }
    }

    private void DrawAccessorySelectionCell(AccessoryOutfitRowState row, bool swap)
    {
        var choices = _accessoryChoices[row.Slot];
        var preview = row.Selection?.Label ?? "Select option";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##AccessorySelection", preview);
        if (!combo.Success) return;
        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##AccessorySelectionFilter", "Search options...", ref _selectionFilter, 128);
        ImGui.Separator();
        var visible = choices.Where(choice => string.IsNullOrWhiteSpace(_selectionFilter) || choice.Label.Contains(_selectionFilter, StringComparison.OrdinalIgnoreCase) || choice.Model.FileName.Contains(_selectionFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults("##AccessorySelectionResults", visible.Length, () =>
        {
            foreach (var choice in visible)
            {
                if (!ImGui.Selectable(choice.Label + "##" + choice.Identity, string.Equals(row.Selection?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase))) continue;
                SetAccessorySelection(row, choice, swap); _selectionFilter = string.Empty;
            }
        });
    }

    private void DrawAccessorySupportBodies(AccessoryOutfitRowState row, bool swap)
    {
        if (row.Selection is null || row.Analysis is null) return;

        var required = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { row.Analysis.PrimarySlot };
        foreach (var evidence in row.Analysis.Slots.Values.Where(evidence => evidence.Primary || evidence.Recommended))
            required.Add(evidence.Slot);

        ImGuiHelpers.ScaledDummy(5f);
        ImGui.PushID($"AccessorySupport_{(swap ? "Swap" : "Convert")}_{row.Slot}");
        if (ImGui.BeginTable("##AccessorySupportBodies", 3, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings | ImGuiTableFlags.RowBg))
        {
            ImGui.TableSetupColumn("Support", ImGuiTableColumnFlags.WidthFixed, 90f * ImGuiHelpers.GlobalScale);
            ImGui.TableSetupColumn("Source", ImGuiTableColumnFlags.WidthStretch, 1f);
            ImGui.TableSetupColumn("Target", ImGuiTableColumnFlags.WidthStretch, 1f);
            ImGui.TableHeadersRow();
            foreach (var bodySlot in BodySlots.All.Where(required.Contains))
            {
                ImGui.PushID(bodySlot);
                ImGui.TableNextRow();
                ImGui.TableNextColumn(); ImGui.AlignTextToFramePadding(); ImGui.TextUnformatted(bodySlot);
                ImGui.TableNextColumn(); DrawAccessoryBodyCell(row, bodySlot, isTarget: false, swap);
                ImGui.TableNextColumn(); DrawAccessoryBodyCell(row, bodySlot, isTarget: true, swap);
                ImGui.PopID();
            }
            ImGui.EndTable();
        }
        ImGui.PopID();
    }

    private void DrawAccessoryBodyCell(AccessoryOutfitRowState row, string bodySlot, bool isTarget, bool swap)
    {
        if (row.Selection is null) return;
        if (!swap && !isTarget && _convertUseVanilla)
        {
            ImGui.SetNextItemWidth(-1);
            using (ImRaii.Disabled(true))
                ImGui.InputText($"##AccessoryVanillaSource_{row.Slot}_{bodySlot}", ref _vanillaSourceDisplay, 64, ImGuiInputTextFlags.ReadOnly);
            return;
        }

        var sourceRaceCode = GetModelRaceCode(row.Selection.Model.GamePath);
        var sourceIdentity = CharacterRaceCatalog.FromCode(sourceRaceCode);
        var sourceGender = sourceIdentity?.Gender;
        var swapTarget = GetSwapTargetIdentity();
        var available = _plugin.Bodies.ForSlot(bodySlot)
            .Where(item => swap
                ? isTarget
                    ? IsValidSwapTargetVariant(item, bodySlot, swapTarget, sourceGender)
                    : IsValidSwapSourceVariant(item, bodySlot, sourceIdentity, swapTarget)
                : item.SupportsGender(sourceGender))
            .ToArray();

        var current = isTarget ? row.Targets[bodySlot] : row.Sources[bodySlot];

        var preview = current is null ? (isTarget ? "Select target" : "Select source") : $"{current.BodyName} / {current.VariantName}";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo($"##Accessory{(isTarget ? "Target" : "Source")}_{bodySlot}", preview);
        if (!combo.Success) return;

        var bodyFilterOwner = $"Accessory:{(swap ? "Swap" : "Convert")}:{row.Slot}:{bodySlot}:{(isTarget ? "Target" : "Source")}";
        if (!string.Equals(_bodyFilterOwner, bodyFilterOwner, StringComparison.Ordinal))
        {
            _bodyFilterOwner = bodyFilterOwner;
            _bodyFilter = string.Empty;
        }

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint($"##AccessoryBodyFilter_{(isTarget ? "Target" : "Source")}_{bodySlot}", "Search bodies...", ref _bodyFilter, 128);
        ImGui.Separator();
        var visibleBodies = available.Where(v => string.IsNullOrWhiteSpace(_bodyFilter)
            || v.BodyName.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)
            || v.VariantName.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)
            || v.Collection.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults($"##AccessoryBodyResults_{(isTarget ? "Target" : "Source")}_{bodySlot}", visibleBodies.Length, () =>
        {
            foreach (var item in visibleBodies)
            {
                var label = $"{item.BodyName} / {item.VariantName}";
                if (!string.Equals(item.Collection, item.BodyName, StringComparison.OrdinalIgnoreCase))
                    label += $"  [{item.Collection}]";
                if (!ImGui.Selectable(label, current == item)) continue;

                if (isTarget)
                {
                    row.Targets[bodySlot] = item;
                }
                else
                {
                    row.Sources[bodySlot] = item;
                    row.SourceUserOverrides.Add(bodySlot);
                }
                _bodyFilter = string.Empty;
            }
        });

        if (available.Length == 0)
            RavaFitUiChrome.DrawMutedText("No compatible bodies found.");
    }

    private void DrawOutfitRows()
    {
        if (ImGui.GetContentRegionAvail().X < 920f * ImGuiHelpers.GlobalScale)
        {
            DrawOutfitRowsNarrow();
            return;
        }

        const ImGuiTableFlags flags = ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.RowBg | ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.NoSavedSettings;
        if (!ImGui.BeginTable("##RavaFitOutfitMapping", 5, flags))
            return;

        ImGui.TableSetupColumn("Refit", ImGuiTableColumnFlags.WidthFixed, 54f * ImGuiHelpers.GlobalScale);
        ImGui.TableSetupColumn("Slot", ImGuiTableColumnFlags.WidthFixed, 72f * ImGuiHelpers.GlobalScale);
        ImGui.TableSetupColumn("Selection", ImGuiTableColumnFlags.WidthStretch, 1.55f);
        ImGui.TableSetupColumn("Source", ImGuiTableColumnFlags.WidthStretch, 1f);
        ImGui.TableSetupColumn("Target", ImGuiTableColumnFlags.WidthStretch, 1f);
        ImGui.TableHeadersRow();

        foreach (var slot in BodySlots.All)
        {
            var row = _outfitRows[slot];
            ImGui.TableNextRow();
            ImGui.TableNextColumn();
            var enabled = row.Enabled;
            using (ImRaii.Disabled(row.Selection is null))
            {
                if (ImGui.Checkbox($"##Refit_{slot}", ref enabled))
                    row.Enabled = enabled;
            }

            ImGui.TableNextColumn();
            ImGui.AlignTextToFramePadding();
            ImGui.TextUnformatted(slot);

            ImGui.TableNextColumn();
            DrawOutfitSelectionCell(row);

            ImGui.TableNextColumn();
            DrawOutfitBodyCell(row, isTarget: false);

            ImGui.TableNextColumn();
            DrawOutfitBodyCell(row, isTarget: true);
        }

        ImGui.EndTable();
    }

    private void DrawOutfitRowsNarrow()
    {
        foreach (var slot in BodySlots.All)
        {
            var row = _outfitRows[slot];
            ImGui.PushID($"outfit-narrow-{slot}");
            ImGui.Separator();
            var enabled = row.Enabled;
            using (ImRaii.Disabled(row.Selection is null))
            {
                if (ImGui.Checkbox("Refit", ref enabled)) row.Enabled = enabled;
            }
            ImGui.SameLine();
            ImGui.TextUnformatted(slot);
            if (ImGui.BeginTable("##fields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##label", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##value", ImGuiTableColumnFlags.WidthStretch, 1f);
                DrawFieldRow("Selection", () => DrawOutfitSelectionCell(row));
                DrawFieldRow("Source", () => DrawOutfitBodyCell(row, false));
                DrawFieldRow("Target", () => DrawOutfitBodyCell(row, true));
                ImGui.EndTable();
            }
            ImGui.PopID();
            ImGuiHelpers.ScaledDummy(4f);
        }
        ImGui.Separator();
    }

    private void DrawOutfitSelectionCell(OutfitRowState row)
    {
        var choices = _outfitChoices[row.Slot];
        var preview = row.Selection?.Label ?? (choices.Count == 0 ? "No model options" : "Select option");
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo($"##Selection_{row.Slot}", preview);
        if (!combo.Success)
            return;

        if (choices.Count == 0)
        {
            RavaFitUiChrome.DrawMutedText("No options found for this slot.");
            return;
        }

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint($"##SelectionFilter_{row.Slot}", "Search options...", ref _selectionFilter, 128);
        ImGui.Separator();
        var visibleChoices = choices.Where(choice => string.IsNullOrWhiteSpace(_selectionFilter)
            || choice.Group.Name.Contains(_selectionFilter, StringComparison.OrdinalIgnoreCase)
            || choice.Option.Name.Contains(_selectionFilter, StringComparison.OrdinalIgnoreCase)
            || choice.Model.FileName.Contains(_selectionFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults($"##SelectionResults_{row.Slot}", visibleChoices.Length, () =>
        {
            foreach (var choice in visibleChoices)
            {
                if (!ImGui.Selectable(choice.Label + "##" + choice.Identity, string.Equals(row.Selection?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase)))
                    continue;
                SetOutfitSelection(row, choice);
                _selectionFilter = string.Empty;
            }
        });
    }

    private void DrawOutfitBodyCell(OutfitRowState row, bool isTarget)
    {
        if (!isTarget && _convertUseVanilla)
        {
            ImGui.SetNextItemWidth(-1);
            using (ImRaii.Disabled(true))
                ImGui.InputText($"##VanillaSource_{row.Slot}", ref _vanillaSourceDisplay, 64, ImGuiInputTextFlags.ReadOnly);
            return;
        }

        var gender = GetOutfitContextGender(row);
        var available = _plugin.Bodies.ForSlot(row.Slot).Where(item => item.SupportsGender(gender)).ToArray();
        var current = isTarget ? row.Target : row.Source;
        var preview = current is null
            ? (!isTarget && row.Analysing ? "Checking..." : "Select body")
            : $"{current.BodyName} / {current.VariantName}";
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo($"##{(isTarget ? "Target" : "Source")}_{row.Slot}", preview);
        if (!combo.Success)
            return;

        var bodyFilterOwner = $"Outfit:{(isTarget ? "Target" : "Source")}:{row.Slot}";
        if (!string.Equals(_bodyFilterOwner, bodyFilterOwner, StringComparison.Ordinal))
        {
            _bodyFilterOwner = bodyFilterOwner;
            _bodyFilter = string.Empty;
        }

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint($"##BodyFilter_{(isTarget ? "Target" : "Source")}_{row.Slot}", "Search bodies...", ref _bodyFilter, 128);
        ImGui.Separator();
        var visibleBodies = available.Where(v => string.IsNullOrWhiteSpace(_bodyFilter)
            || v.BodyName.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)
            || v.VariantName.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)
            || v.Collection.Contains(_bodyFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults($"##BodyResults_{(isTarget ? "Target" : "Source")}_{row.Slot}", visibleBodies.Length, () =>
        {
            foreach (var item in visibleBodies)
            {
                var text = $"{item.BodyName} / {item.VariantName}";
                if (!string.Equals(item.Collection, item.BodyName, StringComparison.OrdinalIgnoreCase))
                    text += $"  [{item.Collection}]";
                if (!ImGui.Selectable(text, current == item))
                    continue;

                if (isTarget)
                {
                    row.Target = item;
                }
                else
                {
                    row.Source = item;
                    row.SourceUserOverride = true;
                    row.SourceInferred = false;
                    row.SourceAutoPriority = 100;
                }
                _bodyFilter = string.Empty;
            }
        });

        if (available.Length == 0)
            RavaFitUiChrome.DrawMutedText($"No {row.Slot.ToLowerInvariant()} variants are loaded for this gender.");
    }

    private void BuildOutfitChoices()
    {
        foreach (var choices in _outfitChoices.Values) choices.Clear();
        foreach (var choices in _accessoryChoices.Values) choices.Clear();
        if (_document is null) return;

        foreach (var group in _document.Groups.Where(group => group.SupportsAppend && group.Options.Count > 0))
        {
            foreach (var option in group.Options)
            {
                foreach (var model in _document.GetOptionModelRedirects(group.StableKey, option.StableKey))
                {
                    if (string.IsNullOrWhiteSpace(GetModelRaceCode(model.GamePath))) continue;
                    var bodySlot = GetModelSlot(model.GamePath);
                    if (bodySlot is not null) _outfitChoices[bodySlot].Add(new OutfitModelChoice(group, option, model, string.Empty));
                    else if (AccessoryModelSlots.FromGamePath(model.GamePath) is { } accessorySlot) _accessoryChoices[accessorySlot].Add(new OutfitModelChoice(group, option, model, string.Empty));
                }
            }
        }

        static void LabelChoices(Dictionary<string, List<OutfitModelChoice>> map, IEnumerable<string> slots)
        {
            foreach (var slot in slots)
            {
                var distinct = map[slot].GroupBy(choice => choice.Identity, StringComparer.OrdinalIgnoreCase).Select(group => group.First()).ToArray();
                var optionCounts = distinct.GroupBy(choice => $"{choice.Group.StableKey}|{choice.Option.StableKey}", StringComparer.OrdinalIgnoreCase).ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);
                var labelled = distinct.Select(choice =>
                {
                    var optionKey = $"{choice.Group.StableKey}|{choice.Option.StableKey}";
                    var label = $"{choice.Group.Name} / {choice.Option.Name}";
                    if (optionCounts[optionKey] > 1) label += $" · {choice.Model.FileName}";
                    return choice with { Label = label };
                }).OrderBy(choice => choice.Group.Name, StringComparer.OrdinalIgnoreCase).ThenBy(choice => choice.Option.Name, StringComparer.OrdinalIgnoreCase).ThenBy(choice => choice.Model.GamePath, StringComparer.OrdinalIgnoreCase).ToArray();
                map[slot].Clear(); map[slot].AddRange(labelled);
            }
        }
        LabelChoices(_outfitChoices, BodySlots.All);
        LabelChoices(_accessoryChoices, AccessoryModelSlots.All);
    }

    private void ResetOutfitRows(bool keepTargets)
    {
        foreach (var accessory in _accessoryRows.Values)
        {
            accessory.Cancellation?.Cancel(); accessory.Cancellation?.Dispose(); accessory.Cancellation = null;
            accessory.Revision++; accessory.Enabled = false; accessory.Selection = null; accessory.Analysis = null; accessory.Analysing = false; accessory.Status = string.Empty;
            foreach (var slot in BodySlots.All) { accessory.Sources[slot] = null; accessory.Targets[slot] = null; }
            accessory.SourceUserOverrides.Clear();
        }
        foreach (var row in _outfitRows.Values)
        {
            row.Cancellation?.Cancel();
            row.Cancellation?.Dispose();
            row.Cancellation = null;
            row.Revision++;
            row.Enabled = false;
            row.Selection = null;
            row.Source = null;
            if (!keepTargets)
                row.Target = null;
            row.Analysis = null;
            row.Analysing = false;
            row.SourceUserOverride = false;
            row.SourceInferred = false;
            row.SourceAutoPriority = 0;
            row.Status = string.Empty;
        }
    }

    private void ResetSwapRows(bool keepTargets)
    {
        foreach (var accessory in _swapAccessoryRows.Values)
        {
            accessory.Cancellation?.Cancel(); accessory.Cancellation?.Dispose(); accessory.Cancellation = null;
            accessory.Revision++; accessory.Enabled = false; accessory.Selection = null; accessory.Analysis = null; accessory.Analysing = false; accessory.Status = string.Empty;
            foreach (var slot in BodySlots.All) { accessory.Sources[slot] = null; accessory.Targets[slot] = null; }
            accessory.SourceUserOverrides.Clear();
        }
        foreach (var row in _swapRows.Values)
        {
            row.Cancellation?.Cancel();
            row.Cancellation?.Dispose();
            row.Cancellation = null;
            row.Revision++;
            row.Enabled = false;
            row.Selection = null;
            row.Source = null;
            if (!keepTargets)
                row.Target = null;
            row.Analysis = null;
            row.Analysing = false;
            row.SourceUserOverride = false;
            row.SourceInferred = false;
            row.SourceAutoPriority = 0;
            row.Status = string.Empty;
        }
    }

    private void SetSwapSelection(OutfitRowState row, OutfitModelChoice choice)
    {
        row.Cancellation?.Cancel();
        row.Cancellation?.Dispose();
        row.Cancellation = null;
        row.Selection = choice;
        row.Enabled = true;
        if (row.Target is not null && !IsValidSwapTargetVariant(row.Target, row.Slot, GetSwapTargetIdentity(), GetModelGender(row)))
            row.Target = null;
        row.Source = null;
        row.Analysis = null;
        row.Analysing = false;
        row.SourceUserOverride = false;
        row.SourceInferred = false;
        row.SourceAutoPriority = 0;
        row.Status = string.Empty;
        row.Revision++;
        StartSwapAnalysis(row);
    }

    private void StartSwapAnalysis(OutfitRowState row)
    {
        if (_selectedMod is null || row.Selection is null)
            return;
        if (!_plugin.ModelBridge.Status.Available || !_plugin.Solver.Ready)
        {
            row.Status = "Choose Source manually.";
            return;
        }

        row.Cancellation?.Cancel();
        row.Cancellation?.Dispose();
        row.Cancellation = new CancellationTokenSource();
        var token = row.Cancellation.Token;
        var revision = ++row.Revision;
        var identity = row.Selection.Identity;
        var mod = _selectedMod;
        var choice = row.Selection;
        row.Analysing = true;
        row.Status = "Checking...";
        _ = AnalyseSwapRowAsync(row.Slot, revision, identity, mod, choice, token);
    }

    private async Task AnalyseSwapRowAsync(string rowSlot, int revision, string identity, PenumbraModInfo mod, OutfitModelChoice choice, CancellationToken cancellationToken)
    {
        try
        {
            var result = await _plugin.Conversion.AnalyseCoverageAsync(mod, choice.Group.StableKey, choice.Option.StableKey, choice.Model, cancellationToken).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                var row = _swapRows[rowSlot];
                if (revision != row.Revision || !string.Equals(row.Selection?.Identity, identity, StringComparison.OrdinalIgnoreCase))
                    return;
                row.Analysing = false;
                row.Analysis = result;
                row.Status = string.Empty;
                if (result.SourceMatches.TryGetValue(rowSlot, out var match))
                {
                    var embedded = match.Reason.StartsWith("Embedded body", StringComparison.OrdinalIgnoreCase)
                        || match.Reason.StartsWith("Penumbra selection name", StringComparison.OrdinalIgnoreCase);
                    ApplyAutomaticSwapSourceMatch(match, embedded ? 4 : 2, inferred: !embedded);
                }
            });
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _uiActions.Enqueue(() =>
            {
                var row = _swapRows[rowSlot];
                if (revision != row.Revision || !string.Equals(row.Selection?.Identity, identity, StringComparison.OrdinalIgnoreCase))
                    return;
                row.Analysing = false;
                row.Status = $"Source detection failed: {ex.Message}";
            });
        }
    }

    private void ApplyAutomaticSwapSourceMatch(SourceBodyMatch match, int priority, bool inferred)
    {
        if (!_swapRows.TryGetValue(match.Slot, out var contextRow) || contextRow.SourceUserOverride)
            return;
        if (contextRow.Source is not null && contextRow.SourceAutoPriority >= priority)
            return;
        contextRow.Source = match.Variant;
        contextRow.SourceInferred = inferred;
        contextRow.SourceAutoPriority = priority;
    }

    private bool TryBuildSwapSlotSelections(OutfitRowState row, out IReadOnlyList<SlotConversionSelection> selections, out string reason)
    {
        selections = Array.Empty<SlotConversionSelection>();
        reason = string.Empty;
        if (row.Selection is null || row.Source is null || row.Target is null)
        {
            reason = $"Complete Selection, Source and Target for {row.Slot}.";
            return false;
        }

        var sourceRaceCode = GetModelRaceCode(row.Selection.Model.GamePath);
        var sourceIdentity = CharacterRaceCatalog.FromCode(sourceRaceCode);
        if (sourceIdentity is null)
        {
            reason = $"{row.Slot} selection does not expose a supported character race.";
            return false;
        }
        var target = GetSwapTargetIdentity();
        var required = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { row.Slot };
        if (row.Analysis is not null)
            foreach (var evidence in row.Analysis.Slots.Values.Where(evidence => evidence.Primary || evidence.Recommended))
                required.Add(evidence.Slot);

        var built = new List<SlotConversionSelection>();
        foreach (var slot in BodySlots.All.Where(required.Contains))
        {
            BodyVariantInfo? source = null;
            BodyVariantInfo? targetVariant = null;
            var context = _swapRows[slot];
            if (context.Source is not null) source = context.Source;
            if (source is null && row.Analysis is not null && row.Analysis.SourceMatches.TryGetValue(slot, out var inferred)) source = inferred.Variant;
            if (source is null) source = string.Equals(slot, row.Slot, StringComparison.OrdinalIgnoreCase) ? row.Source : _plugin.Bodies.FindSibling(row.Source, slot);
            if (context.Target is not null) targetVariant = context.Target;
            if (targetVariant is null) targetVariant = string.Equals(slot, row.Slot, StringComparison.OrdinalIgnoreCase) ? row.Target : _plugin.Bodies.FindSibling(row.Target, slot);
            if (source is null || !IsValidSwapSourceVariant(source, slot, sourceIdentity, target))
            {
                reason = CharacterRaceCatalog.RequiresSmallclothesSource(sourceIdentity, target.Gender, slot)
                    ? $"Select an SFW {slot} Source body for male-to-female conversion."
                    : $"Pick a {sourceIdentity.Gender.ToLowerInvariant()} {slot} source body.";
                return false;
            }
            if (targetVariant is null || !IsValidSwapTargetVariant(targetVariant, slot, target, sourceIdentity.Gender))
            {
                reason = CharacterRaceCatalog.RequiresSmallclothesTarget(sourceIdentity.Gender, target, slot)
                    ? $"Select an SFW {slot} Target body for {target.DisplayName}."
                    : $"Select a {slot} Target body for {target.DisplayName}.";
                return false;
            }
            built.Add(new SlotConversionSelection(slot, source, targetVariant, sourceRaceCode));
        }
        selections = built;
        return true;
    }

    private static string? GetModelGender(OutfitRowState row)
        => CharacterRaceCatalog.FromCode(row.Selection is null ? null : GetModelRaceCode(row.Selection.Model.GamePath))?.Gender;

    private string? GetOutfitContextGender(OutfitRowState row)
    {
        var direct = GetModelGender(row);
        if (!string.IsNullOrWhiteSpace(direct))
            return direct;

        return _outfitRows.Values
            .Where(candidate => candidate.Enabled && candidate.Selection is not null)
            .Select(GetModelGender)
            .FirstOrDefault(gender => !string.IsNullOrWhiteSpace(gender));
    }

    private static bool IsValidSwapSourceVariant(BodyVariantInfo variant, string slot, CharacterRaceIdentity? source, CharacterRaceIdentity target)
        => source is not null
           && variant.SupportsGender(source.Gender)
           && CharacterRaceCatalog.BodySupports(variant, source.Code)
           && (!CharacterRaceCatalog.RequiresSmallclothesSource(source, target.Gender, slot) || variant.IsSmallclothesSupport);

    private static bool IsValidSwapTargetVariant(BodyVariantInfo variant, string slot, CharacterRaceIdentity target, string? sourceGender)
        => variant.SupportsGender(target.Gender)
           && CharacterRaceCatalog.BodySupports(variant, target.Code)
           && (!CharacterRaceCatalog.RequiresSmallclothesTarget(sourceGender, target, slot) || variant.IsSmallclothesSupport);

    private static string GetRaceSwapOutputOptionName(BodyVariantInfo target, CharacterRaceIdentity identity)
        => $"{GetOutputOptionName(target)} - {identity.DisplayName}";

    private void SetAccessorySelection(AccessoryOutfitRowState row, OutfitModelChoice choice, bool swap)
    {
        row.Cancellation?.Cancel(); row.Cancellation?.Dispose(); row.Cancellation = null;
        row.Selection = choice; row.Enabled = true; row.Analysis = null; row.Analysing = false; row.Status = string.Empty; row.Revision++;
        foreach (var slot in BodySlots.All) { row.Sources[slot] = null; row.Targets[slot] = null; }
        row.SourceUserOverrides.Clear();
        if (_selectedMod is null || !_plugin.ModelBridge.Status.Available || !_plugin.Solver.Ready)
        {
            row.Status = "Coverage analysis is not ready.";
            return;
        }
        row.Cancellation = new CancellationTokenSource();
        var token = row.Cancellation.Token; var revision = ++row.Revision; var identity = choice.Identity; var mod = _selectedMod;
        row.Analysing = true; row.Status = "Checking garment coverage...";
        _ = AnalyseAccessoryRowAsync(row.Slot, revision, identity, mod, choice, swap, token);
    }

    private async Task AnalyseAccessoryRowAsync(string accessorySlot, int revision, string identity, PenumbraModInfo mod, OutfitModelChoice choice, bool swap, CancellationToken cancellationToken)
    {
        try
        {
            var result = await _plugin.Conversion.AnalyseCoverageAsync(mod, choice.Group.StableKey, choice.Option.StableKey, choice.Model, cancellationToken, detectSourceBodies: true).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                var row = (swap ? _swapAccessoryRows : _accessoryRows)[accessorySlot];
                if (revision != row.Revision || !string.Equals(row.Selection?.Identity, identity, StringComparison.OrdinalIgnoreCase)) return;
                row.Analysing = false; row.Analysis = result; row.Status = string.Empty;
                foreach (var match in result.SourceMatches.Values)
                {
                    if (row.Sources.ContainsKey(match.Slot) && !row.SourceUserOverrides.Contains(match.Slot))
                        row.Sources[match.Slot] = match.Variant;
                }
            });
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _uiActions.Enqueue(() =>
            {
                var row = (swap ? _swapAccessoryRows : _accessoryRows)[accessorySlot];
                if (revision != row.Revision || !string.Equals(row.Selection?.Identity, identity, StringComparison.OrdinalIgnoreCase)) return;
                row.Analysing = false; row.Status = "Accessory coverage failed: " + ex.Message;
            });
        }
    }

    private bool TryBuildAccessorySlotSelections(AccessoryOutfitRowState row, bool swap, out IReadOnlyList<SlotConversionSelection> selections, out string reason)
    {
        selections = Array.Empty<SlotConversionSelection>(); reason = string.Empty;
        if (row.Selection is null || row.Analysis is null) { reason = $"Wait for {row.Slot} coverage analysis."; return false; }
        var raceCode = GetModelRaceCode(row.Selection.Model.GamePath);
        var sourceIdentity = CharacterRaceCatalog.FromCode(raceCode);
        if (sourceIdentity is null) { reason = $"{row.Slot} selection does not expose a supported human race."; return false; }
        var required = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { row.Analysis.PrimarySlot };
        foreach (var evidence in row.Analysis.Slots.Values.Where(e => e.Primary || e.Recommended)) required.Add(evidence.Slot);
        var built = new List<SlotConversionSelection>();
        if (!swap)
        {
            foreach (var slot in BodySlots.All.Where(required.Contains))
            {
                BodyVariantInfo? source = row.Sources[slot];
                if (source is null && row.Analysis.SourceMatches.TryGetValue(slot, out var inferred)) source = inferred.Variant;
                var target = row.Targets[slot];
                if (!_convertUseVanilla && (source is null || !source.SupportsGender(sourceIdentity.Gender))) { reason = $"Accessory {row.Slot} needs a {slot} source body. Pick it under the accessory."; return false; }
                if (target is null || !target.SupportsGender(sourceIdentity.Gender)) { reason = $"Accessory {row.Slot} needs a {slot} target body. Pick it under the accessory."; return false; }
                built.Add(new SlotConversionSelection(slot, _convertUseVanilla ? null : source, target, raceCode));
            }
        }
        else
        {
            var targetIdentity = GetSwapTargetIdentity();
            foreach (var slot in BodySlots.All.Where(required.Contains))
            {
                BodyVariantInfo? source = row.Sources[slot];
                if (source is null && row.Analysis.SourceMatches.TryGetValue(slot, out var inferred)) source = inferred.Variant;
                var target = row.Targets[slot];
                if (source is null || !IsValidSwapSourceVariant(source, slot, sourceIdentity, targetIdentity)) { reason = $"Accessory {row.Slot} also needs a valid {slot} source body."; return false; }
                if (target is null || !IsValidSwapTargetVariant(target, slot, targetIdentity, sourceIdentity.Gender)) { reason = $"Accessory {row.Slot} also needs a valid {slot} target body."; return false; }
                built.Add(new SlotConversionSelection(slot, source, target, raceCode));
            }
        }
        selections = built; return true;
    }

    private void SetOutfitSelection(OutfitRowState row, OutfitModelChoice choice)
    {
        row.Cancellation?.Cancel();
        row.Cancellation?.Dispose();
        row.Cancellation = null;
        row.Selection = choice;
        row.Enabled = true;
        row.Source = null;
        var gender = GetModelGender(row);
        if (row.Target is not null && !row.Target.SupportsGender(gender))
            row.Target = null;
        row.Analysis = null;
        row.Analysing = false;
        row.SourceUserOverride = false;
        row.SourceInferred = false;
        row.SourceAutoPriority = 0;
        row.Status = string.Empty;
        row.Revision++;
        StartOutfitAnalysis(row);
    }

    private void StartOutfitAnalysis(OutfitRowState row)
    {
        if (_selectedMod is null || row.Selection is null)
            return;
        if (!_plugin.ModelBridge.Status.Available || !_plugin.Solver.Ready)
        {
            row.Status = _convertUseVanilla
                ? "Fit check is not ready yet."
                : "Couldn't detect the source body. Pick one manually.";
            return;
        }

        row.Cancellation?.Cancel();
        row.Cancellation?.Dispose();
        row.Cancellation = new CancellationTokenSource();
        var token = row.Cancellation.Token;
        var revision = ++row.Revision;
        var identity = row.Selection.Identity;
        var mod = _selectedMod;
        var choice = row.Selection;
        var detectSourceBodies = !_convertUseVanilla;
        row.Analysing = true;
        row.Status = detectSourceBodies ? "Checking source body..." : "Checking fit...";
        _ = AnalyseOutfitRowAsync(row.Slot, revision, identity, mod, choice, detectSourceBodies, token);
    }

    private async Task AnalyseOutfitRowAsync(string rowSlot, int revision, string identity, PenumbraModInfo mod, OutfitModelChoice choice, bool detectSourceBodies, CancellationToken cancellationToken)
    {
        try
        {
            var result = await _plugin.Conversion.AnalyseCoverageAsync(mod, choice.Group.StableKey, choice.Option.StableKey, choice.Model, cancellationToken, detectSourceBodies).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                var row = _outfitRows[rowSlot];
                if (revision != row.Revision || !string.Equals(row.Selection?.Identity, identity, StringComparison.OrdinalIgnoreCase))
                    return;

                row.Analysing = false;
                row.Analysis = result;
                row.Status = string.Empty;

                if (detectSourceBodies)
                {
                    // Evidence order is manual > explicit/embedded evidence > garment-only inference, scoped to this row.
                    if (result.SourceMatches.TryGetValue(rowSlot, out var match))
                    {
                        var embedded = match.Reason.StartsWith("Embedded body", StringComparison.OrdinalIgnoreCase)
                            || match.Reason.StartsWith("Penumbra selection name", StringComparison.OrdinalIgnoreCase);
                        ApplyAutomaticSourceMatch(match, embedded ? 4 : 2, inferred: !embedded);
                    }
                }
            });
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _uiActions.Enqueue(() =>
            {
                var row = _outfitRows[rowSlot];
                if (revision != row.Revision || !string.Equals(row.Selection?.Identity, identity, StringComparison.OrdinalIgnoreCase))
                    return;
                row.Analysing = false;
                row.Status = detectSourceBodies ? $"Couldn't detect the source body: {ex.Message}" : $"Couldn't check the fit: {ex.Message}";
                // Keep source selection local to this row; a failed guess must not alter sibling rows.
            });
        }
    }

    private void ApplyAutomaticSourceMatch(SourceBodyMatch match, int priority, bool inferred)
    {
        if (!_outfitRows.TryGetValue(match.Slot, out var contextRow) || contextRow.SourceUserOverride)
            return;
        if (contextRow.Source is not null && contextRow.SourceAutoPriority >= priority)
            return;

        contextRow.Source = match.Variant;
        contextRow.SourceInferred = inferred;
        contextRow.SourceAutoPriority = priority;
    }

    private bool TryBuildSlotSelections(OutfitRowState row, out IReadOnlyList<SlotConversionSelection> selections, out string reason)
    {
        selections = Array.Empty<SlotConversionSelection>();
        reason = string.Empty;
        if (row.Selection is null || row.Target is null)
        {
            reason = _convertUseVanilla ? $"Complete Selection and Target for {row.Slot}." : $"Complete Selection, Source and Target for {row.Slot}.";
            return false;
        }
        var raceCode = GetModelRaceCode(row.Selection.Model.GamePath);
        var identity = CharacterRaceCatalog.FromCode(raceCode);
        if (identity is null)
        {
            reason = $"{row.Slot} selection does not expose a supported c#### human race code.";
            return false;
        }

        if (_convertUseVanilla)
        {
            if (!row.Target.SupportsGender(identity.Gender))
            {
                reason = $"Select a {identity.Gender.ToLowerInvariant()} {row.Slot} Target body.";
                return false;
            }
            if (row.Analysis is null)
            {
                reason = $"Wait for the {row.Slot} fit check to finish.";
                return false;
            }

            var vanillaRequiredSlots = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { row.Slot };
            foreach (var evidence in row.Analysis.Slots.Values.Where(evidence => evidence.Primary || evidence.Recommended))
                vanillaRequiredSlots.Add(evidence.Slot);

            var vanillaBuiltSelections = new List<SlotConversionSelection>();
            foreach (var slot in BodySlots.All.Where(vanillaRequiredSlots.Contains))
            {
                var context = _outfitRows[slot];
                var target = context.Target;
                if (target is null) target = string.Equals(slot, row.Slot, StringComparison.OrdinalIgnoreCase) ? row.Target : _plugin.Bodies.FindSibling(row.Target, slot);
                if (target is null || !target.SupportsGender(identity.Gender))
                {
                    reason = $"This {row.Slot.ToLowerInvariant()} also needs a {slot.ToLowerInvariant()} body. Pick one below.";
                    return false;
                }
                vanillaBuiltSelections.Add(new SlotConversionSelection(slot, null, target, raceCode));
            }
            selections = vanillaBuiltSelections;
            return true;
        }

        if (row.Source is null)
        {
            reason = $"Select a Source body for {row.Slot}.";
            return false;
        }

        var required = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { row.Slot };
        if (row.Analysis is not null)
            foreach (var evidence in row.Analysis.Slots.Values.Where(evidence => evidence.Primary || evidence.Recommended))
                required.Add(evidence.Slot);
        var built = new List<SlotConversionSelection>();
        foreach (var slot in BodySlots.All.Where(required.Contains))
        {
            BodyVariantInfo? source = null;
            BodyVariantInfo? target = null;
            var outfitContext = _outfitRows[slot];
            if (outfitContext.Source is not null) source = outfitContext.Source;
            if (source is null && row.Analysis is not null && row.Analysis.SourceMatches.TryGetValue(slot, out var inferred)) source = inferred.Variant;
            if (source is null) source = string.Equals(slot, row.Slot, StringComparison.OrdinalIgnoreCase) ? row.Source : _plugin.Bodies.FindSibling(row.Source, slot);
            if (outfitContext.Target is not null) target = outfitContext.Target;
            if (target is null) target = string.Equals(slot, row.Slot, StringComparison.OrdinalIgnoreCase) ? row.Target : _plugin.Bodies.FindSibling(row.Target, slot);
            if (source is null || !source.SupportsGender(identity.Gender))
            {
                reason = $"This {row.Slot.ToLowerInvariant()} also needs a {slot.ToLowerInvariant()} source body. Pick one below.";
                return false;
            }
            if (target is null || !target.SupportsGender(identity.Gender))
            {
                reason = $"This {row.Slot.ToLowerInvariant()} also needs a {slot.ToLowerInvariant()} body. Pick one below.";
                return false;
            }
            built.Add(new SlotConversionSelection(slot, source, target, raceCode));
        }
        selections = built;
        return true;
    }

    private static string GetOutputOptionName(BodyVariantInfo target)
        => string.Join(" ", new[] { target.BodyName.Trim(), target.VariantName.Trim() }.Where(value => !string.IsNullOrWhiteSpace(value))).Trim();

    private static string? GetModelRaceCode(string gamePath)
    {
        var path = gamePath.Replace('\\', '/');
        var slash = path.LastIndexOf('/');
        var file = slash >= 0 ? path[(slash + 1)..] : path;
        if (file.Length < 5 || char.ToLowerInvariant(file[0]) != 'c')
            return null;
        var value = file.AsSpan(1, 4);
        return value[0] is >= '0' and <= '9' && value[1] is >= '0' and <= '9' && value[2] is >= '0' and <= '9' && value[3] is >= '0' and <= '9'
            ? value.ToString()
            : null;
    }

    private static string? GetModelSlot(string gamePath)
    {
        var path = gamePath.Replace('\\', '/');
        if (path.EndsWith("_top.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Chest;
        if (path.EndsWith("_dwn.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Legs;
        if (path.EndsWith("_glv.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Hands;
        if (path.EndsWith("_sho.mdl", StringComparison.OrdinalIgnoreCase)) return BodySlots.Feet;
        return null;
    }

    private static string? GetModelContainerSlot(string gamePath)
        => GetModelSlot(gamePath) ?? AccessoryModelSlots.FromGamePath(gamePath);

    private void DrawBodyCatalogueTab()
    {
        RavaFitUiChrome.DrawSectionTitle("Add a body", "Import a body option from an installed Penumbra mod.");
        using (RavaFitUiChrome.BeginCard("##RavaFitBodyCatalogueImportCard", 248f * ImGuiHelpers.GlobalScale, allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            if (!_plugin.Penumbra.Available)
            {
                ImGui.TextUnformatted("Penumbra is not available.");
                if (ImGui.Button("Refresh Penumbra"))
                    _plugin.Penumbra.Refresh();
            }
            else if (ImGui.BeginTable("##RavaFitBodyImportFields", 2, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("##BodyImportLabel", ImGuiTableColumnFlags.WidthFixed, GetFieldColumnWidth());
                ImGui.TableSetupColumn("##BodyImportValue", ImGuiTableColumnFlags.WidthStretch, 1f);
                DrawFieldRow("Body mod", DrawBodyImportModPicker);
                DrawFieldRow("Option", DrawBodyImportChoicePicker);
                DrawFieldRow("Body name", () =>
                {
                    ImGui.SetNextItemWidth(-1);
                    ImGui.InputText("##BodyImportBodyName", ref _bodyImportBodyName, 128);
                });
                DrawFieldRow("Variant", () =>
                {
                    ImGui.SetNextItemWidth(-1);
                    ImGui.InputText("##BodyImportVariantName", ref _bodyImportVariantName, 128);
                });
                ImGui.EndTable();

                ImGuiHelpers.ScaledDummy(6f);
                if (_bodyImportChoice is not null)
                    RavaFitUiChrome.DrawMutedText(DescribeBodyImportChoice(_bodyImportChoice));

                var canImport = !_bodyImportBusy
                    && !_plugin.Conversion.Busy
                    && _plugin.Solver.Ready
                    && _bodyImportMod is not null
                    && _bodyImportChoice is not null
                    && !string.IsNullOrWhiteSpace(_bodyImportBodyName)
                    && !string.IsNullOrWhiteSpace(_bodyImportVariantName);
                using (ImRaii.Disabled(!canImport))
                {
                    if (ImGui.Button(_bodyImportBusy ? "Adding..." : "Add to catalogue", new Vector2(-1, 34f * ImGuiHelpers.GlobalScale)))
                        StartBodyImport();
                }
            }

            DrawMessage();
        }

        RavaFitUiChrome.DrawSectionTitle("Your bodies");
        using (RavaFitUiChrome.BeginCard("##RavaFitCustomBodiesCard", GetWorkCardHeight(180f), allowScroll: true, resetScroll: _resetScrollThisFrame))
        {
            var userVariants = _plugin.Bodies.UserVariants;
            if (userVariants.Count == 0)
            {
                RavaFitUiChrome.DrawMutedText("No custom bodies added yet.");
                return;
            }

            if (ImGui.BeginTable("##RavaFitCustomBodies", 3, ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.RowBg | ImGuiTableFlags.NoSavedSettings))
            {
                ImGui.TableSetupColumn("Body", ImGuiTableColumnFlags.WidthStretch, 1.15f);
                ImGui.TableSetupColumn("Slot", ImGuiTableColumnFlags.WidthFixed, 76f * ImGuiHelpers.GlobalScale);
                ImGui.TableSetupColumn("Variants", ImGuiTableColumnFlags.WidthStretch, 1.5f);
                ImGui.TableHeadersRow();

                foreach (var body in userVariants.GroupBy(v => new { v.BodyId, v.BodyName }).OrderBy(group => group.Key.BodyName, StringComparer.OrdinalIgnoreCase))
                {
                    var firstRow = true;
                    foreach (var slot in BodySlots.All)
                    {
                        var variants = body.Where(v => string.Equals(v.Slot, slot, StringComparison.OrdinalIgnoreCase))
                            .OrderBy(v => v.VariantName, StringComparer.OrdinalIgnoreCase)
                            .ToArray();
                        if (variants.Length == 0)
                            continue;

                        ImGui.TableNextRow();
                        ImGui.TableNextColumn();
                        ImGui.TextUnformatted(firstRow ? body.Key.BodyName : string.Empty);
                        ImGui.TableNextColumn();
                        ImGui.TextUnformatted(slot);
                        ImGui.TableNextColumn();
                        ImGui.TextUnformatted(string.Join(", ", variants.Select(v => v.VariantName)));
                        firstRow = false;
                    }
                }

                ImGui.EndTable();
            }
        }
    }

    private void DrawBodyImportModPicker()
    {
        if (_bodyImportMod is null && _plugin.Penumbra.Mods.Count == 1) SelectBodyImportMod(_plugin.Penumbra.Mods[0]);
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##BodyImportMod", _bodyImportMod?.Name ?? "Select a mod");
        if (!combo.Success)
            return;

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##BodyImportModFilter", "Search mods...", ref _bodyImportModFilter, 128);
        ImGui.Separator();
        var visibleMods = _plugin.Penumbra.Mods.Where(mod => string.IsNullOrWhiteSpace(_bodyImportModFilter) || mod.Name.Contains(_bodyImportModFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults("##BodyImportModResults", visibleMods.Length, () =>
        {
            foreach (var mod in visibleMods)
            {
                if (!ImGui.Selectable(mod.Name, string.Equals(_bodyImportMod?.Directory, mod.Directory, StringComparison.OrdinalIgnoreCase)))
                    continue;
                SelectBodyImportMod(mod);
                _bodyImportModFilter = string.Empty;
            }
        });
    }

    private void DrawBodyImportChoicePicker()
    {
        if (_bodyImportChoice is null && _bodyImportChoices.Count == 1) { _bodyImportChoice = _bodyImportChoices[0]; _bodyImportVariantName = _bodyImportChoice.Option.Name; }
        ImGui.SetNextItemWidth(-1);
        using var combo = ImRaii.Combo("##BodyImportChoice", _bodyImportChoice?.Label ?? (_bodyImportMod is null ? "Select a body mod first" : "Select an option"));
        if (!combo.Success)
            return;

        ImGui.SetNextItemWidth(-1);
        ImGui.InputTextWithHint("##BodyImportChoiceFilter", "Search options...", ref _bodyImportChoiceFilter, 128);
        ImGui.Separator();
        var visibleChoices = _bodyImportChoices.Where(choice => string.IsNullOrWhiteSpace(_bodyImportChoiceFilter)
            || choice.Group.Name.Contains(_bodyImportChoiceFilter, StringComparison.OrdinalIgnoreCase)
            || choice.Option.Name.Contains(_bodyImportChoiceFilter, StringComparison.OrdinalIgnoreCase)).ToArray();
        DrawPinnedSearchResults("##BodyImportChoiceResults", visibleChoices.Length, () =>
        {
            foreach (var choice in visibleChoices)
            {
                if (!ImGui.Selectable(choice.Label, string.Equals(_bodyImportChoice?.Identity, choice.Identity, StringComparison.OrdinalIgnoreCase)))
                    continue;
                _bodyImportChoice = choice;
                _bodyImportVariantName = choice.Option.Name;
                _bodyImportChoiceFilter = string.Empty;
            }
        });

        if (_bodyImportChoices.Count == 0)
            RavaFitUiChrome.DrawMutedText("No body options found in this mod.");
    }

    private static string DescribeBodyImportChoice(BodyImportChoice choice)
        => string.Join(" · ", choice.Models
            .GroupBy(model => model.Slot)
            .OrderBy(group => Array.IndexOf(BodySlots.All, group.Key))
            .Select(group => $"{group.Key}: {string.Join(", ", group.Select(model => model.RaceCode).Distinct().OrderBy(code => code).Select(code => CharacterRaceCatalog.FromCode(code)?.DisplayName ?? $"c{code}"))}"));

    private void DrawRaceSwapConversionControls()
    {
        ImGuiHelpers.ScaledDummy(9f);
        ImGui.Separator();
        ImGuiHelpers.ScaledDummy(7f);

        var ready = CanRaceSwap(out var reason);
        var progress = _plugin.Conversion.Progress;
        if (_plugin.Conversion.Busy || progress.Stage is not ConversionStage.Idle)
        {
            DrawConversionActivity(progress);
            ImGui.ProgressBar(progress.Progress, new Vector2(-1, 0), string.Empty);
            ImGuiHelpers.ScaledDummy(5f);
        }
        else if (!ready && !string.IsNullOrWhiteSpace(reason))
        {
            RavaFitUiChrome.DrawMutedText(reason);
            ImGuiHelpers.ScaledDummy(5f);
        }

        using (ImRaii.Disabled(!ready || _plugin.Conversion.Busy))
        {
            var label = _plugin.Conversion.Busy ? "Working..." : "Convert selected";
            if (ImGui.Button(label, new Vector2(-1, 36f * ImGuiHelpers.GlobalScale)))
                StartRaceSwapConversion();
        }
        DrawMessage();
    }

    private bool CanRaceSwap(out string reason)
    {
        if (_selectedMod is null || _document is null)
        {
            reason = "Select a mod.";
            return false;
        }
        var analysing = _swapRows.Values.FirstOrDefault(row => row.Selection is not null && row.Analysing);
        if (analysing is not null)
        {
            reason = $"Checking {analysing.Slot}...";
            return false;
        }
        var enabledRows = _swapRows.Values.Where(row => row.Enabled).ToArray();
        var enabledAccessories = _swapAccessoryRows.Values.Where(row => row.Enabled).ToArray();
        if (enabledRows.Length == 0 && enabledAccessories.Length == 0)
        {
            reason = "Choose at least one outfit row.";
            return false;
        }
        var identity = GetSwapTargetIdentity();
        foreach (var row in enabledRows)
        {
            if (row.Selection is null || row.Source is null || row.Target is null)
            {
                reason = $"Complete {row.Slot}.";
                return false;
            }
            var sourceIdentity = CharacterRaceCatalog.FromCode(GetModelRaceCode(row.Selection.Model.GamePath));
            if (sourceIdentity is null || !IsValidSwapSourceVariant(row.Source, row.Slot, sourceIdentity, identity))
            {
                reason = sourceIdentity is not null && CharacterRaceCatalog.RequiresSmallclothesSource(sourceIdentity, identity.Gender, row.Slot)
                    ? $"Choose an SFW {row.Slot} source for male-to-female conversion."
                    : $"Choose a valid {row.Slot} source body.";
                return false;
            }
            if (!IsValidSwapTargetVariant(row.Target, row.Slot, identity, GetModelGender(row)))
            {
                reason = CharacterRaceCatalog.RequiresSmallclothesTarget(GetModelGender(row), identity, row.Slot)
                    ? $"Choose an SFW {row.Slot} target for {identity.DisplayName}."
                    : $"Choose a {row.Slot} target for {identity.DisplayName}.";
                return false;
            }
            if (!TryBuildSwapSlotSelections(row, out _, out reason))
                return false;
        }

        var conflict = enabledRows
            .GroupBy(row => $"{row.Selection!.Group.StableKey}|{GetRaceSwapOutputOptionName(row.Target!, identity)}", StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(group => group.Select(row => row.Selection!.Option.StableKey).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1);
        if (conflict is not null)
        {
            reason = "Selected rows use different source options in the same group.";
            return false;
        }
        foreach (var accessory in enabledAccessories)
        {
            if (accessory.Selection is null || accessory.Analysing || accessory.Analysis is null)
            {
                reason = accessory.Analysing ? $"Checking {accessory.Slot} garment coverage..." : $"Choose and analyse a {accessory.Slot} accessory garment.";
                return false;
            }
            if (!TryBuildAccessorySlotSelections(accessory, swap: true, out _, out reason)) return false;
            var target = accessory.Targets[accessory.Analysis.PrimarySlot];
            if (target is null) { reason = $"Pick a {accessory.Analysis.PrimarySlot} target body under {accessory.Slot}."; return false; }
            var outputName = GetRaceSwapOutputOptionName(target, identity);
            if (accessory.Selection.Group.Options.Any(option => string.Equals(option.Name, outputName, StringComparison.OrdinalIgnoreCase)))
            { reason = $"'{outputName}' already exists."; return false; }
        }
        foreach (var row in enabledRows)
        {
            var outputName = GetRaceSwapOutputOptionName(row.Target!, identity);
            if (row.Selection!.Group.Options.Any(option => string.Equals(option.Name, outputName, StringComparison.OrdinalIgnoreCase)))
            {
                reason = $"'{outputName}' already exists.";
                return false;
            }
        }
        if (!_plugin.ModelBridge.Status.Available)
        {
            reason = "Model bridge unavailable.";
            return false;
        }
        if (!_plugin.Solver.Ready || !_plugin.Solver.ConversionReady)
        {
            reason = "Solver not ready.";
            return false;
        }
        reason = string.Empty;
        return true;
    }

    private void DrawConversionControls()
    {
        ImGuiHelpers.ScaledDummy(9f);
        ImGui.Separator();
        ImGuiHelpers.ScaledDummy(7f);

        var ready = CanConvert(out var reason);
        var progress = _plugin.Conversion.Progress;
        if (_plugin.Conversion.Busy || progress.Stage is not ConversionStage.Idle)
        {
            DrawConversionActivity(progress);
            ImGui.ProgressBar(progress.Progress, new Vector2(-1, 0), string.Empty);
            ImGuiHelpers.ScaledDummy(5f);
        }
        else if (!ready && !string.IsNullOrWhiteSpace(reason))
        {
            RavaFitUiChrome.DrawMutedText(reason);
            ImGuiHelpers.ScaledDummy(5f);
        }

        using (ImRaii.Disabled(!ready || _plugin.Conversion.Busy))
        {
            var label = _plugin.Conversion.Busy ? "Working..." : "Convert selected";
            if (ImGui.Button(label, new Vector2(-1, 36f * ImGuiHelpers.GlobalScale)))
                StartConversion();
        }

        DrawMessage();
    }

    private void DrawConversionActivity(ConversionProgress progress)
    {
        if (_plugin.Conversion.Busy)
        {
            DrawActivitySpinner();
            ImGui.SameLine();
        }

        var text = GetFriendlyProgressText(progress);
        if (_plugin.Conversion.Busy && _conversionBusyStartedAt >= 0d)
        {
            var elapsed = TimeSpan.FromSeconds(Math.Max(0d, ImGui.GetTime() - _conversionBusyStartedAt));
            text += elapsed.TotalMinutes >= 1d
                ? $"  ·  {elapsed.Minutes:00}:{elapsed.Seconds:00} elapsed"
                : $"  ·  {elapsed.Seconds:00}s elapsed";
        }
        RavaFitUiChrome.DrawMutedText(text);
    }

    private static void DrawActivitySpinner()
    {
        var scale = ImGuiHelpers.GlobalScale;
        var radius = 6f * scale;
        var dotRadius = 1.35f * scale;
        var size = new Vector2((radius * 2f) + (4f * scale), (radius * 2f) + (4f * scale));
        var min = ImGui.GetCursorScreenPos();
        var centre = min + (size * 0.5f);
        var draw = ImGui.GetWindowDrawList();
        var phase = (float)(ImGui.GetTime() * 7.5d);
        const int dots = 10;
        for (var i = 0; i < dots; i++)
        {
            var angle = ((MathF.PI * 2f * i) / dots) - (MathF.PI * 0.5f);
            var head = ((phase + i) % dots) / dots;
            var alpha = 0.18f + (0.82f * head);
            var position = centre + new Vector2(MathF.Cos(angle), MathF.Sin(angle)) * radius;
            draw.AddCircleFilled(position, dotRadius, ImGui.GetColorU32(new Vector4(0.72f, 0.50f, 0.95f, alpha)));
        }
        ImGui.Dummy(size);
    }

    private static string GetFriendlyProgressText(ConversionProgress progress)
    {
        var detail = progress.Detail?.Trim() ?? string.Empty;
        var split = detail.Split(" — ", 2, StringSplitOptions.TrimEntries);
        var context = split.Length > 1 ? split[0] : string.Empty;
        return progress.Stage switch
        {
            ConversionStage.Preparing => string.IsNullOrWhiteSpace(context) ? "Preparing outfit..." : $"Preparing {context}...",
            ConversionStage.Exporting => string.IsNullOrWhiteSpace(context) ? "Reading outfit..." : $"Reading {context}...",
            ConversionStage.Fitting => string.IsNullOrWhiteSpace(context) ? "Fitting outfit..." : $"Fitting {context}...",
            ConversionStage.Building => string.IsNullOrWhiteSpace(context) ? "Building final model..." : $"Building {context}...",
            ConversionStage.UpdatingPenumbra => "Updating mod...",
            ConversionStage.Complete => "Done",
            ConversionStage.Failed => "Conversion failed",
            _ => string.IsNullOrWhiteSpace(detail) ? "Working..." : detail,
        };
    }

    private void DrawProgress()
    {
        if (_plugin.Conversion.Progress.Stage is ConversionStage.Idle)
            return;
        ImGuiHelpers.ScaledDummy(7f);
        ImGui.ProgressBar(_plugin.Conversion.Progress.Progress, new Vector2(-1, 0), _plugin.Conversion.Progress.Detail);
    }

    private void DrawMessage()
    {
        if (string.IsNullOrWhiteSpace(_uiMessage))
            return;
        ImGuiHelpers.ScaledDummy(6f);
        ImGui.PushStyleColor(ImGuiCol.Text, _uiMessageError ? ErrorColour : SuccessColour);
        ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + MathF.Max(1f, ImGui.GetContentRegionAvail().X));
        ImGui.TextWrapped(_uiMessage);
        ImGui.PopTextWrapPos();
        ImGui.PopStyleColor();
    }

    private bool CanConvert(out string reason)
    {
        if (_selectedMod is null || _document is null)
        {
            reason = "Select a mod.";
            return false;
        }

        var analysingContext = _outfitRows.Values.FirstOrDefault(row => row.Selection is not null && row.Analysing);
        if (analysingContext is not null)
        {
            reason = _convertUseVanilla
                ? $"Checking {analysingContext.Slot} fit..."
                : $"Checking {analysingContext.Slot} source body...";
            return false;
        }

        var enabledRows = _outfitRows.Values.Where(row => row.Enabled).ToArray();
        var enabledAccessories = _accessoryRows.Values.Where(row => row.Enabled).ToArray();
        if (enabledRows.Length == 0 && enabledAccessories.Length == 0)
        {
            reason = "Choose at least one outfit row to refit.";
            return false;
        }
        foreach (var row in enabledRows)
        {
            if (row.Selection is null)
            {
                reason = $"Choose a {row.Slot} Selection.";
                return false;
            }
            if (row.Analysing)
            {
                reason = _convertUseVanilla ? $"Checking {row.Slot} fit..." : $"Checking {row.Slot} source body...";
                return false;
            }
            if (row.Target is null || (!_convertUseVanilla && row.Source is null))
            {
                reason = _convertUseVanilla ? $"Pick a target body for {row.Slot}." : $"Pick source and target bodies for {row.Slot}.";
                return false;
            }
            if (!TryBuildSlotSelections(row, out _, out reason))
                return false;
        }

        foreach (var accessory in enabledAccessories)
        {
            if (accessory.Selection is null || accessory.Analysing || accessory.Analysis is null)
            {
                reason = accessory.Analysing ? $"Checking {accessory.Slot} garment coverage..." : $"Choose and analyse a {accessory.Slot} accessory garment.";
                return false;
            }
            if (!TryBuildAccessorySlotSelections(accessory, swap: false, out _, out reason)) return false;
            var target = accessory.Targets[accessory.Analysis.PrimarySlot];
            if (target is null) { reason = $"Pick a {accessory.Analysis.PrimarySlot} target body under {accessory.Slot}."; return false; }
            var outputName = GetOutputOptionName(target);
            if (accessory.Selection.Group.Options.Any(option => string.Equals(option.Name, outputName, StringComparison.OrdinalIgnoreCase)))
            { reason = $"'{outputName}' already exists in '{accessory.Selection.Group.Name}'."; return false; }
        }

        var conflict = enabledRows
            .GroupBy(row => $"{row.Selection!.Group.StableKey}|{GetOutputOptionName(row.Target!)}", StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(group => group.Select(row => row.Selection!.Option.StableKey).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1);
        if (conflict is not null)
        {
            reason = "Those rows use different options from the same Penumbra group. Pick one.";
            return false;
        }

        foreach (var row in enabledRows)
        {
            var outputName = GetOutputOptionName(row.Target!);
            if (row.Selection!.Group.Options.Any(option => string.Equals(option.Name, outputName, StringComparison.OrdinalIgnoreCase)))
            {
                reason = $"'{outputName}' already exists in '{row.Selection.Group.Name}'.";
                return false;
            }
        }

        if (!_plugin.ModelBridge.Status.Available)
        {
            reason = "Penumbra model bridge is unavailable.";
            return false;
        }
        if (!_plugin.Solver.Ready || !_plugin.Solver.ConversionReady)
        {
            reason = "Solver runtime is not ready for conversion.";
            return false;
        }

        reason = string.Empty;
        return true;
    }

    private void ReconcileSelectedPenumbraMod()
    {
        if (_selectedMod is null || !_plugin.Penumbra.Available) return;

        var refreshed = _plugin.Penumbra.Mods.FirstOrDefault(mod => string.Equals(mod.Directory, _selectedMod.Directory, StringComparison.OrdinalIgnoreCase));
        if (refreshed is not null)
        {
            _selectedMod = refreshed;
            return;
        }

        _plugin.PreviewTextures.Clear();
        _selectedMod = null;
        _document = null;
        ResetOutfitRows(keepTargets: false);
        ResetSwapRows(keepTargets: false);
        _swapTargetRaceOverride = null;
        _animationSourceRaceOverride = null;
        _animationTargetRaceOverride = null;
        _animationSkeletonChoiceId = AnimationSkeletonService.StandardChoiceId;
        _customiseSelection = null;
        _customisePiercingBody = null;
        _customiseParts = [];
        _customisePreviewTriangles = [];
        _customisePreviewEdges = [];
        _customiseSelectedParts.Clear();
        _cleanupCandidates = [];
        _customiseCleanupSelected.Clear();
        _animationCleanupSelected.Clear();
        _selectionFilter = string.Empty;
        _bodyFilter = string.Empty;
        _bodyFilterOwner = string.Empty;
        _uiMessage = string.Empty;
        _uiMessageError = false;
        _plugin.AnimationPort.ResetStatus();
        RequestScrollTop();
    }

    private void SelectMod(PenumbraModInfo mod)
    {
        RequestScrollTop();
        _plugin.PreviewTextures.Clear();
        var modChanged = _selectedMod is null || !string.Equals(_selectedMod.Directory, mod.Directory, StringComparison.OrdinalIgnoreCase);
        _selectedMod = mod;
        _document = null;
        ResetOutfitRows(keepTargets: !modChanged);
        ResetSwapRows(keepTargets: !modChanged);
        _animationSourceRaceOverride = null;
        if (modChanged)
        {
            _swapTargetRaceOverride = null;
            _animationTargetRaceOverride = null;
            _animationSkeletonChoiceId = AnimationSkeletonService.StandardChoiceId;
            _selectionFilter = string.Empty;
            _bodyFilter = string.Empty;
            _bodyFilterOwner = string.Empty;
        }
        _modFilter = string.Empty;
        _customiseSelection = null;
        _customisePiercingBody = null;
        _customisePiercingBodies = [];
        _customisePiercingBodiesLoading = false;
        _customisePiercingBodiesKey = string.Empty;
        _customiseParts = [];
        _customisePreviewTriangles = [];
        _customisePreviewEdges = [];
        _customisePreviewMin = Vector3.Zero;
        _customisePreviewMax = Vector3.One;
        _customiseSelectedParts.Clear();
        _customisePartToggleNames.Clear();
        _customisePartColourSlots.Clear();
        _customiseNextColourSlot = 0;
        _customiseCombinedToggleName = string.Empty;
        _customiseAccessoryTarget = null;
        _customiseAccessoryFilter = string.Empty;
        _customiseAccessoryOptionName = string.Empty;
        _customiseHoveredPart = -1;
        _customisePreviewPan = Vector2.Zero;
        _customisePreviewFocusCentre = null;
        _customisePreviewFocusRadius = null;
        _customisePreviewZoom = 1f;
        _customiseInspecting = false;
        _customiseInspectionComplete = false;
        _cleanupCandidates = [];
        _customiseCleanupSelected.Clear();
        _animationCleanupSelected.Clear();
        _uiMessage = string.Empty;
        _uiMessageError = false;
        _plugin.AnimationPort.ResetStatus();

        try
        {
            _document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
            BuildOutfitChoices();
            RefreshCleanupCandidates();
        }
        catch (Exception ex)
        {
            if (_currentTab != MainTab.AnimationPort)
                SetMessage(ex.Message, true);
        }
    }

    private void SelectBodyImportMod(PenumbraModInfo mod)
    {
        RequestScrollTop();
        _bodyImportMod = mod;
        _bodyImportDocument = null;
        _bodyImportChoices.Clear();
        _bodyImportChoice = null;
        _bodyImportBodyName = mod.Name;
        _bodyImportVariantName = string.Empty;
        SetMessage(string.Empty, false);

        try
        {
            _bodyImportDocument = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
            BuildBodyImportChoices();
            if (_bodyImportChoices.Count == 1)
            {
                _bodyImportChoice = _bodyImportChoices[0];
                _bodyImportVariantName = _bodyImportChoice.Option.Name;
            }
        }
        catch (Exception ex)
        {
            SetMessage(ex.Message, true);
        }
    }

    private void BuildBodyImportChoices()
    {
        _bodyImportChoices.Clear();
        if (_bodyImportDocument is null)
            return;

        var defaultOption = new V4OptionInfo(null, "Default", new JsonObject());
        var defaultGroup = new V4GroupInfo(null, "Base body", "Single", new JsonObject(), [defaultOption]);
        AddBodyImportChoice(defaultGroup, defaultOption, _bodyImportDocument.GetDefaultModelRedirects());

        foreach (var group in _bodyImportDocument.Groups)
        {
            foreach (var option in group.Options)
                AddBodyImportChoice(group, option, _bodyImportDocument.GetOptionModelRedirects(group.StableKey, option.StableKey));
        }

        _bodyImportChoices.Sort((left, right) => string.Compare(left.Label, right.Label, StringComparison.OrdinalIgnoreCase));
    }

    private void AddBodyImportChoice(V4GroupInfo group, V4OptionInfo option, IReadOnlyList<ModelRedirect> redirects)
    {
        if (_bodyImportDocument is null)
            return;
        var models = new List<BodyImportModelChoice>();
        foreach (var model in redirects)
        {
            if (!IsCatalogueBodyModelPath(model.GamePath))
                continue;
            var slot = GetModelSlot(model.GamePath);
            var raceCode = GetModelRaceCode(model.GamePath);
            if (slot is null || raceCode is null)
                continue;
            try
            {
                var physical = _bodyImportDocument.ResolvePhysicalPath(model);
                if (File.Exists(physical))
                    models.Add(new BodyImportModelChoice(model, slot, raceCode, physical));
            }
            catch
            {
                // Skip broken redirects in the body picker.
            }
        }

        if (models.Count == 0)
            return;
        var slots = string.Join(", ", models.Select(model => model.Slot).Distinct().OrderBy(slot => Array.IndexOf(BodySlots.All, slot)));
        _bodyImportChoices.Add(new BodyImportChoice(group, option, models, $"{group.Name} / {option.Name} · {slots}"));
    }

    private static bool IsCatalogueBodyModelPath(string gamePath)
    {
        var path = gamePath.Replace('\\', '/');
        return path.Contains("/obj/body/", StringComparison.OrdinalIgnoreCase)
            || path.Contains("/equipment/e0000/", StringComparison.OrdinalIgnoreCase);
    }

    private void StartBodyImport()
    {
        if (_bodyImportBusy || _bodyImportMod is null || _bodyImportChoice is null)
            return;

        var mod = _bodyImportMod;
        var choice = _bodyImportChoice;
        var request = new CustomBodyImportRequest(
            _bodyImportBodyName.Trim(),
            _bodyImportVariantName.Trim(),
            mod.Name,
            choice.Group.Name,
            choice.Option.Name,
            choice.Models.Select(model => new CustomBodyImportModel(model.Slot, model.RaceCode, model.Model.GamePath, model.Model.RelativePath, model.PhysicalPath)).ToArray());

        _bodyImportBusy = true;
        _ = RunUiTask(async () =>
        {
            try
            {
                var result = await _plugin.Bodies.ImportCustomBodyAsync(request).ConfigureAwait(false);
                _uiActions.Enqueue(() => SetMessage($"Added {result.BodyName} / {result.VariantName} to your catalogue.", false));
            }
            finally
            {
                _uiActions.Enqueue(() => _bodyImportBusy = false);
            }
        });
    }

    private void StartRaceSwapConversion()
    {
        if (_selectedMod is null || _document is null)
            return;

        var selectedMod = _selectedMod;
        var identity = GetSwapTargetIdentity();
        var requests = new List<RaceSwapConversionRequest>();
        foreach (var row in _swapRows.Values.Where(row => row.Enabled))
        {
            if (row.Selection is null || row.Target is null)
            {
                SetMessage($"{row.Slot} is not ready to convert.", true);
                return;
            }
            if (!TryBuildSwapSlotSelections(row, out var slots, out var reason))
            {
                SetMessage(string.IsNullOrWhiteSpace(reason) ? $"{row.Slot} is not ready to convert." : reason, true);
                return;
            }
            requests.Add(new RaceSwapConversionRequest(selectedMod.Directory, selectedMod.Name, selectedMod.ModRoot,
                row.Selection.Group.StableKey, row.Selection.Option.StableKey, [row.Selection.Model], slots,
                GetRaceSwapOutputOptionName(row.Target, identity), identity.Code));
        }

        foreach (var accessory in _swapAccessoryRows.Values.Where(row => row.Enabled))
        {
            if (accessory.Selection is null || accessory.Analysis is null)
            {
                SetMessage($"{accessory.Slot} accessory is not ready to port.", true);
                return;
            }
            if (!TryBuildAccessorySlotSelections(accessory, swap: true, out var slots, out var reason))
            {
                SetMessage(string.IsNullOrWhiteSpace(reason) ? $"{accessory.Slot} accessory is not ready to port." : reason, true);
                return;
            }
            var target = accessory.Targets[accessory.Analysis.PrimarySlot];
            if (target is null) { SetMessage($"Pick a {accessory.Analysis.PrimarySlot} target body under {accessory.Slot}.", true); return; }
            requests.Add(new RaceSwapConversionRequest(selectedMod.Directory, selectedMod.Name, selectedMod.ModRoot,
                accessory.Selection.Group.StableKey, accessory.Selection.Option.StableKey, [accessory.Selection.Model], slots,
                GetRaceSwapOutputOptionName(target, identity), identity.Code, accessory.Analysis.PrimarySlot, accessory.Analysis.SourceContainsBody, false));
        }

        _ = RunUiTask(async () =>
        {
            var results = await _plugin.Conversion.RunRaceSwapOutfitConversionAsync(requests).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                ReloadSwapRowsAfterWrite(selectedMod);
                ResetOutfitRows(keepTargets: true);
                SetMessage($"Created {identity.DisplayName} port.", false);
            });
        });
    }

    private bool ReloadSwapRowsAfterWrite(PenumbraModInfo mod)
    {
        var snapshots = _swapRows.ToDictionary(pair => pair.Key, pair => new
        {
            Identity = pair.Value.Selection?.Identity,
            pair.Value.Enabled,
            pair.Value.Source,
            pair.Value.Target,
            pair.Value.Analysis,
            pair.Value.SourceUserOverride,
            pair.Value.SourceInferred,
            pair.Value.SourceAutoPriority,
        }, StringComparer.OrdinalIgnoreCase);

        try
        {
            _document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
            BuildOutfitChoices();
            foreach (var (slot, snapshot) in snapshots)
            {
                var row = _swapRows[slot];
                row.Selection = snapshot.Identity is null ? null : _outfitChoices[slot].FirstOrDefault(choice => string.Equals(choice.Identity, snapshot.Identity, StringComparison.OrdinalIgnoreCase));
                row.Enabled = snapshot.Enabled && row.Selection is not null;
                row.Source = snapshot.Source;
                row.Target = snapshot.Target;
                row.Analysis = snapshot.Analysis;
                row.SourceUserOverride = snapshot.SourceUserOverride;
                row.SourceInferred = snapshot.SourceInferred;
                row.SourceAutoPriority = snapshot.SourceAutoPriority;
                row.Analysing = false;
                row.Status = string.Empty;
            }
            return true;
        }
        catch (Exception ex)
        {
            SetMessage($"Penumbra reloaded, but RavaFit could not restore the race-swap selection: {ex.Message}", true);
            return false;
        }
    }

    private void StartConversion()
    {
        if (_selectedMod is null || _document is null)
            return;

        var selectedMod = _selectedMod;
        var requests = new List<ConversionRequest>();
        foreach (var row in _outfitRows.Values.Where(row => row.Enabled))
        {
            if (row.Selection is null || row.Target is null)
            {
                SetMessage($"{row.Slot} is not ready to convert.", true);
                return;
            }

            if (!TryBuildSlotSelections(row, out var slots, out var reason))
            {
                SetMessage(string.IsNullOrWhiteSpace(reason) ? $"{row.Slot} is not ready to convert." : reason, true);
                return;
            }

            requests.Add(new ConversionRequest(selectedMod.Directory, selectedMod.Name, selectedMod.ModRoot,
                row.Selection.Group.StableKey, row.Selection.Option.StableKey, [row.Selection.Model], slots, GetOutputOptionName(row.Target)));
        }

        foreach (var accessory in _accessoryRows.Values.Where(row => row.Enabled))
        {
            if (accessory.Selection is null || accessory.Analysis is null)
            {
                SetMessage($"{accessory.Slot} accessory is not ready to convert.", true);
                return;
            }
            if (!TryBuildAccessorySlotSelections(accessory, swap: false, out var slots, out var reason))
            {
                SetMessage(string.IsNullOrWhiteSpace(reason) ? $"{accessory.Slot} accessory is not ready to convert." : reason, true);
                return;
            }
            var target = accessory.Targets[accessory.Analysis.PrimarySlot];
            if (target is null) { SetMessage($"Pick a {accessory.Analysis.PrimarySlot} target body under {accessory.Slot}.", true); return; }
            requests.Add(new ConversionRequest(selectedMod.Directory, selectedMod.Name, selectedMod.ModRoot,
                accessory.Selection.Group.StableKey, accessory.Selection.Option.StableKey, [accessory.Selection.Model], slots, GetOutputOptionName(target),
                accessory.Analysis.PrimarySlot, accessory.Analysis.SourceContainsBody, false));
        }

        _ = RunUiTask(async () =>
        {
            var results = await _plugin.Conversion.RunOutfitConversionAsync(requests).ConfigureAwait(false);
            _uiActions.Enqueue(() =>
            {
                ReloadOutfitRowsAfterWrite(selectedMod);
                ResetSwapRows(keepTargets: true);
                SetMessage($"Converted {requests.Count} outfit item{(requests.Count == 1 ? string.Empty : "s")} and added {results.Count} target option{(results.Count == 1 ? string.Empty : "s")}.", false);
            });
        });
    }

    private bool ReloadOutfitRowsAfterWrite(PenumbraModInfo mod)
    {
        var snapshots = _outfitRows.ToDictionary(pair => pair.Key, pair => new
        {
            Identity = pair.Value.Selection?.Identity,
            pair.Value.Enabled,
            pair.Value.Source,
            pair.Value.Target,
            pair.Value.Analysis,
            pair.Value.SourceUserOverride,
            pair.Value.SourceInferred,
            pair.Value.SourceAutoPriority,
        }, StringComparer.OrdinalIgnoreCase);

        try
        {
            _document = PenumbraV4Document.Load(Path.Combine(mod.ModRoot, "meta.json"));
            BuildOutfitChoices();
            foreach (var (slot, snapshot) in snapshots)
            {
                var row = _outfitRows[slot];
                row.Selection = snapshot.Identity is null ? null : _outfitChoices[slot].FirstOrDefault(choice => string.Equals(choice.Identity, snapshot.Identity, StringComparison.OrdinalIgnoreCase));
                row.Enabled = snapshot.Enabled && row.Selection is not null;
                row.Source = snapshot.Source;
                row.Target = snapshot.Target;
                row.Analysis = snapshot.Analysis;
                row.SourceUserOverride = snapshot.SourceUserOverride;
                row.SourceInferred = snapshot.SourceInferred;
                row.SourceAutoPriority = snapshot.SourceAutoPriority;
                row.Analysing = false;
                row.Status = string.Empty;
            }

            return true;
        }
        catch (Exception ex)
        {
            SetMessage($"Penumbra reloaded, but RavaFit could not refresh the outfit rows: {ex.Message}", true);
            return false;
        }
    }

    private async Task RunUiTask(Func<Task> action)
    {
        SetMessage(string.Empty, false);
        try
        {
            await action().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _uiActions.Enqueue(() => SetMessage(ex.Message, true));
        }
    }

    private void SetMessage(string message, bool error)
    {
        _uiMessage = message;
        _uiMessageError = error;
    }
}
