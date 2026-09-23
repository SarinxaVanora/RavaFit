using Dalamud.Game.ClientState.Objects.Enums;
using Dalamud.Plugin.Services;
using RavaFit.Core.Models;

namespace RavaFit.Services;

internal sealed class CharacterRaceService
{
    private readonly IObjectTable _objects;

    public CharacterRaceService(IObjectTable objects) => _objects = objects;

    public CharacterRaceIdentity? Current
    {
        get
        {
            var player = _objects.LocalPlayer;
            if (player is null)
                return null;
            var customize = player.Customize;
            var required = Math.Max((int)CustomizeIndex.Race, Math.Max((int)CustomizeIndex.Gender, (int)CustomizeIndex.Tribe));
            if (customize.Length <= required)
                return null;
            return CharacterRaceCatalog.FromCustomize(
                customize[(int)CustomizeIndex.Race],
                customize[(int)CustomizeIndex.Gender],
                customize[(int)CustomizeIndex.Tribe]);
        }
    }
}
