using System.Runtime.InteropServices;
using Dalamud.Plugin.Services;
using FFXIVClientStructs.FFXIV.Client.Game.Character;
using FFXIVClientStructs.FFXIV.Client.Graphics.Scene;
using FFXIVClientStructs.Havok.Animation;
using FFXIVClientStructs.Havok.Animation.Rig;
using FFXIVClientStructs.Havok.Common.Base.Math.QsTransform;
using FFXIVClientStructs.Havok.Common.Base.Object;
using FFXIVClientStructs.Havok.Common.Base.Types;
using FFXIVClientStructs.Havok.Common.Serialize.Resource;
using FFXIVClientStructs.Havok.Common.Serialize.Util;
using RavaFit.Core.Models;

namespace RavaFit.Services.Animation;

internal sealed class AnimationSkeletonService
{
    public const string StandardChoiceId = "standard";
    public const string CurrentChoiceId = "current";

    private readonly IFramework _framework;
    private readonly IDataManager _dataManager;
    private readonly IObjectTable _objects;
    private readonly IPluginLog _log;
    private readonly CharacterRaceService _characterRace;

    public AnimationSkeletonService(IFramework framework, IDataManager dataManager, IObjectTable objects, CharacterRaceService characterRace, IPluginLog log)
    {
        _framework = framework;
        _dataManager = dataManager;
        _objects = objects;
        _characterRace = characterRace;
        _log = log;
    }

    public IReadOnlyList<AnimationSkeletonChoice> GetChoices(CharacterRaceIdentity target)
    {
        var choices = new List<AnimationSkeletonChoice>
        {
            new(StandardChoiceId, "Standard player skeleton", $"The normal {target.DisplayName} game skeleton.", false),
        };
        if (_characterRace.Current is { } current && string.Equals(current.Code, target.Code, StringComparison.OrdinalIgnoreCase) && _objects.LocalPlayer is not null)
            choices.Add(new AnimationSkeletonChoice(CurrentChoiceId, "My current skeleton", "Uses the skeleton currently loaded on your character, including compatible extra partial skeletons.", true));
        return choices;
    }

    public async Task<IReadOnlyList<AnimationTargetSkeletonSnapshot>> LoadAsync(CharacterRaceIdentity target, string? choiceId, CancellationToken cancellationToken)
    {
        if (string.Equals(choiceId, CurrentChoiceId, StringComparison.OrdinalIgnoreCase))
        {
            if (_characterRace.Current is not { } current || !string.Equals(current.Code, target.Code, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("My current skeleton can only be used when the target character matches your current character.");
            return await RunOnFrameworkThread(CaptureCurrentSkeletons).ConfigureAwait(false);
        }

        var gamePath = GetStandardSkeletonPath(target.Code);
        var sklbBytes = ReadGameFileBytes(gamePath);
        var hkx = ExtractHkxFromSklb(sklbBytes);
        var tempHkx = Path.Combine(Path.GetTempPath(), $"ravafit-skeleton-{Guid.NewGuid():N}.hkx");
        try
        {
            await File.WriteAllBytesAsync(tempHkx, hkx, cancellationToken).ConfigureAwait(false);
            var snapshot = await RunOnFrameworkThread(() => LoadSkeletonSnapshotFromHkx(tempHkx, gamePath)).ConfigureAwait(false);
            return [snapshot];
        }
        finally
        {
            try { File.Delete(tempHkx); } catch { }
        }
    }

    public static string GetStandardSkeletonPath(string raceCode)
        => $"chara/human/c{raceCode}/skeleton/base/b0001/skl_c{raceCode}b0001.sklb";

    private async Task<T> RunOnFrameworkThread<T>(Func<T> func)
    {
        if (_framework.IsInFrameworkUpdateThread) return func();
        return await _framework.RunOnFrameworkThread(func).ConfigureAwait(false);
    }

    private unsafe IReadOnlyList<AnimationTargetSkeletonSnapshot> CaptureCurrentSkeletons()
    {
        var player = _objects.LocalPlayer;
        if (player is null || player.Address == nint.Zero)
            throw new InvalidOperationException("Your character is not currently available.");

        var character = (Character*)player.Address;
        var drawObject = character->GameObject.DrawObject;
        if (drawObject == null)
            throw new InvalidOperationException("Your character skeleton is not currently loaded.");

        var chara = (CharacterBase*)drawObject;
        if (chara->GetModelType() != CharacterBase.ModelType.Human || chara->Skeleton == null)
            throw new InvalidOperationException("Your current character is not using a human animation skeleton.");

        var skeleton = chara->Skeleton;
        var resHandles = skeleton->SkeletonResourceHandles;
        if (resHandles == null || skeleton->PartialSkeletonCount <= 0 || skeleton->PartialSkeletonCount > 64)
            throw new InvalidOperationException("Your current character skeleton could not be read.");

        var output = new List<AnimationTargetSkeletonSnapshot>();
        for (var i = 0; i < skeleton->PartialSkeletonCount; i++)
        {
            var handle = *(resHandles + i);
            if ((nint)handle == nint.Zero || handle->BoneCount <= 0 || handle->BoneCount > 4096 || handle->HavokSkeleton == null)
                continue;
            var resourcePath = handle->FileName.ToString();
            var internalName = handle->HavokSkeleton->Name.String;
            var skeletonName = string.IsNullOrWhiteSpace(internalName) ? resourcePath : internalName;
            var boneNames = new string?[handle->BoneCount];
            var boneLookup = new Dictionary<string, short>(StringComparer.OrdinalIgnoreCase);
            for (short bone = 0; bone < handle->BoneCount; bone++)
            {
                var name = handle->HavokSkeleton->Bones[bone].Name.String;
                if (string.IsNullOrWhiteSpace(name)) continue;
                boneNames[bone] = name;
                boneLookup.TryAdd(name, bone);
            }
            hkQsTransformf[] pose = [];
            var referencePose = handle->HavokSkeleton->ReferencePose;
            if (referencePose.Data != null && referencePose.Length > 0 && referencePose.Length <= 4096)
            {
                pose = new hkQsTransformf[referencePose.Length];
                new ReadOnlySpan<hkQsTransformf>(referencePose.Data, referencePose.Length).CopyTo(pose);
            }
            var snapshot = new AnimationTargetSkeletonSnapshot(resourcePath, skeletonName, boneNames, boneLookup, pose);
            if (snapshot.IsHumanAnimationSkeleton) output.Add(snapshot);
        }

        if (output.Count == 0)
            throw new InvalidOperationException("No usable human animation skeleton was found on your current character.");
        return output.OrderByDescending(x => x.BoneCount).ToArray();
    }

    private unsafe AnimationTargetSkeletonSnapshot LoadSkeletonSnapshotFromHkx(string hkxPath, string resourcePath)
    {
        var ansi = Marshal.StringToHGlobalAnsi(hkxPath);
        hkResource* resource = null;
        try
        {
            var loadOptions = stackalloc hkSerializeUtil.LoadOptions[1];
            loadOptions->TypeInfoRegistry = hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry();
            loadOptions->ClassNameRegistry = hkBuiltinTypeRegistry.Instance()->GetClassNameRegistry();
            loadOptions->Flags = new hkFlags<hkSerializeUtil.LoadOptionBits, int> { Storage = (int)hkSerializeUtil.LoadOptionBits.Default };
            resource = hkSerializeUtil.LoadFromFile((byte*)ansi, null, loadOptions);
            if (resource == null) throw new InvalidDataException($"Could not load target skeleton {resourcePath}.");

            hkaSkeleton* skeleton = null;
            var rootName = "hkRootLevelContainer"u8;
            fixed (byte* root = rootName)
            {
                var container = (hkRootLevelContainer*)resource->GetContentsPointer(root, hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry());
                if (container != null)
                {
                    var animationName = "hkaAnimationContainer"u8;
                    fixed (byte* name = animationName)
                    {
                        var animationContainer = (hkaAnimationContainer*)container->findObjectByName(name, null);
                        if (animationContainer != null && animationContainer->Skeletons.Length > 0 && animationContainer->Skeletons.Length <= 64)
                        {
                            for (var i = 0; i < animationContainer->Skeletons.Length; i++)
                            {
                                var candidate = animationContainer->Skeletons[i].ptr;
                                if (candidate == null || candidate->Bones.Length <= 0 || candidate->Bones.Length > 4096) continue;
                                skeleton = candidate;
                                break;
                            }
                        }
                    }

                    if (skeleton == null)
                    {
                        var skeletonType = "hkaSkeleton"u8;
                        fixed (byte* type = skeletonType)
                            skeleton = (hkaSkeleton*)container->findObjectByType(type, null);
                    }
                }
            }
            if (skeleton == null)
            {
                var skeletonType = "hkaSkeleton"u8;
                fixed (byte* type = skeletonType)
                    skeleton = (hkaSkeleton*)resource->GetContentsPointer(type, hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry());
            }
            if (skeleton == null || skeleton->Bones.Length <= 0 || skeleton->Bones.Length > 4096)
                throw new InvalidDataException($"Target skeleton {resourcePath} did not contain a readable player hkaSkeleton.");

            var boneNames = new string?[skeleton->Bones.Length];
            var lookup = new Dictionary<string, short>(StringComparer.OrdinalIgnoreCase);
            for (short i = 0; i < skeleton->Bones.Length; i++)
            {
                var name = skeleton->Bones[i].Name.String;
                if (string.IsNullOrWhiteSpace(name)) continue;
                boneNames[i] = name;
                lookup.TryAdd(name, i);
            }
            hkQsTransformf[] pose = [];
            if (skeleton->ReferencePose.Data != null && skeleton->ReferencePose.Length > 0 && skeleton->ReferencePose.Length <= 4096)
            {
                pose = new hkQsTransformf[skeleton->ReferencePose.Length];
                new ReadOnlySpan<hkQsTransformf>(skeleton->ReferencePose.Data, skeleton->ReferencePose.Length).CopyTo(pose);
            }
            var skeletonName = skeleton->Name.String;
            return new AnimationTargetSkeletonSnapshot(resourcePath, string.IsNullOrWhiteSpace(skeletonName) ? Path.GetFileNameWithoutExtension(resourcePath) : skeletonName, boneNames, lookup, pose);
        }
        finally
        {
            if (resource != null)
                ((hkReferencedObject*)resource)->RemoveReference();
            Marshal.FreeHGlobal(ansi);
        }
    }

    public bool GameFileExists(string gamePath) => _dataManager.FileExists(gamePath);

    public byte[] ReadGameFileBytes(string gamePath)
    {
        var resource = _dataManager.GetFile(gamePath)
            ?? throw new FileNotFoundException($"Could not read game file {gamePath}.");
        return resource.Data;
    }

    private static byte[] ExtractHkxFromSklb(byte[] sklb)
    {
        using var stream = new MemoryStream(sklb, writable: false);
        using var reader = new BinaryReader(stream);
        if (reader.ReadInt32() != 0x736B6C62) throw new InvalidDataException("Target skeleton had invalid SKLB magic.");
        _ = reader.ReadInt16();
        var versionTwo = reader.ReadInt16();
        int havokOffset;
        if (versionTwo == 0x3132)
        {
            _ = reader.ReadInt16();
            havokOffset = reader.ReadInt16();
        }
        else
        {
            _ = reader.ReadInt32();
            havokOffset = reader.ReadInt32();
        }
        if (havokOffset <= 0 || havokOffset >= sklb.Length) throw new InvalidDataException("Target skeleton had an invalid Havok payload offset.");
        return sklb.AsSpan(havokOffset).ToArray();
    }
}
