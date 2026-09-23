using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using Dalamud.Plugin.Services;
using FFXIVClientStructs.Havok.Animation;
using FFXIVClientStructs.Havok.Animation.Animation;
using FFXIVClientStructs.Havok.Animation.Rig;
using FFXIVClientStructs.Havok.Common.Base.Container.Array;
using FFXIVClientStructs.Havok.Common.Base.Math.QsTransform;
using FFXIVClientStructs.Havok.Common.Base.Math.Vector;
using FFXIVClientStructs.Havok.Common.Base.Math.Quaternion;
using FFXIVClientStructs.Havok.Common.Base.Object;
using FFXIVClientStructs.Havok.Common.Base.System.IO.OStream;
using FFXIVClientStructs.Havok.Common.Base.Types;
using FFXIVClientStructs.Havok.Common.Serialize.Resource;
using FFXIVClientStructs.Havok.Common.Serialize.Util;

namespace RavaFit.Services.Animation;

internal sealed class AnimationPapRetargeter
{
    private enum PapContainerReadStatus { Valid = 0, PassThroughOriginal = 1, Invalid = 2 }
    private const int MaxHavokBytes = 32 * 1024 * 1024;
    private const string HavokInterleavedConversionCtorSig = "48 89 5C 24 ?? 48 89 6C 24 ?? 56 57 41 55 41 56 41 57 48 83 EC ?? 0F 29 74 24";
    private static readonly SemaphoreSlim NativeHavokGate = new(1, 1);
    private readonly IFramework _framework;
    private readonly IPluginLog _log;
    private readonly HavokInterleavedConversionCtorDelegate? _havokInterleavedConversionCtor;

    public AnimationPapRetargeter(IFramework framework, IPluginLog log, ISigScanner sigScanner)
    {
        _framework = framework;
        _log = log;
        _havokInterleavedConversionCtor = ResolveHavokInterleavedConversionCtor(sigScanner);
        if (_havokInterleavedConversionCtor == null)
            _log.Warning("RavaFit animation porting could not resolve the compressed Havok conversion constructor. Interleaved PAPs can still be ported; compressed PAPs requiring track removal will fail closed.");
    }

    public async Task<AnimationPapRewriteResult> RewriteFileAsync(string sourcePath, string destinationPath, IReadOnlyList<AnimationTargetSkeletonSnapshot> sourceSkeletons, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons, CancellationToken token)
    {
        if (!File.Exists(sourcePath)) throw new FileNotFoundException("Animation PAP was not found.", sourcePath);
        if (sourceSkeletons is null || sourceSkeletons.Count == 0) throw new InvalidOperationException("Source skeleton was unavailable.");
        if (targetSkeletons is null || targetSkeletons.Count == 0) throw new InvalidOperationException("Target skeleton was unavailable.");

        var preparation = PreparePapRewrite(sourcePath);
        if (preparation.ImmediateResult is { } immediate)
        {
            if (immediate.Status == PapOutcomeStatus.OriginalSafe)
            {
                CopyFileAtomic(sourcePath, destinationPath);
                return new AnimationPapRewriteResult(AnimationPapRewriteStatus.OriginalSafe, 0, 0, immediate.BindingCount, immediate.Reason);
            }
            return new AnimationPapRewriteResult(AnimationPapRewriteStatus.Blocked, 0, 0, immediate.BindingCount, immediate.Reason);
        }

        var prepared = preparation.Context ?? throw new InvalidOperationException("Animation rewrite preparation did not produce a context.");
        try
        {
            PapRewriteFrameworkResult frameworkResult;
            await NativeHavokGate.WaitAsync(token).ConfigureAwait(false);
            try
            {
                frameworkResult = await RunOnFrameworkThread(() => RewritePapForTargetFramework(prepared, sourceSkeletons, targetSkeletons)).ConfigureAwait(false);
            }
            finally { NativeHavokGate.Release(); }

            if (frameworkResult.TerminalResult is { } terminal)
            {
                if (terminal.Status == PapOutcomeStatus.OriginalSafe)
                {
                    CopyFileAtomic(sourcePath, destinationPath);
                    return new AnimationPapRewriteResult(AnimationPapRewriteStatus.OriginalSafe, 0, 0, terminal.BindingCount, terminal.Reason);
                }
                return new AnimationPapRewriteResult(AnimationPapRewriteStatus.Blocked, 0, 0, terminal.BindingCount, terminal.Reason);
            }

            if (frameworkResult.RewrittenHkx is null || frameworkResult.RewrittenHkx.Length == 0)
                return new AnimationPapRewriteResult(AnimationPapRewriteStatus.Blocked, 0, 0, frameworkResult.BindingCount, "Animation conversion produced no Havok payload.");

            var rewrittenPap = BuildPapFromRewrittenHkx(prepared.PapContainer, frameworkResult.RewrittenHkx);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath) ?? throw new InvalidOperationException("Animation output path has no parent directory."));
            var temp = destinationPath + $".ravafit-{Guid.NewGuid():N}.tmp";
            await File.WriteAllBytesAsync(temp, rewrittenPap, token).ConfigureAwait(false);
            try
            {
                await NativeHavokGate.WaitAsync(token).ConfigureAwait(false);
                try
                {
                    var verification = await RunOnFrameworkThread(() => TryVerifyMaterializedPap(temp, targetSkeletons, frameworkResult.BindingCount, out var failure) ? (true, string.Empty) : (false, failure)).ConfigureAwait(false);
                    if (!verification.Item1)
                        return new AnimationPapRewriteResult(AnimationPapRewriteStatus.Blocked, 0, 0, frameworkResult.BindingCount, verification.Item2);
                }
                finally { NativeHavokGate.Release(); }

                File.Move(temp, destinationPath, true);
            }
            finally { try { File.Delete(temp); } catch { } }

            return new AnimationPapRewriteResult(AnimationPapRewriteStatus.Converted, frameworkResult.RemappedTracks, frameworkResult.DroppedTracks, frameworkResult.BindingCount, frameworkResult.Reason);
        }
        finally { CleanupPreparedPapRewrite(prepared); }
    }

    private async Task<T> RunOnFrameworkThread<T>(Func<T> func)
    {
        if (_framework.IsInFrameworkUpdateThread) return func();
        return await _framework.RunOnFrameworkThread(func).ConfigureAwait(false);
    }

    private static void CopyFileAtomic(string source, string destination)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destination) ?? throw new InvalidOperationException("Animation output path has no parent directory."));
        var temp = destination + $".ravafit-{Guid.NewGuid():N}.tmp";
        try { File.Copy(source, temp, true); File.Move(temp, destination, true); }
        finally { try { File.Delete(temp); } catch { } }
    }

    private PapRewritePreparation PreparePapRewrite(string sourcePath)
    {
        try
        {
            var status = TryReadPapContainer(sourcePath, out var papContainer, out var readFailure);
            if (status == PapContainerReadStatus.Invalid)
                return new PapRewritePreparation(null, PapRewriteOutcome.Blocked(readFailure));
            if (status == PapContainerReadStatus.PassThroughOriginal)
                return new PapRewritePreparation(null, PapRewriteOutcome.Blocked(string.IsNullOrWhiteSpace(readFailure) ? "Animation Havok format could not be verified for the selected target skeleton." : readFailure));
            if (!papContainer.IsHumanPap)
                return new PapRewritePreparation(null, PapRewriteOutcome.Original(0, "Animation does not use the human PAP container and was copied unchanged."));

            var tempHkxPath = Path.Combine(Path.GetTempPath(), $"ravafit-animation-{Guid.NewGuid():N}.hkx");
            File.WriteAllBytes(tempHkxPath, papContainer.HavokBytes);
            return new PapRewritePreparation(new PapRewritePreparedContext(Path.GetFileName(sourcePath), sourcePath, papContainer, tempHkxPath), null);
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "RavaFit could not prepare animation PAP {Path}", sourcePath);
            return new PapRewritePreparation(null, PapRewriteOutcome.Blocked($"Could not read animation PAP: {ex.Message}"));
        }
    }

    private PapRewriteFrameworkResult RewritePapForTargetFramework(PapRewritePreparedContext prepared, IReadOnlyList<AnimationTargetSkeletonSnapshot> sourceSkeletons, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons)
    {
        var tempHkxPathAnsi = Marshal.StringToHGlobalAnsi(prepared.TempHkxPath);

        try
        {
            unsafe
            {
                var loadOptions = stackalloc hkSerializeUtil.LoadOptions[1];
                loadOptions->TypeInfoRegistry = hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry();
                loadOptions->ClassNameRegistry = hkBuiltinTypeRegistry.Instance()->GetClassNameRegistry();
                loadOptions->Flags = new hkFlags<hkSerializeUtil.LoadOptionBits, int>
                {
                    Storage = (int)hkSerializeUtil.LoadOptionBits.Default
                };

                var resource = hkSerializeUtil.LoadFromFile((byte*)tempHkxPathAnsi, null, loadOptions);
                if (resource == null)
                {
                    return PapRewriteFrameworkResult.FromTerminal(
                        PapRewriteOutcome.Blocked(prepared.OriginalHash, "Animation Havok payload could not be loaded, so RavaFit will not create an unverified target animation."));
                }

                var rootLevelName = "hkRootLevelContainer"u8;
                fixed (byte* rootName = rootLevelName)
                {
                    var container = (hkRootLevelContainer*)resource->GetContentsPointer(rootName, hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry());
                    if (container == null)
                    {
                        return PapRewriteFrameworkResult.FromTerminal(
                            PapRewriteOutcome.Blocked(prepared.OriginalHash, "Animation Havok root container was unavailable, so RavaFit will not create an unverified target animation."));
                    }

                    var animationName = "hkaAnimationContainer"u8;
                    fixed (byte* animName = animationName)
                    {
                        var animationContainer = (hkaAnimationContainer*)container->findObjectByName(animName, null);
                        if (animationContainer == null)
                        {
                            return PapRewriteFrameworkResult.FromTerminal(
                                PapRewriteOutcome.Blocked(prepared.OriginalHash, "Animation container was unavailable, so RavaFit will not create an unverified target animation."));
                        }

                        var analysis = AnalyzeAnimationBindings(animationContainer, sourceSkeletons, targetSkeletons, prepared.SourcePath);
                        if (analysis.Blocked)
                            return PapRewriteFrameworkResult.FromTerminal(PapRewriteOutcome.Blocked(prepared.OriginalHash, analysis.Reason));

                        if (!analysis.Changed)
                        {
                            _log.Debug("PAP {hash} required no transform-track pruning for current skeleton", prepared.OriginalHash);
                            return PapRewriteFrameworkResult.FromTerminal(PapRewriteOutcome.Original(prepared.OriginalHash, prepared.SourcePath, analysis.BindingCount, analysis.Reason));
                        }

                        var allocatedPatchBuffers = new List<nint>();
                        try
                        {
                            if (!ApplyBindingPatchesInMemory(animationContainer, analysis.BindingPatches, allocatedPatchBuffers, out var applyFailure))
                                return PapRewriteFrameworkResult.FromTerminal(PapRewriteOutcome.Blocked(prepared.OriginalHash, applyFailure));

                            if (!TryValidatePatchedBindingsInMemory(animationContainer, targetSkeletons, analysis.BindingCount, out var inMemoryVerifyFailure))
                                return PapRewriteFrameworkResult.FromTerminal(PapRewriteOutcome.Blocked(prepared.OriginalHash, inMemoryVerifyFailure));

                            if (!TrySaveResourceToHkx(resource, out var rewrittenHkx, out var saveFailure))
                                return PapRewriteFrameworkResult.FromTerminal(PapRewriteOutcome.Blocked(prepared.OriginalHash, saveFailure));

                            return PapRewriteFrameworkResult.FromSanitized(rewrittenHkx, analysis.RemappedTracks, analysis.DroppedTracks, analysis.BindingCount, analysis.Reason);
                        }
                        finally
                        {
                            foreach (var buffer in allocatedPatchBuffers)
                            {
                                if (buffer != 0)
                                    Marshal.FreeHGlobal(buffer);
                            }
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "Failed to sanitize PAP {hash} at {path}; leaving the original PAP untouched", prepared.OriginalHash, prepared.SourcePath);
            return PapRewriteFrameworkResult.FromTerminal(
                PapRewriteOutcome.OriginalFallback(prepared.OriginalHash, prepared.SourcePath, 0, "PAP sanitizer hit an internal error, so the mod will be disabled for the current target skeleton"));
        }
        finally
        {
            Marshal.FreeHGlobal(tempHkxPathAnsi);
        }
    }

    private static void CleanupPreparedPapRewrite(PapRewritePreparedContext? prepared)
        => CleanupPreparedPapRewrite(prepared?.TempHkxPath);

    private static void CleanupPreparedPapRewrite(string? tempHkxPath)
    {
        if (string.IsNullOrWhiteSpace(tempHkxPath))
            return;

        try { File.Delete(tempHkxPath); } catch { }
    }


    private enum PapOutcomeStatus { OriginalSafe, Blocked }
    private sealed record PapRewriteOutcome(PapOutcomeStatus Status, int BindingCount, string Reason)
    {
        public static PapRewriteOutcome Original(string originalHash, string effectivePath, int bindingCount, string reason) => new(PapOutcomeStatus.OriginalSafe, bindingCount, reason);
        public static PapRewriteOutcome Original(string originalHash, string effectivePath, int bindingCount, string reason, bool unused = false) => new(PapOutcomeStatus.OriginalSafe, bindingCount, reason);
        public static PapRewriteOutcome Original(string originalHash, int bindingCount, string reason) => new(PapOutcomeStatus.OriginalSafe, bindingCount, reason);
        public static PapRewriteOutcome Original(int bindingCount, string reason) => new(PapOutcomeStatus.OriginalSafe, bindingCount, reason);
        public static PapRewriteOutcome Blocked(string originalHash, string reason) => new(PapOutcomeStatus.Blocked, 0, reason);
        public static PapRewriteOutcome Blocked(string reason) => new(PapOutcomeStatus.Blocked, 0, reason);
        public static PapRewriteOutcome OriginalFallback(string originalHash, string effectivePath, int bindingCount, string reason) => new(PapOutcomeStatus.Blocked, bindingCount, reason);
    }

    private sealed record PapContainerData(int HeaderSize, int HavokOffset, int FooterOffset, bool IsHumanPap, byte[] HeaderBytes, byte[] HavokBytes, byte[] FooterBytes);
    private sealed record PapRewritePreparation(PapRewritePreparedContext? Context, PapRewriteOutcome? ImmediateResult);
    private sealed record PapRewritePreparedContext(string OriginalHash, string SourcePath, PapContainerData PapContainer, string TempHkxPath);
    private sealed record PapRewriteFrameworkResult(PapRewriteOutcome? TerminalResult, byte[]? RewrittenHkx, int RemappedTracks, int DroppedTracks, int BindingCount, string Reason)
    {
        public static PapRewriteFrameworkResult FromTerminal(PapRewriteOutcome result)
            => new(result, null, 0, 0, result.BindingCount, result.Reason ?? string.Empty);

        public static PapRewriteFrameworkResult FromSanitized(byte[] rewrittenHkx, int remappedTracks, int droppedTracks, int bindingCount, string reason)
            => new(null, rewrittenHkx, remappedTracks, droppedTracks, bindingCount, reason);
    }
    private sealed record BindingPatchPlan(int BindingIndex, nint AnimationAddress, string SourceSkeletonKey, string TargetSkeletonKey, AnimationTargetSkeletonSnapshot SourceSkeleton, AnimationTargetSkeletonSnapshot TargetSkeleton, short[] OriginalTracks, short[] PatchedTracks, int[] KeptTrackIndices, int NewTrackCount);
    private readonly record struct RebuildCacheKey(nint AnimationAddress, string PlanKey, string SourceSkeletonKey, string TargetSkeletonKey);
    private sealed record AnalysisOutcome(bool Changed, bool Blocked, int RemappedTracks, int DroppedTracks, int BindingCount, string Reason, IReadOnlyList<BindingPatchPlan> BindingPatches);
    [StructLayout(LayoutKind.Sequential)]
    private struct HkaInterleavedUncompressedAnimation
    {
        public hkaAnimation Animation;
        public hkArray<hkQsTransformf> Transforms;
        public hkArray<float> Floats;
    }

    [StructLayout(LayoutKind.Explicit, Size = 0xC0)]
    private unsafe struct HkaPredictiveCompressedAnimation
    {
        [FieldOffset(0x00)] public hkaAnimation Animation;
        [FieldOffset(0xB0)] public hkaSkeleton* Skeleton;
    }

    [StructLayout(LayoutKind.Explicit, Size = 0x58)]
    private unsafe struct HkaQuantizedCompressedAnimation
    {
        [FieldOffset(0x00)] public hkaAnimation Animation;
        [FieldOffset(0x50)] public hkaSkeleton* Skeleton;
    }

    private unsafe delegate HkaInterleavedUncompressedAnimation* HavokInterleavedConversionCtorDelegate(HkaInterleavedUncompressedAnimation* destinationAnimation, hkaAnimation* sourceAnimation);


    private static void AddTargetLookupEntry(Dictionary<string, AnimationTargetSkeletonSnapshot> output, string key, AnimationTargetSkeletonSnapshot snapshot)
    {
        if (string.IsNullOrWhiteSpace(key))
            return;

        if (output.TryGetValue(key, out var existing) && existing.BoneCount >= snapshot.BoneCount)
            return;

        output[key] = snapshot;
    }

    private static string CreateTargetSkeletonCacheKey(AnimationTargetSkeletonSnapshot targetSkeleton)
    {
        if (targetSkeleton == null)
            return string.Empty;

        return targetSkeleton.NormalizedResourceName
            + "|"
            + targetSkeleton.NormalizedSkeletonName
            + "|"
            + targetSkeleton.BoneCount;
    }

    private unsafe bool TryDetectAlreadySafePap(hkaAnimationContainer* animationContainer, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons, string sourcePath, out int bindingCount, out string reason)
    {
        bindingCount = 0;
        reason = "PAP already matched target skeleton";

        if (animationContainer == null)
            return false;

        bindingCount = animationContainer->Bindings.Length;
        if (bindingCount <= 0 || bindingCount > 256)
            return false;

        var targetLookup = BuildTargetLookup(targetSkeletons);
        var unionMaxIndex = targetSkeletons.Count == 0 ? -1 : targetSkeletons.Max(static s => s.BoneCount - 1);

        for (int bindingIndex = 0; bindingIndex < bindingCount; bindingIndex++)
        {
            var binding = animationContainer->Bindings[bindingIndex].ptr;
            if (binding == null)
                continue;

            var transformTracks = binding->TransformTrackToBoneIndices;
            if (transformTracks.Length <= 0 || transformTracks.Length > 8192)
                return false;

            short maxAnimIndex = -1;
            for (int trackIndex = 0; trackIndex < transformTracks.Length; trackIndex++)
            {
                var boneIndex = transformTracks[trackIndex];
                if (boneIndex > maxAnimIndex)
                    maxAnimIndex = boneIndex;
            }

            var originalSkeletonName = binding->OriginalSkeletonName.String ?? string.Empty;
            var normalizedOriginalSkeleton = XivAnimationSkeletonIdentity.NormalizeSkeletonKey(originalSkeletonName);
            var targetSkeleton = ResolveTargetSkeleton(originalSkeletonName, normalizedOriginalSkeleton, targetLookup, targetSkeletons, maxAnimIndex);
            if (targetSkeleton == null)
            {
                if (maxAnimIndex > unionMaxIndex)
                {
                    _log.Debug(
                        "PAP {path} fast safety scan could not match binding {bindingIndex} to a target skeleton for referenced bone index {maxAnimIndex}",
                        sourcePath,
                        bindingIndex,
                        maxAnimIndex);
                }

                return false;
            }

            var animation = (hkaAnimation*)binding->Animation.ptr;
            if (animation == null)
                return false;

            if (animation->NumberOfTransformTracks > 0 && animation->NumberOfTransformTracks != transformTracks.Length)
                return false;

            if (animation->Type == hkaAnimation.AnimationType.InterleavedAnimation
                && !TryValidateInterleavedFrameLayout(animation, bindingIndex, out _))
            {
                return false;
            }

            if (!TryValidateBindingSideArrayCounts(binding, animation, bindingIndex, out _))
                return false;

            for (int trackIndex = 0; trackIndex < transformTracks.Length; trackIndex++)
            {
                short originalBoneIndex = transformTracks[trackIndex];
                if (originalBoneIndex >= 0 && originalBoneIndex >= targetSkeleton.BoneCount)
                    return false;
            }
        }

        reason = "PAP already matched target skeleton (sender-side fast scan)";
        return true;
    }

    private unsafe AnalysisOutcome AnalyzeAnimationBindings(hkaAnimationContainer* animationContainer, IReadOnlyList<AnimationTargetSkeletonSnapshot> sourceSkeletons, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons, string sourcePath)
    {
        var bindingCount = animationContainer->Bindings.Length;
        if (bindingCount <= 0 || bindingCount > 256)
            return new AnalysisOutcome(false, false, 0, 0, 0, "PAP had no animation bindings", Array.Empty<BindingPatchPlan>());

        var sourceLookup = BuildTargetLookup(sourceSkeletons);
        var targetLookup = BuildTargetLookup(targetSkeletons);
        var bindingPatches = new List<BindingPatchPlan>(bindingCount);
        int remappedTracks = 0;
        int droppedTracks = 0;

        for (int bindingIndex = 0; bindingIndex < bindingCount; bindingIndex++)
        {
            var binding = animationContainer->Bindings[bindingIndex].ptr;
            if (binding == null)
                continue;

            var transformTracks = binding->TransformTrackToBoneIndices;
            if (transformTracks.Length <= 0 || transformTracks.Length > 8192)
                continue;

            var originalTracks = new short[transformTracks.Length];
            short maxAnimIndex = -1;
            for (int trackIndex = 0; trackIndex < transformTracks.Length; trackIndex++)
            {
                var boneIndex = transformTracks[trackIndex];
                originalTracks[trackIndex] = boneIndex;
                if (boneIndex > maxAnimIndex)
                    maxAnimIndex = boneIndex;
            }

            var originalSkeletonName = binding->OriginalSkeletonName.String ?? string.Empty;
            var normalizedOriginalSkeleton = XivAnimationSkeletonIdentity.NormalizeSkeletonKey(originalSkeletonName);
            var sourceSkeleton = ResolveTargetSkeleton(originalSkeletonName, normalizedOriginalSkeleton, sourceLookup, sourceSkeletons, maxAnimIndex);
            if (sourceSkeleton == null)
            {
                var why = string.IsNullOrWhiteSpace(originalSkeletonName)
                    ? "Animation binding could not be matched to the source skeleton"
                    : $"Animation binding skeleton '{originalSkeletonName}' could not be matched to the source skeleton";
                return new AnalysisOutcome(false, true, 0, 0, bindingCount, why, Array.Empty<BindingPatchPlan>());
            }

            var targetSkeleton = ResolveTargetSkeleton(originalSkeletonName, normalizedOriginalSkeleton, targetLookup, targetSkeletons, maxAnimIndex);
            if (targetSkeleton == null)
            {
                var why = string.IsNullOrWhiteSpace(originalSkeletonName)
                    ? "Animation binding could not be matched to the selected target skeleton"
                    : $"Animation binding skeleton '{originalSkeletonName}' could not be matched to the selected target skeleton";
                return new AnalysisOutcome(false, true, 0, 0, bindingCount, why, Array.Empty<BindingPatchPlan>());
            }

            var animation = (hkaAnimation*)binding->Animation.ptr;
            if (animation == null)
                return new AnalysisOutcome(false, true, 0, 0, bindingCount, $"Animation binding skeleton '{originalSkeletonName}' did not reference an animation object", Array.Empty<BindingPatchPlan>());
            if (!TryValidateBindingSideArrayCounts(binding, animation, bindingIndex, out var sideArrayFailure))
                return new AnalysisOutcome(false, true, 0, 0, bindingCount, sideArrayFailure, Array.Empty<BindingPatchPlan>());
            if (animation->NumberOfTransformTracks > 0 && animation->NumberOfTransformTracks != originalTracks.Length)
                return new AnalysisOutcome(false, true, 0, 0, bindingCount, $"Animation binding skeleton '{originalSkeletonName}' had mismatched track metadata", Array.Empty<BindingPatchPlan>());

            var patchedTracks = new List<short>(originalTracks.Length);
            var keptTrackIndices = new List<int>(originalTracks.Length);
            for (int trackIndex = 0; trackIndex < originalTracks.Length; trackIndex++)
            {
                var sourceBoneIndex = originalTracks[trackIndex];
                if (sourceBoneIndex < 0)
                {
                    patchedTracks.Add(sourceBoneIndex);
                    keptTrackIndices.Add(trackIndex);
                    continue;
                }

                if (sourceBoneIndex >= sourceSkeleton.BoneCount)
                {
                    droppedTracks++;
                    continue;
                }

                var boneName = sourceSkeleton.BoneNamesByIndex[sourceBoneIndex];
                if (string.IsNullOrWhiteSpace(boneName) || !targetSkeleton.BoneNameToIndex.TryGetValue(boneName, out var targetBoneIndex))
                {
                    droppedTracks++;
                    continue;
                }

                if (sourceBoneIndex >= sourceSkeleton.ReferencePoseByIndex.Count || targetBoneIndex >= targetSkeleton.ReferencePoseByIndex.Count)
                {
                    return new AnalysisOutcome(false, true, 0, 0, bindingCount,
                        $"Animation bone '{boneName}' had no complete source/target reference pose, so a proportion-safe retarget could not be proven.", Array.Empty<BindingPatchPlan>());
                }

                patchedTracks.Add(targetBoneIndex);
                keptTrackIndices.Add(trackIndex);
                remappedTracks++;
            }

            if (patchedTracks.Count <= 0)
                return new AnalysisOutcome(false, true, 0, 0, bindingCount,
                    $"Animation binding skeleton '{originalSkeletonName}' had no common transform bones on the target skeleton", Array.Empty<BindingPatchPlan>());

            if (animation->Type != hkaAnimation.AnimationType.InterleavedAnimation && !IsSupportedSenderSideRebuildAnimationType(animation->Type))
                return new AnalysisOutcome(false, true, 0, 0, bindingCount,
                    $"Animation binding skeleton '{originalSkeletonName}' requires true retargeting, but Havok animation type '{animation->Type}' is not supported by the sender-side rebuild path", Array.Empty<BindingPatchPlan>());

            var keptTrackIndexArray = keptTrackIndices.ToArray();
            bindingPatches.Add(new BindingPatchPlan(
                bindingIndex,
                (nint)animation,
                CreateTargetSkeletonCacheKey(sourceSkeleton),
                CreateTargetSkeletonCacheKey(targetSkeleton),
                sourceSkeleton,
                targetSkeleton,
                originalTracks,
                patchedTracks.ToArray(),
                keptTrackIndexArray,
                keptTrackIndexArray.Length));
        }

        if (bindingPatches.Count == 0)
            return new AnalysisOutcome(false, false, 0, 0, bindingCount, "PAP had no transform bindings requiring retargeting", bindingPatches);

        var reason = droppedTracks > 0
            ? $"Retargeted {remappedTracks} transform tracks by bone name/reference pose and removed {droppedTracks} bones unavailable on the target skeleton"
            : $"Retargeted {remappedTracks} transform tracks by bone name and source/target reference pose";
        return new AnalysisOutcome(true, false, remappedTracks, droppedTracks, bindingCount, reason, bindingPatches);
    }

    private unsafe bool ApplyBindingPatchesInMemory(hkaAnimationContainer* animationContainer, IReadOnlyList<BindingPatchPlan> bindingPatches, List<nint> allocatedBuffers, out string failure)
    {
        failure = string.Empty;
        var rebuiltAnimationLookup = new Dictionary<RebuildCacheKey, nint>();

        foreach (var patch in bindingPatches)
        {
            if (patch.BindingIndex < 0 || patch.BindingIndex >= animationContainer->Bindings.Length)
                continue;

            var binding = animationContainer->Bindings[patch.BindingIndex].ptr;
            if (binding == null)
                continue;

            var rebuildCacheKey = new RebuildCacheKey(
                patch.AnimationAddress,
                CreateTrackPlanKey(patch.KeptTrackIndices) + ">" + string.Join(",", patch.PatchedTracks),
                patch.SourceSkeletonKey,
                patch.TargetSkeletonKey);

            hkaAnimation* animation;
            if (rebuiltAnimationLookup.TryGetValue(rebuildCacheKey, out var rebuiltAnimationAddress))
            {
                animation = (hkaAnimation*)rebuiltAnimationAddress;
                binding->Animation.ptr = animation;
            }
            else
            {
                animation = (hkaAnimation*)binding->Animation.ptr;
                if (animation == null)
                {
                    failure = $"Animation binding {patch.BindingIndex} lost its animation reference during sender-side rebuild";
                    return false;
                }

                if (!TryRebuildAnimationPayload(animationContainer, binding, animation, patch, allocatedBuffers, out var rebuiltAnimation, out failure))
                    return false;

                animation = rebuiltAnimation;
                rebuiltAnimationLookup[rebuildCacheKey] = (nint)animation;
                binding->Animation.ptr = animation;
            }

            if (animation == null)
            {
                failure = $"Animation binding {patch.BindingIndex} did not produce a rebuilt animation object";
                return false;
            }

            if (!TryCompactBindingPartitionIndices(binding, patch.KeptTrackIndices, patch.OriginalTracks.Length, animation->NumberOfFloatTracks, allocatedBuffers, out failure))
                return false;

            binding->TransformTrackToBoneIndices = CreateOwnedArray<short>(patch.PatchedTracks, allocatedBuffers);
        }

        return true;
    }

    private unsafe bool TryRebuildAnimationPayload(hkaAnimationContainer* animationContainer, hkaAnimationBinding* binding, hkaAnimation* animation, BindingPatchPlan patch, List<nint> allocatedBuffers, out hkaAnimation* rebuiltAnimation, out string failure)
    {
        rebuiltAnimation = animation;
        failure = string.Empty;

        if (animation == null)
        {
            failure = "Animation was null during sender-side rebuild";
            return false;
        }

        if (animation->Type == hkaAnimation.AnimationType.InterleavedAnimation)
        {
            if (!TryCloneInterleavedAnimationPayload(animation, allocatedBuffers, out var clonedAnimation, out failure))
                return false;

            return TryRebuildInterleavedAnimationPayload(clonedAnimation, patch, allocatedBuffers, out rebuiltAnimation, out failure);
        }

        if (!IsSupportedSenderSideRebuildAnimationType(animation->Type))
        {
            failure = $"Unsupported Havok animation type '{animation->Type}' for sender-side transform rebuild";
            return false;
        }

        return TryRebuildCompressedAnimationPayload(animationContainer, binding, animation, patch, allocatedBuffers, out rebuiltAnimation, out failure);
    }

    private unsafe bool TryRebuildCompressedAnimationPayload(hkaAnimationContainer* animationContainer, hkaAnimationBinding* binding, hkaAnimation* sourceAnimation, BindingPatchPlan patch, List<nint> allocatedBuffers, out hkaAnimation* rebuiltAnimation, out string failure)
    {
        rebuiltAnimation = null;
        failure = string.Empty;

        if (_havokInterleavedConversionCtor == null)
        {
            failure = "Compressed animation rebuild was unavailable because the Havok conversion constructor could not be resolved";
            return false;
        }

        if (!IsSupportedSenderSideRebuildAnimationType(sourceAnimation->Type))
        {
            failure = $"Unsupported Havok animation type '{sourceAnimation->Type}' for sender-side compressed rebuild";
            return false;
        }

        hkaSkeleton* injectedSkeleton = null;
        if (!TryEnsureCompressedAnimationSkeleton(binding, sourceAnimation, patch.SourceSkeleton, allocatedBuffers, out injectedSkeleton, out failure))
            return false;

        var interleavedSize = Marshal.SizeOf<HkaInterleavedUncompressedAnimation>();
        var convertedInterleaved = (HkaInterleavedUncompressedAnimation*)Marshal.AllocHGlobal(interleavedSize);
        new Span<byte>(convertedInterleaved, interleavedSize).Clear();

        HkaInterleavedUncompressedAnimation* ctorResult;
        try
        {
            ctorResult = _havokInterleavedConversionCtor(convertedInterleaved, sourceAnimation);
        }
        catch (Exception ex)
        {
            Marshal.FreeHGlobal((nint)convertedInterleaved);
            failure = $"Compressed animation conversion to interleaved form failed: {ex.Message}";
            return false;
        }
        finally
        {
            if (injectedSkeleton != null)
                ClearInjectedCompressedAnimationSkeleton(sourceAnimation, injectedSkeleton);
        }

        if (ctorResult == null)
        {
            Marshal.FreeHGlobal((nint)convertedInterleaved);
            failure = "Compressed animation conversion produced no interleaved animation object";
            return false;
        }

        if (ctorResult->Animation.Type != hkaAnimation.AnimationType.InterleavedAnimation)
        {
            Marshal.FreeHGlobal((nint)ctorResult);
            failure = $"Compressed animation conversion produced unexpected Havok animation type '{ctorResult->Animation.Type}'";
            return false;
        }

        if (ctorResult->Animation.NumberOfTransformTracks <= 0 || ctorResult->Transforms.Length <= 0)
        {
            Marshal.FreeHGlobal((nint)ctorResult);
            failure = "Compressed animation conversion produced no transform data";
            return false;
        }

        if (ctorResult->Animation.NumberOfTransformTracks != patch.OriginalTracks.Length)
        {
            Marshal.FreeHGlobal((nint)ctorResult);
            failure = $"Compressed animation conversion changed transform track count unexpectedly ({ctorResult->Animation.NumberOfTransformTracks} vs expected {patch.OriginalTracks.Length})";
            return false;
        }

        if (!TryValidateInterleavedFrameLayout(&ctorResult->Animation, patch.BindingIndex, out failure))
        {
            Marshal.FreeHGlobal((nint)ctorResult);
            return false;
        }

        allocatedBuffers.Add((nint)ctorResult);

        rebuiltAnimation = &ctorResult->Animation;
        if (!TryRebuildInterleavedAnimationPayload(rebuiltAnimation, patch, allocatedBuffers, out rebuiltAnimation, out failure))
            return false;

        return true;
    }

    private static string CreateTrackPlanKey(ReadOnlySpan<int> keptTrackIndices)
    {
        if (keptTrackIndices.Length == 0)
            return string.Empty;

        var builder = new StringBuilder(checked(keptTrackIndices.Length * 6));
        for (int i = 0; i < keptTrackIndices.Length; i++)
        {
            if (i > 0)
                builder.Append(',');

            builder.Append(keptTrackIndices[i]);
        }

        return builder.ToString();
    }

    private static unsafe bool TryCloneInterleavedAnimationPayload(hkaAnimation* sourceAnimation, List<nint> allocatedBuffers, out hkaAnimation* clonedAnimation, out string failure)
    {
        clonedAnimation = null;
        failure = string.Empty;

        if (sourceAnimation == null)
        {
            failure = "Interleaved animation clone source was null";
            return false;
        }

        if (sourceAnimation->Type != hkaAnimation.AnimationType.InterleavedAnimation)
        {
            failure = $"Unsupported Havok animation type '{sourceAnimation->Type}' for sender-side interleaved clone";
            return false;
        }

        if (!TryValidateInterleavedFrameLayout(sourceAnimation, -1, out failure))
            return false;

        var sourceInterleaved = (HkaInterleavedUncompressedAnimation*)sourceAnimation;
        var cloneSize = Marshal.SizeOf<HkaInterleavedUncompressedAnimation>();
        var clonedInterleaved = (HkaInterleavedUncompressedAnimation*)Marshal.AllocHGlobal(cloneSize);
        new Span<byte>(clonedInterleaved, cloneSize).Clear();
        *clonedInterleaved = *sourceInterleaved;
        allocatedBuffers.Add((nint)clonedInterleaved);

        clonedInterleaved->Transforms = CreateOwnedArray<hkQsTransformf>(AsSpan(sourceInterleaved->Transforms), allocatedBuffers);
        clonedInterleaved->Floats = CreateOwnedArray<float>(AsSpan(sourceInterleaved->Floats), allocatedBuffers);
        clonedInterleaved->Animation.AnnotationTracks = CreateOwnedArray<hkaAnnotationTrack>(AsSpan(sourceAnimation->AnnotationTracks), allocatedBuffers);

        clonedAnimation = &clonedInterleaved->Animation;
        return true;
    }

    private static unsafe ReadOnlySpan<T> AsSpan<T>(hkArray<T> array) where T : unmanaged
    {
        if (array.Length <= 0 || array.Data == null)
            return ReadOnlySpan<T>.Empty;

        return new ReadOnlySpan<T>(array.Data, array.Length);
    }

    private static unsafe bool TryRebuildInterleavedAnimationPayload(hkaAnimation* animation, BindingPatchPlan patch, List<nint> allocatedBuffers, out hkaAnimation* rebuiltAnimation, out string failure)
    {
        rebuiltAnimation = animation;
        failure = string.Empty;

        if (animation->Type != hkaAnimation.AnimationType.InterleavedAnimation)
        {
            failure = $"Unsupported Havok animation type '{animation->Type}' for sender-side transform rebuild";
            return false;
        }

        var oldTrackCount = patch.OriginalTracks.Length;
        if (oldTrackCount <= 0)
        {
            failure = "Interleaved animation had no transform tracks";
            return false;
        }

        if (!TryValidateInterleavedFrameLayout(animation, patch.BindingIndex, out failure))
            return false;

        if (animation->NumberOfTransformTracks != oldTrackCount)
        {
            failure = $"Interleaved animation track count {animation->NumberOfTransformTracks} did not match expected track count {oldTrackCount}";
            return false;
        }

        var interleaved = (HkaInterleavedUncompressedAnimation*)animation;
        var transformCount = interleaved->Transforms.Length;

        if (patch.KeptTrackIndices.Length <= 0)
        {
            failure = "Track stripping removed every transform track";
            return false;
        }

        var frameCount = transformCount / oldTrackCount;
        var compactedTransformCount = checked(frameCount * patch.KeptTrackIndices.Length);
        var compactedTransforms = ArrayPool<hkQsTransformf>.Shared.Rent(compactedTransformCount);

        try
        {
            var compactedTransformIndex = 0;
            for (int frame = 0; frame < frameCount; frame++)
            {
                var frameOffset = frame * oldTrackCount;
                for (int i = 0; i < patch.KeptTrackIndices.Length; i++)
                {
                    var sourceTrackIndex = patch.KeptTrackIndices[i];
                    var transform = interleaved->Transforms[frameOffset + sourceTrackIndex];
                    var sourceBoneIndex = patch.OriginalTracks[sourceTrackIndex];
                    var targetBoneIndex = patch.PatchedTracks[i];
                    compactedTransforms[compactedTransformIndex++] = sourceBoneIndex < 0 || targetBoneIndex < 0
                        ? transform
                        : RetargetTransform(transform, patch.SourceSkeleton.ReferencePoseByIndex[sourceBoneIndex], patch.TargetSkeleton.ReferencePoseByIndex[targetBoneIndex]);
                }
            }

            if (!TryCompactAnnotationTracksForStrippedTransforms(animation, patch.KeptTrackIndices, oldTrackCount, allocatedBuffers, out failure))
                return false;

            interleaved->Transforms = CreateOwnedArray<hkQsTransformf>(compactedTransforms.AsSpan(0, compactedTransformCount), allocatedBuffers);
            animation->NumberOfTransformTracks = patch.NewTrackCount;
            rebuiltAnimation = animation;
            return true;
        }
        finally
        {
            ArrayPool<hkQsTransformf>.Shared.Return(compactedTransforms);
        }
    }

    private static hkQsTransformf RetargetTransform(hkQsTransformf animated, hkQsTransformf sourceReference, hkQsTransformf targetReference)
    {
        var sourceRotation = NormalizeQuaternion(sourceReference.Rotation);
        var targetRotation = NormalizeQuaternion(targetReference.Rotation);
        var animatedRotation = NormalizeQuaternion(animated.Rotation);
        var rotationDelta = MultiplyQuaternion(targetRotation, InverseQuaternion(sourceRotation));
        var retargetedRotation = NormalizeQuaternion(MultiplyQuaternion(rotationDelta, animatedRotation));

        var dx = animated.Translation.X - sourceReference.Translation.X;
        var dy = animated.Translation.Y - sourceReference.Translation.Y;
        var dz = animated.Translation.Z - sourceReference.Translation.Z;
        var sourceLength = MathF.Sqrt((sourceReference.Translation.X * sourceReference.Translation.X) + (sourceReference.Translation.Y * sourceReference.Translation.Y) + (sourceReference.Translation.Z * sourceReference.Translation.Z));
        var targetLength = MathF.Sqrt((targetReference.Translation.X * targetReference.Translation.X) + (targetReference.Translation.Y * targetReference.Translation.Y) + (targetReference.Translation.Z * targetReference.Translation.Z));
        var lengthRatio = sourceLength > 1e-5f && targetLength > 1e-5f ? Math.Clamp(targetLength / sourceLength, 0.20f, 5.00f) : 1f;
        RotateVector(rotationDelta, dx * lengthRatio, dy * lengthRatio, dz * lengthRatio, out var rx, out var ry, out var rz);

        return new hkQsTransformf
        {
            Translation = new hkVector4f
            {
                X = targetReference.Translation.X + rx,
                Y = targetReference.Translation.Y + ry,
                Z = targetReference.Translation.Z + rz,
                W = animated.Translation.W,
            },
            Rotation = retargetedRotation,
            Scale = new hkVector4f
            {
                X = SafeScale(animated.Scale.X, sourceReference.Scale.X, targetReference.Scale.X),
                Y = SafeScale(animated.Scale.Y, sourceReference.Scale.Y, targetReference.Scale.Y),
                Z = SafeScale(animated.Scale.Z, sourceReference.Scale.Z, targetReference.Scale.Z),
                W = animated.Scale.W,
            },
        };
    }

    private static float SafeScale(float animated, float sourceReference, float targetReference)
        => MathF.Abs(sourceReference) > 1e-6f ? targetReference * (animated / sourceReference) : animated;

    private static hkQuaternionf NormalizeQuaternion(hkQuaternionf value)
    {
        var length = MathF.Sqrt((value.X * value.X) + (value.Y * value.Y) + (value.Z * value.Z) + (value.W * value.W));
        if (length <= 1e-8f)
            return new hkQuaternionf { X = 0f, Y = 0f, Z = 0f, W = 1f };
        var inv = 1f / length;
        return new hkQuaternionf { X = value.X * inv, Y = value.Y * inv, Z = value.Z * inv, W = value.W * inv };
    }

    private static hkQuaternionf InverseQuaternion(hkQuaternionf value)
    {
        var normalized = NormalizeQuaternion(value);
        return new hkQuaternionf { X = -normalized.X, Y = -normalized.Y, Z = -normalized.Z, W = normalized.W };
    }

    private static hkQuaternionf MultiplyQuaternion(hkQuaternionf left, hkQuaternionf right)
        => new()
        {
            X = (left.W * right.X) + (left.X * right.W) + (left.Y * right.Z) - (left.Z * right.Y),
            Y = (left.W * right.Y) - (left.X * right.Z) + (left.Y * right.W) + (left.Z * right.X),
            Z = (left.W * right.Z) + (left.X * right.Y) - (left.Y * right.X) + (left.Z * right.W),
            W = (left.W * right.W) - (left.X * right.X) - (left.Y * right.Y) - (left.Z * right.Z),
        };

    private static void RotateVector(hkQuaternionf q, float x, float y, float z, out float rx, out float ry, out float rz)
    {
        q = NormalizeQuaternion(q);
        var tx = 2f * ((q.Y * z) - (q.Z * y));
        var ty = 2f * ((q.Z * x) - (q.X * z));
        var tz = 2f * ((q.X * y) - (q.Y * x));
        rx = x + (q.W * tx) + ((q.Y * tz) - (q.Z * ty));
        ry = y + (q.W * ty) + ((q.Z * tx) - (q.X * tz));
        rz = z + (q.W * tz) + ((q.X * ty) - (q.Y * tx));
    }

    private static unsafe bool TryValidateInterleavedFrameLayout(hkaAnimation* animation, int bindingIndex, out string failure)
    {
        failure = string.Empty;

        if (animation == null)
        {
            failure = bindingIndex >= 0
                ? $"Patched PAP binding {bindingIndex} had a null interleaved animation"
                : "Interleaved animation pointer was null";
            return false;
        }

        if (animation->Type != hkaAnimation.AnimationType.InterleavedAnimation)
            return true;

        var interleaved = (HkaInterleavedUncompressedAnimation*)animation;
        if (animation->NumberOfTransformTracks <= 0)
        {
            failure = bindingIndex >= 0
                ? $"Patched PAP binding {bindingIndex} had no transform tracks after interleaved rebuild"
                : "Interleaved animation had no transform tracks";
            return false;
        }

        if (interleaved->Transforms.Length <= 0 || interleaved->Transforms.Length % animation->NumberOfTransformTracks != 0)
        {
            failure = bindingIndex >= 0
                ? $"Patched PAP binding {bindingIndex} had an invalid interleaved transform buffer length {interleaved->Transforms.Length} for track count {animation->NumberOfTransformTracks}"
                : $"Interleaved transform buffer length {interleaved->Transforms.Length} was not divisible by track count {animation->NumberOfTransformTracks}";
            return false;
        }

        var frameCount = interleaved->Transforms.Length / animation->NumberOfTransformTracks;
        if (frameCount <= 0)
        {
            failure = bindingIndex >= 0
                ? $"Patched PAP binding {bindingIndex} had no animation frames after interleaved rebuild"
                : "Interleaved animation had no animation frames";
            return false;
        }

        if (animation->NumberOfFloatTracks <= 0)
        {
            if (interleaved->Floats.Length != 0)
            {
                failure = bindingIndex >= 0
                    ? $"Patched PAP binding {bindingIndex} kept float samples without any float tracks"
                    : "Interleaved animation had float samples without any float tracks";
                return false;
            }

            return true;
        }

        if (interleaved->Floats.Length <= 0 || interleaved->Floats.Length % animation->NumberOfFloatTracks != 0)
        {
            failure = bindingIndex >= 0
                ? $"Patched PAP binding {bindingIndex} had an invalid float buffer length {interleaved->Floats.Length} for float track count {animation->NumberOfFloatTracks}"
                : $"Interleaved float buffer length {interleaved->Floats.Length} was not divisible by float track count {animation->NumberOfFloatTracks}";
            return false;
        }

        var floatFrameCount = interleaved->Floats.Length / animation->NumberOfFloatTracks;
        if (floatFrameCount != frameCount)
        {
            failure = bindingIndex >= 0
                ? $"Patched PAP binding {bindingIndex} had mismatched transform frame count {frameCount} and float frame count {floatFrameCount}"
                : $"Interleaved transform frame count {frameCount} did not match float frame count {floatFrameCount}";
            return false;
        }

        return true;
    }

    private static unsafe bool TryValidateBindingSideArrayCounts(hkaAnimationBinding* binding, hkaAnimation* animation, int bindingIndex, out string failure)
    {
        failure = string.Empty;

        if (binding == null || animation == null)
            return true;

        var expectedTransformTrackCount = animation->NumberOfTransformTracks;
        var expectedCombinedTrackCount = expectedTransformTrackCount + Math.Max(0, animation->NumberOfFloatTracks);

        if (animation->AnnotationTracks.Length > 0
            && animation->AnnotationTracks.Length != expectedTransformTrackCount
            && animation->AnnotationTracks.Length != expectedCombinedTrackCount)
        {
            failure = $"PAP binding {bindingIndex} had an invalid annotation track count {animation->AnnotationTracks.Length} for transform/float track counts {expectedTransformTrackCount}/{animation->NumberOfFloatTracks}";
            return false;
        }

        if (binding->PartitionIndices.Length > 0
            && binding->PartitionIndices.Length != expectedTransformTrackCount
            && binding->PartitionIndices.Length != expectedCombinedTrackCount)
        {
            failure = $"PAP binding {bindingIndex} had an invalid partition index count {binding->PartitionIndices.Length} for transform/float track counts {expectedTransformTrackCount}/{animation->NumberOfFloatTracks}";
            return false;
        }

        return true;
    }

    private static unsafe bool TryCompactAnnotationTracksForStrippedTransforms(hkaAnimation* animation, ReadOnlySpan<int> keptTrackIndices, int oldTrackCount, List<nint> allocatedBuffers, out string failure)
    {
        failure = string.Empty;

        var annotationTrackCount = animation->AnnotationTracks.Length;
        if (annotationTrackCount == 0)
            return true;

        if (annotationTrackCount != oldTrackCount && annotationTrackCount != oldTrackCount + animation->NumberOfFloatTracks)
        {
            failure = $"Annotation track count {annotationTrackCount} did not align with transform/float track counts ({oldTrackCount}, {animation->NumberOfFloatTracks})";
            return false;
        }

        var compactedAnnotationTrackCount = checked(keptTrackIndices.Length + Math.Max(0, annotationTrackCount - oldTrackCount));
        var compactedAnnotationTracks = ArrayPool<hkaAnnotationTrack>.Shared.Rent(compactedAnnotationTrackCount);

        try
        {
            var compactedIndex = 0;
            for (int i = 0; i < keptTrackIndices.Length; i++)
                compactedAnnotationTracks[compactedIndex++] = animation->AnnotationTracks[keptTrackIndices[i]];

            for (int i = oldTrackCount; i < annotationTrackCount; i++)
                compactedAnnotationTracks[compactedIndex++] = animation->AnnotationTracks[i];

            animation->AnnotationTracks = CreateOwnedArray<hkaAnnotationTrack>(compactedAnnotationTracks.AsSpan(0, compactedAnnotationTrackCount), allocatedBuffers);
            return true;
        }
        finally
        {
            ArrayPool<hkaAnnotationTrack>.Shared.Return(compactedAnnotationTracks);
        }
    }

    private static unsafe bool TryCompactBindingPartitionIndices(hkaAnimationBinding* binding, ReadOnlySpan<int> keptTrackIndices, int oldTrackCount, int floatTrackCount, List<nint> allocatedBuffers, out string failure)
    {
        failure = string.Empty;

        var partitionCount = binding->PartitionIndices.Length;
        if (partitionCount == 0)
            return true;

        if (partitionCount != oldTrackCount && partitionCount != oldTrackCount + floatTrackCount)
        {
            failure = $"Partition index count {partitionCount} did not align with transform/float track counts ({oldTrackCount}, {floatTrackCount})";
            return false;
        }

        var compactedPartitionCount = checked(keptTrackIndices.Length + Math.Max(0, partitionCount - oldTrackCount));
        var compactedPartitionIndices = ArrayPool<short>.Shared.Rent(compactedPartitionCount);

        try
        {
            var compactedIndex = 0;
            for (int i = 0; i < keptTrackIndices.Length; i++)
                compactedPartitionIndices[compactedIndex++] = binding->PartitionIndices[keptTrackIndices[i]];

            for (int i = oldTrackCount; i < partitionCount; i++)
                compactedPartitionIndices[compactedIndex++] = binding->PartitionIndices[i];

            binding->PartitionIndices = CreateOwnedArray<short>(compactedPartitionIndices.AsSpan(0, compactedPartitionCount), allocatedBuffers);
            return true;
        }
        finally
        {
            ArrayPool<short>.Shared.Return(compactedPartitionIndices);
        }
    }

    private static bool IsSupportedSenderSideRebuildAnimationType(hkaAnimation.AnimationType animationType)
    {
        return animationType is hkaAnimation.AnimationType.SplineCompressedAnimation
            or hkaAnimation.AnimationType.PredictiveCompressedAnimation
            or hkaAnimation.AnimationType.QuantizedCompressedAnimation;
    }

    private static HavokInterleavedConversionCtorDelegate? ResolveHavokInterleavedConversionCtor(ISigScanner sigScanner)
    {
        try
        {
            var ctorAddress = sigScanner.ScanText(HavokInterleavedConversionCtorSig);
            return Marshal.GetDelegateForFunctionPointer<HavokInterleavedConversionCtorDelegate>(ctorAddress);
        }
        catch
        {
            return null;
        }
    }

    private static bool RequiresDummySkeletonForCompressedConversion(hkaAnimation.AnimationType animationType)
    {
        return animationType is hkaAnimation.AnimationType.PredictiveCompressedAnimation
            or hkaAnimation.AnimationType.QuantizedCompressedAnimation;
    }

    private static unsafe bool TryEnsureCompressedAnimationSkeleton(hkaAnimationBinding* binding,hkaAnimation* animation,AnimationTargetSkeletonSnapshot targetSkeleton,List<nint> allocatedBuffers,out hkaSkeleton* injectedSkeleton,out string failure)
    {
        injectedSkeleton = null;
        failure = string.Empty;

        if (animation == null)
        {
            failure = "Compressed animation pointer was null";
            return false;
        }

        if (!RequiresDummySkeletonForCompressedConversion(animation->Type))
            return true;

        var skeletonField = GetCompressedAnimationSkeletonField(animation);
        if (skeletonField == null)
        {
            failure = $"Compressed animation type '{animation->Type}' did not expose a skeleton field for conversion";
            return false;
        }

        if (*skeletonField != null)
            return true;

        if (targetSkeleton == null || targetSkeleton.ReferencePoseByIndex.Count <= 0)
        {
            failure = $"Compressed animation rebuild could not inject a dummy target skeleton because '{targetSkeleton?.SkeletonName ?? "<unknown>"}' had no reference pose data";
            return false;
        }

        var requiredBoneCount = Math.Max(1, GetMaxReferencedBoneIndex(binding) + 1);
        injectedSkeleton = AllocateDummyCompressedAnimationSkeleton(targetSkeleton.ReferencePoseByIndex, requiredBoneCount, allocatedBuffers);
        if (injectedSkeleton == null)
        {
            failure = "Compressed animation rebuild could not allocate a dummy target skeleton for conversion";
            return false;
        }

        *skeletonField = injectedSkeleton;
        return true;
    }

    private static unsafe hkaSkeleton** GetCompressedAnimationSkeletonField(hkaAnimation* animation)
    {
        if (animation == null)
            return null;

        return animation->Type switch
        {
            hkaAnimation.AnimationType.PredictiveCompressedAnimation => &((HkaPredictiveCompressedAnimation*)animation)->Skeleton,
            hkaAnimation.AnimationType.QuantizedCompressedAnimation => &((HkaQuantizedCompressedAnimation*)animation)->Skeleton,
            _ => null,
        };
    }

    private static unsafe hkaSkeleton* AllocateDummyCompressedAnimationSkeleton(IReadOnlyList<hkQsTransformf> targetReferencePoseByIndex,int requiredBoneCount,List<nint> allocatedBuffers)
    {
        requiredBoneCount = Math.Max(1, requiredBoneCount);

        var skeleton = (hkaSkeleton*)Marshal.AllocHGlobal(sizeof(hkaSkeleton));
        new Span<byte>(skeleton, sizeof(hkaSkeleton)).Clear();
        allocatedBuffers.Add((nint)skeleton);

        var poseBuffer = (hkQsTransformf*)Marshal.AllocHGlobal(requiredBoneCount * sizeof(hkQsTransformf));
        new Span<byte>(poseBuffer, requiredBoneCount * sizeof(hkQsTransformf)).Clear();
        allocatedBuffers.Add((nint)poseBuffer);

        var identityTransform = new hkQsTransformf
        {
            Translation = new hkVector4f { X = 0f, Y = 0f, Z = 0f, W = 0f },
            Rotation = new hkQuaternionf { X = 0f, Y = 0f, Z = 0f, W = 1f },
            Scale = new hkVector4f { X = 1f, Y = 1f, Z = 1f, W = 0f },
        };

        var sharedCount = Math.Min(targetReferencePoseByIndex.Count, requiredBoneCount);
        for (int i = 0; i < sharedCount; i++)
            poseBuffer[i] = targetReferencePoseByIndex[i];

        for (int i = sharedCount; i < requiredBoneCount; i++)
            poseBuffer[i] = identityTransform;

        skeleton->ReferencePose = new hkArray<hkQsTransformf>
        {
            Data = poseBuffer,
            Length = requiredBoneCount,
            CapacityAndFlags = requiredBoneCount | unchecked((int)hkArray<hkQsTransformf>.hkArrayFlags.DontDeallocate),
        };

        return skeleton;
    }

    private static unsafe void ClearInjectedCompressedAnimationSkeleton(hkaAnimation* animation, hkaSkeleton* injectedSkeleton)
    {
        if (animation == null || injectedSkeleton == null)
            return;

        var skeletonField = GetCompressedAnimationSkeletonField(animation);
        if (skeletonField != null && *skeletonField == injectedSkeleton)
            *skeletonField = null;
    }

    private static unsafe int GetMaxReferencedBoneIndex(hkaAnimationBinding* binding)
    {
        if (binding == null)
            return -1;

        var transformTracks = binding->TransformTrackToBoneIndices;
        var maxReferencedIndex = -1;
        for (int trackIndex = 0; trackIndex < transformTracks.Length; trackIndex++)
        {
            var boneIndex = transformTracks[trackIndex];
            if (boneIndex > maxReferencedIndex)
                maxReferencedIndex = boneIndex;
        }

        return maxReferencedIndex;
    }
    private static unsafe hkArray<T> CreateOwnedArray<T>(ReadOnlySpan<T> data, List<nint> allocatedBuffers) where T : unmanaged
    {
        var count = data.Length;
        T* buffer = null;

        if (count > 0)
        {
            var bytes = checked(count * sizeof(T));
            buffer = (T*)Marshal.AllocHGlobal(bytes);
            data.CopyTo(new Span<T>(buffer, count));
            allocatedBuffers.Add((nint)buffer);
        }

        return new hkArray<T>
        {
            Data = buffer,
            Length = count,
            CapacityAndFlags = count | unchecked((int)hkArray<T>.hkArrayFlags.DontDeallocate),
        };
    }

    private static Dictionary<string, AnimationTargetSkeletonSnapshot> BuildTargetLookup(IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons)
    {
        var output = new Dictionary<string, AnimationTargetSkeletonSnapshot>(StringComparer.Ordinal);
        foreach (var skeleton in targetSkeletons.OrderByDescending(static s => s.BoneCount))
        {
            AddTargetLookupEntry(output, skeleton.NormalizedSkeletonName, skeleton);
            AddTargetLookupEntry(output, skeleton.NormalizedResourceName, skeleton);
            AddTargetLookupEntry(output, skeleton.HumanAnimationFamilyKey, skeleton);
        }

        return output;
    }

    private static AnimationTargetSkeletonSnapshot? ResolveTargetSkeleton(string originalSkeletonName, string normalizedOriginalSkeleton, IReadOnlyDictionary<string, AnimationTargetSkeletonSnapshot> targetLookup, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons, int maxReferencedIndex)
    {
        if (!string.IsNullOrWhiteSpace(normalizedOriginalSkeleton) && targetLookup.TryGetValue(normalizedOriginalSkeleton, out var exact))
            return exact;

        var markerBoneName = ExtractBindingMarkerBoneName(originalSkeletonName);
        if (!string.IsNullOrWhiteSpace(markerBoneName))
        {
            var markerMatch = ChooseBestTargetSkeletonCandidate(
                targetSkeletons.Where(snapshot => snapshot.BoneNameToIndex.ContainsKey(markerBoneName)),
                maxReferencedIndex);
            if (markerMatch != null)
                return markerMatch;
        }

        var humanAnimationFamilyKey = XivAnimationSkeletonIdentity.NormalizeHumanAnimationFamilyKey(originalSkeletonName);
        if (!string.IsNullOrWhiteSpace(humanAnimationFamilyKey))
        {
            var familyMatch = ChooseBestTargetSkeletonCandidate(
                targetSkeletons.Where(snapshot => string.Equals(snapshot.HumanAnimationFamilyKey, humanAnimationFamilyKey, StringComparison.Ordinal)),
                maxReferencedIndex);
            if (familyMatch != null)
                return familyMatch;

            if (humanAnimationFamilyKey.StartsWith("human-partial:", StringComparison.Ordinal))
                return null;
        }

        if (!string.IsNullOrWhiteSpace(normalizedOriginalSkeleton))
        {
            var fuzzyMatch = ChooseBestTargetSkeletonCandidate(
                targetSkeletons.Where(target =>
                    target.NormalizedSkeletonName.Contains(normalizedOriginalSkeleton, StringComparison.Ordinal)
                    || normalizedOriginalSkeleton.Contains(target.NormalizedSkeletonName, StringComparison.Ordinal)
                    || target.NormalizedResourceName.Contains(normalizedOriginalSkeleton, StringComparison.Ordinal)
                    || normalizedOriginalSkeleton.Contains(target.NormalizedResourceName, StringComparison.Ordinal)),
                maxReferencedIndex);
            if (fuzzyMatch != null)
                return fuzzyMatch;
        }

        var preferredHumanAnimation = ChooseBestTargetSkeletonCandidate(
            targetSkeletons.Where(static s => s.IsHumanAnimationSkeleton),
            maxReferencedIndex);
        if (preferredHumanAnimation != null)
            return preferredHumanAnimation;

        if (targetSkeletons.Count == 1)
            return targetSkeletons[0];

        return null;
    }

    private static string ExtractBindingMarkerBoneName(string? skeletonName)
    {
        if (string.IsNullOrWhiteSpace(skeletonName))
            return string.Empty;

        var normalized = skeletonName.Replace('\\', '/').Trim();
        var lastColon = normalized.LastIndexOf(':');
        if (lastColon >= 0 && lastColon < normalized.Length - 1)
            return normalized[(lastColon + 1)..].Trim().ToLowerInvariant();

        return XivAnimationSkeletonIdentity.NormalizeSkeletonKey(normalized);
    }

    private static AnimationTargetSkeletonSnapshot? ChooseBestTargetSkeletonCandidate(IEnumerable<AnimationTargetSkeletonSnapshot> candidates, int maxReferencedIndex)
    {
        AnimationTargetSkeletonSnapshot? bestAdequate = null;
        AnimationTargetSkeletonSnapshot? bestFallback = null;

        foreach (var candidate in candidates)
        {
            if (candidate == null || candidate.BoneCount <= 0)
                continue;

            if (candidate.BoneCount > maxReferencedIndex)
            {
                if (bestAdequate == null || candidate.BoneCount < bestAdequate.BoneCount)
                    bestAdequate = candidate;
            }
            else
            {
                if (bestFallback == null || candidate.BoneCount > bestFallback.BoneCount)
                    bestFallback = candidate;
            }
        }

        return bestAdequate ?? bestFallback;
    }

    private static PapContainerReadStatus TryReadPapContainer(string filePath, out PapContainerData papContainer, out string failure)
    {
        papContainer = default!;
        failure = string.Empty;

        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(filePath);
        }
        catch (Exception ex)
        {
            failure = $"Could not read PAP from disk: {ex.Message}";
            return PapContainerReadStatus.Invalid;
        }

        if (bytes.Length < 64)
        {
            failure = "PAP file was too small to be valid";
            return PapContainerReadStatus.Invalid;
        }

        using var reader = new BinaryReader(new MemoryStream(bytes, writable: false));

        if (reader.ReadUInt32() != 0x20706170)
        {
            failure = "PAP magic was invalid";
            return PapContainerReadStatus.Invalid;
        }

        reader.ReadUInt16();
        reader.ReadUInt16();
        reader.ReadUInt16();
        reader.ReadUInt16();

        byte type = reader.ReadByte();
        reader.ReadByte();

        int headerSize = reader.ReadInt32();
        int havokOffset = reader.ReadInt32();
        int footerOffset = reader.ReadInt32();

        if (headerSize <= 0 || headerSize > havokOffset)
        {
            failure = "PAP header contained an invalid header size";
            return PapContainerReadStatus.Invalid;
        }

        if (havokOffset <= 0 || footerOffset <= 0 || footerOffset <= havokOffset)
        {
            failure = "PAP header contained invalid Havok offsets";
            return PapContainerReadStatus.Invalid;
        }

        if (havokOffset >= bytes.Length || footerOffset > bytes.Length)
        {
            failure = "PAP offsets pointed outside the file";
            return PapContainerReadStatus.Invalid;
        }

        int havokSize = footerOffset - havokOffset;
        if (havokSize <= 32 || havokSize > MaxHavokBytes)
        {
            failure = "PAP Havok section had an invalid size";
            return PapContainerReadStatus.Invalid;
        }

        var havokBytes = new byte[havokSize];
        Buffer.BlockCopy(bytes, havokOffset, havokBytes, 0, havokSize);

        var headerBytes = new byte[havokOffset];
        Buffer.BlockCopy(bytes, 0, headerBytes, 0, havokOffset);

        var footerBytes = new byte[bytes.Length - footerOffset];
        Buffer.BlockCopy(bytes, footerOffset, footerBytes, 0, footerBytes.Length);

        papContainer = new PapContainerData(headerSize, havokOffset, footerOffset, type == 0, headerBytes, havokBytes, footerBytes);
        return PapContainerReadStatus.Valid;
    }


    private static byte[] BuildPapFromRewrittenHkx(PapContainerData originalPap, byte[] rewrittenHkx)
    {
        var unpaddedFooterOffset = originalPap.HavokOffset + rewrittenHkx.Length;
        var originalFooterRemainder = originalPap.FooterOffset & 3;
        var padding = (originalFooterRemainder - (unpaddedFooterOffset & 3) + 4) & 3;
        var rewrittenFooterOffset = unpaddedFooterOffset + padding;

        var output = new byte[originalPap.HeaderBytes.Length + rewrittenHkx.Length + padding + originalPap.FooterBytes.Length];
        Buffer.BlockCopy(originalPap.HeaderBytes, 0, output, 0, originalPap.HeaderBytes.Length);
        Buffer.BlockCopy(rewrittenHkx, 0, output, originalPap.HeaderBytes.Length, rewrittenHkx.Length);
        Buffer.BlockCopy(originalPap.FooterBytes, 0, output, rewrittenFooterOffset, originalPap.FooterBytes.Length);
        BinaryPrimitives.WriteInt32LittleEndian(output.AsSpan(22, sizeof(int)), rewrittenFooterOffset);
        return output;
    }

    private unsafe bool TryVerifyMaterializedPap(string rewrittenPath, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons, int expectedBindingCount, out string failure)
    {
        failure = string.Empty;
        var readStatus = TryReadPapContainer(rewrittenPath, out var papContainer, out var readFailure);
        if (readStatus != PapContainerReadStatus.Valid)
        {
            failure = readStatus == PapContainerReadStatus.PassThroughOriginal
                ? "Rewritten PAP verification produced an unsupported Havok payload"
                : readFailure;
            return false;
        }

        var tempHkxPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName()) + ".hkx";
        File.WriteAllBytes(tempHkxPath, papContainer.HavokBytes);
        var tempHkxPathAnsi = Marshal.StringToHGlobalAnsi(tempHkxPath);
        try
        {
            var loadOptions = stackalloc hkSerializeUtil.LoadOptions[1];
            loadOptions->TypeInfoRegistry = hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry();
            loadOptions->ClassNameRegistry = hkBuiltinTypeRegistry.Instance()->GetClassNameRegistry();
            loadOptions->Flags = new hkFlags<hkSerializeUtil.LoadOptionBits, int>
            {
                Storage = (int)hkSerializeUtil.LoadOptionBits.Default
            };

            var resource = hkSerializeUtil.LoadFromFile((byte*)tempHkxPathAnsi, null, loadOptions);
            if (resource == null)
            {
                failure = "Rewritten PAP verification could not reload the emitted Havok payload";
                return false;
            }

            var rootLevelName = "hkRootLevelContainer"u8;
            fixed (byte* rootName = rootLevelName)
            {
                var container = (hkRootLevelContainer*)resource->GetContentsPointer(rootName, hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry());
                if (container == null)
                {
                    failure = "Rewritten PAP verification could not resolve the Havok root container";
                    return false;
                }

                var animationName = "hkaAnimationContainer"u8;
                fixed (byte* animName = animationName)
                {
                    var animationContainer = (hkaAnimationContainer*)container->findObjectByName(animName, null);
                    if (animationContainer == null)
                    {
                        failure = "Rewritten PAP verification could not resolve the animation container";
                        return false;
                    }

                    if (!TryValidatePatchedBindingsInMemory(animationContainer, targetSkeletons, expectedBindingCount, out failure))
                        return false;
                }
            }

            return true;
        }
        finally
        {
            Marshal.FreeHGlobal(tempHkxPathAnsi);
            try { File.Delete(tempHkxPath); } catch { }
        }
    }

    private static unsafe bool TrySaveResourceToHkx(hkResource* resource, out byte[] rewrittenHkx, out string failure)
    {
        rewrittenHkx = Array.Empty<byte>();
        failure = string.Empty;

        string tempFile = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName()) + ".hkx";
        var tempFileAnsi = Marshal.StringToHGlobalAnsi(tempFile);

        try
        {
            var rootLevelName = "hkRootLevelContainer"u8;
            fixed (byte* rootName = rootLevelName)
            {
                var rootPointer = (hkRootLevelContainer*)resource->GetContentsPointer(rootName, hkBuiltinTypeRegistry.Instance()->GetTypeInfoRegistry());
                if (rootPointer == null)
                {
                    failure = "Havok root pointer was unavailable during save";
                    return false;
                }

                var rootClass = hkBuiltinTypeRegistry.Instance()->GetClassNameRegistry()->GetClassByName(rootName);
                if (rootClass == null)
                {
                    failure = "Havok root class was unavailable during save";
                    return false;
                }

                hkOstream* outputStream = stackalloc hkOstream[1];
                outputStream->Ctor((byte*)tempFileAnsi);

                try
                {
                    hkResult* result = stackalloc hkResult[1];
                    var options = new hkSerializeUtil.SaveOptions
                    {
                        Flags = new hkFlags<hkSerializeUtil.SaveOptionBits, int>
                        {
                            Storage = (int)hkSerializeUtil.SaveOptionBits.Default
                        }
                    };

                    hkSerializeUtil.Save(result, rootPointer, rootClass, outputStream->StreamWriter.ptr, options);
                    if (result->Result != hkResult.hkResultEnum.Success)
                    {
                        failure = "Havok serializer returned failure";
                        return false;
                    }
                }
                finally
                {
                    outputStream->Dtor();
                }
            }

            rewrittenHkx = File.ReadAllBytes(tempFile);
            if (rewrittenHkx.Length == 0)
            {
                failure = "Havok serializer returned an empty HKX payload";
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            failure = ex.Message;
            return false;
        }
        finally
        {
            Marshal.FreeHGlobal(tempFileAnsi);
            try { File.Delete(tempFile); } catch { }
        }
    }


    private unsafe bool TryValidatePatchedBindingsInMemory(hkaAnimationContainer* animationContainer, IReadOnlyList<AnimationTargetSkeletonSnapshot> targetSkeletons, int expectedBindingCount, out string failure)
    {
        failure = string.Empty;

        if (animationContainer == null)
        {
            failure = "Animation container was null during in-memory verification";
            return false;
        }

        var actualBindingCount = animationContainer->Bindings.Length;
        if (expectedBindingCount > 0 && actualBindingCount != expectedBindingCount)
        {
            failure = $"Patched animation binding count changed unexpectedly ({expectedBindingCount} -> {actualBindingCount})";
            return false;
        }

        for (int bindingIndex = 0; bindingIndex < actualBindingCount; bindingIndex++)
        {
            var binding = animationContainer->Bindings[bindingIndex].ptr;
            if (binding == null)
                continue;

            var animation = (hkaAnimation*)binding->Animation.ptr;
            if (animation == null)
            {
                failure = $"Patched PAP binding {bindingIndex} lost its animation reference";
                return false;
            }

            if (animation->NumberOfTransformTracks != binding->TransformTrackToBoneIndices.Length)
            {
                failure = $"Patched PAP binding {bindingIndex} track map length {binding->TransformTrackToBoneIndices.Length} did not match animation track count {animation->NumberOfTransformTracks}";
                return false;
            }

            if (animation->Type == hkaAnimation.AnimationType.InterleavedAnimation
                && !TryValidateInterleavedFrameLayout(animation, bindingIndex, out failure))
            {
                return false;
            }

            if (!TryValidateBindingSideArrayCounts(binding, animation, bindingIndex, out failure))
                return false;
        }

        var targetLookup = BuildTargetLookup(targetSkeletons);
        for (int bindingIndex = 0; bindingIndex < actualBindingCount; bindingIndex++)
        {
            var binding = animationContainer->Bindings[bindingIndex].ptr;
            if (binding == null)
                continue;
            var tracks = binding->TransformTrackToBoneIndices;
            short maxIndex = -1;
            for (int i = 0; i < tracks.Length; i++)
                if (tracks[i] > maxIndex) maxIndex = tracks[i];
            var originalSkeletonName = binding->OriginalSkeletonName.String ?? string.Empty;
            var targetSkeleton = ResolveTargetSkeleton(originalSkeletonName, XivAnimationSkeletonIdentity.NormalizeSkeletonKey(originalSkeletonName), targetLookup, targetSkeletons, maxIndex);
            if (targetSkeleton == null)
            {
                failure = $"Patched PAP binding {bindingIndex} could not be matched to a target skeleton during verification";
                return false;
            }
            for (int i = 0; i < tracks.Length; i++)
            {
                var boneIndex = tracks[i];
                if (boneIndex >= 0 && boneIndex >= targetSkeleton.BoneCount)
                {
                    failure = $"Patched PAP binding {bindingIndex} still referenced target bone index {boneIndex} outside skeleton '{targetSkeleton.SkeletonName}'";
                    return false;
                }
            }
        }

        return true;
    }

}
