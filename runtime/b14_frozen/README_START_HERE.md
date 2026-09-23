# LoBoFit FFXIV — EXACT B14 reproducer

This is the recovered **B14 production solver**, not the earlier B07/B08 approximation.

For the preserved Original Outfit + Selected Body test pair, a clean source-only recovery run produced the original B14 files **byte-for-byte**:

- `B14_final_garment.glb` SHA256: `1da2405dddd6ea47ff3bdf8e31af9ea3d579e5fafc0021cf057d5de78a165255`
- `B14_with_Selected_Body.glb` SHA256: `86e9b17ce475008c6307a1d354f327f1fadd3f66544138cd9d10dc884435ba78`

The integrity audit passes with zero opposed/degenerate faces on every garment mesh and an exact Selected Body transplant.

## Important research rule

The production solver reads only:

- `inputs/Original_outfit.glb`
- `inputs/Selected_Body.glb`

It does **not** read `reference/`, YAB, the hand-fitted control, or any previous generated candidate. The reference folder is optional and exists only so `verify_B14_reference.py` can compare a completed/frozen result afterward.

## Windows quick start

1. Extract Parts 01 and 02 into the same parent folder so `B14_EXACT_REPRODUCER_RELEASE` merges.
2. Run `SETUP_WINDOWS.bat` once.
3. Run `RUN_B14.bat`.
4. Outputs appear in `candidates/`:
   - `B14_final_garment.glb`
   - `B14_with_Selected_Body.glb`

Part 03 is optional. Merge it too if you want the frozen golden B14 outputs and post-run regression verifier. Then run `VERIFY_GOLDEN.bat`.

## Command line

```bash
python scripts/build_B14_fresh.py
```

The launcher:

1. builds a fresh source/target body correspondence cache;
2. solves every garment mesh in an isolated process;
3. routes meshes by generic geometry/contact behaviour (`stand_off_structured_shell`, `constructed_close_shell`, `body_following_flexible_layer`, `conservative_component_assembly`);
4. assembles a fresh garment GLB;
5. exact-transplants the Selected Body;
6. performs the B14 integrity audit.

## Using another outfit / body

Replace the two files under `inputs/` while keeping these filenames:

- `Original_outfit.glb`
- `Selected_Body.glb`

Then run B14 again. There is no fitted comparison/reference required at inference time.

## Source provenance

B14 source was recovered from:

- the byte-verified original B07 workspace;
- the surviving B08/B14 checkpoints;
- the prior-chat B11→B14 development transcript containing the actual production source-creation blocks.

The recovery was accepted only after a clean solve with the validation/control asset absent generated both original B14 GLBs with their exact historical SHA256 hashes.
