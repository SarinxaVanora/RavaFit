# RavaFit LoBoMap Production Lane

This lane is intentionally additive. `production_b14.py` remains the trusted strict-B14 checkpoint path and is not modified by this work.

## Production principle

**One coherent source-to-target deformation field moves the outfit. Geometry type changes preservation constraints, not deformation authority.**

The lane follows the representation described by *LoBoFit: Flexible Garment Refitting via Local Bone Mapping Blending*:

1. Encode rendered garment vertices in sparse source bone-local coordinates.
2. Decode the whole garment through matching target bone-local frames for a coherent initialization.
3. Refine body fit with localized residuals while preserving source fit style.
4. Infer authored components/layers from untouched source geometry and make their construction an explicit solve constraint.
5. Resolve literal target-body contact against target triangles without sacrificing component topology.
6. Restore exact raw MDL storage only after solved rendered topology is frozen.

## RavaFit-specific rules

- Production inputs are only untouched source garment/body + untouched target body.
- Authored YAB/hand-fitted controls are validation-only and never enter solve geometry, objectives, anchors, warm starts, or correspondences.
- No garment-name/type conditionals.
- Connectivity is authoritative for component identity; material, body-support region, clearance and mirrored-peer evidence infer larger authored layers.
- Rigid ornaments preserve one uniform source-derived reference shape and then move by SE(3); they are not freely vertex-warped.
- Semi-rigid/flexible pieces retain the coherent target initialisation as differential-structure authority.
- Thin-wall and inter-layer relationships are source-derived and operate by coherent component/layer motion, not target-body embossing.
- Bilateral components retain independent identity.
- Structural quality metrics are acceptance checks, not excuses to add an arbitrary global post-finalizer.

## Current implementation

`lobomap_production.py` provides:

- hierarchical LoBoMap frame construction;
- sparse top-K local-coordinate compilation;
- exact source-frame round-trip;
- coherent target-frame decode;
- RBODY/FFXIV body-correspondence frames from skin-weighted source/target local similarities;
- strict rendered-topology `LoBoMapMeshInput` views with exact raw-storage vertex IDs retained for later restoration;
- direct collection of real RavaFit/FFXIV GLB meshes with caller-supplied body-mesh exclusion rather than garment-name logic;
- whole-outfit initialization in one shared frame field.

`lobomap_refinement.py` provides the localized body-fit data term:

- source-space body correspondence selected once and carried to the target body;
- skin-affinity-biased spatial matching to reduce cross-limb ambiguity;
- source signed normal clearance retained as the fit-style signal;
- coarse-to-fine residual diffusion over the authored mesh graph;
- stronger boundary data fidelity;
- source-relative and initial-relative diagnostics for edge strain, triangle area, normalized-Laplacian/detail drift, near-collapse and triangle orientation/flips;
- `refine_ffxiv_batch_structured`, which composes body fit with the authored-structure and literal collision constraints.
- `solve_ffxiv_source_structured`, the one-call real-FFXIV path: coherent source initialization, exact RBODY-backed target collision surface, authored-structure solve and convergent collision.

`lobomap_structure.py` adds the first explicit authored-structure solve:

- topology-first connected-component inference;
- deterministic stable component/layer IDs;
- geometry-only rigid / semi-rigid / flexible classification;
- body-geometry-derived bilateral symmetry plane (no garment/body names);
- layer grouping from material continuity, similar source clearance, overlapping source support region and mirrored-peer evidence;
- source body-relative ordering relationships between overlapping layers;
- close parallel thin-wall component relations;
- rigid component source-shape preservation through one uniform source->target reference followed by SE(3) projection;
- coherent clearance translation for rigid components without changing their shape;
- semi-rigid SE(3) bias plus target-initial edge constraints;
- flexible target-initial edge constraints;
- local topology guard that removes flips/collapse by rolling back only threatening neighbourhoods rather than the whole garment;
- exact target triangle collision using the original body topology plus RBODY target vertices/normals;
- topology-aware collision: local correction first, coherent component translation only when required to clear remaining real penetration.

## Real FFXIV fixture status

The embedded untouched strict-B14 outfit fixture remains the integration check. With the three body meshes excluded, this checkpoint processes:

- **8 garment meshes**;
- **38,471 rendered garment vertices**;
- **804 inferred connected authored components**;
- **6 inferred authored layers**;
- **723 rigid**, **67 semi-rigid**, **14 flexible** components;
- **4 inferred thin-wall relations**;
- **7 inferred inter-layer ordering relationships**.

Measured on the current development worker:

- coherent initialisation: approximately **0.54 s**;
- plain local residual stage: approximately **0.46 s**;
- full structure + literal collision stage: approximately **12.89 s** before performance optimisation.

Fit/structure results on this fixture:

- worst per-mesh p95 body-relative clearance mismatch starts at approximately **25.398 mm** after coherent initialisation;
- plain residual fitting can drive that metric to approximately **0.066 mm**, but on this fixture it also creates **39 triangle flips total** and **4 near-collapse triangles** across the two hardest meshes;
- the structured solve finishes with **zero triangle flips**, **zero near-collapse triangles**, and **zero actual target-body penetrations**;
- final worst per-mesh p95 body-relative clearance mismatch is approximately **3.713 mm**. This is intentionally looser than the unconstrained residual because rigid/semi-rigid construction and topology are now allowed to overrule pointwise body-fit perfection;
- the hardest top mesh drops from approximately **66.94%** p95 initial-to-residual edge strain to approximately **32.73%**, while its **38 flips -> 0** and **2 near-collapse triangles -> 0**;
- the bra mesh drops from approximately **29.73%** p95 initial-to-residual edge strain to approximately **21.53%**, with **2 near-collapse triangles -> 0**.

The collision solver takes two passes on this fixture to reach zero true penetration. A small number of vertices can remain inside the optional **0.35 mm safety margin** while still being outside the target surface; those are reported separately as margin violations and are not mislabeled as penetrations.

## Safety / regression status

- `python -m unittest discover -v` in `runtime/solver`: **21 / 21 tests passed**.
- Every one of the **20 Python solver test modules** was also executed directly; all completed successfully, including the script-style regressions that `unittest` does not count.
- `production_b14.py` remains byte-for-byte identical to the uploaded local-residual checkpoint: `SHA-256 0e01487b63d2049a495f6ce3154094e5aadfcedb6215a5af370eb35987645541`.
- The LoBoMap lane remains additive; the trusted strict-B14 production path has not been replaced.

## Next stage

Do not add another global modeller or metric-chasing repair pass. The next useful work is:

1. optimise component/layer inference and exact triangle contact so the structured stage approaches interactive runtime;
2. freeze the rendered solve and restore exact raw FFXIV storage through the retained raw vertex IDs, preserving dead storage rows untouched;
3. carry authored skinning/garment-bone authority through the final geometry freeze rather than deriving new weights from proximity;
4. run a completely fresh untouched-source -> untouched-target Hot Topic/Neolithe acceptance conversion using the unified `Bodies.rbody` and inspect the exported result in-game/Blender;
5. only after that frozen candidate exists, compare against validation/control fits to identify generic solver improvements without feeding control geometry back into inference.
