# SolverHost

`server.py` is the local JSON-lines endpoint used by the plugin.

The private Python runtime is built with `scripts/Install-DevRuntime.ps1`. Third-party wheels are not stored in the repository.

Main commands: `health`, `catalogue`, `prepare_cache`, `extract_payload`, `analyze_coverage`, `detect_source_bodies`, `convert` and Customise operations.

The frozen B14 reference remains under `runtime/b14_frozen`.

Final target-body clearance runs on a shared surface for source-proven seams,
including split boundaries inside one mesh. It samples triangle interiors as
well as vertices and edges. Contact displacement is then spread over connected
cloth with a 60 mm compact kernel in `cloth_clearance_envelope.py`. This prevents
small body relief embedded in the main body material from being embossed by the
last collision pass. Only the correction is filtered; the fitted garment remains
the reference shape. Corrections retain the 8 mm movement limit and triangle
orientation guard, followed by a literal-body residual clearance check.

`test_cloth_clearance_envelope.py` covers relief on a rounded cup, disconnected
nearby layers, unchanged noncolliding detail, and consistent physical spread
across mesh resolutions. `test_final_shared_seam_and_clearance.py` covers seam
junctions and contacts missed by vertex/midpoint-only collision sampling.
