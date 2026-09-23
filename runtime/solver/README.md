# SolverHost

`server.py` is the local JSON-lines endpoint used by the plugin.

The private Python runtime is built with `scripts/Install-DevRuntime.ps1`. Third-party wheels are not stored in the repository.

Main commands: `health`, `catalogue`, `prepare_cache`, `extract_payload`, `analyze_coverage`, `detect_source_bodies`, `convert` and Customise operations.

The frozen B14 reference remains under `runtime/b14_frozen`.

When the source mode is RBODY, the selected complete RBODY defines source
anatomy. The outfit's embedded body cannot reshape that reference. The selected
complete target anatomical surface defines collision, independently of output
body cutouts and optional hair/accessory materials. Garment shape, topology and
skin weights come from the untouched source garment.

Body cutouts are applied only to the output body after fitting. An authored source
cutaway supplies candidates; `body_visibility.py` retains target triangles around
openings unless the fitted garment covers every corner and centre. These holes
never enter fitting or collision queries. Unchanged skinning bypasses seam
arithmetic and accessor repacking, retaining the original JOINTS/WEIGHTS bytes.

Final target-body clearance runs on a shared surface for source-proven seams,
including split boundaries inside one mesh. It samples triangle interiors as
well as vertices and edges. Contact displacement is then spread over connected
cloth with a 60 mm compact kernel in `cloth_clearance_envelope.py`. This prevents
small body relief embedded in the main body material from being embossed by the
last collision pass. Only the correction is filtered; the fitted garment remains
the reference shape. Corrections retain the 8 mm movement limit and triangle
orientation guard. Unsafe local moves are limited without discarding repairs on
other cloth regions. A denser 46-point residual check includes neighbours of
every moved triangle. A conversion with unresolved final penetration fails before
output publication.

`test_cloth_clearance_envelope.py` covers relief on a rounded cup, disconnected
nearby layers, unchanged noncolliding detail, and consistent physical spread
across mesh resolutions. `test_final_shared_seam_and_clearance.py` covers seam
junctions and contacts missed by vertex/midpoint-only collision sampling.
