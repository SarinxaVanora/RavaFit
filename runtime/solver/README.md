# SolverHost

`server.py` is the local JSON-lines endpoint used by the plugin.

The private Python runtime is built with `scripts/Install-DevRuntime.ps1`. Third-party wheels are not stored in the repository.

Main commands: `health`, `catalogue`, `prepare_cache`, `extract_payload`, `analyze_coverage`, `detect_source_bodies`, `convert` and Customise operations.

The frozen B14 reference remains under `runtime/b14_frozen`.
