# Developer assets

`DevAssets` is local cache/build state and is not distributed with the plugin.

`Prepare-DevEnvironment.ps1` validates the body catalogue and SolverHost runtime before normal builds. Missing or stale runtime components are rebuilt by `Install-DevRuntime.ps1`.

Published copies are created under `distribution/assets` by `Build-GitHubAssets.ps1`.
