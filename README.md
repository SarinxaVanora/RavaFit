# RavaFit

RavaFit is a Dalamud plugin for porting FFXIV outfits and animations between supported body shapes, races and genders.

## Build

Requirements: Dalamud API 15 and the .NET 10 SDK.

A normal Visual Studio build checks the local private runtime and body catalogue first. Missing or stale runtime files are provisioned by `scripts/Install-DevRuntime.ps1`; they are never packed into the Dalamud plugin.

Release builds run DalamudPackager and produce a plugin-only ZIP at:

```text
distribution/latest.zip
```

The ZIP contains the RavaFit plugin and its managed dependencies only.

## GitHub assets

Runtime and body data are separate Git LFS distribution files:

```text
distribution/assets/RavaFit.Runtime.win-x64.zip
distribution/assets/Bodies.rbody
```

Build or refresh them with:

```powershell
.\scripts\Build-GitHubAssets.ps1
```

That command validates/provisions `DevAssets`, packages the runtime, copies the current body catalogue and rewrites `distribution/ravafit-assets.json` with the exact sizes and SHA-256 hashes.

RavaFit reads this manifest at runtime:

```text
https://raw.githubusercontent.com/SarinxaVanora/RavaFit/master/distribution/ravafit-assets.json
```

Large asset URLs use GitHub's LFS media endpoint. The plugin downloads to `%LOCALAPPDATA%\RavaFit\Assets`, verifies SHA-256, validates SolverHost/RBODY, then activates the new version. The previous installed version remains active if an update fails.

## Git LFS

Run once in the repository:

```powershell
git lfs install
git add .gitattributes
git add distribution/assets/RavaFit.Runtime.win-x64.zip distribution/assets/Bodies.rbody
```

`DevAssets` is local cache/build state and is intentionally ignored. This avoids tracking individual Torch or llvmlite DLLs through LFS.

## Dalamud repository

Build the plugin and metadata:

```powershell
.\scripts\Build.ps1
```

Commit `distribution/latest.zip`, `distribution/pluginmaster.json` and `distribution/ravafit-assets.json` to `master` together with any updated LFS assets.

Custom repository URL:

```text
https://raw.githubusercontent.com/SarinxaVanora/RavaFit/master/distribution/pluginmaster.json
```

## First 1.0 publish

From the repo root:

```powershell
git lfs install
.\scripts\Build-GitHubAssets.ps1
.\scripts\Build.ps1
```

`Build-GitHubAssets.ps1` must run before the first public plugin build so `ravafit-assets.json` contains the real runtime hash and size. Then commit/push the generated LFS assets, manifest, `distribution/latest.zip` and `distribution/pluginmaster.json` to `master`.

## Tests

```powershell
.\scripts\Test.ps1
python -m pytest runtime\solver\test_*.py
```
