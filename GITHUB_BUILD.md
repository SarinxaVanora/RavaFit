# GitHub-built RavaFit distribution

RavaFit's canonical distribution can be built entirely by GitHub Actions.

## Normal release flow

1. Commit and push source changes to `master`.
2. `.github/workflows/build-distribution.yml` runs on a GitHub-hosted Windows runner.
3. The runner downloads the current stable Dalamud development distribution from the official `dalamud-distrib/latest.zip`, installs it under the normal `%AppData%\XIVLauncher\addon\Hooks\dev` build location, and exports `DALAMUD_HOME`. This supplies the build-only assemblies required by `Dalamud.NET.Sdk`; it does not alter the packaged RavaFit plugin.
4. The workflow invokes `scripts/Build-Distribution.ps1`, which calls the same repository build scripts used locally:
   - `Build-GitHubAssets.ps1`
   - `Build.ps1`
5. The generated `distribution/` files are committed back to `master` by `github-actions[bot]`.
6. The bot-only `distribution/**` commit does not start another build.

The private Python runtime is cached by its source/provisioning inputs, but it is still validated by `Prepare-DevEnvironment.ps1` on every distribution build. The hosted runtime ZIP is generated deterministically so unchanged runtime content does not create a new large Git LFS object merely because the runner's timestamps changed.

## First push / existing Bodies.rbody

`Bodies.rbody` is intentionally a Git LFS hosted asset, not reconstructed from source during an ordinary CI build. The repository therefore needs an existing real `distribution/assets/Bodies.rbody` LFS object for the workflow to hydrate. Do not delete that existing LFS asset when replacing/updating the source tree.

If the catalogue itself changes, publish the new `Bodies.rbody` LFS object once; subsequent GitHub builds will hydrate and retain it automatically.

## Clean shipped source paths

`Directory.Build.props` maps Release-build repository source paths to `/_/`. Portable PDB line information is retained, but a shipped stack trace contains paths such as:

```text
/_/src/RavaFit.Plugin/Services/SolverHostService.cs:line 139
```

rather than a developer workstation path or a GitHub runner checkout path.

Debug builds are deliberately not path-mapped, so local source debugging/breakpoints continue to use the real checkout path.

## Manual run

The same workflow can be started from GitHub's **Actions** tab using **Run workflow**.

To reproduce the canonical pipeline locally, run:

```powershell
.\scripts\Build-Distribution.ps1 -RunTests
```

Local builds continue using your existing Dalamud/XIVLauncher development installation. The automatic Dalamud download is CI-only, because GitHub runners start without XIVLauncher or `Hooks\dev`.

## Repository setting

The workflow requests `contents: write` for `GITHUB_TOKEN` because it commits `distribution/` back to `master`. If repository/branch policy blocks GitHub Actions from writing to `master`, allow the workflow token/bot to push generated distribution commits or change the repository policy accordingly.
