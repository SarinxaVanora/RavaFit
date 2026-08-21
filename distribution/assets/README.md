# Hosted assets

`Build-GitHubAssets.ps1` writes the two large files published through Git LFS:

- `RavaFit.Runtime.win-x64.zip`
- `Bodies.rbody`

The runtime ZIP is generated from the validated local `DevAssets\Runtime`. `Bodies.rbody` is copied from the validated local catalogue. The script also updates `distribution\ravafit-assets.json` with the exact size and SHA-256 of each file.

These files are not part of the Dalamud `latest.zip`.
