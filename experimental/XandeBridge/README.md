# Experimental Xande bridge

This folder is intentionally NOT part of the default RavaFit build.

The historical Xande commit `306003a7fd591cc6854e98165dd9cea00b66d16b` contains a coherent
skeleton-aware `ModelConverter` with both MDL->GLB and GLB->MDL code, but its Havok signatures were
last checked against a 2023 game build and it targets an old Dalamud API.

Do not enable it by simply defining a symbol and hoping. The bridge must be retargeted to API 15,
its current-game Havok calls/signatures verified, and the RavaFit Diagnostics zero-change round-trip
must pass before `ModelBridgeStatus.Validated` may return true.

The public RavaFit plugin intentionally fails closed until that work is verified on the current game.
