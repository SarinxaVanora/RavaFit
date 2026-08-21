#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 <base.rbody> <stage-dir> <output.rbody>" >&2; exit 2; fi
BASE="$1"; STAGE="$2"; OUT="$3"
cp --reflink=auto "$BASE" "$OUT"
(cd "$STAGE" && zip -0 -q -u "$OUT" manifest.json catalogue.json payload_index.json target_options.json payloads/*.mdl)
