from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Set numerical thread limits before importing NumPy/SciPy/Torch.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE.parent
for path in (HERE, RUNTIME_ROOT / "rbody", RUNTIME_ROOT / "b14_frozen" / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from b14_compat import solve_constructed_close_root_only, solve_standoff_root_only


def _safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("peer_shell_worker.py <input.npz> <meta.json> <output.npz> <stage.json>")
    input_path = Path(sys.argv[1]); meta_path = Path(sys.argv[2]); output_path = Path(sys.argv[3]); stage_path = Path(sys.argv[4])
    meta = json.loads(meta_path.read_text(encoding="utf-8")); z = np.load(input_path, allow_pickle=False)
    rig = SimpleNamespace(js=meta["source_js"])
    w = {"V": z["w_V"], "F": z["w_F"], "W": z["w_W"], "joint_names": list(meta["w_joint_names"])}
    Wg=z["Wg"]; base=z["base"]; contact=z["contact"]; X=z["X"]; Y=z["Y"]; BW=z["BW"]; NS=z["NS"]; NT=z["NT"]
    names=list(meta["names"]); axes=z["axes"]; source_tri=z["source_tri"]; target_tri=z["target_tri"]; labels=z["labels"]; features=dict(meta["features"])
    behavior=str(meta["behavior"]); t0=time.perf_counter()
    if behavior == "constructed_close_shell":
        candidate, peer_vertices, _, stage = solve_constructed_close_root_only(rig,rig,w,Wg,base,X,Y,BW,NS,NT,names,axes,cKDTree(X),source_tri,target_tri,labels,features)
    elif behavior == "stand_off_structured_shell":
        candidate, peer_vertices, _, stage = solve_standoff_root_only(rig,w,Wg,base,contact,X,Y,BW,NS,NT,names,axes,source_tri,target_tri,labels,features)
    else:
        raise ValueError(f"Fresh peer worker does not support {behavior}")
    peer_vertices=np.asarray(peer_vertices,dtype=np.int64)
    np.savez(output_path, peer_vertices=peer_vertices, peer_positions=np.asarray(candidate,dtype=np.float64)[peer_vertices])
    stage_path.write_text(json.dumps({"stage":_safe(stage),"elapsed_sec":time.perf_counter()-t0,"pid":os.getpid()},separators=(",",":")),encoding="utf-8")
    return 0


if __name__ == "__main__":
    code=int(main() or 0)
    sys.stdout.flush();sys.stderr.flush()
    os._exit(code)
