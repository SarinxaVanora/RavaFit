from __future__ import annotations

import importlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True

root = pathlib.Path(sys.executable).resolve().parent
packages = (root / "packages").resolve()
rows: dict[str, dict[str, str]] = {}

for name in ("numpy", "scipy", "trimesh", "torch", "numba", "llvmlite"):
    mod = importlib.import_module(name)
    raw_file = getattr(mod, "__file__", None)
    if not raw_file:
        raise RuntimeError(f"{name} imported as a namespace/placeholder package with no __file__: {mod!r}")

    module_file = pathlib.Path(raw_file).resolve()
    try:
        module_file.relative_to(packages)
    except ValueError as ex:
        raise RuntimeError(f"{name} resolved outside Runtime/packages: {module_file}") from ex

    version = getattr(mod, "__version__", None)
    if not version:
        raise RuntimeError(f"{name} imported from {module_file} but exposes no __version__")
    rows[name] = {"version": str(version), "file": str(module_file)}

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import torch
import trimesh
from numba import njit

vertices = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
cKDTree(vertices).query(vertices[:1], k=1)
sp.csr_matrix(np.eye(2)).dot(np.ones(2))
trimesh.Trimesh(
    vertices=vertices,
    faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    process=False,
).vertex_normals

parameter = torch.tensor([1.0], requires_grad=True)
optimiser = torch.optim.AdamW([parameter], lr=0.01)
(parameter * parameter).sum().backward()
torch.nn.utils.clip_grad_norm_([parameter], 1.0)
optimiser.step()

@njit(cache=False)
def _ravafit_numba_probe(x):
    return x * x + 1.0
if abs(float(_ravafit_numba_probe(2.0)) - 5.0) > 1e-12:
    raise RuntimeError("Numba JIT probe returned the wrong result")

print(json.dumps(rows, sort_keys=True))
