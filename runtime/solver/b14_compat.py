from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = _THIS_DIR.parent if _THIS_DIR.name.casefold() == "solver" else _THIS_DIR
B14_SCRIPTS = RUNTIME_ROOT / "b14_frozen" / "scripts"
if str(B14_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(B14_SCRIPTS))

# Keep frozen B14 untouched; production fixes live here.
import b14_mesh_worker as _worker
import collision_eval as _collision_eval
import construction_fields as _construction_fields
import lobofit_official_refine as _lobofit_official
import trimesh
from scipy.spatial import cKDTree

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

_expected_worker = (B14_SCRIPTS / "b14_mesh_worker.py").resolve()
_loaded_worker = Path(getattr(_worker, "__file__", "")).resolve()
if _loaded_worker != _expected_worker:
    raise ImportError(f"RavaFit loaded b14_mesh_worker from {_loaded_worker}, expected {_expected_worker}")

_original_refine_assembly = _worker.refine_assembly
_original_refine_component = _worker.refine_component
_original_refine_lobofit = _worker.refine_lobofit
_original_smooth_collision_polish = _worker.smooth_collision_polish
_original_nearest_surface = _collision_eval.nearest_surface

# Cache immutable nearest-surface acceleration data for the life of one conversion.
_surface_query_cache: dict[int, tuple[np.ndarray, cKDTree, np.ndarray]] = {}
_surface_query_cache_enabled = True

def reset_runtime_caches():
    _surface_query_cache.clear()

def set_surface_query_cache_enabled(enabled: bool):
    global _surface_query_cache_enabled
    _surface_query_cache_enabled = bool(enabled)
    _surface_query_cache.clear()


# trimesh.constants.tol.zero.  Keep this identical to trimesh.triangles.closest_point.
_TRIANGLE_ZERO_TOL = 1e-13

if _NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True)
    def _closest_point_one_triangle(px, py, pz, a, b, c):
        ab0=b[0]-a[0]; ab1=b[1]-a[1]; ab2=b[2]-a[2]
        ac0=c[0]-a[0]; ac1=c[1]-a[1]; ac2=c[2]-a[2]
        ap0=px-a[0]; ap1=py-a[1]; ap2=pz-a[2]
        d1=ab0*ap0+ab1*ap1+ab2*ap2; d2=ac0*ap0+ac1*ap1+ac2*ap2
        if d1 < _TRIANGLE_ZERO_TOL and d2 < _TRIANGLE_ZERO_TOL:
            return a[0],a[1],a[2]
        bp0=px-b[0]; bp1=py-b[1]; bp2=pz-b[2]
        d3=ab0*bp0+ab1*bp1+ab2*bp2; d4=ac0*bp0+ac1*bp1+ac2*bp2
        if d3 > -_TRIANGLE_ZERO_TOL and d4 <= d3:
            return b[0],b[1],b[2]
        vc=d1*d4-d3*d2
        if vc < _TRIANGLE_ZERO_TOL and d1 > -_TRIANGLE_ZERO_TOL and d3 < _TRIANGLE_ZERO_TOL:
            den=d1-d3
            if abs(den) > 1e-300:
                v=d1/den; return a[0]+v*ab0,a[1]+v*ab1,a[2]+v*ab2
        cp0=px-c[0]; cp1=py-c[1]; cp2=pz-c[2]
        d5=ab0*cp0+ab1*cp1+ab2*cp2; d6=ac0*cp0+ac1*cp1+ac2*cp2
        if d6 > -_TRIANGLE_ZERO_TOL and d5 <= d6:
            return c[0],c[1],c[2]
        vb=d5*d2-d1*d6
        if vb < _TRIANGLE_ZERO_TOL and d2 > -_TRIANGLE_ZERO_TOL and d6 < _TRIANGLE_ZERO_TOL:
            den=d2-d6
            if abs(den) > 1e-300:
                w=d2/den; return a[0]+w*ac0,a[1]+w*ac1,a[2]+w*ac2
        va=d3*d6-d5*d4
        if va < _TRIANGLE_ZERO_TOL and (d4-d3) > -_TRIANGLE_ZERO_TOL and (d5-d6) > -_TRIANGLE_ZERO_TOL:
            d43=d4-d3; den=d43+(d5-d6)
            if abs(den) > 1e-300:
                w=d43/den; return b[0]+w*(c[0]-b[0]),b[1]+w*(c[1]-b[1]),b[2]+w*(c[2]-b[2])
        den=va+vb+vc
        if abs(den) > 1e-300:
            inv=1.0/den; v=vb*inv; w=vc*inv
            return a[0]+ab0*v+ac0*w,a[1]+ab1*v+ac1*w,a[2]+ab2*v+ac2*w
        da=(px-a[0])**2+(py-a[1])**2+(pz-a[2])**2
        db=(px-b[0])**2+(py-b[1])**2+(pz-b[2])**2
        dc=(px-c[0])**2+(py-c[1])**2+(pz-c[2])**2
        if da <= db and da <= dc: return a[0],a[1],a[2]
        if db <= dc: return b[0],b[1],b[2]
        return c[0],c[1],c[2]

    @njit(cache=True, nogil=True)
    def _closest_candidate_kernel_serial(P,T,idx):
        n=P.shape[0]; kk=idx.shape[1]
        cp=np.empty((n,3),dtype=np.float64); fi=np.empty(n,dtype=np.int64); dist2=np.empty(n,dtype=np.float64)
        for row in range(n):
            px=P[row,0]; py=P[row,1]; pz=P[row,2]
            best=1.7976931348623157e308; best_face=idx[row,0]; bx=0.0; by=0.0; bz=0.0
            for candidate in range(kk):
                face=idx[row,candidate]
                x,y,z=_closest_point_one_triangle(px,py,pz,T[face,0],T[face,1],T[face,2])
                dx=x-px; dy=y-py; dz=z-pz; dd=dx*dx+dy*dy+dz*dz
                if dd < best:
                    best=dd; best_face=face; bx=x; by=y; bz=z
            cp[row,0]=bx; cp[row,1]=by; cp[row,2]=bz; fi[row]=best_face; dist2[row]=best
        return cp,fi,dist2
else:
    _closest_candidate_kernel_serial = None

def _cached_nearest_surface(P, triangles, k=16):
    P=np.asarray(P,float);T=np.asarray(triangles,float)
    if not np.all(np.isfinite(P)):
        bad=np.argwhere(~np.isfinite(P));first=bad[0].tolist() if len(bad) else None
        raise ValueError(f"RavaFit nearest-surface query received non-finite points: count={len(bad)}, first={first}")
    if not np.all(np.isfinite(T)):
        bad=np.argwhere(~np.isfinite(T));first=bad[0].tolist() if len(bad) else None
        raise ValueError(f"RavaFit nearest-surface target contains non-finite triangles: count={len(bad)}, first={first}")
    if not _surface_query_cache_enabled:
        return _original_nearest_surface(P,T,k=k)
    key=id(T);entry=_surface_query_cache.get(key)
    if entry is None or entry[0] is not T:
        cent=T.mean(1);tree=cKDTree(cent);fn=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-12);entry=(T,tree,fn);_surface_query_cache[key]=entry
    _,tree,fn=entry
    kk=min(int(k),len(T)); n=len(P)
    cp=np.empty((n,3),dtype=np.float64); fi=np.empty(n,dtype=np.int64); dist2=np.empty(n,dtype=np.float64)
    chunk_size=50000 if _closest_candidate_kernel_serial is not None else 6000
    for start in range(0,n,chunk_size):
        stop=min(n,start+chunk_size); Pc=P[start:stop]
        workers=1
        _,idx=tree.query(Pc,k=kk,workers=workers); idx=idx if idx.ndim>1 else idx[:,None]
        m=len(Pc)
        if _closest_candidate_kernel_serial is not None:
            local_cp,local_fi,local_dist2=_closest_candidate_kernel_serial(np.ascontiguousarray(Pc),np.ascontiguousarray(T),np.ascontiguousarray(idx,dtype=np.int64))
            cp[start:stop]=local_cp; fi[start:stop]=local_fi; dist2[start:stop]=local_dist2
        else:
            Tc=T[idx.reshape(-1)]; Q=np.repeat(Pc,kk,axis=0)
            C=trimesh.triangles.closest_point(Tc,Q).reshape(m,kk,3)
            dd=np.sum((C-Pc[:,None,:])**2,axis=2); j=np.argmin(dd,axis=1); rows=np.arange(m)
            cp[start:stop]=C[rows,j]; fi[start:stop]=idx[rows,j]; dist2[start:stop]=dd[rows,j]
    N=fn[fi];signed=np.sum((P-cp)*N,axis=1);dist=np.sqrt(dist2);return cp,N,signed,dist,fi

_collision_eval.nearest_surface = _cached_nearest_surface
_construction_fields.nearest_surface = _cached_nearest_surface
_lobofit_official.nearest_surface = _cached_nearest_surface
_worker.nearest_surface = _cached_nearest_surface


def _clearance_refine_lobofit(*args, **kwargs):
    """Give close-fitting B14 garments a practical cloth-over-skin clearance."""
    kwargs["collision_margin"] = max(float(kwargs.get("collision_margin", 0.00035)), 0.00075)
    kwargs["collision_weight"] = max(float(kwargs.get("collision_weight", 10.0)), 50.0)
    return _original_refine_lobofit(*args, **kwargs)


def _clearance_collision_polish(V, F, triangles, margin=.00035, iterations=18, blend=.55, max_push=.0025, k=16):
    """Use B14's topology-smoothed collision polish with a render-safe clearance."""
    return _original_smooth_collision_polish(
        V, F, triangles,
        margin=max(float(margin), 0.00075),
        iterations=max(int(iterations), 16),
        blend=float(blend),
        max_push=max(float(max_push), 0.0025),
        k=max(int(k), 24),
    )


def _safe_refine_component(Vsrc, Vinit, F, Ncontact, *args, **kwargs):
    """Keep degenerate/face-less inferred components out of frozen B14's structural optimiser.

    Some perfectly valid XIV garments contain welded vertices that survive model import but do not
    own a complete local triangle inside the inferred root component.  Frozen B14's cotangent
    graph computes a percentile over its edge weights; for an empty local face set that becomes
    NumPy's opaque ``index -1 is out of bounds for axis 0 with size 0``.  A component with no
    usable local triangles has no structural surface to refine, so the authoritative safe result is
    the already-computed body-field mapping passed in as Vinit.
    """
    source=np.asarray(Vsrc,dtype=np.float64)
    initial=np.asarray(Vinit,dtype=np.float64)
    faces=np.asarray(F,dtype=np.int64)
    if initial.shape != source.shape:
        raise ValueError(f"B14 component initial/source geometry shape mismatch: {initial.shape} vs {source.shape}")
    if len(source)==0:
        return initial.copy(), [{"iter":0,"loss":0.0,"skipped":True,"reason":"component has no vertices"}]
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        if faces.size == 0:
            faces=np.empty((0,3),dtype=np.int64)
        else:
            raise ValueError(f"B14 component faces must be Nx3, got {faces.shape}")
    if len(faces):
        in_range=np.all((faces>=0)&(faces<len(source)),axis=1)
        nondegenerate=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,2]!=faces[:,0])
        faces=faces[in_range&nondegenerate]
    if len(faces)==0:
        return initial.copy(), [{"iter":0,"loss":0.0,"skipped":True,"reason":"component has no usable local triangles"}]
    return _original_refine_component(source,initial,faces,Ncontact,*args,**kwargs)


def _safe_refine_assembly(Vsrc, Vinit, F, lab, root_component=0, iterations=800, lr=.025, max_gap=.0030, log_every=100):
    """Skip assembly refinement when B14 has no reachable component to optimise."""
    labels = np.asarray(lab)
    if labels.size == 0:
        return np.asarray(Vinit, float).copy(), {}, [], [{
            "iter": 0,
            "loss": 0.0,
            "skipped": True,
            "reason": "assembly labels were empty",
            "gaps": [],
        }]

    cc = int(labels.max()) + 1
    comp_ids = [np.where(labels == c)[0] for c in range(cc)]
    kinds = {
        c: _worker.component_kind_pca(np.asarray(Vsrc)[comp_ids[c]], c == root_component)
        for c in range(cc)
        if len(comp_ids[c])
    }
    edges = _worker.attachment_pairs(np.asarray(Vsrc), labels, max_source_gap=max_gap)

    reachable = {root_component}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge["a"] in reachable and edge["b"] not in reachable:
                reachable.add(edge["b"])
                changed = True
            if edge["b"] in reachable and edge["a"] not in reachable:
                reachable.add(edge["a"])
                changed = True

    optimisable = [c for c in sorted(reachable) if c != root_component and len(comp_ids[c])]
    if not optimisable:
        return np.asarray(Vinit, float).copy(), kinds, edges, [{
            "iter": 0,
            "loss": 0.0,
            "attach": 0.0,
            "frame": 0.0,
            "shape": 0.0,
            "reg": 0.0,
            "skipped": True,
            "reason": "no reachable non-root assembly components",
            "gaps": [],
        }]

    return _original_refine_assembly(
        Vsrc, Vinit, F, labels,
        root_component=root_component,
        iterations=iterations,
        lr=lr,
        max_gap=max_gap,
        log_every=log_every,
    )


# Patch the live import surface only; the frozen B14 files stay untouched.
_worker.refine_component = _safe_refine_component
_worker.refine_assembly = _safe_refine_assembly
_worker.refine_lobofit = _clearance_refine_lobofit
_worker.smooth_collision_polish = _clearance_collision_polish

solve_standoff = _worker.solve_standoff
solve_constructed_close = _worker.solve_constructed_close

solve_flexible = _worker.solve_flexible
solve_conservative = _worker.solve_conservative


def _identity_refine_assembly(Vsrc, Vinit, F, lab, root_component=0, iterations=800, lr=.025, max_gap=.0030, log_every=100):
    return np.asarray(Vinit, float).copy(), {}, [], [{
        "iter": 0, "loss": 0.0, "attach": 0.0, "frame": 0.0, "shape": 0.0, "reg": 0.0,
        "skipped": True, "reason": "independent peer-shell root solve", "gaps": [],
    }]


def _root_only_call(fn, *args, **kwargs):
    old=_worker.refine_assembly
    _worker.refine_assembly=_identity_refine_assembly
    try:return fn(*args, **kwargs)
    finally:_worker.refine_assembly=old


def solve_standoff_root_only(*args, **kwargs):
    return _root_only_call(_worker.solve_standoff, *args, **kwargs)


def solve_constructed_close_root_only(*args, **kwargs):
    return _root_only_call(_worker.solve_constructed_close, *args, **kwargs)


def refine_peer_assembly(w, initial, labels, behavior, root_component):
    """Run only the assembly tail of B14 after an already-computed root solve."""
    if behavior == "constructed_close_shell":
        return _worker.refine_assembly(w['V'],np.asarray(initial,float).copy(),w['F'],labels,root_component=int(root_component),iterations=420,lr=.018,max_gap=.003,log_every=420)
    if behavior == "stand_off_structured_shell":
        return _worker.refine_assembly(w['V'],np.asarray(initial,float).copy(),w['F'],labels,root_component=int(root_component),iterations=450,lr=.018,max_gap=.004,log_every=450)
    raise ValueError(f"Peer assembly reuse is not defined for {behavior}")


def runtime_self_test() -> dict[str, object]:
    """Run the production runtime checks used by SolverHost health and staging."""
    import ffxiv_lobofit as lobofit
    expected_lobofit = (B14_SCRIPTS / "ffxiv_lobofit.py").resolve()
    loaded_lobofit = Path(getattr(lobofit, "__file__", "")).resolve()
    if loaded_lobofit != expected_lobofit:
        raise RuntimeError(f"RavaFit loaded ffxiv_lobofit from {loaded_lobofit}, expected {expected_lobofit}")

    v = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=float)
    f = np.asarray([[0, 1, 2]], dtype=np.int64)
    labels = np.zeros(3, dtype=np.int64)
    out, _, _, history = _safe_refine_assembly(v, v.copy(), f, labels, root_component=0, iterations=1)
    if out.shape != v.shape or not np.allclose(out, v):
        raise RuntimeError("B14 compatibility self-test changed a root-only mesh.")
    if not history or not history[-1].get("skipped"):
        raise RuntimeError("B14 compatibility self-test did not take the root-only empty-assembly guard.")

    # Disconnected components can leave B14 with an empty AdamW parameter list.
    v2 = np.vstack([v, v + np.asarray([1.0, 0.0, 0.0])])
    f2 = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    labels2 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    out2, _, _, history2 = _safe_refine_assembly(v2, v2.copy(), f2, labels2, root_component=0, iterations=1, max_gap=.004)
    if out2.shape != v2.shape or not np.allclose(out2, v2):
        raise RuntimeError("B14 compatibility self-test changed a disconnected assembly.")
    if not history2 or not history2[-1].get("skipped"):
        raise RuntimeError("B14 compatibility self-test did not take the disconnected empty-assembly guard.")

    # Welded XIV meshes can contain an inferred component with vertices but no complete local face.
    # That is a valid no-op structural case and must never reach cotan_graph's empty percentile.
    empty_faces=np.empty((0,3),dtype=np.int64)
    empty_normals=np.zeros_like(v)
    refined_empty,empty_history=_worker.refine_component(v,v.copy(),empty_faces,empty_normals,iterations=1)
    if refined_empty.shape!=v.shape or not np.allclose(refined_empty,v) or not empty_history or not empty_history[-1].get("skipped"):
        raise RuntimeError("B14 compatibility self-test did not guard a face-less inferred component.")

    # Exercise the Torch/SciPy/B14 paths we actually ship, not just imports.
    import torch
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve
    import trimesh

    x = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.AdamW([x], lr=0.01, weight_decay=0.0)
    optimiser.zero_grad()
    loss = torch.sum(x * x) + torch.linalg.norm(torch.linalg.cross(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        dim=0,
    ))
    loss.backward()
    torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
    optimiser.step()
    normalised = torch.nn.functional.normalize(torch.tensor([[3.0, 4.0]], dtype=torch.float64), dim=1)
    if not bool(torch.isfinite(x).all()) or not torch.allclose(torch.linalg.norm(normalised, dim=1), torch.ones(1, dtype=torch.float64)):
        raise RuntimeError("PyTorch optimisation/nn probe produced invalid values.")

    tree = cKDTree(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float))
    distance, _ = tree.query(np.asarray([[0.1, 0.0, 0.0]], dtype=float))
    if not np.isfinite(distance).all():
        raise RuntimeError("SciPy cKDTree probe failed.")
    matrix = csr_matrix(np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=float))
    solved = spsolve(matrix, np.asarray([2.0, 8.0], dtype=float))
    if not np.allclose(solved, np.asarray([1.0, 2.0])):
        raise RuntimeError("SciPy sparse solve probe failed.")
    graph = coo_matrix((np.ones(2), (np.asarray([0, 1]), np.asarray([1, 0]))), shape=(3, 3))
    component_count, _ = connected_components(graph, directed=False)
    if component_count != 2:
        raise RuntimeError("SciPy sparse connected-components probe failed.")

    tri = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if np.asarray(tri.vertex_normals).shape != v.shape:
        raise RuntimeError("trimesh normal-generation probe failed.")
    closest = trimesh.triangles.closest_point(v[f], np.asarray([[0.002, 0.002, 0.003]], dtype=float))
    if np.asarray(closest).shape != (1, 3) or not np.isfinite(closest).all():
        raise RuntimeError("trimesh triangle closest-point probe failed.")

    reset_runtime_caches()
    query_points=np.asarray([[0.002,0.002,0.003],[0.006,0.001,-0.002]],dtype=float);triangles=v[f]
    original=_original_nearest_surface(query_points,triangles,k=1);cached=_cached_nearest_surface(query_points,triangles,k=1);cached_again=_cached_nearest_surface(query_points,triangles,k=1)
    for lhs,rhs in zip(original,cached):
        if not np.array_equal(lhs,rhs):raise RuntimeError("Cached B14 surface query changed nearest-surface output.")
    for lhs,rhs in zip(cached,cached_again):
        if not np.array_equal(lhs,rhs):raise RuntimeError("Repeated cached B14 surface query is not deterministic.")
    set_surface_query_cache_enabled(False)
    reference_via_wrapper=_cached_nearest_surface(query_points,triangles,k=1)
    for lhs,rhs in zip(original,reference_via_wrapper):
        if not np.array_equal(lhs,rhs):raise RuntimeError("Reference B14 surface-query retry path changed output.")
    try:
        _cached_nearest_surface(np.asarray([[np.nan,0.0,0.0]]),triangles,k=1)
        raise RuntimeError("RavaFit finite-value guard accepted NaN query geometry.")
    except ValueError as ex:
        if "non-finite" not in str(ex).casefold():
            raise RuntimeError(f"RavaFit finite-value guard returned an unclear error: {ex}") from ex
    finally:
        set_surface_query_cache_enabled(True)
    reset_runtime_caches()

    return {
        "ok": True,
        "empty_assembly_guard": True,
        "disconnected_assembly_guard": True,
        "torch_optimisation": True,
        "scipy_spatial": True,
        "scipy_sparse": True,
        "scipy_connected_components": True,
        "trimesh_normals": True,
        "trimesh_closest_point": True,
        "surface_query_cache_exact": True,
    }
