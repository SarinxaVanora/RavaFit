from __future__ import annotations

import numpy as np

from unilateral_pair_separation import UnilateralPairSeparationConfig, preserve_source_unilateral_pair_separation


def _band(x0: float, x1: float, y0: float=-0.012, y1: float=0.012, z0: float=0.0, z1: float=0.006, nx: int=7, ny: int=4):
    xs=np.linspace(x0,x1,nx);ys=np.linspace(y0,y1,ny)
    V=np.asarray([[x,y,z0+(z1-z0)*(j/(ny-1))] for j,y in enumerate(ys) for x in xs],dtype=np.float64)
    F=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;b=a+1;c=a+nx;d=c+1
            F.extend(((a,b,c),(b,d,c)))
    return V,np.asarray(F,dtype=np.int64)


def _pair(source_left=(-0.060,-0.030),source_right=(0.030,0.060)):
    L,LF=_band(*source_left);R,RF=_band(*source_right)
    V=np.vstack((L,R));F=np.vstack((LF,RF+len(L)))
    return V,F,len(L)


def _cfg():
    return UnilateralPairSeparationConfig(require_leg_evidence=False, preserve_ratio=.94, absolute_tolerance_m=.001, max_corrective_vertex_movement_m=.010)


def test_collapsed_separate_thigh_bands_restore_target_scaled_inner_gap():
    source,F,boundary=_pair()
    solved=source.copy();solved[:boundary,0]+=0.012;solved[boundary:,0]-=0.012
    out,report=preserve_source_unilateral_pair_separation(source,solved,F,config=_cfg())
    pair=report["pairs"][0]
    assert report["pair_count"]==1 and report["moved_vertices"]>0
    assert float(np.max(out[:boundary,0]))<0.0 and float(np.min(out[boundary:,0]))>0.0
    final_gap=float(np.min(out[boundary:,0])-np.max(out[:boundary,0]))
    assert final_gap+1e-9>=pair["required_inner_gap_mm"]/1000.0
    assert not np.array_equal(out,solved)


def test_genuine_centreline_bridge_is_untouched_and_not_unilateral_pair():
    V,F=_band(-0.040,0.040,nx=9,ny=5)
    solved=V.copy();solved[:,0]*=.80
    out,report=preserve_source_unilateral_pair_separation(V,solved,F,config=_cfg())
    assert report["pair_count"]==0
    assert np.array_equal(out,solved)


def test_target_width_change_scales_required_inner_gap_instead_of_freezing_source_gap():
    source,F,boundary=_pair()
    solved=source.copy();solved[:,0]*=.75
    # Add a small extra medial collapse so the guard has work to do after the target-width scaling.
    solved[:boundary,0]+=0.004;solved[boundary:,0]-=0.004
    out,report=preserve_source_unilateral_pair_separation(source,solved,F,config=_cfg())
    pair=report["pairs"][0]
    assert .65<=pair["lateral_scale"]<1.0
    assert pair["expected_inner_gap_mm"]<pair["source_inner_gap_mm"]
    assert pair["required_inner_gap_mm"]<pair["source_inner_gap_mm"]
    final_gap=float(np.min(out[boundary:,0])-np.max(out[:boundary,0]))*1000.0
    assert final_gap+1e-6>=pair["required_inner_gap_mm"]


def test_already_correct_pair_is_idempotent():
    source,F,_=_pair()
    out,report=preserve_source_unilateral_pair_separation(source,source.copy(),F,config=_cfg())
    assert report["pair_count"]==1
    assert report["moved_vertices"]==0
    assert report["pairs"][0]["status"]=="already_preserved"
    assert np.array_equal(out,source)
    out2,report2=preserve_source_unilateral_pair_separation(source,out,F,config=_cfg())
    assert report2["moved_vertices"]==0
    assert np.array_equal(out2,out)


def test_leg_evidence_gate_rejects_unweighted_symmetric_non_leg_components():
    source,F,_=_pair()
    zero=np.zeros(len(source),dtype=np.float64)
    cfg=UnilateralPairSeparationConfig(require_leg_evidence=True)
    out,report=preserve_source_unilateral_pair_separation(source,source.copy(),F,left_leg_mass=zero,right_leg_mass=zero,config=cfg)
    assert report["pair_count"]==0
    assert np.array_equal(out,source)


def test_leg_evidence_gate_accepts_opposite_unilateral_skinning():
    source,F,boundary=_pair();solved=source.copy();solved[:boundary,0]+=0.010;solved[boundary:,0]-=0.010
    left=np.zeros(len(source));right=np.zeros(len(source));left[:boundary]=.90;right[:boundary]=.05;left[boundary:]=.05;right[boundary:]=.90
    cfg=UnilateralPairSeparationConfig(require_leg_evidence=True)
    out,report=preserve_source_unilateral_pair_separation(source,solved,F,left_leg_mass=left,right_leg_mass=right,config=cfg)
    assert report["pair_count"]==1 and report["moved_vertices"]>0
    assert not np.array_equal(out,solved)


def _open_tube(cx: float, *, y0: float=-0.10, y1: float=0.10, radius: float=0.012, rings: int=7, sides: int=18):
    ys=np.linspace(y0,y1,rings); angles=np.linspace(0.0,2.0*np.pi,sides,endpoint=False)
    V=np.asarray([[cx+radius*np.cos(a),y,radius*np.sin(a)] for y in ys for a in angles],dtype=np.float64)
    F=[]
    for j in range(rings-1):
        for i in range(sides):
            ni=(i+1)%sides;a=j*sides+i;b=j*sides+ni;c=(j+1)*sides+i;d=(j+1)*sides+ni
            F.extend(((a,b,c),(b,d,c)))
    return V,np.asarray(F,dtype=np.int64)


def test_long_unilateral_shell_uses_proximal_boundary_witness_when_distal_geometry_crosses_global_gap():
    L,LF=_open_tube(-0.050);R,RF=_open_tube(0.050)
    # Source-authored distal details approach/cross the centreline, but the proximal openings remain
    # cleanly separated.  A whole-component min/max witness would incorrectly reject this pair.
    L[0,0]=0.0004;R[0,0]=0.0001
    source=np.vstack((L,R));F=np.vstack((LF,RF+len(L)))
    solved=source.copy();solved[:len(L),0]+=0.018;solved[len(L):,0]-=0.018
    left=np.zeros(len(source));right=np.zeros(len(source));left[:len(L)]=.05;right[:len(L)]=.95;left[len(L):]=.95;right[len(L):]=.05
    out,report=preserve_source_unilateral_pair_separation(source,solved,F,left_leg_mass=left,right_leg_mass=right,config=UnilateralPairSeparationConfig(require_leg_evidence=True,max_corrective_vertex_movement_m=.020))
    assert report["pair_count"]==1 and report["moved_vertices"]>0
    pair=report["pairs"][0]
    assert pair["left_witness_mode"]=="proximal_boundary_loop" and pair["right_witness_mode"]=="proximal_boundary_loop"
    assert pair["source_inner_gap_mm"]>50.0
    assert pair["final_inner_gap_mm"]+1e-6>=pair["required_inner_gap_mm"]
    # The correction is medial-only: authored outer solved envelope remains the actual target refit.
    left_outer=int(np.argmin(source[:len(L),0])); right_outer=len(L)+int(np.argmax(source[len(L):,0]))
    assert out[left_outer,0]==solved[left_outer,0] and out[right_outer,0]==solved[right_outer,0]
    assert not np.array_equal(out,solved)


def test_proximal_boundary_witness_scales_with_solved_outer_span_not_source_world_distance():
    L,LF=_open_tube(-0.050);R,RF=_open_tube(0.050);source=np.vstack((L,R));F=np.vstack((LF,RF+len(L)))
    solved=source.copy();solved[:,0]*=.80;solved[:len(L),0]+=0.006;solved[len(L):,0]-=0.006
    left=np.zeros(len(source));right=np.zeros(len(source));left[:len(L)]=.05;right[:len(L)]=.95;left[len(L):]=.95;right[len(L):]=.05
    out,report=preserve_source_unilateral_pair_separation(source,solved,F,left_leg_mass=left,right_leg_mass=right,config=UnilateralPairSeparationConfig(require_leg_evidence=True,max_corrective_vertex_movement_m=.020))
    pair=report["pairs"][0]
    assert .65<pair["lateral_scale"]<1.0
    assert pair["expected_inner_gap_mm"]<pair["source_inner_gap_mm"]
    assert pair["final_inner_gap_mm"]+1e-6>=pair["required_inner_gap_mm"]


def test_proximal_opening_shape_is_restored_even_when_pair_gap_is_already_legal():
    L,LF=_open_tube(-0.055,y0=-0.12,y1=0.12,radius=.018,rings=10,sides=24);R,RF=_open_tube(0.055,y0=-0.12,y1=0.12,radius=.018,rings=10,sides=24)
    source=np.vstack((L,R));F=np.vstack((LF,RF+len(L)));solved=source.copy();solved[:,0]*=.92
    # Damage only the source-proximal region while keeping left/right separation safely legal.
    left_top=np.flatnonzero(L[:,1]>.075);right_top=len(L)+np.flatnonzero(R[:,1]>.075)
    solved[left_top,2]+=np.linspace(-.009,.008,len(left_top));solved[left_top,1]-=.006*np.sin(np.linspace(0,2*np.pi,len(left_top)))
    solved[right_top,2]+=np.linspace(.008,-.009,len(right_top));solved[right_top,1]+=.006*np.sin(np.linspace(0,2*np.pi,len(right_top)))
    left=np.zeros(len(source));right=np.zeros(len(source));left[:len(L)]=.95;right[:len(L)]=.05;left[len(L):]=.05;right[len(L):]=.95
    out,report=preserve_source_unilateral_pair_separation(source,solved,F,left_leg_mass=left,right_leg_mass=right,config=UnilateralPairSeparationConfig(require_leg_evidence=True))
    pair=report['pairs'][0];prox=pair['proximal_structure']
    assert pair['solved_inner_gap_mm']>=pair['required_inner_gap_mm']
    assert prox['left']['moved_vertices']>0 and prox['right']['moved_vertices']>0
    assert prox['left']['post_shape_error_p95_mm']<prox['left']['pre_shape_error_p95_mm']
    assert prox['right']['post_shape_error_p95_mm']<prox['right']['pre_shape_error_p95_mm']
    # Deep/distal shell stays on the actual target solve.
    deep=np.r_[np.flatnonzero(L[:,1]<0.0),len(L)+np.flatnonzero(R[:,1]<0.0)]
    assert np.array_equal(out[deep],solved[deep])
