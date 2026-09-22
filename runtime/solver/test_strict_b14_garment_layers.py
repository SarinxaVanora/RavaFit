
from __future__ import annotations

import inspect
import numpy as np
import production_b14 as p


class Source:
    def __init__(self, meshes):
        self._meshes=meshes
        self.js={"nodes":[],"skins":[]}
    def mesh_names(self):
        return list(self._meshes)
    def data(self,name):
        return self._meshes[name]


def grid(x0: float, x1: float, z: float, nx: int=8, ny: int=8):
    xs=np.linspace(x0,x1,nx);ys=np.linspace(-.04,.04,ny)
    V=np.asarray([[x,y,z] for y in ys for x in xs],dtype=np.float64)
    F=[]
    for y in range(ny-1):
        for x in range(nx-1):
            a=y*nx+x;b=a+1;c=a+nx;d=c+1
            F.extend(((a,b,c),(b,d,c)))
    return V,np.asarray(F,dtype=np.int64)


def combine(parts):
    V=[];F=[];offset=0
    for pv,pf in parts:
        V.append(pv);F.append(pf+offset);offset+=len(pv)
    return np.vstack(V),np.vstack(F)


def mesh_data(V,F,material="cloth"):
    W=np.zeros((len(V),2),dtype=np.float64);W[:,0]=1.0
    return {"V":np.asarray(V,float),"F":np.asarray(F,np.int64),"W":W,"UV":np.zeros((len(V),2),float),"N":np.tile([[0,0,1.]],(len(V),1)),"joint_names":["j_l","j_r"],"material":material,"name":"mesh"}


BODY=np.asarray([[[-.5,-.5,0.0],[.5,-.5,0.0],[-.5,.5,0.0]],[[.5,-.5,0.0],[.5,.5,0.0],[-.5,.5,0.0]]],dtype=np.float64)


def _layer(layer_id,V,F):
    member=p.GarmentLayerMember("m",0,tuple(range(len(V))),tuple(range(len(F))),"cloth","shell",5.0,5.0,5.0)
    faces=np.asarray(F,dtype=np.int64).copy();faces.setflags(write=False)
    return p.GarmentLayer(layer_id,(member,),faces,("cloth",),5.0,5.0,(member.stable_component_id,),"shell")


def _frozen(layer_id,V):
    arr=np.asarray(V,dtype=np.float64).copy();arr.setflags(write=False)
    return p.B14LayerResult(layer_id,arr,"body_following_flexible_layer",5.0,5.0,.01,{})


def test_two_nested_shells_infer_two_different_layers():
    a,fa=grid(-.04,.04,.004)
    b,fb=grid(-.04,.04,.007)
    V,F=combine(((a,fa),(b,fb)))
    source=Source({"m":mesh_data(V,F,"same")})
    layers,_=p._infer_garment_layers(source,set(),None,BODY)
    assert len(layers)==2
    assert all(len(layer.members)==1 for layer in layers)


def test_disconnected_mirrored_fragments_can_remain_one_authored_layer():
    left,fl=grid(-.12,-.04,.006)
    right,fr=grid(.04,.12,.006)
    V,F=combine(((left,fl),(right,fr)))
    source=Source({"m":mesh_data(V,F,"same")})
    layers,_=p._infer_garment_layers(source,set(),None,BODY)
    assert len(layers)==1
    assert len(layers[0].members)==2
    assert len(set(layers[0].connected_components))==2


def test_close_substantial_different_material_structures_stay_separate():
    lower,fl=grid(-.04,.04,.0050)
    upper,fu=grid(-.04,.04,.0057)
    source=Source({"lower":mesh_data(lower,fl,"material_a"),"upper":mesh_data(upper,fu,"material_b")})
    layers,_=p._infer_garment_layers(source,set(),None,BODY)
    assert len(layers)==2


def test_source_inside_outside_ordering_survives_target_fitting():
    inner,fi=grid(-.04,.04,.004)
    outer,fo=grid(-.04,.04,.006)
    source=Source({"inner":mesh_data(inner,fi,"inner_material"),"outer":mesh_data(outer,fo,"outer_material")})
    layers,_=p._infer_garment_layers(source,set(),None,BODY)
    layers,relations=p._infer_layer_order_graph(source,layers,BODY)
    assert len(layers)==2 and len(relations)==1
    relation=relations[0]
    inner_id=relation["inner"];outer_id=relation["outer"]
    source_geometry={layer.stable_id:p._layer_geometry_from_source(source,layer) for layer in layers}
    # Independent B14 fits genuinely cross: outer is 0.2 mm inside the authored inner layer.
    # A smaller-but-positive target-space gap is valid; reconciliation is only for collapse/crossing.
    fitted_outer=source_geometry[outer_id].copy()
    fitted_outer[:,2]=source_geometry[inner_id][:,2]-.0002
    frozen={
        inner_id:_frozen(inner_id,source_geometry[inner_id]),
        outer_id:_frozen(outer_id,fitted_outer),
    }
    final,report=p._reconcile_frozen_b14_layers(layers,frozen,relations,BODY,displacement_cap_m=.002)
    assert report["layer_order_violations_before"]>0
    assert report["layer_order_violations_after"]==0
    source_gap=relation["source_gap"][relation["source_gap"]>0]
    assert len(source_gap)>0
    generated_gap=np.median(final[outer_id][:,2]-final[inner_id][:,2])
    minimum_gap=float(np.median(np.minimum(source_gap*.10,.00025)))
    assert generated_gap>=minimum_gap-.00003


def test_reconciliation_cannot_exceed_displacement_cap():
    inner,fi=grid(-.04,.04,.004)
    outer,fo=grid(-.04,.04,.006)
    li=_layer("inner",inner,fi);lo=_layer("outer",outer,fo)
    fitted_outer=outer.copy();fitted_outer[:,2]=.001
    frozen={"inner":_frozen("inner",inner),"outer":_frozen("outer",fitted_outer)}
    gap=np.full(len(outer),.002,dtype=np.float64);gap.setflags(write=False)
    closest,_,_,_,face_rows=p._b14_nearest_surface(outer,inner[fi],k=32)
    face_rows=np.asarray(face_rows,dtype=np.int64)
    bary=p.trimesh.triangles.points_to_barycentric(inner[fi][face_rows],np.asarray(closest,dtype=np.float64))
    face_rows.setflags(write=False);bary=np.asarray(bary,dtype=np.float64);bary.setflags(write=False)
    relation={"id":"inner->outer","inner":"inner","outer":"outer","source_gap":gap,"inner_face_index":face_rows,"inner_barycentric":bary,"source_gap_p50_mm":2.0,"source_gap_p95_mm":2.0,"overlap_vertices":len(outer),"influence_m":.01,"source_directional_median_mm":2.0}
    final,report=p._reconcile_frozen_b14_layers([li,lo],frozen,[relation],BODY,displacement_cap_m=.002)
    moved=np.linalg.norm(final["outer"]-fitted_outer,axis=1)
    assert float(np.max(moved))<=.0020000001
    assert report["largest_unbounded_requested_correction_mm"]>2.0
    assert report["passed"] is False


def test_already_valid_b14_layer_is_byte_identical_after_reconciliation():
    # A layer already outside the body by only 0.1 mm is still valid; reconciliation must not
    # manufacture a larger safety envelope around it.
    V,F=grid(-.04,.04,.0001)
    layer=_layer("valid",V,F);frozen={"valid":_frozen("valid",V)}
    final,report=p._reconcile_frozen_b14_layers([layer],frozen,[],BODY,displacement_cap_m=.002)
    assert final["valid"].tobytes()==V.tobytes()
    assert final["valid"].flags.writeable is False
    assert frozen["valid"].positions.flags.writeable is False
    assert report["layers"]["valid"]["reconciled_vertices"]==0


def test_post_b14_reconciler_never_calls_macro_or_second_fit(monkeypatch):
    V,F=grid(-.04,.04,.006)
    layer=_layer("valid",V,F);frozen={"valid":_frozen("valid",V)}
    def forbidden(*args,**kwargs):
        raise AssertionError("post-B14 solver/finaliser was called")
    for name in ("_extreme_macro_body_field","_garment_support_proxy","_apply_target_relief_correction","_stabilise_extreme_refit_components","_support_frame_clearance_guard","_literal_body_envelope_guard"):
        monkeypatch.setattr(p,name,forbidden)
    p._reconcile_frozen_b14_layers([layer],frozen,[],BODY,displacement_cap_m=.002)


def test_strict_layer_orchestrator_has_no_post_b14_macro_geometry_lane():
    source=inspect.getsource(p._solve_strict_b14_layers)
    post_freeze=source.split("final_layers,reconciliation=",1)[1]
    for forbidden in ("_extreme_macro_body_field","_garment_support_proxy","_apply_target_relief_correction","_stabilise_extreme_refit_components","_support_frame_clearance_guard","_literal_body_envelope_guard","_preserve_authored_garment_layers"):
        assert forbidden not in post_freeze


def test_skinning_changes_weights_but_never_positions(monkeypatch):
    V,F=grid(-.04,.04,.006)
    source=Source({"m":mesh_data(V,F,"cloth")})
    positions={"m":V.copy()}
    before=positions["m"].copy()
    def fake_retarget(raw_positions,source_positions,source_weights,*args,**kwargs):
        out=np.asarray(source_weights,float).copy()
        out[:,0]=.75;out[:,1]=.25
        return out,{"mode":"test"}
    monkeypatch.setattr(p,"_retarget_garment_skinning",fake_retarget)
    skin,_=p._retarget_frozen_layer_skinning(source,positions,{},BODY)
    assert np.array_equal(positions["m"],before)
    assert np.allclose(skin["m"]["weights"][:,1],.25)


def test_bilateral_components_remain_separate_inside_one_layer():
    left,fl=grid(-.12,-.04,.006)
    right,fr=grid(.04,.12,.006)
    V,F=combine(((left,fl),(right,fr)))
    source=Source({"m":mesh_data(V,F,"same")})
    layers,_=p._infer_garment_layers(source,set(),None,BODY)
    assert len(layers)==1 and len(layers[0].members)==2
    a=set(layers[0].members[0].vertex_ids);b=set(layers[0].members[1].vertex_ids)
    assert a.isdisjoint(b)
    boundary=len(a)
    # Layer-local faces may belong to either peer, but no face may bridge between them.
    for face in np.asarray(layers[0].faces,dtype=np.int64):
        assert (np.all(face<boundary) or np.all(face>=boundary))

def test_layer_component_discovery_uses_welded_render_connectivity():
    # Two triangles are authored as one surface but duplicate a seam vertex pair in raw MDL storage.
    # Welded/rendered connectivity must discover one component, not two unrelated authorities.
    class Source:
        js={"nodes":[],"meshes":[]}
        def mesh_names(self): return ["mesh"]
        def data(self,name):
            return {
                "V":np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,0.],[0.,1.,0.],[-1.,0.,0.]]),
                "F":np.array([[0,1,2],[3,4,5]],dtype=np.int64),
                "W":np.ones((6,1)),"joint_names":["j"],"material":"cloth","UV":np.zeros((6,2)),"N":None,"name":"mesh",
            }
    body=np.array([[[-10.,-10.,-1.],[10.,-10.,-1.],[0.,10.,-1.]]],dtype=float)
    comps=p._discover_garment_components(Source(),set(),None,body)
    assert len(comps)==1
    assert comps[0]["raw_seam_duplicate_count"]>=2
    assert set(comps[0]["vertex_ids"].tolist())==set(range(6))


def _thin_box_grid(nx=5,ny=5,z0=.004,z1=.006,x0=-.04,x1=.04,y0=-.04,y1=.04):
    xs=np.linspace(x0,x1,nx);ys=np.linspace(y0,y1,ny)
    top=np.asarray([[x,y,z1] for y in ys for x in xs],dtype=np.float64)
    bottom=np.asarray([[x,y,z0] for y in ys for x in xs],dtype=np.float64)
    V=np.vstack((top,bottom));F=[];offset=nx*ny
    for y in range(ny-1):
        for x in range(nx-1):
            a=y*nx+x;b=a+1;c=a+nx;d=c+1
            F.extend(((a,b,c),(b,d,c)))
            a+=offset;b+=offset;c+=offset;d+=offset
            F.extend(((a,c,b),(b,c,d)))
    for x in range(nx-1):
        t0=x;t1=x+1;b0=offset+x;b1=offset+x+1
        F.extend(((t0,b0,t1),(t1,b0,b1)))
        row=(ny-1)*nx;t0=row+x;t1=row+x+1;b0=offset+row+x;b1=offset+row+x+1
        F.extend(((t0,t1,b0),(t1,b1,b0)))
    for y in range(ny-1):
        t0=y*nx;t1=(y+1)*nx;b0=offset+y*nx;b1=offset+(y+1)*nx
        F.extend(((t0,t1,b0),(t1,b1,b0)))
        t0=y*nx+nx-1;t1=(y+1)*nx+nx-1;b0=offset+y*nx+nx-1;b1=offset+(y+1)*nx+nx-1
        F.extend(((t0,b0,t1),(t1,b0,b1)))
    return V,np.asarray(F,dtype=np.int64)


def test_thin_closed_volume_exposes_authored_garment_surface():
    V,F=_thin_box_grid()
    w=mesh_data(V,F,"cloth")
    thin=p._infer_thin_volume_surface(w,BODY)
    assert thin is not None
    assert thin["paired_fraction"]>=.90
    assert abs(float(thin["thickness_p50_mm"])-2.0)<1e-6
    assert 0<len(thin["outer_ids"])<len(V)
    assert float(thin["outer_boundary_fraction"])>.025


def test_thin_volume_reconstruction_preserves_source_authored_thickness():
    V,F=_thin_box_grid()
    w=mesh_data(V,F,"cloth")
    thin=p._infer_thin_volume_surface(w,BODY)
    assert thin is not None
    translation=np.asarray([.001,.002,.003],dtype=np.float64)
    solved_outer=V[np.asarray(thin["outer_ids"],dtype=np.int64)]+translation
    rebuilt=p._reconstruct_thin_volume_from_surface(w,thin,solved_outer)
    assert np.max(np.linalg.norm((rebuilt-V)-translation,axis=1))<1e-10


def test_source_relative_body_validation_does_not_repair_authored_negative_sign():
    # Concave/bilateral anatomy can produce a negative nearest-normal sign in the untouched source.
    # The solved layer may preserve that authored relation without being treated as a new penetration.
    V,F=grid(-.04,.04,-.0010)
    layer=_layer("authored-negative",V,F);frozen={"authored-negative":_frozen("authored-negative",V.copy())}
    final,report=p._reconcile_frozen_b14_layers([layer],frozen,[],BODY,displacement_cap_m=.002,source_layer_geometry={"authored-negative":V},source_body_triangles=BODY)
    assert final["authored-negative"].tobytes()==V.tobytes()
    assert report["source_relative_body_validation"] is True
    assert report["body_penetrations_before"]==0
    assert report["body_penetrations_after"]==0


def test_source_relative_body_validation_repairs_only_new_worsening():
    source,F=grid(-.04,.04,-.0010)
    solved=source.copy();solved[:,2]=-.0015
    layer=_layer("worse",source,F);frozen={"worse":_frozen("worse",solved)}
    final,report=p._reconcile_frozen_b14_layers([layer],frozen,[],BODY,displacement_cap_m=.002,source_layer_geometry={"worse":source},source_body_triangles=BODY)
    assert report["body_penetrations_before"]>0
    assert report["body_penetrations_after"]==0
    # It returns to the authored signed relation, not to an invented positive envelope.
    assert np.max(np.abs(final["worse"][:,2]-source[:,2]))<1e-10
    assert report["layers"]["worse"]["max_mm"]<.51


def test_source_relative_body_validation_still_clears_new_literal_crossing():
    source,F=grid(-.04,.04,.0010)
    solved=source.copy();solved[:,2]=-.00050
    layer=_layer("new-crossing",source,F);frozen={"new-crossing":_frozen("new-crossing",solved)}
    final,report=p._reconcile_frozen_b14_layers([layer],frozen,[],BODY,displacement_cap_m=.002,source_layer_geometry={"new-crossing":source},source_body_triangles=BODY)
    assert report["body_penetrations_before"]>0
    assert report["body_penetrations_after"]==0
    assert float(np.min(final["new-crossing"][:,2]))>=.0000199


def test_final_source_attachment_closure_can_cross_mesh_boundaries():
    shell=np.array([[0,0,0],[0,.005,0],[0,.010,0],[-.02,0,0],[-.02,.005,0],[-.02,.010,0],[0,0,.005],[0,.005,.005],[0,.010,.005],[-.02,0,.005],[-.02,.005,.005],[-.02,.010,.005]],float)
    shell_faces=np.array([(0,3,1),(3,4,1),(1,4,2),(4,5,2),(6,7,9),(7,10,9),(7,8,10),(8,11,10),(0,6,3),(3,6,9),(2,5,8),(5,11,8)],dtype=np.int64)
    connector=np.array([[.0005,0,0],[.0005,.005,0],[.0005,.010,0],[.001,.005,.003]],float)
    connector_faces=np.array([(0,1,3),(1,2,3)],dtype=np.int64)
    x=np.linspace(.0012,.1012,8);ribbon=np.vstack([np.c_[x,np.full(8,y),np.zeros(8)] for y in (0,.005,.01)])
    ribbon_faces=[]
    for row in range(2):
        a=row*8;b=(row+1)*8
        for i in range(7):ribbon_faces.extend(((a+i,a+i+1,b+i),(a+i+1,b+i+1,b+i)))
    source=Source({"shell":mesh_data(shell,shell_faces,"shell"),"connector":mesh_data(connector,connector_faces,"hardware"),"ribbon":mesh_data(ribbon,np.asarray(ribbon_faces,dtype=np.int64),"strap")})
    positions={"shell":shell.copy(),"connector":connector.copy(),"ribbon":ribbon.copy()};positions["shell"][:,1]+=.006
    out,report=p._preserve_final_source_attachment_continuity(source,positions)
    assert report["enabled"] is True and report["cross_mesh_capable"] is True
    assert np.array_equal(out["shell"],positions["shell"])
    assert float(np.mean(out["connector"][:,1]-connector[:,1]))>.0055
    assert float(np.mean(out["ribbon"][[0,8,16],1]-ribbon[[0,8,16],1]))>.0050
