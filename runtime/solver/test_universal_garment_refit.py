import numpy as np

from universal_garment_refit import (
    UniversalRefitConfig,
    _graph,
    _structured_macro_transport,
    fit_body_macro_transform,
)


def _two_quads(gap: float):
    # Two disconnected parallel quads whose closest x edges are separated by `gap`.
    left=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[-1.0,1.0,0.0],[0.0,1.0,0.0]],dtype=np.float64)
    right=np.array([[gap,0.0,0.0],[1.0+gap,0.0,0.0],[gap,1.0,0.0],[1.0+gap,1.0,0.0]],dtype=np.float64)
    V=np.vstack((left,right))
    F=np.array([[0,1,2],[1,3,2],[4,6,5],[5,6,7]],dtype=np.int64)
    return V,F


def test_nearby_bilateral_components_do_not_become_attached_by_proximity_alone():
    V,F=_two_quads(.0020)
    _,labels,attachments=_graph(V,F,UniversalRefitConfig())
    assert len(np.unique(labels))==2
    assert attachments==[]


def test_repeated_near_coincident_source_witnesses_confirm_a_real_attachment():
    left=np.array([[-1,0,0],[0,0,0],[-1,.5,0],[0,.5,0],[-1,1,0],[0,1,0]],dtype=np.float64)
    right=np.array([[.00025,0,0],[1.00025,0,0],[.00025,.5,0],[1.00025,.5,0],[.00025,1,0],[1.00025,1,0]],dtype=np.float64)
    V=np.vstack((left,right))
    F=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4],[6,8,7],[7,8,9],[8,10,9],[9,10,11]],dtype=np.int64)
    _,labels,attachments=_graph(V,F,UniversalRefitConfig())
    assert len(np.unique(labels))==2
    assert len(attachments)>=3


def test_body_macro_transform_recovers_shared_affine_frame():
    rng=np.random.default_rng(42)
    X=rng.normal(size=(128,3))*.1
    L=np.array([[.91,.02,0.0],[-.01,.97,.04],[0.0,-.03,.72]],dtype=np.float64)
    t=np.array([.01,-.02,.03],dtype=np.float64)
    Y=X@L.T+t
    fitted,translation,rotation,report=fit_body_macro_transform(X,Y)
    predicted=X@fitted.T+translation
    assert np.max(np.linalg.norm(predicted-Y,axis=1))<1e-9
    assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9)
    assert report['fit_residual_p95_mm']<1e-6


def test_structured_transport_preserves_standoff_more_than_body_compression():
    support=np.array([[-.1,0,0],[.1,0,0],[-.1,.1,0],[.1,.1,0]],dtype=np.float64)
    P=support+np.array([0,0,.02])
    L=np.diag([.9,.95,.64])
    t=np.zeros(3)
    R=np.eye(3)
    cfg=UniversalRefitConfig()
    out=_structured_macro_transport(P,support,L,t,R,.02,cfg)
    mapped_support=support@L.T
    new_clear=np.linalg.norm(out-mapped_support,axis=1)
    fully_scaled=.02*.64
    # sqrt(.64)=.8, so structured clothing keeps more clearance than the body scales by.
    assert np.all(new_clear>fully_scaled)
    assert np.allclose(new_clear,.02*np.sqrt(.64),atol=1e-10)


def _stub_surface_query(target_signed: float):
    def _nearest(V, triangles, k=24):
        V=np.asarray(V,dtype=np.float64)
        is_target=float(np.mean(np.asarray(triangles,dtype=np.float64)))>0.5
        signed=np.full(len(V),target_signed if is_target else .020,dtype=np.float64)
        normals=np.zeros_like(V);normals[:,2]=1.0
        points=V-normals*signed[:,None]
        distance=np.abs(signed)
        face_index=np.zeros(len(V),dtype=np.int64)
        return points,normals,signed,distance,face_index
    return _nearest


def test_standoff_authored_geometry_that_already_clears_target_is_exact_noop():
    from universal_garment_refit import coherent_universal_refit
    P=np.array([[-.05,0,.02],[.05,0,.02],[-.05,.1,.02],[.05,.1,.02]],dtype=np.float64)
    F=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    B=P+np.array([.01,0,0]);C=P+np.array([.02,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64)
    target_tri=np.ones((1,3,3),dtype=np.float64)
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.10,nearest_surface_fn=_stub_surface_query(.005))
    assert np.array_equal(out,P)
    assert report['authored_source_restored'] is True
    assert report['already_fits_target'] is True


def test_standoff_authored_geometry_that_intersects_target_still_refits():
    from universal_garment_refit import coherent_universal_refit
    P=np.array([[-.05,0,.02],[.05,0,.02],[-.05,.1,.02],[.05,.1,.02]],dtype=np.float64)
    F=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    B=P.copy();C=P+np.array([.01,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64)
    target_tri=np.ones((1,3,3),dtype=np.float64)
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.10,nearest_surface_fn=_stub_surface_query(-.002))
    assert not bool(report.get('already_fits_target',False))
    assert not np.array_equal(out,P)



def _rigid_assembly_surface_query(V, triangles, k=24):
    V=np.asarray(V,dtype=np.float64)
    is_target=float(np.mean(np.asarray(triangles,dtype=np.float64)))>0.5
    normals=np.zeros_like(V);normals[:,2]=1.0
    if is_target:
        signed=V[:,2].copy()
    else:
        signed=np.full(len(V),.0065,dtype=np.float64)
    points=V-normals*signed[:,None]
    distance=np.abs(signed)
    face_index=np.zeros(len(V),dtype=np.int64)
    return points,normals,signed,distance,face_index


def _many_disconnected_triangles(component_count: int=40):
    vertices=[];faces=[]
    for i in range(component_count):
        x=float(i)*.01
        base=len(vertices)
        if i==0:
            vertices.extend(((x,0,-.001),(x+.002,0,.0015),(x,.002,.0015)))
        else:
            vertices.extend(((x,0,.0015),(x+.002,0,.0015),(x,.002,.0015)))
        faces.append((base,base+1,base+2))
    return np.asarray(vertices,dtype=np.float64),np.asarray(faces,dtype=np.int64)


def test_multicomponent_shallow_contact_preserves_shape_and_uses_one_rigid_translation():
    from universal_garment_refit import coherent_universal_refit
    P,F=_many_disconnected_triangles(40)
    B=P+np.array([.004,0,0]);C=P+np.array([.008,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64)
    target_tri=np.ones((1,3,3),dtype=np.float64)
    W=np.tile(np.array([[.72,.28]],dtype=np.float64),(len(P),1))
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.10,nearest_surface_fn=_rigid_assembly_surface_query,source_weights=W)
    delta=out-P
    assert report['construction_mode']=='shape-preserving coherent rigid placement'
    assert report['authored_shape_preserved'] is True
    assert report['final_penetrating_vertices']==0
    assert np.max(np.linalg.norm(delta-delta[0],axis=1))<1e-12
    assert np.max(np.abs((out[F[:,1]]-out[F[:,0]])-(P[F[:,1]]-P[F[:,0]])))<1e-12


def test_connected_fitted_shell_does_not_enter_rigid_assembly_lane():
    from universal_garment_refit import coherent_universal_refit
    P=np.array([[-.05,0,-.001],[.05,0,.001],[-.05,.1,.001],[.05,.1,.001]],dtype=np.float64)
    F=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    B=P.copy();C=P+np.array([.01,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64)
    target_tri=np.ones((1,3,3),dtype=np.float64)
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.10,nearest_surface_fn=_rigid_assembly_surface_query)
    assert report.get('construction_mode')!='shape-preserving coherent rigid placement'


def test_multicomponent_cloth_with_distributed_skinning_does_not_enter_rigid_lane():
    from universal_garment_refit import coherent_universal_refit
    P,F=_many_disconnected_triangles(40)
    B=P+np.array([.004,0,0]);C=P+np.array([.008,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64);target_tri=np.ones((1,3,3),dtype=np.float64)
    W=np.zeros((len(P),4),dtype=np.float64)
    quarter=max(1,len(P)//4)
    for i in range(len(P)):
        W[i,i//quarter if i//quarter<4 else 3]=1.0
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.10,nearest_surface_fn=_rigid_assembly_surface_query,source_weights=W)
    assert report.get('construction_mode')!='shape-preserving coherent rigid placement'


def test_clear_standoff_with_small_body_motion_uses_transport_not_exact_source_noop():
    from universal_garment_refit import coherent_universal_refit
    P=np.array([[-.05,0,.02],[.05,0,.02],[-.05,.1,.02],[.05,.1,.02]],dtype=np.float64)
    F=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    B=P.copy();C=P+np.array([.004,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64);target_tri=np.ones((1,3,3),dtype=np.float64)
    macro=(np.eye(3),np.array([.004,0,0],dtype=np.float64),np.eye(3),{})
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.004,nearest_surface_fn=_stub_surface_query(.005),macro_transform=macro)
    assert not bool(report.get('already_fits_target',False))
    assert np.max(np.linalg.norm(out-P,axis=1))>0.001


def test_embedded_body_asset_disables_standoff_exact_source_noop():
    from universal_garment_refit import coherent_universal_refit
    P=np.array([[-.05,0,.02],[.05,0,.02],[-.05,.1,.02],[.05,.1,.02]],dtype=np.float64)
    F=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    B=P.copy();C=P+np.array([.02,0,0])
    source_tri=np.zeros((1,3,3),dtype=np.float64);target_tri=np.ones((1,3,3),dtype=np.float64)
    macro=(np.eye(3),np.array([.02,0,0],dtype=np.float64),np.eye(3),{})
    out,report=coherent_universal_refit(P,F,B,C,source_tri,target_tri,body_motion_p95_m=.10,nearest_surface_fn=_stub_surface_query(.005),macro_transform=macro,allow_authored_standoff_noop=False)
    assert not bool(report.get('already_fits_target',False))
    assert np.max(np.linalg.norm(out-P,axis=1))>0.001
