import numpy as np
import production_b14 as prod


def _legacy_macro(points, garment_weights, cache):
    X=np.asarray(cache["X"],dtype=np.float64);Y=np.asarray(cache["Y"],dtype=np.float64);BW=np.asarray(cache["BW"],dtype=np.float64);P=np.asarray(points,dtype=np.float64);Wg=np.asarray(garment_weights,dtype=np.float64)
    body_move=np.linalg.norm(Y-X,axis=1);p95=float(np.percentile(body_move,95)) if len(body_move) else 0.0
    if p95<.055 or len(X)<8:return None,{"enabled":False,"body_move_p95_mm":p95*1000.0}
    span=np.ptp(X,axis=0);sigma=float(np.clip(np.max(span)*.014,.012,.024));k=min(128,len(X));tree=prod.cKDTree(X);distance,index=tree.query(P,k=k)
    if distance.ndim==1:distance=distance[:,None];index=index[:,None]
    alignment=np.einsum("nk,nqk->nq",Wg,BW[index]);weight=np.exp(-((distance/max(sigma,1e-6))**2))*np.clip(alignment,.05,1.0)
    cross=(np.abs(P[:,0,None])>.020)&(np.sign(P[:,0,None])!=np.sign(X[index][:,:,0]));weight*=np.where(cross,np.exp(-5.0),1.0);weight/=np.maximum(weight.sum(axis=1,keepdims=True),1e-12)
    mapped=P+np.sum((Y-X)[index]*weight[:,:,None],axis=1);best=np.argmax(weight,axis=1);contact=index[np.arange(len(P)),best]
    return (mapped,contact,np.sum(distance*weight,axis=1),np.sum(alignment*weight,axis=1),weight,index),{"enabled":True,"body_move_p95_mm":p95*1000.0,"sigma_mm":sigma*1000.0,"samples":int(k)}


def test_extreme_macro_chunking_is_exactly_output_preserving():
    rng=np.random.default_rng(112358);n=180;q=23
    X=rng.normal(size=(n,3))*.08;Y=X+rng.normal(size=(n,3))*.075;BW=rng.random((n,q));BW/=BW.sum(axis=1,keepdims=True)
    P=rng.normal(size=(2100,3))*.09;Wg=rng.random((len(P),q));Wg/=Wg.sum(axis=1,keepdims=True);cache={"X":X,"Y":Y,"BW":BW}
    old,_=_legacy_macro(P,Wg,cache);new,report=prod._extreme_macro_body_field(P,Wg,cache)
    assert report["chunk_rows"]==1024
    for a,b in zip(old,new):assert np.array_equal(a,b)


def test_fragmented_decorated_shell_gate_is_extreme_and_generic():
    details=[{"kind":"shell","n":1400,"extent":[.2,.2,.2]}]+[{"kind":"rigid","n":1,"extent":[.001,.001,.001]} for _ in range(3740)]
    rue={"component_count":3741,"root_vertices":1378,"root_fraction":.123}
    assert prod._use_fragmented_decorated_shell_frame("stand_off_structured_shell",rue,details)
    assert not prod._use_fragmented_decorated_shell_frame("conservative_component_assembly",rue,details)
    assert not prod._use_fragmented_decorated_shell_frame("stand_off_structured_shell",{"component_count":31,"root_vertices":2856,"root_fraction":.17},details[:31])
    assert not prod._use_fragmented_decorated_shell_frame("stand_off_structured_shell",{"component_count":700,"root_vertices":1500,"root_fraction":.40},details[:700])
