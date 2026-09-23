from __future__ import annotations
import numpy as np
import production_b14 as p

class _Source:
    def __init__(self, meshes): self._meshes=meshes
    def mesh_names(self): return list(self._meshes)
    def data(self,name): return self._meshes[name]

def _grid(n=10, step=.010, z=0.0):
    xs=(np.arange(n)-(n-1)/2)*step; ys=(np.arange(n)-(n-1)/2)*step
    V=np.asarray([[x,y,z] for y in ys for x in xs],float);F=[]
    for y in range(n-1):
        for x in range(n-1):
            a=y*n+x;b=a+1;c=a+n;d=c+1;F.extend(((a,b,d),(a,d,c)))
    return V,np.asarray(F,np.int64)

def test_target_centric_source_cutaway_fallback_suppresses_small_explicit_hole():
    V,F=_grid()
    # Physical source body explicitly cuts a compact central patch. The canonical source reference
    # remains complete; this is exactly the case where a small cutaway may fail the old >=20 vertex gate.
    centre=np.zeros(len(V),bool)
    xy=np.abs(V[:,:2]);centre=(xy[:,0]<=.0151)&(xy[:,1]<=.0151)
    keep_face=~np.any(centre[F],axis=1)
    actualF=F[keep_face]
    garmentV=V.copy();garmentV[:,2]=.005
    source=_Source({"body":{"V":V,"F":actualF},"garment":{"V":garmentV,"F":F}})
    W=np.ones((len(V),1),float)
    cache={"slot_pairs":[{
        "slot":"Legs","source_literal_V":V,"source_literal_F":F,"target_literal_V":V,"target_literal_F":F,
        "X":V,"Y":V,"BW":W,"target_literal_W":W,
        "target_mesh_records":[{"mesh_index":0,"face_offset":0,"face_count":len(F)}],
    }]}
    plan=p._source_body_suppression_plan(source,cache,{"Legs":[{"mesh_name":"body"}]},["Legs"],{"body"},None)
    row=plan["slots"][0]
    assert row["status"]=="source body explicitly removes garment-covered geometry"
    assert row.get("target_centric_fallback") is True
    assert row["suppressed_target_triangles"]>0
    assert plan["native_suppression"]
