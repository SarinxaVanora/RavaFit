from types import SimpleNamespace

import numpy as np

import strict_fit_authority


def _grid(n=13, spacing=.006):
    xs=(np.arange(n)-(n-1)/2)*spacing
    ys=(np.arange(n)-(n-1)/2)*spacing
    V=np.asarray([[x,y,0.] for y in ys for x in xs],dtype=float)
    F=[]
    for y in range(n-1):
        for x in range(n-1):
            a=y*n+x;b=a+1;c=a+n;d=c+1
            F.extend(((a,b,d),(a,d,c)))
    return V,np.asarray(F,dtype=np.int64)


class _SurfaceProd:
    @staticmethod
    def _triangles_from_surface(vertices, faces):
        return np.asarray(vertices,dtype=float)[np.asarray(faces,dtype=np.int64)]


def test_strict_cache_uses_macro_shape_but_keeps_literal_collision():
    source,faces=_grid()
    r=np.linalg.norm(source[:,:2],axis=1)
    broad=.009+.004*np.exp(-(r/.045)**2)
    cleft=-.006*np.exp(-((source[:,0]/.004)**2+(source[:,1]/.009)**2))
    literal=source.copy();literal[:,2]+=broad+cleft
    weights=np.ones((len(source),1),dtype=float)
    cache={
        'X':source.copy(),'Y':literal.copy(),'BW':weights.copy(),'target_correspondence_W':weights.copy(),
        'NS':np.tile([0.,0.,1.],(len(source),1)),'NT':np.tile([0.,0.,1.],(len(source),1)),
        'names':['j_test'],'parts':np.asarray(['Legs']*len(source),object),
        '_ravafit_strict_source_surface_V':source.copy(),'_ravafit_strict_source_surface_F':faces.copy(),
        '_ravafit_strict_target_surface_V':literal.copy(),'_ravafit_strict_target_surface_F':faces.copy(),
        'source_surface_V':source.copy(),'source_surface_F':faces.copy(),'source_surface_W':weights.copy(),
        'target_surface_V':literal.copy(),'target_surface_F':faces.copy(),'target_surface_W':weights.copy(),
        'source_support_V':source.copy(),'source_support_F':faces.copy(),'target_support_V':literal.copy(),'target_support_F':faces.copy(),
        'target_relief_C':np.zeros_like(source),
    }
    report=strict_fit_authority.prepare_strict_cache(_SurfaceProd(),cache)
    centre=int(np.argmin(r))

    assert report['enabled'] and report['strict_lane']
    # Macro shaping has bridged a meaningful amount of the literal cleft.
    assert cache['_ravafit_strict_target_surface_V'][centre,2] > literal[centre,2] + .0015
    # Literal collision bytes/geometry remain separately authoritative.
    np.testing.assert_allclose(cache['_ravafit_literal_strict_target_surface_V'],literal)
    np.testing.assert_allclose(cache['_ravafit_target_collision_triangles'],literal[faces])
    # B14 target support now shares source topology rather than literal target topology/detail.
    np.testing.assert_array_equal(cache['_ravafit_strict_target_surface_F'],faces)
    np.testing.assert_allclose(cache['target_support_V'],cache['_ravafit_strict_target_surface_V'])


class _PlaneSource:
    def __init__(self, vertices, faces):
        self.vertices=vertices;self.faces=faces
    def data(self,name):
        return {'V':self.vertices,'F':self.faces}


class _PlaneProd:
    @staticmethod
    def _triangles_from_surface(vertices,faces):
        return np.asarray(vertices,float)[np.asarray(faces,np.int64)]

    @staticmethod
    def _b14_nearest_surface(points,triangles,k=32):
        points=np.asarray(points,float)
        z=float(np.mean(np.asarray(triangles,float)[:,:,2]))
        closest=points.copy();closest[:,2]=z
        normals=np.tile([0.,0.,1.],(len(points),1))
        signed=points[:,2]-z
        distance=np.abs(signed)
        return closest,normals,signed,distance,np.zeros(len(points),dtype=np.int64)


def test_strict_standoff_restores_source_authored_close_cloth_gap():
    body,body_faces=_grid(n=7,spacing=.012)
    garment=body.copy();garment[:,2]=.0010
    solved=body.copy();solved[:,2]=.00020
    garment_faces=body_faces.copy()
    source=_PlaneSource(garment,garment_faces)
    cache={
        '_ravafit_strict_source_surface_V':body.copy(),'_ravafit_strict_source_surface_F':body_faces.copy(),
        '_ravafit_strict_target_surface_V':body.copy(),'_ravafit_strict_target_surface_F':body_faces.copy(),
    }
    out,report=strict_fit_authority.enforce_source_standoff(_PlaneProd(),source,cache,{'cloth':solved})
    assert report['moved_vertices'] == len(garment)
    # 1.0 mm authored source gap + the 0.15 mm safety pad is restored, rather than accepting 0.2 mm.
    assert np.percentile(out['cloth'][:,2],5) >= .00114
    assert report['maximum_move_mm'] >= .9
