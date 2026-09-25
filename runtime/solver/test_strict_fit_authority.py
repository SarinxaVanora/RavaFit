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


def test_strict_standoff_ignores_far_vertices_in_same_connected_shell():
    body,body_faces=_grid(n=9,spacing=.010)
    garment=body.copy();garment[:,2]=.0010
    # Same connected raw shell, but one edge represents a collar/strap region far from the body.
    far=np.flatnonzero(garment[:,1] > np.percentile(garment[:,1],75))
    garment[far,2]=.050
    solved=body.copy();solved[:,2]=.00020
    solved[far,2]=.010
    source=_PlaneSource(garment,body_faces.copy())
    cache={
        '_ravafit_strict_source_surface_V':body.copy(),'_ravafit_strict_source_surface_F':body_faces.copy(),
        '_ravafit_strict_target_surface_V':body.copy(),'_ravafit_strict_target_surface_F':body_faces.copy(),
    }
    out,report=strict_fit_authority.enforce_source_standoff(_PlaneProd(),source,cache,{'cloth':solved})
    # Close cloth regains its authored body spacing.
    close=np.setdiff1d(np.arange(len(garment)),far)
    assert np.percentile(out['cloth'][close,2],5) >= .00114
    # Far authored regions are not shrink-wrap/standoff targets and remain untouched.
    np.testing.assert_allclose(out['cloth'][far],solved[far],atol=1e-12)
    component=report['meshes'][0]['components'][0]
    assert component['supported_vertices'] < component['vertices']
    assert component['support_limit_mm'] <= 12.0


class _VetoProd(_PlaneProd):
    @staticmethod
    def _coupled_topology_safe_alpha(source,before,proposal,faces):
        # Simulate a topology veto: no geometry may move.
        return np.asarray(before,float).copy(),0.0,{},{}

    @staticmethod
    def _target_fit_collision_triangles(cache):
        V=np.asarray(cache['_ravafit_strict_target_surface_V'],float)
        F=np.asarray(cache['_ravafit_strict_target_surface_F'],np.int64)
        return V[F]

    @staticmethod
    def _final_target_body_clearance(source,positions,target_collision,target_support,margin=.00035):
        return positions,{'enabled':True}


def test_topology_veto_reports_remaining_standoff_instead_of_false_success(monkeypatch):
    import pytest
    body,body_faces=_grid(n=7,spacing=.012)
    garment=body.copy();garment[:,2]=.0010
    solved=body.copy();solved[:,2]=.00020
    source=_PlaneSource(garment,body_faces.copy())
    cache={
        '_ravafit_strict_source_surface_V':body.copy(),'_ravafit_strict_source_surface_F':body_faces.copy(),
        '_ravafit_strict_target_surface_V':body.copy(),'_ravafit_strict_target_surface_F':body_faces.copy(),
    }
    projected,report=strict_fit_authority.enforce_source_standoff(_VetoProd(),source,cache,{'cloth':solved})
    np.testing.assert_allclose(projected['cloth'],solved)
    assert report['maximum_remaining_shortfall_mm'] > .9

    # Occupancy being clear is not enough: unresolved authored standoff must prevent publication.
    monkeypatch.setattr(strict_fit_authority.final_occupancy_guard,'_dense_penetration_report',
        lambda *args,**kwargs:{'penetrating_samples':0,'minimum_signed_clearance_mm':1.0})
    with pytest.raises(ValueError,match='standoff'):
        strict_fit_authority.finalize_strict_solution(_VetoProd(),source,cache,{'cloth':solved})
