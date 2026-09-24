import numpy as np

import core_fit_authority
from macro_body_authority import macro_body_correspondence
from source_standoff_authority import authored_clearance_floor, outward_standoff_deficit


def _grid(n=25, spacing=.004):
    xs=(np.arange(n)-(n-1)/2)*spacing
    ys=(np.arange(n)-(n-1)/2)*spacing
    V=np.asarray([[x,y,0.] for y in ys for x in xs],float)
    F=[]
    for y in range(n-1):
        for x in range(n-1):
            a=y*n+x;b=a+1;c=a+n;d=c+1
            F.extend(((a,b,d),(a,d,c)))
    return V,np.asarray(F,np.int64)


def test_macro_body_keeps_broad_change_but_rejects_local_cleft():
    source,faces=_grid()
    r=np.linalg.norm(source[:,:2],axis=1)
    broad=.010 + .006*np.exp(-(r/.060)**2)
    cleft=-.008*np.exp(-((source[:,0]/.005)**2 + (source[:,1]/.010)**2))
    literal=source.copy();literal[:,2]+=broad+cleft
    macro,normals,report=macro_body_correspondence(source,literal,faces,radius_m=.022)
    centre=np.argmin(r)
    far=np.argmin(np.abs(r-.045))
    assert macro[far,2] > .010
    literal_depth=(broad[centre]+cleft[centre])-broad[centre]
    macro_residual=macro[centre,2]-broad[centre]
    assert abs(macro_residual) < abs(literal_depth)*.45
    assert report['rejected_local_relief_max_mm'] > 2.0
    assert np.all(np.isfinite(normals))


def test_authored_standoff_is_a_floor_not_a_target_shrinkwrap():
    source=np.asarray([.0010,.0008,.0005,.0040])
    current=np.asarray([.0001,.0004,.0008,.0060])
    deficit,desired=outward_standoff_deficit(current,source,minimum_clearance=.00035)
    np.testing.assert_allclose(desired,source)
    np.testing.assert_allclose(current+deficit,np.asarray([.0010,.0008,.0008,.0060]))
    assert deficit[2] == 0.0 and deficit[3] == 0.0


def test_clearance_floor_handles_bad_samples_without_destroying_authored_gap():
    source=np.asarray([np.nan,-1.0,.0002,.002])
    desired=authored_clearance_floor(source,.00035,.030)
    np.testing.assert_allclose(desired,np.asarray([.00035,.00035,.00035,.002]))


class _Prod:
    def _target_skin_weights_at_points(self, points, cache, source_weights=None, source_joint_names=None):
        out=np.asarray(source_weights,dtype=float).copy()
        # A real paired-body change: first half transfers 25% from bone 0 to bone 1.
        half=len(out)//2
        moved=np.minimum(out[:half,0],.25)
        out[:half,0]-=moved;out[:half,1]+=moved
        return out,np.zeros(len(out),dtype=float)


def test_prepare_cache_changes_the_real_support_frame_and_body_skin_field():
    source,faces=_grid(n=17,spacing=.005)
    r=np.linalg.norm(source[:,:2],axis=1)
    broad=.008+.004*np.exp(-(r/.050)**2)
    cleft=-.006*np.exp(-((source[:,0]/.005)**2+(source[:,1]/.009)**2))
    literal=source.copy();literal[:,2]+=broad+cleft
    weights=np.zeros((len(source),2),float);weights[:,0]=1.0
    cache={
        'X':source.copy(),'Y':literal.copy(),'NS':np.tile([0.,0.,1.],(len(source),1)),'NT':np.tile([0.,0.,1.],(len(source),1)),
        'BW':weights.copy(),'target_correspondence_W':weights.copy(),'names':['a','b'],'parts':np.asarray(['Legs']*len(source),object),
        'source_surface_V':source.copy(),'source_surface_F':faces.copy(),'source_surface_W':weights.copy(),
        'target_surface_V':literal.copy(),'target_surface_F':faces.copy(),'target_surface_W':weights.copy(),
        'source_support_V':source.copy(),'source_support_F':faces.copy(),'target_support_V':literal.copy(),'target_support_F':faces.copy(),
        'target_relief_C':np.ones_like(source)*.001,
    }
    report=core_fit_authority.prepare_cache(_Prod(),cache)
    assert report['enabled']
    assert report['macro_support_surface']['enabled']
    centre=np.argmin(r)
    assert cache['target_support_V'][centre,2] > literal[centre,2]
    assert np.max(np.abs(cache['target_relief_C'])) == 0.0
    assert report['target_skin_field']['paired_body_delta_l1_p95'] > .1
    assert not np.array_equal(cache['target_correspondence_W'],weights)


def test_runtime_support_guard_restores_source_authored_gap_on_real_production_signature():
    # Flat body/support frame with a garment that the base solver has shrink-wrapped too close.
    source,faces=_grid(n=7,spacing=.01)
    labels=np.zeros(len(source),dtype=np.int64)
    source_garment=source.copy();source_garment[:,2]=.0010
    mapped=source.copy();mapped[:,2]=.00025
    cache={'Y':source.copy(),'NT':np.tile([0.,0.,1.],(len(source),1)),'_ravafit_core_fit_report':{'enabled':True}}
    blend=np.ones((len(source),1),float);blend_ids=np.arange(len(source),dtype=np.int64)[:,None]

    class Prod:
        def _apply_target_relief_correction(self,*args):return args[2],{'enabled':True}
        def _support_frame_clearance_guard(self,w,mapped,blend,blend_ids,cache,labels,classes,source_body_triangles,minimum_clearance=.00035):return np.asarray(mapped,float).copy(),{'base':True}
        def _b14_nearest_surface(self,points,triangles,k=32):
            p=np.asarray(points,float);cp=p.copy();cp[:,2]=0.;normal=np.tile([0.,0.,1.],(len(p),1));distance=np.abs(p[:,2]);return cp,normal,p[:,2],distance,np.zeros(len(p),dtype=np.int64)
        def _coupled_topology_safe_alpha(self,source,before,proposal,faces):return proposal,1.0,{},{}
    prod=Prod();core_fit_authority.install_runtime_authority(prod)
    out,report=prod._support_frame_clearance_guard({'V':source_garment,'F':faces},mapped,blend,blend_ids,cache,labels,{0:'shell'},source[faces])
    assert np.percentile(out[:,2],5) >= .00110
    assert report['source_standoff']['adjusted_vertices'] == len(source)
