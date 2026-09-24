import numpy as np

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
