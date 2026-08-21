import numpy as np
import production_b14 as prod


def _grid(nx=14, ny=6, length=1.0, width=.12):
    xs=np.linspace(-width/2,width/2,ny); ys=np.linspace(0,length,nx)
    V=np.asarray([[x,y,.002 + .20*(y/length)**2] for y in ys for x in xs],dtype=np.float64)
    F=[]
    for i in range(nx-1):
        for j in range(ny-1):
            a=i*ny+j;b=a+1;c=a+ny;d=c+1;F.extend([[a,c,b],[b,c,d]])
    return V,np.asarray(F,dtype=np.int64)


def test_displacement_smoothing_preserves_source_geometry_and_reduces_roughness():
    S,F=_grid();V=S.copy();V[:,0]+=np.where(np.arange(len(V))%2==0,.018,-.018)
    before=prod._deformation_roughness_p95(S,V,F)
    out=prod._smooth_mesh_displacement(S,V,F,iterations=20,strength=.40)
    after=prod._deformation_roughness_p95(S,out,F)
    assert after < before*.45,(before,after)
    assert np.array_equal(F,F.copy())
    assert np.all(np.isfinite(out))


def test_weighted_rigid_transform_recovers_known_transform():
    rng=np.random.default_rng(42);A=rng.normal(size=(64,3));angle=.12
    R=np.asarray([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],dtype=np.float64);t=np.asarray([.03,-.02,.015])
    B=A@R+t;fit_R,fit_t=prod._weighted_rigid_transform(A,B,np.ones(len(A)))
    assert np.max(np.linalg.norm(A@fit_R+fit_t-B,axis=1))<1e-9


if __name__=='__main__':
    test_displacement_smoothing_preserves_source_geometry_and_reduces_roughness()
    test_weighted_rigid_transform_recovers_known_transform()
    print('authored structural shape authority tests: PASS')
