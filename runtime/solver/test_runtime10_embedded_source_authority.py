import numpy as np

from pathlib import Path
import sys
RUNTIME_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNTIME_ROOT / "rbody"))
from rbody_b14_adapter import apply_embedded_source_authority


def _grid(nx=8,ny=8,z=0.0):
    V=[]
    for y in range(ny):
        for x in range(nx):V.append(((x-(nx-1)/2)*.004,(y-(ny-1)/2)*.004,z))
    F=[]
    for y in range(ny-1):
        for x in range(nx-1):
            a=y*nx+x;b=a+1;c=a+nx;d=c+1;F.extend(((a,b,d),(a,d,c)))
    V=np.asarray(V,dtype=np.float64);F=np.asarray(F,dtype=np.int64)
    UV=V[:,:2].copy();UV-=UV.min(0);UV/=np.maximum(UV.max(0),1e-9)
    N=np.zeros_like(V);N[:,2]=1.0;W=np.ones((len(V),1),dtype=np.float64)
    return {'V':V,'F':F,'UV':UV,'N':N,'W':W,'joint_names':['j_test'],'payload_id':'catalogue','slot':'Chest','mesh_records':[]}


def test_runtime10_exact_embedded_topology_is_literal_source_authority():
    canonical=_grid();embedded={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in canonical.items()}
    embedded['V']=canonical['V'].copy();embedded['V'][:,2]+=.003*np.exp(-((embedded['V'][:,0]/.010)**2+(embedded['V'][:,1]/.010)**2))
    result,report=apply_embedded_source_authority(canonical,embedded)
    assert report['enabled'] is True
    assert report['mode']=='exact-embedded-topology'
    assert np.allclose(result['V'],embedded['V'])
    assert result['payload_id'].startswith('embedded-local:')
    assert report['move_max_mm']>2.0


def test_runtime10_partial_embedded_body_preserves_catalogue_completion():
    canonical=_grid(10,10);keep=np.arange(80,dtype=np.int64)
    embedded={
        'V':canonical['V'][keep].copy(),
        'UV':canonical['UV'][keep].copy(),
        'N':canonical['N'][keep].copy(),
        'W':canonical['W'][keep].copy(),
        'joint_names':['j_test'],'payload_id':'embedded','slot':'Chest',
    }
    # Build faces using only the retained region and compact its indices.
    lut=np.full(len(canonical['V']),-1,dtype=np.int64);lut[keep]=np.arange(len(keep),dtype=np.int64)
    mask=np.all(lut[canonical['F']]>=0,axis=1);embedded['F']=lut[canonical['F'][mask]]
    embedded['V'][:,2]+=.00030
    result,report=apply_embedded_source_authority(canonical,embedded)
    assert report['enabled'] is True
    assert report['mode']=='hybrid-embedded-with-rbody-completion'
    # The well-covered majority follows the literal source body.
    assert float(np.median(result['V'][:70,2]))>.00025
    # The deliberately absent final rows retain meaningful catalogue fallback rather than snapping remotely.
    assert float(np.median(np.abs(result['V'][90:,2]-canonical['V'][90:,2])))<.00012
    assert report['completion_fraction']>0.0
