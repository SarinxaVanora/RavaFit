import numpy as np

from production_b14 import (
    _coupled_edge_metric,
    _coupled_topology_safe_alpha,
    _preserve_local_authored_garment_layers,
    _structural_laplacian_regularize,
)


def _grid(nx=8, ny=8, z=0.0):
    vertices=[]
    for y in range(ny):
        for x in range(nx):
            vertices.append(((x-(nx-1)/2)*0.004, (y-(ny-1)/2)*0.004, z))
    faces=[]
    for y in range(ny-1):
        for x in range(nx-1):
            a=y*nx+x;b=a+1;c=a+nx;d=c+1
            faces.append((a,b,d));faces.append((a,d,c))
    return np.asarray(vertices,dtype=np.float64),np.asarray(faces,dtype=np.int64)


def test_runtime9_source_differential_reconstruction_rejects_single_vertex_spear():
    source,faces=_grid(9,9,0.0)
    macro=source.copy();macro[:,0]*=1.08;macro[:,1]*=.94;macro[:,2]=.002*np.cos(macro[:,0]*70.0)
    centre=(len(source)//2);spiked=macro.copy();spiked[centre,2]+=.028
    before=_coupled_edge_metric(source,spiked,faces)
    solved=_structural_laplacian_regularize(source,spiked,faces,24.0)
    after=_coupled_edge_metric(source,solved,faces)
    assert np.all(np.isfinite(solved))
    assert after["max"] < before["max"]
    assert np.linalg.norm(solved[centre]-macro[centre]) < np.linalg.norm(spiked[centre]-macro[centre])
    # Away from the pathological point, the supplied macro target remains the positional authority.
    keep=np.ones(len(source),dtype=bool);keep[centre]=False
    assert float(np.percentile(np.linalg.norm(solved[keep]-macro[keep],axis=1),95)) < .0045


def test_runtime9_topology_alpha_guards_extreme_edge_stretch():
    source,faces=_grid(4,4,0.0);before=source.copy();proposed=source.copy();proposed[5]+=np.array([.040,0.0,.020])
    accepted,alpha,_,edge=_coupled_topology_safe_alpha(source,before,proposed,faces)
    assert alpha < 1.0
    assert edge["max"] <= 3.25 + 1e-9
    assert np.all(np.isfinite(accepted))


def test_runtime9_detects_and_restores_same_mesh_component_layering():
    inner,faces=_grid(8,8,.0100);outer,_=_grid(8,8,.0110);offset=len(inner)
    source=np.vstack((inner,outer));combined_faces=np.vstack((faces,faces+offset))
    # B14-like intermediate has locally collapsed the outer layer almost onto the inner layer.
    current=source.copy();current[offset:,2]=.01018
    body,body_faces=_grid(10,10,0.0);body_tri=body[body_faces]
    out,changed,report=_preserve_local_authored_garment_layers({"cloth":{"V":source,"F":combined_faces}},{"cloth":current},body_tri,body_tri)
    assert report["relation_count"] >= 1
    assert "cloth" in changed
    gap=float(np.median(out["cloth"][offset:,2]-out["cloth"][:offset,2]))
    assert gap >= .00070
    edge=_coupled_edge_metric(source,out["cloth"],combined_faces)
    assert edge["max"] < 1.35
