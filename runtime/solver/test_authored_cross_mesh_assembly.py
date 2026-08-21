from __future__ import annotations
import numpy as np
import production_b14 as prod


def strip(y0: float, gap: float = 0.0):
    # Two-row open strip with a four-vertex open end at x=1.
    V=np.asarray([[0,y0,0],[1,y0,0],[0,y0+.1,0],[1,y0+.1,0]],dtype=np.float64)
    F=np.asarray([[0,1,2],[1,3,2]],dtype=np.int64)
    return V,F


def main():
    A,FA=strip(0.0);B,FB=strip(0.0)
    n=16
    ys=np.linspace(0.0,.15,n)
    A=np.column_stack((np.ones(n),ys,np.zeros(n)))
    B=A.copy()
    # Fan strips around the boundary points so every point is genuinely on an open boundary.
    A2=np.column_stack((np.full(n,.95),ys,np.zeros(n)));B2=np.column_stack((np.full(n,1.05),ys,np.zeros(n)))
    VA=np.vstack((A,A2));VB=np.vstack((B,B2))
    FA=[];FB=[]
    for i in range(n-1):
        FA.extend(((i,n+i,i+1),(i+1,n+i,n+i+1)))
        FB.extend(((i,i+1,n+i),(i+1,n+i+1,n+i)))
    FA=np.asarray(FA,dtype=np.int64);FB=np.asarray(FB,dtype=np.int64)
    source={'a':{'V':VA,'F':FA},'b':{'V':VB,'F':FB}}
    solved={'a':VA.copy(),'b':VB.copy()}
    solved['b'][:n,0]+=0.003
    out,changed,report=prod._preserve_authored_cross_mesh_assembly(source,solved,set())
    assert report['relation_count']==1,report
    before=np.median(np.linalg.norm(solved['b'][:n]-solved['a'][:n],axis=1))
    after=np.median(np.linalg.norm(out['b'][:n]-out['a'][:n],axis=1))
    assert after<before*.35,(before,after,report)
    assert changed=={'a','b'},changed

    far={'a':{'V':VA,'F':FA},'b':{'V':VB+np.asarray([0.02,0,0]),'F':FB}}
    _,changed2,report2=prod._preserve_authored_cross_mesh_assembly(far,{'a':VA.copy(),'b':VB.copy()+np.asarray([0.02,0,0])},set())
    assert report2['relation_count']==0,report2
    assert not changed2,changed2
    print('authored cross-mesh assembly: PASS')

if __name__=='__main__':main()
