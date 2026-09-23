#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import production_b14 as prod


def test_normal_frame_transport():
    v=np.asarray([[.002,.003,.004]],dtype=np.float64)
    ns=np.asarray([[0.0,0.0,1.0]],dtype=np.float64)
    nt=np.asarray([[0.0,1.0,0.0]],dtype=np.float64)
    out=prod._transport_vectors_between_normals(v,ns,nt)[0]
    # Shortest rotation maps +Z to +Y and keeps the perpendicular X component unchanged.
    assert np.allclose(out,np.asarray([.002,.004,-.003]),atol=1e-12),out


def test_cross_fold_constraint_reconciliation():
    faces=np.asarray([[0,1,2],[0,1,2]],dtype=np.int64)
    bary=np.asarray([[1.0,0.0,0.0],[1.0,0.0,0.0]],dtype=np.float64)
    errors=np.asarray([[.002,0.0,0.0],[-.006,0.0,0.0]],dtype=np.float64)
    sns=np.asarray([[0.0,0.0,1.0],[0.0,0.0,-1.0]],dtype=np.float64)
    tns=sns.copy()
    vns=np.asarray([[0.0,0.0,1.0]]*3,dtype=np.float64)
    vnt=vns.copy()
    legacy=(errors[0]+errors[1])*.5
    assert legacy[0]<0.0
    field,report=prod._coherent_coverage_vertex_field(3,faces,bary,errors,sns,tns,vns,vnt)
    assert field[0,0]>0.0019,field[0]
    assert np.allclose(field[1:],0.0,atol=1e-12),field
    assert report['cross_fold_rejected']>=1,report


def test_same_facing_literal_winner_replaces_opposite_fold():
    up=np.asarray([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],dtype=np.float64)
    down=np.asarray([[0.0,1.0,.001],[1.0,0.0,.001],[0.0,0.0,.001]],dtype=np.float64)
    literal=np.asarray([up,down],dtype=np.float64)
    support=np.asarray([up],dtype=np.float64)
    p=np.asarray([[.20,.20,.0020]],dtype=np.float64)
    _,base_n,base_signed,base_dist,_=prod._nearest_surface_reference_chunked(p,literal,k=2)
    assert base_signed[0]<-.00075,(base_n,base_signed,base_dist)
    _,n,signed,distance,face_index,rejected=prod._nearest_literal_surface_consistent_with_support(p,literal,support,exact_base=True)
    assert rejected==1,(rejected,n,signed,distance,face_index)
    assert face_index[0]==0,face_index
    assert signed[0]>.0019,signed
    assert np.dot(n[0],np.asarray([0.0,0.0,1.0]))>.99,n




def test_planar_coverage_transfer_completes_without_loss():
    n=10
    xy=np.asarray([(i/(n-1),j/(n-1)) for j in range(n) for i in range(n)],dtype=np.float64)*.09
    x=np.column_stack([xy,np.zeros(len(xy))]);y=x.copy();y[:,2]=.010
    ns=np.tile([0.0,0.0,1.0],(len(x),1));nt=ns.copy();src=np.column_stack([xy,np.full(len(xy),.001)])
    faces=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;b=a+1;c=a+n;d=c+1;faces.extend(((a,b,d),(a,d,c)))
    faces=np.asarray(faces,dtype=np.int64)
    out,report=prod._preserve_source_authored_body_coverage(src,src.copy(),faces,x,y,ns,nt,source_limit=.004,margin=.00065)
    assert report['source_covered_points']==100,report
    assert report['undercovered_before']==100,report
    assert report['undercovered_after']==0,report
    assert np.all(np.isfinite(out))
    assert float(np.median(out[:,2]))>.0079


def test_deep_low_alignment_collision_uses_bisector_push():
    support=np.asarray([[0.0,0.0,1.0]],dtype=np.float64)
    literal=np.asarray([[1.0,0.0,0.0]],dtype=np.float64)
    signed=np.asarray([-.0020],dtype=np.float64)
    out,report=prod._clearance_push_normals_with_bisector(support,literal,signed)
    assert report['used']==1,report
    # Expect a true bisector: equal positive X and Z components.
    assert out[0,0]>.70 and out[0,2]>.70,out
    assert abs(out[0,0]-out[0,2])<1e-6,out


def test_false_opposite_literal_collision_does_not_move_cloth():
    src=np.asarray([[0.0,0.0,.001],[.08,0.0,.001],[0.0,.08,.001]],dtype=np.float64);faces=np.asarray([[0,1,2]],dtype=np.int64)
    source_body=np.asarray([[[0.0,0.0,0.0],[.10,0.0,0.0],[0.0,.10,0.0]]],dtype=np.float64)
    intended=np.asarray([[0.0,0.0,.010],[.10,0.0,.010],[0.0,.10,.010]],dtype=np.float64)
    opposite=np.asarray([[0.0,.10,.011],[.10,0.0,.011],[0.0,0.0,.011]],dtype=np.float64)
    current=src.copy();current[:,2]=.012
    out,report=prod._source_covered_face_clearance(src,current,faces,source_body,np.asarray([intended,opposite]),margin=.00065,target_support_triangles=np.asarray([intended]))
    assert report['opposite_facing_literal_rejections']>0,report
    assert report['affected_faces']==0,report
    assert np.allclose(out,current,atol=1e-12),(out,current,report)


def test_genuine_same_facing_collision_still_clears():
    src=np.asarray([[0.0,0.0,.001],[.08,0.0,.001],[0.0,.08,.001]],dtype=np.float64);faces=np.asarray([[0,1,2]],dtype=np.int64)
    source_body=np.asarray([[[0.0,0.0,0.0],[.10,0.0,0.0],[0.0,.10,0.0]]],dtype=np.float64)
    target=np.asarray([[[0.0,0.0,.010],[.10,0.0,.010],[0.0,.10,.010]]],dtype=np.float64)
    current=src.copy();current[:,2]=.009
    out,report=prod._source_covered_face_clearance(src,current,faces,source_body,target,margin=.00065,target_support_triangles=target)
    assert report['affected_faces']==1,report
    assert report['opposite_facing_literal_rejections']==0,report
    assert report['sample_min_after_mm']>.65,report
    assert float(np.min(out[:,2]))>.0106,(out,report)

def main():
    test_normal_frame_transport()
    test_cross_fold_constraint_reconciliation()
    test_same_facing_literal_winner_replaces_opposite_fold()
    test_planar_coverage_transfer_completes_without_loss()
    test_deep_low_alignment_collision_uses_bisector_push()
    test_false_opposite_literal_collision_does_not_move_cloth()
    test_genuine_same_facing_collision_still_clears()
    print('coherent coverage frame tests: PASS')


if __name__=='__main__':
    main()
