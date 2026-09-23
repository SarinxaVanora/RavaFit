from __future__ import annotations
import unittest
import numpy as np
from b14_local_retarget import LocalRetargetConfig, retarget_b14_local_components, retarget_attached_ribbons_to_assembly, preserve_source_proven_attachment_continuity, _source_attachment_relations


def _plane_nearest(vertices, triangles, k=24):
    v=np.asarray(vertices,float);q=v.copy();q[:,2]=0.0;d=v[:,2].copy();n=np.tile(np.array([0.0,0.0,1.0]),(len(v),1));idx=np.zeros(len(v),int);return q,n,d,idx,np.zeros(len(v))


def _polish(vertices, faces, triangles, margin=.00015, iterations=8, blend=.35, max_push=.0015):
    out=np.asarray(vertices,float).copy();bad=out[:,2]<margin;out[bad,2]=margin;return out,{"affected":int(np.count_nonzero(bad))}


class B14LocalRetargetTests(unittest.TestCase):
    def test_non_selected_geometry_is_exactly_untouched(self):
        # One long open ribbon and one compact closed-ish triangle island. Only the ribbon has evidence.
        x=np.linspace(0,0.10,8);top=np.c_[x,np.zeros_like(x),np.full_like(x,.001)];bottom=np.c_[x,np.full_like(x,.01),np.full_like(x,.001)];ribbon=np.vstack((top,bottom));faces=[]
        for i in range(7):faces.extend(((i,i+1,8+i),(i+1,9+i,8+i)))
        detail=np.array([[.2,0,.01],[.21,0,.01],[.2,.01,.01]],float);offset=len(ribbon);faces.extend(((offset,offset+1,offset+2),));P=np.vstack((ribbon,detail));F=np.asarray(faces,int);B=P.copy();B[:len(ribbon),2]-=.003;C=P.copy()
        target=np.zeros((1,3,3),float);source=np.zeros((1,3,3),float)
        out,report=retarget_b14_local_components(P,F,B,C,source,target,nearest_surface_fn=_plane_nearest,collision_polish_fn=_polish,config=LocalRetargetConfig(min_vertices=8,min_b14_clearance_p95_error_m=.0005,required_clearance_p95_ratio=.8))
        self.assertTrue(np.array_equal(out[len(ribbon):],B[len(ribbon):]))
        self.assertGreater(report["changed_vertex_count"],0)

    def test_rejects_when_correspondence_is_not_better(self):
        x=np.linspace(0,0.10,8);a=np.c_[x,np.zeros_like(x),np.full_like(x,.002)];b=np.c_[x,np.full_like(x,.01),np.full_like(x,.002)];P=np.vstack((a,b));F=[]
        for i in range(7):F.extend(((i,i+1,8+i),(i+1,9+i,8+i)))
        F=np.asarray(F,int);B=P.copy();C=B.copy();out,report=retarget_b14_local_components(P,F,B,C,np.zeros((1,3,3)),np.zeros((1,3,3)),nearest_surface_fn=_plane_nearest,collision_polish_fn=_polish,config=LocalRetargetConfig(min_b14_clearance_p95_error_m=.0001))
        self.assertTrue(np.array_equal(out,B));self.assertEqual(report["selected_components"],0)

    def test_broad_source_tight_shell_follows_body_correspondence_without_moving_boundary(self):
        # A broad tight shell with a bad local B14 bulge.  The body correspondence is smooth,
        # so the whole supported interior should move coherently toward it while the authored
        # outer boundary remains exact.  No control/reference garment participates.
        n=25
        xs=np.linspace(-.10,.10,n);ys=np.linspace(-.10,.10,n)
        xx,yy=np.meshgrid(xs,ys,indexing="xy")
        P=np.c_[xx.ravel(),yy.ravel(),np.full(n*n,.005)]
        faces=[]
        for y in range(n-1):
            for x in range(n-1):
                a=y*n+x;b=a+1;c=a+n;d=c+1
                faces.extend(((a,b,c),(b,d,c)))
        F=np.asarray(faces,int)
        B=P.copy();r2=xx.ravel()**2+yy.ravel()**2;B[:,2]+=.005*np.exp(-r2/(2*.012**2))
        C=P.copy()  # smooth body-derived destination for this synthetic case
        source=np.zeros((1,3,3),float);target=np.zeros((1,3,3),float)
        out,report=retarget_b14_local_components(P,F,B,C,source,target,nearest_surface_fn=_plane_nearest,collision_polish_fn=_polish)
        fit=report["body_guided_shell_fit"]
        self.assertGreater(fit["changed_vertex_count"],0)
        self.assertTrue(fit["components"][0]["selected"])
        centre=(n//2)*n+n//2
        self.assertLess(out[centre,2],B[centre,2])
        # Corners and authored boundary stay exact.
        self.assertTrue(np.array_equal(out[[0,n-1,n*(n-1),n*n-1]],B[[0,n-1,n*(n-1),n*n-1]]))
        self.assertEqual(report["selected_components"],0)


    def test_final_attachment_closure_keeps_compact_connector_and_ribbon_attached(self):
        # Large stable shell -> compact disconnected connector -> long disconnected ribbon.  The
        # final assembly deliberately leaves the connector/ribbon behind after the shell moves,
        # reproducing the exact class of post-reconciliation detach this pass must close.
        shell=np.array([[0,0,0],[0,.005,0],[0,.010,0],[-.02,0,0],[-.02,.005,0],[-.02,.010,0],[0,0,.005],[0,.005,.005],[0,.010,.005],[-.02,0,.005],[-.02,.005,.005],[-.02,.010,.005]],float)
        faces=[(0,3,1),(3,4,1),(1,4,2),(4,5,2),(6,7,9),(7,10,9),(7,8,10),(8,11,10),(0,6,3),(3,6,9),(2,5,8),(5,11,8)]
        connector=np.array([[.0005,0,0],[.0005,.005,0],[.0005,.010,0],[.001,.005,.003]],float);co=len(shell)
        faces.extend(((co,co+1,co+3),(co+1,co+2,co+3)))
        x=np.linspace(.0012,.1012,8);ribbon=np.vstack([np.c_[x,np.full(8,y),np.zeros(8)] for y in (0,.005,.01)]);ro=co+len(connector)
        for row in range(2):
            a=ro+row*8;b=ro+(row+1)*8
            for i in range(7):faces.extend(((a+i,a+i+1,b+i),(a+i+1,b+i+1,b+i)))
        P=np.vstack((shell,connector,ribbon));F=np.asarray(faces,int);assembly=P.copy();assembly[:len(shell),1]+=.006
        out,report=preserve_source_proven_attachment_continuity(P,F,assembly,config=LocalRetargetConfig(min_vertices=8,min_extent_m=.04))
        self.assertTrue(report["enabled"])
        self.assertGreater(report["changed_vertex_count"],0)
        # Stable shell remains authority and is bit-identical.
        self.assertTrue(np.array_equal(out[:len(shell)],assembly[:len(shell)]))
        # Connector follows the shell's final local frame.
        self.assertGreater(float(np.mean(out[co:ro,1]-P[co:ro,1])),.0055)
        # The attached ribbon end follows the now-final connector rather than remaining behind.
        ribbon_start=np.array([ro,ro+8,ro+16])
        self.assertGreater(float(np.mean(out[ribbon_start,1]-P[ribbon_start,1])),.0050)

    def test_final_attachment_closure_hard_pins_source_witness_gap(self):
        # Stable shell -> compact loop -> long ribbon.  The loop and ribbon are deliberately pulled
        # apart in the solved assembly; final closure must restore the actual source witness gap,
        # not merely move their centroids in roughly the same direction.
        shell=np.array([[0,0,0],[0,.005,0],[0,.010,0],[-.02,0,0],[-.02,.005,0],[-.02,.010,0],[0,0,.005],[0,.005,.005],[0,.010,.005],[-.02,0,.005],[-.02,.005,.005],[-.02,.010,.005]],float)
        faces=[(0,3,1),(3,4,1),(1,4,2),(4,5,2),(6,7,9),(7,10,9),(7,8,10),(8,11,10),(0,6,3),(3,6,9),(2,5,8),(5,11,8)]
        loop=np.array([[.0006,0,0],[.0006,.005,0],[.0006,.010,0],[.0012,.005,.003]],float);lo=len(shell)
        faces.extend(((lo,lo+1,lo+3),(lo+1,lo+2,lo+3)))
        x=np.linspace(.0014,.1014,8);ribbon=np.vstack([np.c_[x,np.full(8,y),np.zeros(8)] for y in (0,.005,.01)]);ro=lo+len(loop)
        for row in range(2):
            a=ro+row*8;b=ro+(row+1)*8
            for i in range(7):faces.extend(((a+i,a+i+1,b+i),(a+i+1,b+i+1,b+i)))
        P=np.vstack((shell,loop,ribbon));F=np.asarray(faces,int);assembly=P.copy();assembly[:len(shell),1]+=.004;assembly[lo:ro,1]+=.004;assembly[ro:,1]-=.005
        cfg=LocalRetargetConfig(min_vertices=8,min_extent_m=.04)
        out,report=preserve_source_proven_attachment_continuity(P,F,assembly,config=cfg)
        labels,relations,_=_source_attachment_relations(P,F,cfg)
        rel=next(r for r in relations if set(r["components"])=={1,2})
        src=np.asarray([np.linalg.norm(P[a]-P[b]) for a,b,_ in rel["pairs"]])
        before=np.asarray([np.linalg.norm(assembly[a]-assembly[b]) for a,b,_ in rel["pairs"]])
        after=np.asarray([np.linalg.norm(out[a]-out[b]) for a,b,_ in rel["pairs"]])
        self.assertGreater(float(np.median(before)),float(np.median(src))+.003)
        self.assertLess(abs(float(np.median(after))-float(np.median(src))),.00025)
        self.assertTrue(report["enabled"])

    def test_attached_ribbon_rebinds_to_final_source_proven_endpoint_frames(self):
        x=np.linspace(0,.10,8);rows=[np.c_[x,np.full_like(x,y),np.full_like(x,.001)] for y in (0.0,.005,.01)];ribbon=np.vstack(rows);faces=[]
        for row in range(2):
            a0=row*8;b0=(row+1)*8
            for i in range(7):faces.extend(((a0+i,a0+i+1,b0+i),(a0+i+1,b0+i+1,b0+i)))
        # Two compact disconnected attachment pieces, with three close witnesses at each ribbon end.
        left=np.array([[-.001,.000,.001],[-.001,.005,.001],[-.001,.010,.001],[-.002,.005,.002]],float)
        right=np.array([[.101,.000,.001],[.101,.005,.001],[.101,.010,.001],[.102,.005,.002]],float)
        lo=len(ribbon);ro=lo+len(left);faces.extend(((lo,lo+1,lo+3),(lo+1,lo+2,lo+3),(ro,ro+1,ro+3),(ro+1,ro+2,ro+3)))
        P=np.vstack((ribbon,left,right));F=np.asarray(faces,int);B=P.copy();B[:len(ribbon),2]-=.003;C=P.copy();assembly=P.copy();assembly[lo:ro,1]+=.004;assembly[ro:,1]+=.008
        cfg=LocalRetargetConfig(min_vertices=8,min_b14_clearance_p95_error_m=.0005,required_clearance_p95_ratio=.8)
        out,report=retarget_attached_ribbons_to_assembly(P,F,B,C,assembly,np.zeros((1,3,3)),np.zeros((1,3,3)),nearest_surface_fn=_plane_nearest,collision_polish_fn=_polish,config=cfg)
        self.assertEqual(report["selected_component_count"],1)
        # Endpoint columns follow the independently moved source-proven attachment frames.
        left_end=np.array([0,8,16]);right_end=np.array([7,15,23])
        self.assertGreater(float(np.mean(out[left_end,1])-np.mean(P[left_end,1])),.002)
        self.assertGreater(float(np.mean(out[right_end,1])-np.mean(P[right_end,1])),.005)
        self.assertTrue(np.array_equal(out[lo:],assembly[lo:]))

if __name__=="__main__":unittest.main()
