from __future__ import annotations
import numpy as np
from production_b14 import _preserve_authored_weld_splits

pos={'mesh':np.array([[0.,0.,0.],[0.002,0.,0.],[1.,0.,0.],[0.,1.,0.]])}
ctx={'mesh':{'w':{'raw_to_weld':np.array([0,0,1,2])},'data':{'F':np.array([[0,2,3],[1,2,3]]),'W':np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.]])}}}
out,changed,report=_preserve_authored_weld_splits(pos,ctx)
assert 'mesh' in changed and np.allclose(out['mesh'][0],out['mesh'][1])
assert report['corrected_group_count']==1

pos={'mesh':np.array([[0.,0.,0.],[0.002,0.,0.],[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]])}
ctx={'mesh':{'w':{'raw_to_weld':np.array([0,0,1,2,3,4])},'data':{'F':np.array([[0,2,3],[1,4,5]]),'W':np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.]])}}}
out,changed,report=_preserve_authored_weld_splits(pos,ctx)
assert 'mesh' not in changed and not np.allclose(out['mesh'][0],out['mesh'][1])
assert report['corrected_group_count']==0

# Shared topology but meaningfully different authored skinning: keep split.
pos={'mesh':np.array([[0.,0.,0.],[0.002,0.,0.],[1.,0.,0.],[0.,1.,0.]])}
ctx={'mesh':{'w':{'raw_to_weld':np.array([0,0,1,2])},'data':{'F':np.array([[0,2,3],[1,2,3]]),'W':np.array([[1.,0.],[0.,1.],[1.,0.],[1.,0.]])}}}
out,changed,report=_preserve_authored_weld_splits(pos,ctx)
assert 'mesh' not in changed and report['corrected_group_count']==0
print('authored weld seam tests: PASS')
