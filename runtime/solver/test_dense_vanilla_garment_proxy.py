from __future__ import annotations
import numpy as np
import production_b14 as prod


def main():
    data={
        'V':np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float64),
        'F':np.array([[0,1,2]],dtype=np.int64),
        'N':np.array([[0.,0.,1.]]*3,dtype=np.float64),
        'UV':np.array([[0.,0.],[1.,0.],[0.,1.]],dtype=np.float64),
        'W':np.array([[1.,0.],[0.5,0.5],[0.,1.]],dtype=np.float64),
        'joint_names':['a','b'],
    }
    dense=prod._subdivide_vanilla_garment_data_once(data)
    assert len(dense['V']) == 6 and len(dense['F']) == 4
    assert np.array_equal(dense['V'][:3],data['V'])
    assert np.array_equal(dense['N'][:3],data['N'])
    assert np.array_equal(dense['UV'][:3],data['UV'])
    assert np.array_equal(dense['W'][:3],data['W'])
    print('dense vanilla garment proxy topology: PASS')

if __name__ == '__main__':
    main()
