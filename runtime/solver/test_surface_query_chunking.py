from __future__ import annotations
import numpy as np
import b14_compat

def main():
    rng=np.random.default_rng(724)
    triangles=rng.normal(size=(317,3,3))*0.2
    points=rng.normal(size=(13001,3))*0.15
    expected=b14_compat._original_nearest_surface(points,triangles,k=32)
    b14_compat.reset_runtime_caches()
    actual=b14_compat._cached_nearest_surface(points,triangles,k=32)
    for i in range(4):
        if not np.array_equal(expected[i],actual[i]):
            raise AssertionError(f'cached/chunked nearest-surface numeric mismatch at output {i}: max={np.max(np.abs(expected[i]-actual[i]))}')
    if not np.array_equal(expected[4],actual[4]):
        raise AssertionError('cached/chunked nearest-surface face ids changed')
    print('surface query chunking equivalence: PASS')

if __name__=='__main__':main()
