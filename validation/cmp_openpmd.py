"""compare two openPMD series (HDF5) of QSRZ runs, e.g. serial vs MPI: usage cmp_openpmd.py dirA dirB"""
import sys, numpy as np, openpmd_api as io
A = io.Series(sys.argv[1] + '/openpmd/data_%06T.h5', io.Access.read_only)
B = io.Series(sys.argv[2] + '/openpmd/data_%06T.h5', io.Access.read_only)
worst = 0.0
for n, ia in A.iterations.items():
    ib = B.iterations[n]
    assert ia.time == ib.time
    for name, m in ia.meshes.items():
        for c, rc in m.items():
            a = rc.load_chunk(); b = ib.meshes[name][c].load_chunk(); A.flush(); B.flush()
            assert a.shape == b.shape
            worst = max(worst, np.abs(a - b).max() / (np.abs(a).max() or 1))
    for name, sp in ia.particles.items():
        for rn in ('position', 'momentum'):
            for c, rc in sp[rn].items():
                a = rc.load_chunk(); b = ib.particles[name][rn][c].load_chunk(); A.flush(); B.flush()
                worst = max(worst, np.abs(np.sort(a) - np.sort(b)).max() / (np.abs(a).max() or 1))
print(f'{sys.argv[1]} vs {sys.argv[2]}: iterations {list(A.iterations)}, max rel diff {worst:.2e}')
