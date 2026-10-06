import numpy as np
def read_lazy(fn):
    """memory-mapped reader of fields_*.bin (format v2): returns dict with r, xi and memmapped components"""
    with open(fn, 'rb') as f:
        ver, M, K, m1 = np.fromfile(f, np.int32, 4); t = np.fromfile(f, np.float64, 1)[0]
        r = np.fromfile(f, np.float64, M); xi = np.fromfile(f, np.float64, K); nc = np.fromfile(f, np.int32, 1)[0]
        off = f.tell()
    out = {'r': r, 'xi': xi, 't': t}
    for c in range(nc):
        with open(fn, 'rb') as f:
            f.seek(off); name = f.read(16).split(b'\0')[0].decode()
        out[name] = np.memmap(fn, np.float64, 'r', off + 16, (K, M))
        off += 16 + 8 * K * M
    return out
