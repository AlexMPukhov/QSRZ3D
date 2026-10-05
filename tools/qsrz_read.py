"""Readers for QSRZ output files (optional helper; formats are documented in README.md).

fields_NNNNNN.bin (version 2):
    int32 version(=2), int32 M (radial nodes), int32 K (xi slices), int32 mode1, float64 t,
    float64 r[M], float64 xi[K], int32 ncomp, then ncomp x { char name[16], float64 data[K][M] }
    mode 0 names: psi ez er eth br bth bz ne ni rhob
    mode 1 names: <name>_c, <name>_s with  f(r, theta) = f0 + f_c cos(theta) + f_s sin(theta)
beam_<name>_NNNNNN.bin:
    int32 n, then n records of 7 float64: x, y, p_x, p_y, p_z, xi, w
"""
import numpy as np


def read_fields(fn):
    with open(fn, "rb") as f:
        ver, M, K, m1 = np.fromfile(f, dtype=np.int32, count=4)
        if ver != 2:
            raise ValueError("unknown QSRZ field file version %d" % ver)
        t = np.fromfile(f, dtype=np.float64, count=1)[0]
        r = np.fromfile(f, dtype=np.float64, count=M)
        xi = np.fromfile(f, dtype=np.float64, count=K)
        nc = np.fromfile(f, dtype=np.int32, count=1)[0]
        out = {"t": t, "r": r, "xi": xi, "mode1": bool(m1)}
        for _ in range(nc):
            name = f.read(16).split(b"\0")[0].decode()
            out[name] = np.fromfile(f, dtype=np.float64, count=M * K).reshape(K, M)
    return out


def xz_plane(d, name):
    """Field in the (xi, x) plane through the axis (theta = 0 for x > 0, theta = pi for x < 0).
    Returns x (2M-1 points, -R..R) and array [K, 2M-1]."""
    r = d["r"]
    f0 = d[name]
    fc = d.get(name + "_c", np.zeros_like(f0))
    right = f0 + fc
    left = f0 - fc
    x = np.concatenate([-r[:0:-1], r])
    return x, np.concatenate([left[:, :0:-1], right], axis=1)


def read_beam(fn):
    with open(fn, "rb") as f:
        n = np.fromfile(f, dtype=np.int32, count=1)[0]
        a = np.fromfile(f, dtype=np.float64, count=7 * n).reshape(n, 7)
    return dict(zip(["x", "y", "px", "py", "pz", "xi", "w"], a.T))


def read_beamlog(fn):
    """beams.txt -> dict beam_name -> columns"""
    cols = ["step", "t", "alive", "npart", "charge", "gamma_mean", "gamma_rms", "xi_mean", "xi_rms",
            "r_rms", "emit_nx", "x_mean", "y_mean", "emit_ny", "sigma_x", "sigma_y"]
    data = {}
    for line in open(fn):
        if line.startswith("#"):
            continue
        p = line.split()
        data.setdefault(p[0], []).append([float(x) for x in p[1:]])
    return {k: dict(zip(cols, np.array(v).T)) for k, v in data.items()}


def read_slices(fn):
    a = np.loadtxt(fn)
    return dict(zip(["xi", "charge", "x_mean", "y_mean", "sigma_x", "sigma_y", "gamma_mean", "emit_nx"], a.T))
