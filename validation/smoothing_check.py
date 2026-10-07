"""Section 14: radial smoothing of the plasma sources (plasma.smooth_length) regularizes the axis
caustic at the bubble back. usage: smoothing_check.py cold_ref smoothed_run...
Prints for each run the E_z minimum and the axis density maximum in the closure region (xi 8.5..9.6),
and the r.m.s. change of E_z(0), psi(0) behind the bubble (9.2..11.9) and around the witness
(6.4..7.2) relative to the cold, unsmoothed reference. Then the spread of the E_z minimum over the
smoothed runs (different axis cells / xi steps: should be small, the spike is now resolved)."""
import sys

import numpy as np


def load(d):
    a = np.loadtxt(d + "/axis_000000.txt", comments="#")
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def l2(x0, f0, x1, f1, a, b):
    xs = np.arange(a, b, 0.005)
    r0, r1 = np.interp(xs, x0, f0), np.interp(xs, x1, f1)
    return np.sqrt(np.mean((r1 - r0) ** 2) / np.mean(r0 ** 2))


xr, er, pr, nr = load(sys.argv[1])
m = (xr > 8.5) & (xr < 9.6)
print("  %-16s closure: min Ez %7.1f  max ne(0) %9.0f   (cold, not smoothed)" % (sys.argv[1], er[m].min(), nr[m].max()))
mins = []
for d in sys.argv[2:]:
    x, e, p, n = load(d)
    m = (x > 8.5) & (x < 9.6)
    mins.append(e[m].min())
    print("  %-16s closure: min Ez %7.1f  max ne(0) %9.0f | vs cold: behind Ez %.1e psi %.1e, witness Ez %.1e psi %.1e"
          % (d, e[m].min(), n[m].max(), l2(xr, er, x, e, 9.2, 11.9), l2(xr, pr, x, p, 9.2, 11.9),
             l2(xr, er, x, e, 6.4, 7.2), l2(xr, pr, x, p, 6.4, 7.2)))
print("  smoothed runs: min Ez %.2f +- %.2f (spread over axis cells and xi steps)" % (np.mean(mins), np.std(mins)))
