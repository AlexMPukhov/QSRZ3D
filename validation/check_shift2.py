"""m = 1 validation by translation invariance (see check_shift.py): relative L2 errors
of the m = 1 amplitudes of a displaced driver against -d * d/dr of the centred solution.
usage: python check_shift2.py ref_dir shift_dir d rmax [ximin ximax]"""
import sys
import numpy as np
sys.path.insert(0, "../tools")
from quarz_read import read_fields
ref = read_fields(sys.argv[1] + "/fields_000000.bin")
sh = read_fields(sys.argv[2] + "/fields_000000.bin")
d, rmax = float(sys.argv[3]), float(sys.argv[4])
x0, x1 = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (-1e9, 1e9)
r, xi = ref["r"], ref["xi"]
dr = lambda f: np.gradient(f, r, axis=1)
orr = lambda f: np.where(r > 0, f / np.where(r > 0, r, 1), 0)
pred = {"psi_c": -d * dr(ref["psi"]), "ez_c": -d * dr(ref["ez"]), "ne_c": -d * dr(ref["ne"]),
        "er_c": -d * dr(ref["er"]), "eth_s": d * orr(ref["er"]), "bth_c": -d * dr(ref["bth"]), "br_s": -d * orr(ref["bth"])}
rmin = float(sys.argv[7]) if len(sys.argv) > 7 else 0.02
m = (r[None, :] > rmin) & (r[None, :] < rmax) & (xi[:, None] > x0) & (xi[:, None] < x1)
out = []
for n, p in pred.items():
    out.append("%s %.1e" % (n, np.sqrt(((sh[n] - p)[m] ** 2).sum() / (p[m] ** 2).sum())))
zero = max(np.abs(sh[n][m]).max() for n in ["psi_s", "ez_s", "er_s", "bth_s", "ne_s"])
print("  L2 rel. errors: " + ", ".join(out) + "   | max of sin parts that must vanish: %.1e" % zero)
