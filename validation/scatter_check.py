"""Section 18: multiple Coulomb scattering of beam particles on the plasma.
free OUT STEP T LI LE Z ZETA FACTOR : weak zero-emittance witness in a uniform plasma (n_e = n_i = 1):
    <p_x^2> = <p_y^2> = D t, D = k_p r_e factor (Z^2 n_i L_i + n_e L_e + (Z - zeta) n_i L_e); |p| conserved.
channel OUT GAMMA LI FACTOR : matched witness in a pure ion channel (n_i = 1, Z = 1):
    d eps_n/dt = D / sqrt(2 gamma), D = k_p r_e factor L_i (Kirby et al., PAC 2007)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from quarz_read import read_beam, read_beamlog  # noqa: E402

N0 = 1e17
KAPPA = 5.64146e4 * np.sqrt(N0) / 2.99792458e10 * 2.8179403262e-13
mode = sys.argv[1]
if mode == "free":
    out, step, T, Li, Le, Z, zeta, fac = sys.argv[2], int(sys.argv[3]), *map(float, sys.argv[4:10])
    b = read_beam(f"{out}/beam_witness_{step:06d}.bin")
    w = b["w"] > 0
    D = KAPPA * fac * (Z * Z * Li + Le + (Z - zeta) * Le)
    vx, vy = np.var(b["px"][w]), np.var(b["py"][w])
    g = np.sqrt(1 + b["px"] ** 2 + b["py"] ** 2 + b["pz"] ** 2)[w]
    print(f"  Z = {Z:g}, ion charge {zeta:g}: <p_x^2>/(D t) = {vx / (D * T):.4f}, <p_y^2>/(D t) = {vy / (D * T):.4f}"
          f" ({w.sum()} particles, statistical error {np.sqrt(2 / w.sum()):.4f});"
          f" gamma conserved to {np.max(np.abs(g / g.mean() - 1)):.0e}")
elif mode == "channel":
    out, gam, Li, fac = sys.argv[2], *map(float, sys.argv[3:6])
    b = read_beamlog(f"{out}/beams.txt")["witness"]
    t, e = b["t"], 0.5 * (b["emit_nx"] + b["emit_ny"])
    slope = np.polyfit(t, e, 1)[0]
    th = KAPPA * fac * Li / np.sqrt(2 * gam)
    print(f"  matched witness, gamma {gam:g}: d eps_n/dt = {slope:.4e}, theory D/sqrt(2 gamma) = {th:.4e},"
          f" ratio {slope / th:.4f}  (eps_n {e[0]:.5f} -> {e[-1]:.5f})")
