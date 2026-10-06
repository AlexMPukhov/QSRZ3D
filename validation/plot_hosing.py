import sys; sys.path.insert(0, '../tools')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quarz_read import read_slices, read_fields, xz_plane
run = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/hosing'
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for n, c in [(0, '#888888'), (40, '#4C78A8'), (80, '#F58518'), (120, '#E45756')]:
    s = read_slices(f'{run}/slices_driver_{n:06d}.txt')
    ax[0].plot(s['xi'], s['x_mean'], color=c, lw=1.8, label=f't = {5 * n}')
ax[0].set_xlabel(r'$\xi\ [c/\omega_p]$'); ax[0].set_ylabel(r'slice centroid $\langle x\rangle\ [c/\omega_p]$')
ax[0].set_title('Electron hosing: driver slice centroids'); ax[0].legend(frameon=False)
d = read_fields(f'{run}/fields_000120.bin')
x, ne = xz_plane(d, 'ne'); _, rb = xz_plane(d, 'rhob')
a = ax[1]
m = a.pcolormesh(d['xi'], x, ne.T, shading='auto', cmap='Greys', vmin=0, vmax=3)
a.set_ylim(-2.5, 2.5); a.set_xlabel(r'$\xi$'); a.set_ylabel(r'$x$ (plane $\theta=0,\pi$)')
a.set_title(f'plasma electrons, t = {d["t"]:g}')
fig.colorbar(m, ax=a, label=r'$n_e/n_0$')
a = ax[2]
# display only: average over 25 slices (0.25 c/wp) against the particle noise of the fine near-axis cells
kern = np.ones(25) / 25
rbs = np.apply_along_axis(lambda c: np.convolve(c, kern, mode='same'), 0, -rb)
m = a.pcolormesh(d['xi'], x, rbs.T, shading='auto', cmap='magma', vmin=0, vmax=6)
a.set_ylim(-0.8, 0.8); a.set_xlabel(r'$\xi$'); a.set_ylabel(r'$x$')
a.set_title(f'driver density (m=0 + m=1, 0.25-averaged in $\\xi$), t = {d["t"]:g}')
fig.colorbar(m, ax=a, label=r'$n_b/n_0$')
fig.savefig('quarz_hosing.png', dpi=130)
print('ok')
