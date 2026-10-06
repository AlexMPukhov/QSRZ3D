import os, sys; _R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.join(_R, 'tools'))
from quarz_read import read_beamlog
import numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.6,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'legend.frameon': False})
runs = [('u0.01', 'uniform, 800 cells', '#8fb3e8', 'o', '--'), ('u0.005', 'uniform, 1600 cells', '#1f5fbf', 'o', '-'),
        ('s1', 'stretched, 512 cells', '#c4511a', 's', '-'), ('s0.5', 'stretched, 967 cells', 'k', '', ':')]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for n, lab, c, mk, ls in runs:
    if not os.path.exists(f'evo_{n}/beams.txt'): continue
    w = read_beamlog(f'evo_{n}/beams.txt')['witness']
    t = w['t']; em = 0.5 * (w['emit_nx'] + w['emit_ny'])
    axs[0].plot(t, em / em[0], ls, color=c, lw=1.2, marker=mk, ms=3, markevery=10, mfc='white' if mk == 'o' else c, label=lab)
    axs[1].plot(t, w['r_rms'] / w['r_rms'][0], ls, color=c, lw=1.2, marker=mk, ms=3, markevery=10, mfc='white' if mk == 'o' else c)
axs[0].set_ylabel(r'$\varepsilon_n / \varepsilon_{n0}$'); axs[1].set_ylabel(r'$r_\mathrm{rms} / r_\mathrm{rms,0}$')
for ax, l in zip(axs, 'ab'):
    ax.set_xlabel(r'$t\ (\omega_p^{-1})$'); ax.set_xlim(0, 1000); ax.grid(True, lw=0.3, color='0.85')
    ax.set_title(f'({l}) witness ' + ('emittance' if l == 'a' else 'rms radius'), fontsize=8, loc='left')
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=4, fontsize=7.5)
fig.savefig(os.path.join(_R, 'paper', 'tex', 'figs', 'fig_evolution.pdf')); fig.savefig('fig_evolution.png', dpi=160)
