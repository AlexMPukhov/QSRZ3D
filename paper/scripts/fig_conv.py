import os; _R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'legend.frameon': False})
E = json.load(open('errs_s0.03125.json'))
U = ['u0.02', 'u0.01', 'u0.005', 'u0.0025', 'u0.00125']
S = ['s4', 's2', 's1', 's0.5', 's0.25', 's0.125']
CU, CS = '#1f5fbf', '#c4511a'
q = [('ni0', r'(a) ion density $n_i(r{=}0)$'), ('wr01', r'(b) focusing $W_r(r{=}0.01)$'), ('ez0', r'(c) $E_z(r{=}0)$')]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
for ax, (k, title) in zip(axs, q):
    nu = np.array([E[n]['N'] for n in U]); eu = np.array([E[n][k] for n in U])
    ns = np.array([E[n]['N'] for n in S]); es = np.array([E[n][k] for n in S])
    ax.loglog(nu, eu, 'o-', color=CU, mfc='white', ms=4.5, lw=1.2, label='uniform')
    ax.loglog(ns, es, 's-', color=CS, ms=4, lw=1.2, label='stretched')
    x = np.array([300, 2400]); ax.loglog(x, 3e-2 * (x / 300.) ** -2, 'k--', lw=0.7)
    ax.text(330, 3e-2 * 3.0 ** -2 * 0.25, r'$\propto N^{-2}$', fontsize=7)
    ax.set_title(title, fontsize=8, loc='left')
    ax.set_xlabel('radial cells $N$'); ax.set_xlim(120, 1.1e4); ax.set_ylim(5e-6, 0.6)
    ax.grid(True, which='major', lw=0.3, color='0.85')
    # mark the grids of Table II
    for n, c in (('u0.005', CU), ('s1', CS)):
        ax.annotate(f"{E[n]['N']}", (E[n]['N'], E[n][k]), xytext=(4, 4), textcoords='offset points', fontsize=6.5, color=c)
axs[0].set_ylabel('max. relative error in the witness')
axs[0].legend(loc='lower left', fontsize=7.5)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(os.path.join(_R, 'paper', 'tex', 'figs', 'fig_convergence.pdf')); fig.savefig('fig_convergence.png', dpi=160)
