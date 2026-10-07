# Paper scripts

Scripts that produced the figures of `paper/tex/quarz_paper.tex` (run from a scratch directory
containing the run outputs `out_<case>/`, `log_<case>.txt`):

- `evo.sh` — the four evolution runs (Fig. evolution); `QUARZ=/path/to/quarz` selects the executable.
- `errs.py ref` — errors of the convergence series against the reference run (Fig. convergence,
  Table); writes `errs_ref.json`. `errs_s0.03125.json` is the result used in the paper.
- `fig_conv.py`, `fig_maps.py`, `fig_evo.py` — the figures, written to `paper/tex/figs/`.
- `closure.sh`, `fig_closure.py` — runs and figure of the bubble-closure subsection (Fig. closure):
  the pinched-witness deck with the box extended to xi = 12, cold vs `plasma.smooth_length = 0.005`
  (~3 min on 2 cores).
- `rf.py` — helper.

The input decks of all paper runs are in `paper/inputs/` (see its README).
