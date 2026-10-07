# QUARZ — notes for Claude sessions

Read this first in every session. It records the project state, so work does not depend on
conversation history (which can be lost, e.g. when a turn is cut off). README.md is the full
user documentation (physics, numerics, inputs, outputs, validation); this file is the
working state and the rules for developing.

## Workflow rules
- **Start of a session:** read this file, then `git status` and `git log --oneline | head`.
  Uncommitted changes = an earlier session was interrupted: inspect them (`git diff`) and
  verify before building on them; tell Alexander what was found.
- **At every milestone** (a feature works, a bug is fixed, validation passes): commit with a
  descriptive message and update "Status" / "Open issues" below in the same commit.
  Commit unfinished work too, as `WIP: ...`, before long runs or at the end of a turn.
- Never report a feature as done without running the relevant validation section.
- **Remote:** `origin` = https://github.com/AlexMPukhov/QUARZ (PUBLIC; renamed from QSRZ3D on
  2026-10-06, GitHub redirects the old URL).
  Push after every commit (`git push origin main`); a new session starts by cloning it into
  `/home/claude/quarz` (`git clone <url> quarz`). Commit email:
  alex.m.pukhov@gmail.com (GitHub account of Alexander).
- Deliverable to Alexander: `quarz.zip` of the repo (no build dirs, no outputs), see below.

## Project
**Name:** QUARZ = Quasistatic Arbitrary-resolution RZ code (chosen by Alexander and Thomas,
2026-10-06; formerly QSRZ). Executable `quarz`, namespace `quarz`, macros/CMake options
`QUARZ_*`. Old names still accepted: CMake `-DQSRZ_*` options (deprecation warning),
`tools/qsrz_read.py` (shim), `QSRZ_*` variables in `tools/build_gpu.sh`. Input keys and output
formats unchanged (openPMD attributes renamed `qsrz_*` -> `quarz_*`; no reader depends on them).
Quasi-static axisymmetric (m = 0, optional m = 1) PIC code for plasma wakefield
acceleration, LCODE-like, written in C++17 with Kokkos (OpenMP / CUDA / HIP), MPI
decomposition along xi with pipelined time steps, optional openPMD output.
Distinctive: arbitrary non-uniform radial grid (finite-element Laplacian on hat functions),
explicit B_perp solve, Adams–Bashforth plasma push, several beam pushers. Alexander plans to
publish it.

Code map: README §10. Key files: `src/Simulation.cpp` (sweep, MPI hand-off, output),
`src/PlasmaSpecies.cpp` (deposit, push, ionization kernel), `src/Beam.cpp`,
`src/FieldSolver.cpp`, `src/Laser.cpp` (envelope solver), `src/Ionization.*` (ADK, Bethe,
element table, counter-based RNG).

## Build and test (this container)
```bash
cmake -B build -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ROOT=/home/claude/kokkos-install -DopenPMD_ROOT=/home/claude/openpmd-install
cmake --build build -j2 && (cd build && ctest)
export OMP_PROC_BIND=false            # needed here, otherwise OpenMP runs badly
cd validation && ./run_validation.sh -q -j 2 -c       # quick regression, ~2 min: after EVERY code change
cd validation && ./run_validation.sh -j 2 -c          # full suite (sections 1-14), ~7-10 min: milestones / new physics
```
CUDA compile check (no GPU here):
`PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/bin:$PATH CUDA_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/cu13 make -C build-cuda -j2 quarz`

MPI tests: `mpirun --allow-run-as-root --oversubscribe -np P -x OMP_NUM_THREADS=1 ...`,
compare with `validation/cmp_runs.py serial_dir mpi_dir 1e-6`. Serial reference needs
`beams.xi_shape=ngp` for bit-identity.

Package for delivery (includes .git, so the history survives the container):
`cd /home/claude && zip -qr quarz.zip quarz -x 'quarz/build/*' 'quarz/build-*/*' 'quarz/gpu-deps/*' '*/__pycache__/*' 'quarz/validation/out_*' 'quarz/validation/q_*' 'quarz/validation/d_*'`

## Invariants (do not break)
- MPI results bit-identical to serial (with ngp beam shape) unless beam particles move
  between ranks. Particle creation uses prefix sums for indices; random numbers come from
  a counter-based hash of (seed, step, slice, species, particle, draw), never from
  thread-dependent generators.
- E_z is the exact discrete xi-derivative of psi; Gauss's law holds in the laser region.
- Every new feature gets a validation section in `run_validation.sh` against an independent
  (analytic or Python) reference, a quick variant (`$(q FULL QUICK)`), entries in both
  reference files (`-u` after checking the numbers), and a row in README §7.

## Validation policy (keep it fast)
- After a code change: quick suite with `-c` (2 min). Only the sections the change can affect
  if it is clearly local (laser -> 11, ionization -> 12, MPI/output -> 9 10, m = 1 -> 6 7,
  grid/solver -> 1 2 5, plasma push / sub-slicing -> 13, smoothing -> 14). Full suite with `-c` at milestones and before delivering physics results.
- Run long suites in the background (`nohup ... &`) and poll the process, not `pgrep -f`
  with a pattern that matches the polling shell itself.
- CUDA compile check (~10 min) only when device code changed (kernels, Types.hpp, views).
- Never edit `run_validation.sh` while it is running (bash reads scripts incrementally).

## Status (2026-10-07)
Done and validated (README §7, `validation/reference_full.txt`):
- core solver, non-uniform grid, m = 1 mode, mobile ions, parsed profiles, MPI, openPMD,
  CUDA compiles and runs on a Blackwell GB202 GPU (Thomas, 2026-10-06: ~50x vs one thread of a
  Xeon E5620; no systematic benchmark and no GPU-vs-CPU file comparison yet);
- laser envelope solver (Crank–Nicolson in t, trapezoid in xi), ponderomotive force on
  plasma and beams; `pusher.max_qsa_factor` (gamma/Delta > 35 removed as trapped);
- ionization: ADK by plasma/beam fields, period-averaged ADK by the laser (numerical
  quadrature, drift momentum from sampled birth phase), Bethe beam impact ionization 0 -> 1,
  `nsplit` quanta for small fractions; examples `awake_impact_ionization.in`,
  `lwfa_ionization.in`. Immobile ionizable ions do not contribute to chi.
- adaptive sub-slicing of the plasma push (`pusher.max_cells_per_step`, default off): sub-slices
  with full deposit + field solve, beam sources extrapolated with their xi-derivative (guard row
  now carries all 15 beam components), variable-step Adams-Bashforth (history xi positions and
  push counter travel with the particles between ranks; step growth <= 2x), MPI bit-identical.
  Section 13, unit test test_ab. `output.field_files = 0` writes only axis files.
  Finding (2026-10-06): the per-particle sub-stepping first proposed did NOT help (frozen fields
  in xi); the dxi error near the axis is (a) the closure spike = near-singular caustic, not
  convergent at any dxi, and (b) the wake behind it: 0.3 % at dxi 0.005, 2 % at 0.01 -> 0.2 %
  with sub-slicing. Witness region unaffected (1e-5).
- regularization of the bubble-back singularity: `plasma.smooth_length` = a, radial filter
  (1 - a^2 Lap)^-1 on all plasma sources + background (FieldSolver::filter, per angular number,
  n = 0 zero-flux wall, n >= 1 axis/wall rows pass-through: the wall value of S enters the B_theta
  flux condition — forcing it to 0 was a bug). Spike converged (a = 0.005: peak Ez -8.8 on all
  grids, dxi <= a); changes vs cold O(a): 0.4-0.8 % behind the bubble, 2e-4 at the witness.
  Temperature alone (uth) is not a reliable regularizer (seed dependent). Section 14, test_filter.
  Agreed with Alexander 2026-10-07 (the spike is unphysical: cold plasma, perfect cylinder).
- `paper/tex/` (LaTeX source, figures, PDF of the PRAB draft), `paper/scripts/` (figure and
  error scripts, path-independent).
- `paper/inputs/`: self-contained decks of all paper cases (generated by make_decks.py,
  verified to reproduce the convergence errors exactly), `run_paper_cases.sh`, README with
  CPU timings; given to Thomas Wilson for GPU performance tests.

## Open issues
- Build pitfall (solved 2026-10-05): on Alexander's cluster CMake took /usr/bin/c++ = GCC 8.5 while
  the gcc/15.3 module's libstdc++ was loaded at run time -> crash in std::filesystem::path.
  CMake now refuses GCC < 9; use CXX=$(which g++). Branch wip/posix-io (POSIX writes,
  output.io switch) was a wrong lead; delete it on GitHub (session cannot delete branches).
- Wake-T 0.9.1 disagrees with QUARZ by up to 11 % (a0 = 2) and 27 % (a0 = 4) for a single
  laser wake; QUARZ satisfies Gauss's law, Wake-T does not in the laser region. Needs a
  full-PIC comparison (e.g. FBPIC) before trusting a0 >~ 2.
- Bethe M^2, C tabulated only for Ar; Rb (AWAKE) needs literature values.
- Laser: m = 0 envelope only, no d^2/dt^2, no phase correction for strong red-shift,
  no ionization energy loss.
- No collisional ionization by plasma electrons, no recombination; impact only level 0 -> 1.
- GPU: only Thomas' first speed number (~50x vs one old CPU thread). Needed for the paper: deck,
  time per sweep GPU vs all cores of a current CPU, cmp_runs.py GPU vs CPU (TODO in the .tex).

- Paper: Alexander submits v1 to arXiv now (priority for the non-uniform radial grid); referee
  round used for items 1-3 below. The manuscript cites https://github.com/AlexMPukhov/QUARZ
  (public). Everything pushed is visible to everyone: no unpublished results in the repo.

## Next steps (agreed 2026-10-06, before the revised manuscript)
1. Cross-code benchmarks: beam-driven cases vs LCODE 2D and QPAD/HiPACE++; laser vs FBPIC.
2. DONE (see Status): adaptive sub-slicing + error analysis. Candidate for the revised paper:
   a short paragraph/figure on the closure spike vs the converged wake behind it, and the
   smoothing regularization (plasma.smooth_length) with its O(a) cost.
3. Long-term stability/noise behind long drivers (AWAKE-length).
Later: m >= 2 modes; trapped electrons -> beam particles; phase-corrected / m = 1 laser
envelope; checkpoint/restart, beams from openPMD files, warm plasma, Python/PICMI interface;
GPU benchmark vs a full current CPU node, fused slice kernels; Bethe data beyond Ar.

## Earlier proposals
- Convert trapped plasma electrons into beam particles (charge w * dt per step) so that
  ionization injection can be followed through acceleration.
- Benedetti phase-corrected envelope; m = 1 laser envelope.
