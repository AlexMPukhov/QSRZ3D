# QUARZ — Quasistatic Arbitrary-resolution RZ code

*Quasi-static PIC code in (r, θ, ξ) with azimuthal modes m = 0, 1 and a flexible non-uniform radial grid.*
(Formerly QSRZ; renamed in October 2026. Input files and output formats are unchanged.)

QUARZ is a quasi-static particle-in-cell code for plasma wakefield
acceleration in the spirit of LCODE-2D, written in C++17 on top of
[Kokkos](https://github.com/kokkos/kokkos). One source compiles for CPUs
(OpenMP or Serial) and for GPUs (CUDA, HIP, SYCL). With MPI, the ξ box is
split between ranks (CPU cores or GPUs) and the time steps are pipelined,
as in QuickPIC and HiPACE++.

Its main addition over a uniform-grid code is the **radial grid**: any
monotone set of nodes can be used, from a region-based generator with a
growth-ratio limit or from a file. You can put 5·10⁻⁴ c/ω_p cells on the axis
to resolve a strongly pinched, low-emittance witness while keeping 0.05 c/ω_p
cells near the wall. On the pinched-witness test below, 512 stretched cells
give more accurate near-axis fields than 1600 uniform ones, at a third of the
cost.

The fields are decomposed in **azimuthal Fourier modes**. With `modes = 0`
the code is purely axisymmetric, as in LCODE-2D. With `modes = 1` the m = 1
mode is added, the same approach QPAD uses. That covers offset, tilted and
hosing beams, beam–plasma misalignment, and centroid dynamics, at the cost of
a truncated azimuthal expansion.

```
driver / witness beams (3D macro-particles, Vay/HC/IMP pushers, large Δt)
        │  deposit ρ_b, J_b on the (ξ, r) grid, modes m = 0 (,1)
        ▼
for each time step:  sweep plasma slice by slice   ξ = ξ_min → ξ_max
        deposit plasma sources ─► ψ, E_z (Poisson) ─► B_z ─► S ─► B⊥ (Helmholtz) ─► push plasma (Adams–Bashforth in ξ)
        store E, B (ξ, r, mode)
        ▼
push beams with the stored fields, t += Δt
```

---------------------------------------------------------------------------

## 1. Physics model

**Units.** Densities are in units of n₀, time in 1/ω_p, lengths in c/ω_p,
momenta in m_s c (m_s is the mass of each species), and fields in
m_e c ω_p / e. The co-moving variable is **ξ = t − z**: the head of a beam is
at small ξ, and the plasma sweeps from small to large ξ.

**Azimuthal modes.** Every real scalar is written as
f(r,θ) = f₀(r) + 2 Re[f₁(r) e^{iθ}] = f₀ + f_c cos θ + f_s sin θ. Transverse
vectors (J⊥, B⊥, E⊥) go through the complex combination V₊ = V_x + iV_y =
v₀(r) + v₁(r) e^{iθ} + v₂(r) e^{2iθ}. Here v₁ = V_r0 + iV_θ0 is the
axisymmetric part, while v₀ and v₂ carry m = 1. Each coefficient satisfies a
scalar radial equation with L_k f = (1/r)(r f′)′ − k² f/r². The truncation
at m ≤ 1 is the only approximation beyond the quasi-static one.

**Fields.** With ψ = φ − A_z, W⊥ = E⊥ + ẑ × B⊥ = −∇⊥ψ (the transverse force on
a particle with v_z = c):

| equation | modes | boundary conditions |
|---|---|---|
| ∇⊥²ψ = −(ρ − J_z) | ψ₀: L₀, ψ₁: L₁ | regular / ψ₁(0)=0; ψ(R) = 0 |
| ∇⊥²E_z = −∂_ξ(ρ − J_z) | L₀, L₁ | E_z(R) = 0 |
| ∇⊥²B_z = −(∇⊥ × J⊥)_z | L₀, L₁ | B_z(R) = 0 |
| (∇⊥² − χ) B₊ = i[(∂_x + i∂_y) J_z + S₊] | b_k: L_k, k = 0, 1, 2 | b₁(0)=b₂(0)=0; b₀, b₂, B_r0 = 0 at R; (1/r)∂_r(rB_θ0)\|_R = J_z0(R) |
| E₊ = W₊ − iB₊ | | |

The awkward term ∂_ξ J⊥ in the B⊥ equation is computed **explicitly from the
particle equations of motion**, following Baxevanis & Stupakov (PRAB 21,
071301, 2018), the scheme also used by HiPACE++'s explicit solver. Its part
proportional to B⊥ is −iχB₊, with χ = Σ q²w/(m(γ − p_z)) ≥ 0. The remainder,
S₊, is deposited from the particles.

- **m = 0:** the axisymmetric part χ₀ goes into the operator. The resulting
  Helmholtz problem is negative definite, so no iterations are needed.
- **m = 1:** the products χ₁b_{k±1} couple the modes. They are moved to the
  right-hand side and handled with a few Picard iterations (`solver.picard`,
  default 3; converged to < 0.3 % in the tests).
- **With `modes = 0`,** everything reduces to the familiar axisymmetric set
  (ψ, E_z, E_r = B_θ − ∂_rψ, B_θ with the Helmholtz term). It reproduces the
  previous ring-based version of the code to 10⁻¹⁰–10⁻¹³.

**Plasma macro-particles** are Cartesian in the transverse plane:
(x, y, p_x, p_y, Δ = γ − p_z, w).
- **modes = 0:** each particle represents a ring, and the fields are
  axisymmetric.
- **modes = 1:** every ring is split into `ntheta` particles (default 8) at
  equidistant azimuths.

The equations of motion, with q̂ = q/m:

```
dx/dξ   = p_x / Δ                      dy/dξ   = p_y / Δ
dp_x/dξ = q̂ [ γ W_x/Δ + B_y + p_y B_z/Δ ]
dp_y/dξ = q̂ [ γ W_y/Δ − B_x − p_x B_z/Δ ]
dΔ/dξ   = q̂ [ (p⊥·W⊥)/Δ − E_z ]
```

These are integrated with Adams–Bashforth of order 1–5 (default 3). Each
particle deposits ρ − J_z = q w, J = q w p/Δ and χ, with azimuthal weights
e^{−ikθ}.

Two derivatives are deposited as the **exact ξ-derivatives of the discrete
deposits**; the radial and azimuthal motion of the particle both enter:
- ∂_ξ(ρ − J_z);
- ∂_ξ J₊.

This makes E_z the exact discrete ξ-derivative of ψ in every mode. Particles
cross the axis freely, since they are Cartesian, and are reflected
specularly at the wall. Particles whose Δ drops below `pusher.delta_min`, or
whose quasi-static weight γ/Δ = 1/(1 − v_z) exceeds `pusher.max_qsa_factor`
(default 35, as in HiPACE++), are trapped (v_z → c). Such particles cannot be
treated quasi-statically and would make the density and currents singular at
the closure of a strong bubble, so they are removed and counted.

**Species.** Any number of species can be defined (electrons, mobile ions of
any charge and mass). Immobile species are added as a static background. If
no positive species is defined, a neutralising immobile background is built
from the *same* loaded electron rings, so the initial state is exactly
neutral.

**Beams** are 3D macro-particles (x, y, p_x, p_y, p_z, ξ, w) pushed in lab
time. The fields at a particle are the mode sums at its azimuth.
- **Quiet start:** `nsym` places every sampled particle at `nsym` azimuths
  around the centroid, which removes the m = 1 sampling noise of a symmetric
  beam exactly.
- **Offsets and tilts:** a beam can be given a transverse offset (`x0`,
  `y0`), a centroid tilt (`x_slope`, `y_slope`) and an angle (`xp0`, `yp0`).
- **Analytic rigid beams** (`analytic = 1`) are deposited noise-free directly
  on the grid. For a displaced Gaussian the m = 0 and m = 1 parts are exact
  (modified Bessel functions I₀, I₁).

The momentum pusher is selected with `pusher.beam` (see `src/Pushers.hpp`):

| `pusher.beam` | scheme | force-free E + v×B = 0 kept |
|---|---|---|
| `vay` (default) | Vay (2008): trapezoidal velocity average (v⁺ + v⁻)/2 | yes |
| `hc` | Higuera & Cary (2017) | yes |
| `imp` | A. Pukhov's note, eqs. (19)–(36): implicit midpoint in u, solved explicitly | yes |
| `imp_rr` | Pukhov's note, eqs. (2)–(18): trapezoidal velocity average plus implicit radiation-reaction damping −νp | yes |
| `boris` | Boris (1970), for comparison only | **no** |

`tests/test_pushers.cpp` verifies two identities to round-off: `imp` ≡ `hc`,
and `imp_rr` with ν = 0 ≡ `vay`.

Boris must not be used for these beams. An ultra-relativistic bunch sits in
its own huge E_r ≈ B_θ. For the matched pinched witness at Δt = 20, Boris
gives 8 % spurious emittance growth and 40 % error in r_rms. All the other
schemes agree with each other to ≤ 10⁻⁷.

**Radiation reaction** (`imp_rr` only) uses the leading Landau–Lifshitz term,
ν = (2/3) r_e k_p (Z²/M) (Z/M)² γ [(E + v×B)² − (v·E)²]. It is evaluated
from the momentum before the push and applied implicitly as in the note. To
enable it, set `pusher.rr_n0_cm3` to the physical plasma density.

Leapfrog start-up uses a half kick. The quantity 1 − v_z is evaluated as
(1+p⊥²)/(γ(γ+p_z)) to avoid cancellation.

**Laser driver (envelope model).** The laser vector potential is
a = Re[â(r, ξ, t) e^{−i k₀ ξ}] with k₀ = ω₀/ω_p. Dropping only the second
time derivative in the frame moving with c, the envelope obeys

```
[ ∇⊥² + 2 ∂_t (i k₀ − ∂_ξ) ] â = χ â ,      χ = Σ_s q_s² n_s / (m_s γ_s)
```

χ is the same quantity Σ q²w/(mΔ) per volume that the B⊥ solve uses. The
mixed term ∂_t ∂_ξ â keeps the group velocity, the pulse slippage and the
frequency shift (depletion).

- **Discretization** (`src/Laser.*`, m = 0 envelope on the non-uniform grid):
  - for the increment X = â^{n+1} − â^n over one beam step Δt, Crank–Nicolson
    in t gives dX/dξ = i k₀ X + (Δt/4)(∇⊥² − χ)(X + 2â^n);
  - this is integrated in ξ with the trapezoidal rule, slice by slice, inside
    the plasma sweep from the head of the box, where the laser must vanish;
  - ∇⊥² is the m = 0 operator L₀ of the field solver, so each slice is one
    complex tridiagonal solve.

  Both directions are centred, so the scheme is second order. In vacuum every
  Fourier mode keeps its amplitude, and the march in ξ damps rather than
  amplifies. One-sided ξ differences (backward Euler, BDF2) are unconditionally
  unstable with this mixed term; they were tried and rejected.
- **Ponderomotive force on plasma particles:** with ⟨a²⟩ = |â|²/2 (linear
  polarization; |â|² for circular):
  - γ = (1 + p⊥² + q̂²⟨a²⟩ + Δ²)/(2Δ);
  - dp⊥/dξ gets −q̂² ∇⊥⟨a²⟩/(2Δ), and the same term enters the explicit B⊥ source S₊;
  - Δ = γ − p_z is unchanged, because (∂_t + ∂_z)⟨a²⟩ = 0 for a function of ξ.

  Ions (q̂ = Z/M) feel it automatically, reduced by (Z/M)².
- **Ponderomotive force on beam particles:** the kick
  du/dt = −q̂² ∇⟨a²⟩/(2γ̄), with γ̄² = 1 + u² + q̂²⟨a²⟩ and ∂_z = −∂_ξ, is
  applied as two half kicks around the Lorentz push.
- **MPI:** a rank passes â^n of its last two slices and the ξ-integration
  carry (X, G) of its last slice to the next rank (8 M numbers per step).
- **Limitations:**
  - only the m = 0 envelope; with `modes = 1` the laser stays axisymmetric;
  - no second time derivative, so no backward-propagating light;
  - the ξ discretization assumes the local wavenumber stays close to k₀,
    |k − k₀| Δξ ≪ 1. Strongly depleted, red-shifted pulses need a finer Δξ.
    Benedetti's phase-corrected scheme would remove this restriction.

**Ionization** (`src/Ionization.*`; species with `<name>.element` or
`<name>.ionization_energies_eV`). An ionizable species is a gas of atoms or ions:
each macro-particle carries its own charge state z (charge z e, mass M), and neutral
atoms (z = 0) neither deposit nor feel forces. Ionization is done slice by slice in
the sweep, after the fields of the slice are known and before the push. Each
ionization event creates a macro-electron of the same weight at the ion's position, in
the product species (`<name>.ionization.product`, default `electrons`; `ppc = 0` makes
an initially empty species). Three channels:

- **Field ionization by the plasma and beam fields:** ADK rate for the static field
  |E| = (E_x² + E_y² + E_z²)^{1/2}, with l = m = 0 and l* = n* − 1 as in
  WarpX/HiPACE++ (Chen et al., JCP 236, 220 (2013)):
  W = Ip C² [2(2Ip)^{3/2}/E]^{2n*−1} exp[−2(2Ip)^{3/2}/(3E)] (atomic units),
  C² = 2^{2n*}/(n* Γ(2n*)), n* = Z/(2Ip)^{1/2}.
- **Field ionization by the laser (envelope):**
  - the ADK rate of the total field is averaged over the laser period:
    E(φ) = E_slow + E_L cos φ e_x (linear) or E_slow + E_L(cos φ, sin φ) (circular),
    with E_L = k₀|â|;
  - the average is a quadrature around each field maximum: Gauss–Hermite scaled with
    the Laplace width for narrow peaks, otherwise Gauss–Legendre over the lobe;
  - the usual closed form W(E_L)[3E_L/(π(2Ip)^{3/2})]^{1/2} is the Laplace limit of the
    same average. It is 5–40 % too high in the field range where the ionization
    actually happens, so it is not used;
  - an electron born in the laser gets the residual drift p⊥ = −a(φ_birth). The birth
    phase is sampled exactly from W(E(φ)) by rejection; Massimo et al., PRE 102, 033204
    (2020) use its Gaussian approximation. Its Δ = γ − p_z equals the parent's (Δ = 1
    for an atom at rest), so the cycle-averaged p_z follows from the envelope γ.
- **Impact ionization by the beams** (neutral → 1+):
  - Bethe cross-section in the form of Rieke & Prepejchal, PRA 6, 1507 (1972):
    σ = 1.874·10⁻²⁰ cm² z_b² β⁻² [M²(ln β²γ² − β²) + C],
    or a constant `<name>.impact.sigma_cm2`;
  - an atom at rest sees dP = σ n_b β dξ per slice, where n_b is the beam density in the box;
  - the beams deposit three target-independent sums Σz²βn, Σz²Ln/β and Σz²n/β,
    so every beam particle has its own γ;
  - M² = 4.22, C = 37.93 are built in for Ar. They reproduce the measured primary
    ionization of minimum-ionizing particles in Ar (about 25 ion pairs/cm at STP).
    For other gases give `impact.M2` and `impact.C`, or `impact.sigma_cm2`.

**Monte Carlo.**
- Per slice, the transition z → z+1 happens with probability P = 1 − exp(−W dt − R dξ),
  where dt = dξ γ/Δ is the time the ion spends in the slice (dξ at rest).
- Several levels may be crossed in one slice. The ionization step at slice k represents
  [ξ_k − dξ/2, ξ_k + dξ/2].
- **Splitting** (`ionization.nsplit` = n > 1, first ionization of each loaded atom only):
  the expected ionized weight dN = wP of a slice is emitted as a new ion + electron pair.
  The pair has weight dN if dN ≥ w₀/n; otherwise it has weight w₀/n, with probability
  dN n/w₀ (unbiased). This resolves small ionization fractions without the
  all-or-nothing noise of whole rings; the AWAKE case has fractions of ~10⁻⁴.
- Random numbers come from a counter-based hash of (seed, step, slice, species,
  particle, draw). New particles are appended at offsets from a prefix sum. Results
  therefore do not depend on the thread count or the MPI decomposition.
- Particles born in a slice, and ions whose charge changed, restart their Adams–Bashforth
  history.

## 2. Numerics on the non-uniform grid

- **Nodes** 0 = r₀ < r₁ < … < r_N = R. All quantities live on the nodes.
- **Deposition and gather** use linear hat functions W_j(r) on the
  non-uniform cells.
- **Node volumes** are the exact hat-function volumes,
  V_j = 2π∫W_j r dr. This generalises Verboncoeur's cylindrical correction to
  non-uniform cells.
- **Plasma loading.** Every cell is split into `ppc` sub-rings, and each ring
  sits at the **volume centroid** of its sub-ring. With this loading a uniform
  plasma deposits an exactly uniform density on every node, including r = 0
  (unit test: error 5·10⁻¹³). Mid-point loading gives errors of order 10 %
  at the axis.
- **Laplacian.** Lumped-mass linear finite elements, which is the same as
  the conservative flux form, with the same V_j. The discrete Gauss law holds
  exactly.
- **Operators L₁, L₂** (for m = 1 and for B⊥). Conservative finite volumes
  on L_n f = r^{n−1} ∂_r[r^{1−2n} ∂_r(rⁿ f)]. For n = 1 this is the usual
  ∂_r[(1/r)∂_r(rB)].
- **Gradients.** 3-point, second order on non-uniform grids.
- **Tridiagonal solves.** Thomas on CPU. On GPU, parallel cyclic reduction
  runs inside one team (one thread block). Both are selectable with
  `solver.tridiag`.
- **Convergence.** Second order on uniform and smoothly stretched grids.
  Measured orders are 1.9–2.0 for L₀, L₁ and L₂; see `tests/test_solvers.cpp`.
- **Grid ratio.** Keep neighbouring cell ratios at or below about 1.1
  (`grid.max_ratio`). Abrupt jumps reduce accuracy locally to first order, and
  the code warns if the ratio exceeds 1.25.

## 3. Parallelization: GPU and MPI

### GPU (Kokkos)

- All particle loops (deposit, gather/push, beam push and deposit) are Kokkos
  `parallel_for` loops using atomics. They carry no host code, virtual calls
  or `std::` containers.
- **Data layout.** Particles are stored as structure-of-arrays. The fields
  for the whole box are stored as (ξ, r) arrays.
- **Host–device traffic.** Data moves between host and device only for
  loading the fresh plasma slice (once per time step), for diagnostics and
  for output.
- **Kernel count.** With `modes = 0`, one slice takes about 12 kernels plus
  5 tridiagonal solves. With `modes = 1`, it takes about 20 kernels plus
  ~25 solves (3 Picard iterations). In
  2D the per-slice work is small (a few 10⁴ rings), so on a GPU the sweep is
  latency-bound. The big gains come from the beams (10⁶–10⁷ particles) and
  from many radial cells.

**GPU status:** the code (including the m = 1 and the MPI versions) compiles and links for CUDA 13.4 /
sm_80 (A100) with Kokkos 5.2.2 (MPI: OpenMPI 4). It **runs on an NVIDIA Blackwell GPU (GB202)**
(T. C. Wilson, October 2026): about 50 times faster than one thread of an Intel Xeon E5620.
A systematic GPU benchmark (all cores of a current CPU as the reference) and a file-by-file
comparison of GPU and CPU results are still to be done. All physics tests were run on the OpenMP
backend. The PCR solver (the GPU default) was tested on CPU and agrees with
Thomas to round-off. The first thing to do on a GPU machine is to run
`ctest` and `validation/run_validation.sh`.

**Possible next steps for GPU speed:**

- a persistent single-kernel slice sweep, with team-local scratch arrays for
  the sources and the PCR;
- batching independent runs (parameter scans) into one launch.

### MPI: decomposition along ξ with pipelined time steps

```
            ξ_min ─────────────────────────────────────────────► ξ_max
            │  rank 0   │  rank 1   │  rank 2   │  rank 3   │
 wall time  │ step n+3  │ step n+2  │ step n+1  │ step n    │
```

Each rank owns a contiguous block of ξ slices (rank 0 at the head of the box)
and keeps the beam particles of its slices. The plasma enters at the head, so
rank r can only sweep its slices for step n after rank r−1 has done step n;
but at that moment rank r−1 can already start step n+1. After a fill phase
of P−1 sweeps all ranks work at the same time, and the speed-up approaches P
for runs with many time steps (it is (N_steps+1)/(N_steps+P) × P).

**What is sent** (once per time step, from rank r to r+1, one message):

- the plasma particles leaving the last local slice: positions, momenta,
  weights, and the Adams–Bashforth history (`ab_order` previous derivatives);
- the B⊥ of the last slice (warm start of the B⊥ solve);
- ρ_b − J_z,b of the last slice (for ∂ξ of the beam sources in the next rank);
- the beam particles that slipped into the next rank's slices during the push
  (beam particles only move backwards in ξ, dξ/dt = 1 − v_z ≥ 0).

**Beam shape in ξ.** For the pipeline, the beam deposit and field gather of a
slice must not need slices of other ranks at other time levels. The beam
particles therefore use the nearest-slice (NGP) shape in ξ
(`beams.xi_shape = ngp`, the default when running with more than one rank).
The transverse (radial) shape is unchanged. A serial run with
`beams.xi_shape = ngp` does exactly the same arithmetic, so serial and MPI
runs can be compared directly (see Validation). The default of serial runs
stays `linear`, which gives somewhat smoother beam sources for coarse Δξ.

**GPUs.** One GPU per rank. A rank uses GPU number (rank within the node) mod
(GPUs per node), unless `--kokkos-device-id`, `KOKKOS_DEVICE_ID` or a per-rank
`CUDA_VISIBLE_DEVICES` says otherwise. Messages go through host buffers, so
any MPI library works (no CUDA-aware MPI needed). The message is small
compared with the sweep work (one plasma slice, ~10⁴ particles).

**Running.**

```bash
mpirun -np 4 ./quarz run.in                       # 4 CPU ranks (set OMP_NUM_THREADS per rank)
mpirun -np 4 -x OMP_NUM_THREADS=8 ./quarz run.in  # hybrid MPI + OpenMP
mpirun -np 8 ./quarz run.in                       # 2 nodes x 4 GPUs (CUDA build)
srun -n 8 --gpus-per-task=1 ./quarz run.in        # SLURM: each rank sees one GPU
```

Each rank needs at least 2 slices; in practice use blocks of ≥ 50 slices, so
that a rank's sweep is long compared with the message latency. The load is
balanced when the plasma work per slice is similar along the box, which is the
usual case (the plasma dominates). Beams concentrated in a few ranks make those
ranks slower; the pipeline then runs at the speed of the slowest rank.

## 4. Building

Requirements: CMake ≥ 3.18, a C++17 compiler (GCC ≥ 9, Clang ≥ 9), and Kokkos ≥ 4.0.
CMake uses the compiler `c++` found in the PATH; on clusters this is often the old system
compiler even with a newer gcc module loaded. Then set it explicitly in a fresh build
directory: `CXX=$(which g++) CC=$(which gcc) cmake -B build ...` (CMake stops with an error for
GCC < 9, whose `std::filesystem` is incompatible with newer runtime libraries). If Kokkos is
not found, CMake downloads Kokkos 4.4.01 and builds it with the options you
pass.

MPI is optional: if CMake finds an MPI library, the MPI version is built
(`-- QUARZ: MPI enabled` in the CMake output); otherwise a serial version is
built with a warning. `-DQUARZ_ENABLE_MPI=OFF` switches MPI off. Any MPI-3
library works (OpenMPI, MPICH, Intel MPI, Cray MPICH). The MPI executable also
runs without `mpirun` as a single rank.

openPMD output is optional too: if CMake finds
[openPMD-api](https://github.com/openPMD/openPMD-api) ≥ 0.15
(`-DopenPMD_ROOT=/path/to/install`), the openPMD writer is built
(`-- QUARZ: openPMD output enabled`). For MPI runs openPMD-api must be built with
MPI (and with parallel HDF5 or ADIOS2). `-DQUARZ_ENABLE_OPENPMD=OFF` switches it off.

```bash
# CPU (OpenMP)
cmake -B build -DKokkos_ENABLE_OPENMP=ON
cmake --build build -j
cd build && ctest            # unit tests

# NVIDIA GPU: simplest is the script below (no root rights needed, see "GPU build" below)
tools/build_gpu.sh

# NVIDIA GPU by hand, e.g. A100 (use Kokkos >= 5 for CUDA 13; Kokkos 4.x is fine for CUDA 11/12)
cmake -B build-cuda -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
      -DCMAKE_CXX_COMPILER=$PWD/build-cuda/_deps/kokkos-src/bin/nvcc_wrapper
#   or with an installed Kokkos:  -DKokkos_ROOT=/path/to/kokkos -DCMAKE_CXX_COMPILER=<kokkos>/bin/nvcc_wrapper

# AMD GPU
cmake -B build-hip -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON -DCMAKE_CXX_COMPILER=hipcc
```

**GPU build (`tools/build_gpu.sh`).** Kokkos' `nvcc_wrapper` is not a
separate compiler: it is a small script shipped inside the Kokkos sources
(`kokkos/bin/nvcc_wrapper`) that calls NVIDIA's normal `nvcc`. The one thing
you need is a CUDA compiler. The script handles that:

1. **Finding nvcc.** It uses an existing `nvcc` (from the PATH, `$CUDA_HOME`
   or `/usr/local/cuda`, e.g. after `module load cuda`).
2. **Installing nvcc if missing.** It installs NVIDIA's official CUDA 13
   compiler wheels from PyPI (`nvidia-cuda-nvcc`, `-runtime`, `-cccl`,
   `-crt`, ~0.5 GB) into `./gpu-deps`, in user space. These need an NVIDIA
   driver ≥ 580; for older drivers the script prints how to get a CUDA 12
   toolkit (module, conda, or NVIDIA runfile with `--toolkitpath`).
3. **Architecture.** It detects the GPU architecture with `nvidia-smi`
   (override with `QUARZ_ARCH=HOPPER90`, etc.).
4. **Building.** It builds Kokkos (5.2.2 for CUDA 13, 4.4.01 for CUDA 12)
   and QUARZ into `build-gpu/`.
5. **Checking.** It runs `ctest` and a short blowout run on the GPU.

Tested here from a clean copy in compile-only mode
(`QUARZ_COMPILE_ONLY=1 QUARZ_ARCH=AMPERE80`, no GPU in the test machine):
pip toolkit → Kokkos 5.2.2 → QUARZ with sm_80 device code, about 12 minutes on
2 cores. Running on an actual GPU has not been tested.

Run with `./quarz input.in [key=value ...]`. Any input key can be overridden
on the command line, for example `./quarz run.in grid.dr=0.005 time.steps=10`.
Threads are set with `OMP_NUM_THREADS`, and Kokkos options (e.g.
`--kokkos-device-id=1`) are passed through.

**Troubleshooting builds.**

- *Segmentation fault right after the start-up summary* (after the `Beam ...` lines), with a
  backtrace in `std::filesystem::path::~path` and headers from `/usr/include/c++/8`: the code
  was compiled by an old system GCC 8 but runs with the `libstdc++` of a newer GCC (e.g. from a
  loaded module, visible with `ldd quarz | grep stdc++`). GCC 8's `std::filesystem` is not
  binary compatible with GCC >= 9. CMake now refuses GCC < 9; build in a fresh directory with
  `CXX=$(which g++) CC=$(which gcc) cmake -B build ...` and check that CMake reports the
  intended compiler (`The CXX compiler identification is GNU 15...`).
- CMake caches the compiler of a build directory: after loading a different compiler module,
  always configure a new directory instead of re-running CMake in the old one.
- The MPI version is built whenever CMake finds MPI, also when you run without `mpirun`; use
  `-DQUARZ_ENABLE_MPI=OFF` for a purely serial (OpenMP-only) build.

## 5. Input reference

Input files are plain `key = value` lines; `#` starts a comment. Keys that are
never read are reported at the end of the run, which catches typos.

**Radial grid**

| key | default | meaning |
|---|---|---|
| `grid.type` | `uniform` | `uniform` \| `regions` \| `file` |
| `grid.rmax` | — | outer radius R (conducting wall) |
| `grid.dr` | — | spacing for `uniform` |
| `grid.regions` | — | `r_end:h` pairs, e.g. `0.05:0.0005 2.5:0.01 8:0.05`. Spacing h is used for r < r_end. Transitions are geometric. A coarse→fine transition starts early enough to reach the fine spacing at the boundary. |
| `grid.max_ratio` | 1.05 | maximum ratio of neighbouring cells |
| `grid.file` | — | text file with one node radius per line (first 0, last R) |

The generated grid is written to `out/grid.txt` (j, r_j, h_j, V_j).

**Box, time, pusher, solver**

| key | default | meaning |
|---|---|---|
| `modes` | 0 | 0 = axisymmetric (m = 0), 1 = modes m = 0 and 1 |
| `solver.picard` | 3 (modes=1) | Picard iterations for the χ₁ mode coupling of B⊥ |
| `xi.min`, `xi.max`, `xi.step` | 0, —, — | box and slice spacing Δξ |
| `time.dt`, `time.steps`, `time.start` | 0, 0, 0 | beam time step and number of steps (0 = one quasi-static solve) |
| `pusher.ab_order` | 3 | Adams–Bashforth order 1–5 for plasma rings. 2–3 is most robust at bubble closure; 5 can blow up there. |
| `pusher.delta_min` | 10⁻³ | rings with γ − p_z below this are removed (trapped) |
| `pusher.max_qsa_factor` | 35 | rings with γ/(γ − p_z) above this are removed (trapped; the quasi-static weight diverges) |
| `solver.tridiag` | `auto` | `auto` (Thomas on host, PCR on device) \| `thomas` \| `pcr` |
| `beams.xi_shape` | `linear` (1 rank), `ngp` (MPI) | longitudinal shape of beam particles for deposit and field gather: `linear` (between two slices) or `ngp` (nearest slice). MPI runs need `ngp`; a serial run with `ngp` gives the same results as an MPI run. |
| `pusher.beam` | `vay` | beam momentum pusher: `vay` \| `hc` \| `imp` \| `imp_rr` \| `boris`. Can be overridden per beam with `<beam>.pusher`. |
| `pusher.rr_n0_cm3` | 0 | plasma density in cm⁻³. With `imp_rr`, a value > 0 switches on radiation reaction. |
| `pusher.rr_scale` | 1 | multiplies the radiation-reaction strength (testing only) |

**Function parser (HiPACE++ style).** Profiles can be given as formulas.
- **Compilation.** Each expression is compiled once into a small stack
  program, which runs on the CPU and inside GPU kernels.
- **Constant folding.** Parts without variables are evaluated at compile
  time.
- **Syntax and names.** Quotes around expressions are optional. The variable
  names are taken from the key, e.g. `plasma.density(x,y,z)`.

```
my_constants.L    = 200            # user constants; may use each other, in any order
my_constants.kp   = 2*pi/L
plasma.density(x,y,z)   = "if(z < L, sin(0.5*pi*max(z,0)/L)^2, 1) * (1 + 0.2*(x^2+y^2)/4)"
driver.profile           = parsed
driver.density(x,y,xi)   = "6*exp(-(x^2+y^2)/(2*0.25^2)) * (xi > 1)*(xi < 4)"
witness.uz_mean(x,y,xi)  = "2000 - 40*(xi - 6.3)"      # energy chirp
witness.uz_std(x,y,xi)   = "2 + 10*abs(xi - 6.3)"       # slice energy spread varying along the bunch
witness.ux_std(x,y,xi)   = "0.05*(1 + (x^2+y^2)/0.05^2)" # transverse temperature growing with radius
electrons.uth(x,y,z)     = "0.001*(1 + z/1000)"         # plasma temperature
```

| feature | details |
|---|---|
| operators | `+ - * /`, `^` or `**` (right-assoc.), unary `- + !`, `< > <= >= == !=`, `&&`, `\|\|` (true = 1, false = 0) |
| functions | `sqrt exp log log10 sin cos tan asin acos atan sinh cosh tanh abs floor ceil erf heaviside(=step)`, `pow atan2 min max mod`, `if(cond, a, b)` |
| constants | `pi`, `e`, and every `my_constants.<name>` |

Errors name the position, e.g. `parser: unknown symbol 'q' … at position 2 in "q*2"`.

**Plasma**

| key | default | meaning |
|---|---|---|
| `plasma.density(x,y,z)` | — | parsed density / n₀. It replaces `plasma.radial`, `rmax`, `channel_*` and the tables. z is the lab position of the box head (the plasma entering the box). With `modes = 0` its azimuthal average is used; with `modes = 1` its value at each particle, so its m = 1 part is kept. |
| `<species>.density(x,y,z)` | — | the same, for one species only (e.g. a different ion profile) |
| `plasma.species` | `electrons` | list of species names |
| `plasma.radial` | `uniform` | `uniform` \| `channel` (n = 1 + depth·(r/r_c)²) \| `table` |
| `plasma.channel_depth`, `plasma.channel_radius` | | for `channel` |
| `plasma.radial_table` | | `r:n` pairs, piecewise linear |
| `plasma.rmax` | ∞ | plasma column radius |
| `plasma.z_table` | | `z:n` pairs; longitudinal profile at the lab position of the box front |
| `plasma.neutralize` | yes if no positive species | add an immobile neutralising background |
| `plasma.seed` | 1 | RNG seed (thermal plasma) |
| `<species>.charge` | −1 | charge in units of e |
| `<species>.mass` | 1 | mass in units of m_e |
| `<species>.mobile` | 1 | 0 = static background |
| `<species>.ppc` | 4 | rings per radial cell. Fine cells automatically get more rings per unit volume. 0 = initially empty (mobile species, e.g. to receive ionization electrons). |
| `<species>.ntheta` | 1 (modes=0), 8 (modes=1) | particles per ring, at equidistant azimuths (≥ 4 with modes = 1) |
| `<species>.uth` or `<species>.uth(x,y,z)` | 0 | isotropic thermal momentum spread (m_s c): a number or a function of position (z = lab position). With `modes = 0`, the rms over the ring is used. |
| `<species>.density` | 1 | density multiplier relative to the profile (e.g. 1/Z for ions) |

**Laser** (envelope model; switched on by `laser.a0`)

| key | default | meaning |
|---|---|---|
| `laser.a0` | — | peak normalised vector potential at focus |
| `laser.k0` | — | ω₀/ω_p; or `laser.lambda0_um` (wavelength in µm) together with `units.n0_cm3` |
| `laser.w0` | — | spot size at focus: â ∝ exp(−r²/w0²) |
| `laser.L0` | — | length: â ∝ exp(−(ξ − ξ0)²/L0²) (intensity FWHM = 1.177 L0) |
| `laser.xi0` | — | centre of the pulse in ξ (keep ≳ 3 L0 from the head of the box) |
| `laser.focus` | 0 | distance from the initial position to the focal plane (> 0: the pulse is still converging) |
| `laser.polarization` | `linear` | `linear` (⟨a²⟩ = \|â\|²/2) or `circular` (⟨a²⟩ = \|â\|²) |
| `units.n0_cm3` | — | plasma density (needed with `laser.lambda0_um`; also gives SI units in openPMD output) |

The time step `time.dt` is shared with the beams. The laser accuracy is
controlled by Δt relative to the Rayleigh length k₀w0²/2 and to the
dephasing/depletion length. In the tests below, Δt ≤ Z_R/8 was sufficient.

**Ionization** (per ionizable plasma species; needs `units.n0_cm3`)

| key | default | meaning |
|---|---|---|
| `<species>.element` | — | H, He, Li, C, N, O, Ne, Ar (all levels), Kr (8), Rb (3), Xe (3), Cs (2): ionization energies and mass |
| `<species>.ionization_energies_eV` | from the element | list of ionization energies (overrides the table) |
| `<species>.z0`, `<species>.z_max` | 0, all levels | initial and highest charge state |
| `<species>.mass` | from the element | in electron masses |
| `<species>.mobile` | 1 | 0: ions stay at their positions (still macro-particles: their charge changes) |
| `<species>.ionization.product` | `electrons` | species that receives the new electrons (mobile, negative; `ppc = 0` for an initially empty one) |
| `<species>.ionization.field` | 1 | ADK by the plasma, beam and laser fields |
| `<species>.ionization.impact` | 1 if a cross-section is known | beam impact ionization 0 → 1 |
| `<species>.impact.M2`, `<species>.impact.C` | Ar: 4.22, 37.93 | Bethe parameters |
| `<species>.impact.sigma_cm2` | — | constant impact cross-section instead |
| `<species>.ionization.nsplit` | 1 | > 1: split the first ionization into quanta of w₀/nsplit. Use it when the ionized fraction is small; choose nsplit ≳ 1/P_total. The particle count grows up to nsplit-fold where the gas is fully ionized. |

The `<species>.density` of an ionizable species is the atom density. Neutral gas does not
count as a positive species, so a pre-ionized background electron species still gets its
neutralizing background. A partly pre-ionized gas (`z0 > 0`) needs its electrons as a
separate species.

**Beams** (`beams = name1 name2 …`)

| key | default | meaning |
|---|---|---|
| `<beam>.charge`, `<beam>.mass` | −1, 1 | e.g. protons: 1, 1836.15 |
| `<beam>.profile` | `gaussian` | `gaussian` \| `flattop` \| `file` \| `parsed` |
| `<beam>.n0` | — | peak density / n₀ (of the uncut bunch) |
| `<beam>.sigma_r` | — | rms size per transverse coordinate (σ_x = σ_y) |
| `<beam>.sigma_xi` / `<beam>.length` | — | gaussian rms length / flat-top length |
| `<beam>.xi0` | — | centre position |
| `<beam>.xi_cut_head` | — | drop particles with ξ < value (seeded SMI) |
| `<beam>.gamma` | — | mean γ |
| `<beam>.espread` | 0 | rms relative energy spread |
| `<beam>.emit_n` | 0 | normalised rms emittance per plane (c/ω_p) |
| `<beam>.cut` | 3 | truncation in σ (radial and longitudinal) |
| `<beam>.nparticles` | 100000 | macro-particles |
| `<beam>.seed` | 12345 | RNG seed |
| `<beam>.rigid` | 0 | 1 = frozen beam |
| `<beam>.analytic` | 0 | 1 = noise-free rigid beam deposited analytically on the grid (gaussian/flattop, offsets and slopes allowed) |
| `<beam>.x0`, `<beam>.y0` | 0 | transverse centroid offset (modes = 1) |
| `<beam>.x_slope`, `<beam>.y_slope` | 0 | centroid tilt: x_c(ξ) = x0 + x_slope (ξ − ξ0) (modes = 1) |
| `<beam>.xp0`, `<beam>.yp0` | 0 | centroid angle, p_x = p_z·xp0 (modes = 1) |
| `<beam>.nsym` | 1 | quiet start: replicate every particle at nsym azimuths (nparticles must be a multiple) |
| `<beam>.mirror_y` | 0 | quiet start: also add the mirror image y → −y (keeps an x-only offset/tilt exactly y-symmetric) |
| `<beam>.file` | | for `profile = file`: columns x y p_x p_y p_z ξ w |

**Parsed beams** (`<beam>.profile = parsed`)

| key | default | meaning |
|---|---|---|
| `<beam>.density(x,y,xi)` | — | beam density / n₀; the third variable is the co-moving ξ |
| `<beam>.xi_range` | — | ξ interval that is sampled |
| `<beam>.r_max` | — | radius that is sampled |
| `<beam>.ppc` | `1 1` | sub-rings per radial grid cell, sub-slices per ξ cell |
| `<beam>.ntheta` | 1 (modes=0), 8 (modes=1) | particles per ring |
| `<beam>.ux_mean(x,y,xi)`, `uy_mean`, `uz_mean` | 0, 0, √(γ²−1) | mean momenta p/mc (a plain number is accepted too); `uz_mean` or `gamma` is required |
| `<beam>.ux_std(x,y,xi)`, `uy_std`, `uz_std` | 0 | Gaussian spread of p_x, p_y, p_z. Each may depend on position, e.g. `ux_std(x,y,xi) = "emit/sr * (1 + 0.5*xi)"`; a plain number is accepted. Negative values count as 0. |
| `<beam>.u_std` | `0 0 0` | shorthand: three constant spreads (overridden component-wise by `ux_std`, etc.) |
| `<beam>.analytic = 1` | | rigid and noise-free: the parsed density is deposited directly on the grid (32-point azimuthal quadrature for the m = 0, 1 parts); needs `gamma` |

The particles are placed deterministically (quiet start), so no sampling
noise is created:
- **radial:** on the radial cells of the simulation grid, which means a fine
  grid near the axis also gives fine sampling there;
- **ξ:** on sub-slices of the ξ grid;
- **azimuth:** at equidistant angles.

The weight of each particle is density × volume, and particles with zero
density are skipped.

**Output**

| key | default | meaning |
|---|---|---|
| `output.dir` | `out` | output directory |
| `output.every` | 1 | write fields every n steps (0 = never) |
| `output.beam_every` | 0 | write beam particles every n steps |
| `output.beam_slices` | 0 | number of ξ-bins for per-slice beam diagnostics (centroid, size, γ, ε) written at field outputs |
| `output.beam_slices_range` | box | `lo hi`: ξ-range of these bins (default: the whole box); the bins are fixed, so files at different steps line up |
| `output.format` | `native` | `native` (QUARZ binary/text files), `openpmd`, or `both`. With `openpmd`, the native field files and particle dumps are not written; `axis_*`, `beams.txt` and `slices_*` always are. |
| `output.openpmd_backend` | `h5` | `h5` (HDF5), `bp` (ADIOS2) or `json` |
| `output.openpmd_file` | `openpmd/data_%06T` | file name pattern inside `output.dir` (`%T` = step: one file per output) |
| `output.openpmd_grid` | `uniform` | `uniform`: fields interpolated onto a uniform radial grid (openPMD requires uniform spacing; readable by openPMD-viewer, yt, …); `native`: the code's non-uniform nodes, exact, geometry `other` |
| `output.openpmd_dr`, `output.openpmd_rmax` | h_min, R | spacing and extent of the uniform radial grid (default h_min, at most 8192 points) |
| `output.openpmd_options` | `{}` | JSON/TOML options passed to openPMD-api (compression, ADIOS2 engine, …) |
| `units.n0_cm3` | — | plasma density for SI units in the openPMD files; without it the data are in normalised units (all `unitSI` = 1) |

**Physical units.** At plasma density n₀ [cm⁻³]:

- c/ω_p = 5.31·10⁵ / √n₀ cm
- E₀ = 0.96 √n₀ V/m
- a beam weight w corresponds to w·n₀(c/ω_p)³ particles.

## 6. Output

- **`fields_NNNNNN.bin`** (format version 2). Header: int32 version = 2,
  int32 M, int32 K, int32 mode1, float64 t, r[M], ξ[K], int32 ncomp. Then
  ncomp blocks of {char name[16], float64 [K][M]}.
  - m = 0 components: `psi ez er eth br bth bz ne ni rhob`.
  - m = 1 components (when present): the same names with `_c` / `_s`, the
    cos / sin amplitudes in f = f₀ + f_c cos θ + f_s sin θ.
- **`axis_NNNNNN.txt`**: ξ, E_z(0), ψ(0), n_e(0), ρ_b(0), and the focusing
  slope (E_r − B_θ)/r at the first node. With m = 1 it adds W_x(0) and W_y(0),
  the transverse force on the axis. Fixed-width columns (11 significant digits).
- **`beams.txt`**, one line per beam per step, with columns:
  step, t, alive, N, charge, ⟨γ⟩, σ_γ, ⟨ξ⟩, σ_ξ, r_rms, ε_n,x, ⟨x⟩, ⟨y⟩, ε_n,y,
  σ_x, σ_y. The emittances are about the centroid.
- **`slices_<beam>_NNNNNN.txt`**: per-ξ-bin charge, ⟨x⟩, ⟨y⟩, σ_x, σ_y, ⟨γ⟩
  and ε_n,x, for hosing studies. Bins are `output.beam_slices` equal bins over
  `output.beam_slices_range` (empty bins are skipped).
- **Laser runs** add `laser.txt`, one line per step: step, t, max |â|,
  w_rms = (2⟨r²⟩)^{1/2}, ⟨ξ⟩, L_rms (all weighted with |â|²) and ∫|â|² dV.
  ∫|â|² dV is not conserved in a plasma: during depletion the photon number is
  conserved and the frequency drops. The field files contain the envelope as
  `a_re`, `a_im` (openPMD: mesh `laserEnvelope`, components `re`, `im`).
- **Ionization runs** add `ionization.txt`, one line per step, with four columns per ionizable species:
  - the weight of the electrons born in the sweep, in units of n₀(c/ω_p)² per c/ω_p;
  - the same as electrons per metre of plasma;
  - their mean p⊥² at birth;
  - the mean charge state of the column at the end of the box.

  The field files contain the ion charge density `nz_<species>` = Σ z n (m = 0;
  openPMD: mesh `n_z_<species>`).
- **MPI runs** write the same files. Field and axis files are written by all
  ranks directly into one file (each rank at its own byte offset). `beams.txt`,
  the slice files and the particle dumps are first written per rank
  (`*.partN`) and merged into the usual files by rank 0 at the end of the run;
  if a run is killed, the `.partN` files hold the data.
- **`beam_<name>_NNNNNN.bin`**: int32 n, then n × (x, y, p_x, p_y, p_z, ξ, w).
- **openPMD** (`output.format = openpmd` or `both`): one file per output step,
  `openpmd/data_NNNNNN.h5` (or `.bp`, `.json`), following the
  [openPMD standard 1.1](https://github.com/openPMD/openPMD-standard).
  - Field meshes `E`, `B` (components `r`, `t`, `z`), `psi`, `n_e`, `n_i`,
    `rho_beam`, in geometry `thetaMode`, axes (r, z), shape [modes][r][z].
    The modes are (f₀) or (f₀, f_c, f_s) with f = f₀ + f_c cos θ + f_s sin θ
    (geometryParameters `m=2;imag=+`), the convention of openPMD-viewer, FBPIC and WarpX.
  - **Longitudinal axis:** the lab-frame z = t − ξ, stored in increasing z.
  - **Radial grid:** openPMD requires uniform spacing, so by default the fields are
    interpolated (piecewise linearly, consistent with the hat functions) onto a
    uniform radial grid. The default spacing is h_min, which can make the files much
    larger than the native ones; the code prints the size per output and a hint.
    `output.openpmd_grid = native` writes the exact data on the non-uniform nodes
    instead (geometry `other`, node positions in the mesh attribute `quarz_r_nodes`);
    standard readers then do not know the radial positions.
  - **Particles:** every beam is a species with `position` (x, y, z = t − ξ),
    `positionOffset` (0), `momentum` (per physical particle), `weighting`
    (physical particles per macro-particle), `charge` and `mass`.
  - **Units:** SI when `units.n0_cm3` is given, otherwise normalised units with
    `unitSI` = 1 (the series attribute `quarz_units` says which).
  - **MPI:** openPMD writes are collective. In the ξ pipeline the ranks reach a
    step at different times, so each openPMD output step drains the pipeline
    (about P − 1 block sweeps). Example, 2 ranks, output every 2nd of 10 steps:
    10.1 s (native) vs 14.0 s (openPMD). For frequent output with many ranks the
    native format, which needs no synchronisation, is cheaper.
  - Reading: `openpmd_viewer.OpenPMDTimeSeries('out/openpmd/')`, e.g.
    `ts.get_field('E', 'z', iteration=n, m='all', theta=0)`.
- **Python helpers.** `tools/quarz_read.py` reads all of these;
  `xz_plane()` builds the field in the (ξ, x) plane from the modes.
  `tools/plot_fields.py` makes a quick-look PNG.

## 7. Validation (OpenMP backend, 2 cores)

All results can be reproduced with `validation/run_validation.sh`.

| test | result |
|---|---|
| Unit: pushers (`test_pushers`) | Identities: `imp` = HC to 5·10⁻¹⁴ and `imp_rr`(ν=0) = Vay exactly, over 10⁶ random states. Force-free γ = 2·10⁴ particle: Vay/HC/IMP exact, Boris drifts. Radiation-reaction energy loss equals ∫ν u·v dt to 0.05 %. |
| Unit: tridiagonal (M = 1537) | Thomas and PCR both at round-off level (4·10⁻¹⁶, 7·10⁻¹⁶) |
| Unit: L₀, L₁ (flux and Dirichlet wall), L₂ with χ, manufactured solutions | second order: measured 1.9–2.0 on uniform and 1.98–2.0 on stretched grids (h₀ = 0.01 → 0.00125, ratio ≤ 1.05) |
| Cartesian particle rewrite, `modes = 0`, vs the earlier ring version | linear wake and blowout fields agree to 10⁻¹⁰–10⁻¹³; beam energies identical |
| `modes = 1`, centred beam (quiet start) | m = 0 identical to `modes = 0` to 10⁻¹³; all m = 1 fields ≤ 5·10⁻¹⁵ |
| Unit: uniform plasma deposit, stretched grid | exact to 5·10⁻¹³, including the axis node |
| Linear wake, n_b = 10⁻³, vs Green's function in a conducting cylinder | E_z(0): 0.13 % (uniform 0.01), 0.12 % (stretched 0.002→0.05, 389 cells). The error scales ∝ n_b, i.e. it is the nonlinearity, not the numerics. |
| Blowout, n_b = 10, σ_r = 0.3 | (E_r − B_θ)/r = 0.5000 inside the bubble. E_z at ξ = 5, 7, 8 converged to 4 digits across Δr = 0.01→0.0025 and Δξ = 0.01→0.0025. Stretched grid agrees to 0.08 %. PCR = Thomas. |
| Betatron oscillation, ion channel, γ = 1000 | r_rms(t) = r₀\|cos(t/√(2γ))\| to 0.26 % over one period |
| Witness energy gain, rigid driver | d⟨γ⟩/dt = 0.4477, vs −E_z = 0.4471 from the wake |
| Matched witness (σ_r = 0.01, ε_n = 0.01, γ = 2·10⁴), immobile ions, 400/ω_p | ε_n preserved (0.0097 → 0.0097); identical for Δt = 20 and 10. Boris gave about 9 % spurious growth. |
| `modes = 1`: displaced driver vs translated axisymmetric solution, linear regime (n_b = 0.01, d = 0.01) | m = 1 amplitudes = −d ∂_r(m = 0 fields): L2 errors ψ 0.08 %, E_z 0.4 %, E_r 0.3 %, E_θ 0.16 %, B_θ 0.19 %, B_r 0.13 %. The sin parts, which must vanish, are ≤ 6·10⁻¹⁴. The m = 1 density is noisier (~14 %), because the check differentiates a particle-noisy n_e. |
| `modes = 1`: same test in the blowout (n_b = 10, d = 0.002), inside the bubble | ψ 0.06 %, E_r 1.1 %, E_θ 0.6 %, B_θ 1.0 %, B_r 0.5 %. The transverse force on the axis is W_x = −d/2, as expected for a shifted ion column. |
| Parser unit test (`test_parser`) | 15 expressions vs C++ at 200 points: exact. Device evaluation identical to host. Constant folding, `my_constants` in any order, circular definitions and syntax errors reported. |
| Parsed plasma vs built-in profiles | uniform "1": 10⁻¹⁴; parabolic channel: 5·10⁻¹⁰; finite column via `if()`: identical; z up-ramp via `if()` vs `plasma.z_table`: identical witness energies over 8 steps |
| Parsed analytic driver vs built-in analytic Gaussian | m = 0: 10⁻¹⁴. Offset driver with `modes = 1`: 10⁻⁸ (the built-in uses a 2·10⁻⁷ Bessel approximation, so the parsed one is the more accurate). |
| Parsed particle driver (quiet start, ppc 2 2, 4.8·10⁵ particles) vs analytic | ρ_beam 3·10⁻⁴, fields ≤ 1 %. The random Gaussian loader with 10⁶ particles: ρ_beam 200 % local noise, fields up to 7 %. |
| Position-dependent spreads (`test_profiles`) | beam std(p_x) follows 0.01 + 0.5\|x\| within 1–3 %, std(p_z) follows 10(ξ − 4) within 0.5 % together with a chirp, constant `uy_std` within 3 %; plasma std(p_x)/uth(r, z) = 1.00 ± 0.03 (modes 0 and 1); constant spreads reproduce earlier runs bit for bit |
| Asymmetric plasma n = 1 + 0.2x with `modes = 1` | n_e0 = 1 to 2·10⁻¹¹; n_e,c = 0.2 r to 6·10⁻⁵; n_e,s ≤ 10⁻¹³ |
| `modes = 1`: offset beam (x₀ = 0.05) in an ion channel, γ = 1000 | centroid x_c(t) = x₀ cos(t/√(2γ)) to 0.26 %; ⟨y⟩ ≤ 2·10⁻⁷ |

**openPMD output** (`validation/check_openpmd.py`, `validation/cmp_openpmd.py`):
written together with the native files (`output.format = both`) and compared with them.

| check | result |
|---|---|
| openPMD on native nodes vs native files (hosing, m = 1, SI units) | fields, positions, weights identical (0) |
| uniform openPMD grid vs native data interpolated in Python | 4·10⁻¹⁶ |
| openPMD-viewer `get_field('E','r', m='all', theta)` vs f₀ + f_c cos θ + f_s sin θ from the native file | 2·10⁻¹⁶–5·10⁻¹⁶ (θ = 0, 1, π/2); viewer reads u_z = 497 for a γ = 500 beam (momentum and mass units consistent) |
| MPI 2 and 3 ranks vs serial (HDF5, all meshes and particles) | bit-identical |
| backends | HDF5 (serial, parallel), ADIOS2 BP (parallel), JSON (serial) written and read back |

**Laser envelope solver** (`validation/laser_*.in`, `laser_check.py`, section 11 of `run_validation.sh`):

| test | result |
|---|---|
| Diffraction in vacuum through the focus (k₀ = 20, w0 = 2, t = −Z_R … +Z_R) | spot size and peak amplitude within 7·10⁻⁴ of the Gaussian-beam formulas; ∫\|â\|² conserved exactly; the centroid slips by 0.0501 vs the group-velocity value t/(k₀w0)² = 0.0500 (the ∂_t∂_ξ term) |
| Linear wake, a0 = 0.05, w0 = 3 | ψ and E_z vs (∂²_ξ + 1)ψ = ⟨a²⟩/2: 1.4·10⁻³; 2.2·10⁻⁴ for a0 = 0.02, i.e. the O(a0²) nonlinearity. The same on a stretched grid. |
| Matched parabolic channel n = 1 + 4r²/w0⁴, stretched grid, 6.7 Z_R | spot size constant to 2·10⁻⁴; slippage 2.1653 vs (1 + 4/w0²) t/(2k₀²) = 2.1667 (Δt = 2.5; 2.5 % low at Δt = 5) |
| Relativistic self-focusing of a long pulse, w0 = 20, over one Z_R | vacuum 1 → 0.707 (exact); P/P_c = 0.5: diffraction compensated (1.00); P/P_c = 2: focusing, a → 1.58 a0 |
| Ponderomotive kick on a γ = 5 beam inside an a0 = 1 pulse | dp_x/dt and dp_z/dt vs −∇⟨a²⟩/(2γ̄): 4·10⁻⁴, 1.2·10⁻³ |
| MPI 2, 3 ranks vs serial (LWFA example, laser + witness) | bit-identical, incl. `laser.txt` |
| Single solve vs Wake-T 0.9.1 (`validation/laser_vs_waket.py`; same laser, plasma, resolution) | a0 = 0.3, 1: E_z on axis agrees to 10⁻³ and 3·10⁻⁴. a0 = 2: up to 11 % near the bubble closure. a0 = 4: up to 27 % from the laser peak on (both codes converged in Δξ, Δr and ppc). QUARZ's fields satisfy Gauss's law in the laser region to the accuracy of the finite-difference check (e.g. div E = 0.371 vs ρ = 0.369 at the peak, r = 0.5). Wake-T's do not there (−0.014 vs 0.128), which points to the Wake-T side but has not been settled; a full PIC comparison is needed for a0 ≳ 2. |

**LWFA example** (`examples/lwfa_laser.in`, a0 = 4, 100 steps = 10.6 mm, 77 s on 2 cores):
- **laser:** self-guided, with the peak amplitude between 3 and 6.5;
- **witness:** accelerated from 100 MeV to 0.94 GeV (γ = 200 → 1832) with 2.4 % rms energy spread;
- **witness emittance:** ε_n 0.0098 → 0.0113.

**Ionization** (`validation/ion_*.in`, `ion_check.py`, section 12 of `run_validation.sh`). The reference
calculations are independent Python implementations: ADK in SI units, the exact laser field,
and analytic beam densities.

| test | result |
|---|---|
| Field ionization of dilute H and He by the field of a rigid electron beam (E up to 0.27 a.u.), 16 rings/cell, ions immobile | total ionized charge vs the expected charge of every ring (Markov chain of the per-slice probabilities, also = continuous rate equations): He 2.0434·10⁻³ vs 2.0361·10⁻³ ± 1.7·10⁻⁵ (+0.45 σ), H −1.0 σ. Radial bands agree within the Monte Carlo error. |
| The same with `nsplit = 64` | He +0.06 σ, H +0.15 σ (unbiased, less noise; 46× the run time, since every atom emits up to 64 pairs) |
| Laser ionization of N up to N⁵⁺ (0.8 µm, a0 = 0.1, wide spot) | ⟨z⟩(ξ) on the axis vs the ADK rate integrated over the true oscillating field: within 0.033 everywhere (Monte Carlo noise of the 4800 rings near the axis; converged in Δξ). Mean p⊥² of the born electrons vs test particles (p = −a(t_birth)): ratio 0.98. With the usual Laplace closed form the rates were 5–40 % high and p⊥² 13 % high, so it was replaced. |
| Impact ionization of Ar by a rigid 400 GeV proton beam (σ = 1.59·10⁻¹⁸ cm²), `nsplit = 2000` | ionized fraction vs 1 − exp(−σ n₀ k_p⁻¹ β ∫n_b dξ) in radial bands along the beam: within 1–3 %. Total at the end of the box: +0.9 %. |
| MPI 2, 3 ranks vs serial, with laser ionization of N (LWFA example + 1 % N, `nsplit = 4`) and with impact ionization (proton bunch in Ar) | all outputs incl. `ionization.txt` bit-identical |
| MPI 2, 3 ranks vs serial, `modes = 1`, field ionization of H/He (8 particles per ring) by an offset driver | bit-identical |
| All earlier validation sections | unchanged |

**MPI vs serial** (`validation/cmp_runs.py` compares every output file; serial runs with
`beams.xi_shape = ngp`, one OpenMP thread per rank, 2 cores; P = 3, 4 oversubscribed):

| case | ranks | result | time, serial → 2 ranks |
|---|---|---|---|
| witness gain (`gain.in`, 20 steps, 2000 slices) | 2 | all outputs **bit-identical** | 36.2 s → 20.5 s (1.77×) |
| ion motion (`pwfa_ion_motion.in`, mobile ions, 10 steps) | 2 | bit-identical | 14.2 s → 9.0 s |
| parsed profiles (z-dependent plasma, parsed beams) | 2 | bit-identical | |
| analytic offset driver, `modes = 1` | 2, 3, 4 | bit-identical | |
| hosing (`modes = 1`, Picard 3, quiet-start driver, 12 steps) | 2, 3 | identical to step 8; then fields 10⁻¹⁰, beam moments 5·10⁻⁹ | 87.5 s → 48.9 s (1.79×) |
| γ = 5 beam slipping 2 slices per step through all ranks + γ = 200 driver, mobile ions, `modes = 1`, 24 steps | 2, 3, 4 | fields 5·10⁻⁸, particles 4·10⁻⁹, beam moments 10⁻¹¹ | 41.8 s → 22.9 s (1.83×) |

Bit-identity is lost only when beam particles move between ranks: they are
appended to the receiving rank's arrays, so the summation order in the deposit
changes (round-off). The ideal pipeline speed-up for these runs is 1.83–1.92×.
A serial run with `ngp` instead of `linear` changes the witness energy gain by
6·10⁻⁹ (relative) and the local beam sources by particle-sampling noise.
The serial `linear` results are bit-identical to the version before MPI.

**Pinched witness with mobile hydrogen ions** (`examples/pwfa_ion_motion.in`).
The reference is a uniform grid with Δr = 0.00025 (N = 32000, 53 s per
sweep). In the witness the ion density on axis rises to 2.3 n₀, and the
focusing gradient at r = 0.01 is 0.88 instead of 0.5.

| grid | cells | sweep time | error n_i(0) | error W_r(r=0.01) | error W_r(r=0.03) | error E_z(0) |
|---|---|---|---|---|---|---|
| uniform 0.01 | 800 | 1.5 s | 26 % | 7.1 % | 1.6 % | 1.7 % |
| uniform 0.005 | 1600 | 3.1 s | 12 % | 2.5 % | 0.58 % | 0.47 % |
| **stretched 0.0005 → 0.01 → 0.05** | **512** | **1.1 s** | 5.3 % | **0.33 %** | **0.06 %** | **0.10 %** |

## 8. Examples

- `examples/blowout_basic.in`: single quasi-static solve of a blowout.
- `examples/pwfa_ion_motion.in`: driver plus a strongly pinched,
  low-emittance witness with mobile ions (100 steps, a few minutes on
  2 cores).
- `examples/parsed_profiles.in`: all profiles from formulas.
  - plasma: a half-sine up-ramp, a parabolic channel and a finite column;
  - driver: trapezoidal current;
  - witness: energy chirp.

  It takes about 10 s for 20 steps on 2 cores.
- `examples/hosing.in`: `modes = 1`. A long flat-top electron driver,
  slightly tilted in x, drives a blowout; the tail centroid oscillation grows
  (electron hosing). Output includes per-slice centroids
  (`slices_driver_*.txt`). In the test run (120 steps, ~8 min on 2 cores)
  the tail centroid grows from 0.006 to 0.12 c/ω_p in about three betatron
  periods, while ⟨y⟩ stays at 10⁻¹⁴ thanks to `mirror_y`. See
  `validation/quarz_hosing.png`. It is a qualitative demonstration and has not
  been compared quantitatively with hosing theory.
- **Cost of m = 1:** about 10× an axisymmetric run: 8 particles per ring,
  plus the extra field solves and Picard iterations (blowout test: 11 s vs
  1 s per sweep).
- `examples/awake_seeded_smi.in`: AWAKE-like seeded self-modulation of a
  long proton bunch (n₀ = 7·10¹⁴ cm⁻³, 6000 slices, Δt = 4 cm). It is set to
  3 steps; 250 steps give 10 m, in about 6 min on 2 cores. In a 16 m test
  run the on-axis wakefield 6 cm behind the seed grows from 5 MV/m to a peak
  of 240 MV/m at z ≈ 4 m and then decays. That is the expected SMI
  growth-and-saturation behaviour, but it has **not** been benchmarked
  quantitatively against LCODE.

- `examples/lwfa_laser.in`: laser wakefield acceleration with the envelope
  solver (0.8 µm, a0 = 4, k_p w0 = 4 matched, n0 = 10¹⁸ cm⁻³) and an external
  witness in the first bucket, 100 steps over 10.6 mm (about 1 min on
  2 cores; see the laser validation for results).

- `examples/awake_impact_ionization.in`: a 400 GeV proton bunch (3·10¹¹,
  σ_r = 0.2 mm, σ_z = 6 cm) in neutral argon (7·10¹⁴ cm⁻³), with impact ionization only
  (the bunch field is about 13 MV/m, far below tunnelling).
  - The ionized fraction grows along the bunch to 2.0·10⁻⁴ near the axis (mean over
    r < 0.3 c/ω_p at the tail). The analytic value σN/(2πσ_r²) is 1.9·10⁻⁴.
  - The total, 3.4·10¹⁰ electrons per metre, agrees with σ n_gas N (1.19·10⁻³ in code
    units) to 1 %.
  - The new electrons fall into the proton potential: the on-axis electron density
    reaches 100× the ionization density.

  It takes 4 s per step on 2 cores (7200 slices).
- `examples/lwfa_ionization.in`: the laser of `lwfa_laser.in` (a0 = 4) in neutral helium
  (5·10¹⁷ cm⁻³) with 1 % nitrogen.
  - He and N up to N⁵⁺ are ionized in the leading edge; N⁶⁺ and N⁷⁺ only about one
    c/ω_p before the peak.
  - About 1000 plasma electrons per step are trapped and removed (ionization
    injection, which a quasi-static code cannot follow).
  - It takes 3.4 s per step on 2 cores. The plasma particle count is about three times
    that of the pre-ionized run.

`validation/quarz_validation_summary.png` shows the pinched-witness grid
comparison, the blowout on the stretched grid, and the SMI run.

## 9. Relation to LCODE, limitations and road map

**Same approach as LCODE-2D:**

- quasi-static model, axisymmetric with `modes = 0`;
- plasma rings advanced in ξ;
- a beam that evolves with a large time step;
- mobile ions;
- radial and longitudinal plasma profiles;
- seeded long beams.

**Different from LCODE:**

- the field solver: explicit B⊥ (no iterations for m = 0), finite-element Laplacian on
  arbitrary radial nodes, all six field components, and the optional m = 1 mode (QPAD-like);
- the particle integrator (Adams–Bashforth);
- the beam pushers (Vay by default; HC/IMP and IMP with radiation reaction available).

**Not (yet) included:**

- Ionization:
  - trapped electrons (e.g. ionization injection) are removed, not followed. Converting
    them into beam particles would be the natural next step;
  - the laser loses no energy to ionization: there is no ionization current in the
    envelope equation;
  - only tunnelling (ADK) is modelled: no barrier-suppression correction, no
    multiphoton regime (Keldysh γ > 1);
  - no collisional ionization by plasma electrons (no cascades) and no recombination;
  - impact ionization only for the first level. In `modes = 1` the impact sources and
    `nz_<species>` are m = 0 only.
- Laser: the m = 1 part of the envelope (offset or tilted lasers), the
  ∂²_t term, and Benedetti's phase correction for strongly red-shifted pulses.
- Azimuthal modes m ≥ 2 (the transverse field of a beam displaced by more
  than a fraction of its size, or of an elliptical beam, is only
  approximated).
- Plasma-particle sub-stepping near the axis, and noise-reduction options
  (LCODE's "noise reducer").
- Adaptive Δξ.
- Beam macro-particle splitting.
- Checkpoint/restart.
- Transverse (r) decomposition: MPI splits only the ξ box. For a single
  time step (no pipelining possible) MPI therefore gives no speed-up.
- Dynamic load balancing between MPI ranks (the slice blocks are equal and fixed).

**Physics caveats:**

- A quasi-static code cannot follow trapped plasma electrons; they are
  removed (Δ < `delta_min` or γ/Δ > `max_qsa_factor`) and counted in the log. This includes
  ionization-injected electrons.
- With `modes = 1`, keep beam offsets small compared with the beam and
  bubble sizes; the m ≥ 2 content is dropped. Use `nsym` (≥ 2) or
  `analytic = 1` beams so that sampling noise does not seed spurious m = 1.
- The m = 1 wall conditions are Dirichlet, so R should be large compared
  with the bubble.
- The E_z spike at the closure of a strongly nonlinear bubble is
  resolution-dependent, as in every quasi-static code. Converge it with
  Δξ and h₀ if it matters.

## 10. Code map

| file | contents |
|---|---|
| `src/RadialGrid.*` | grid generators, hat-function volumes, device grid (binary-search `locate`) |
| `src/FieldSolver.*` | radial operators L₀, L₁, L₂ with χ, gradients |
| `src/SliceData.hpp`, `src/FieldStore.hpp` | mode layout of sources and fields |
| `src/Tridiag.*` | Thomas and PCR (one team) solvers |
| `src/PlasmaSpecies.*` | loading, mode deposits (sources, S₊), Adams–Bashforth push in ξ (with the ponderomotive force) |
| `src/Laser.*` | laser envelope solver (complex tridiagonal per slice), ponderomotive arrays, laser diagnostics |
| `src/Ionization.*` | element tables, ADK and cycle-averaged laser rates, Bethe impact cross-section; the ionization kernel is in `PlasmaSpecies.cpp` |
| `src/Beam.*` | beam loading, 2D deposit, push, diagnostics |
| `src/Pushers.hpp` | Boris, Vay, Higuera–Cary and the two schemes of the IMP note (+ radiation reaction) |
| `src/Simulation.*` | slice sweep, pipelined time loop, MPI messages, output |
| `src/Parallel.*` | thin MPI layer (point-to-point messages, offset file writes; serial stub without MPI) |
| `src/OpenPMDWriter.*`, `src/FieldTable.hpp` | openPMD output through openPMD-api (fields in thetaMode, beam particles; stub without openPMD-api) |
| `src/Parser.*` | function parser (compiler to a device-evaluable stack program), `my_constants` |
| `src/Profiles.*`, `src/Config.*` | density profiles, input parser |
| `tests/test_solvers.cpp`, `tests/test_pushers.cpp`, `tests/test_parser.cpp`, `tests/test_profiles.cpp` | unit tests (run by `ctest`) |
| `validation/` | physics validation inputs and scripts |
| `tools/` | Python readers and plotting (optional) |
