#include "Simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace quarz {

namespace {
const char* const ION_LOG_NOTE =
    "\n# born_weight: sum of the weights of the electrons created in the sweep (n0 (c/omega_p)^2 per c/omega_p)"
    "\n# born_per_m: the same as electrons per metre of plasma; born_p2: mean p_perp^2 at birth (m_e c)^2"
    "\n# z_tail: mean charge state at the end of the box\n";
void ion_line(std::ostream& o, int n, double t, const std::vector<std::array<double, 4>>& A, double perm) {
    o << n << " " << std::setprecision(10) << t;
    for (const auto& a : A)
        o << " " << a[0] << " " << a[0] * perm << " " << (a[0] > 0 ? a[1] / a[0] : 0.0) << " " << (a[3] > 0 ? a[2] / a[3] : 0.0);
    o << "\n";
}
} // namespace

Simulation::Simulation(const Config& cfg) : cfg_(cfg), comm_(Comm::world()) {
    const int modes = cfg.get_int("modes", 0);
    if (modes != 0 && modes != 1) throw std::runtime_error("modes must be 0 (axisymmetric) or 1 (m = 0 and 1)");
    m1_ = (modes == 1);
    picard_ = cfg.get_int("solver.picard", m1_ ? 3 : 1);
    max_cells_ = cfg.get_double("pusher.max_cells_per_step", 0.0);
    substep_max_ = cfg.get_int("pusher.substep_max", 64);
    if (max_cells_ < 0 || substep_max_ < 1)
        throw std::runtime_error("pusher.max_cells_per_step must be >= 0 and pusher.substep_max >= 1");

    grid_ = std::make_unique<RadialGrid>(cfg);
    const std::string tm = cfg.get_string("solver.tridiag", "auto");
    TridiagMethod method = TridiagMethod::Auto;
    if (tm == "thomas") method = TridiagMethod::Thomas;
    else if (tm == "pcr") method = TridiagMethod::PCR;
    else if (tm != "auto") throw std::runtime_error("solver.tridiag must be auto|thomas|pcr");
    solver_ = std::make_unique<FieldSolver>(*grid_, method);
    // radial smoothing of the plasma sources (regularizes the axis caustic at the bubble back)
    smooth_a_ = cfg.get_double("plasma.smooth_length", 0.0);
    if (smooth_a_ < 0) throw std::runtime_error("plasma.smooth_length must be >= 0");
    if (smooth_a_ > 0) solver_->set_filter(std::vector<Real>(grid_->N, smooth_a_ * smooth_a_));

    box_.xi_min = cfg.get_double("xi.min", 0.0);
    const Real xi_max = cfg.get_double("xi.max");
    box_.dxi = cfg.get_double("xi.step");
    box_.nxi = static_cast<int>(std::lround((xi_max - box_.xi_min) / box_.dxi)) + 1;
    dt_ = cfg.get_double("time.dt", 0.0);
    nsteps_ = cfg.get_int("time.steps", 0);
    t_ = cfg.get_double("time.start", 0.0);
    out_every_ = cfg.get_int("output.every", 1);
    beam_dump_every_ = cfg.get_int("output.beam_every", 0);
    slice_bins_ = cfg.get_int("output.beam_slices", 0);
    {
        const auto rg = cfg.get_list("output.beam_slices_range");
        slice_lo_ = rg.size() == 2 ? std::stod(rg[0]) : box_.xi_min;
        slice_hi_ = rg.size() == 2 ? std::stod(rg[1]) : xi_max;
    }
    outdir_ = cfg.get_string("output.dir", "out");
    seed_ = cfg.get_int("plasma.seed", 1);

    // ---- decomposition along xi: contiguous blocks of slices, rank 0 at the head
    const int P = comm_.size(), r = comm_.rank(), K = box_.nxi;
    if (P > K) throw std::runtime_error("more MPI ranks than xi slices");
    box_.k0 = static_cast<int>((static_cast<long long>(K) * r) / P);
    box_.nloc = static_cast<int>((static_cast<long long>(K) * (r + 1)) / P) - box_.k0;
    if (box_.nloc < 2 && P > 1) throw std::runtime_error("each rank needs at least 2 xi slices");
    const std::string shape = cfg.get_string("beams.xi_shape", P > 1 ? "ngp" : "linear");
    if (shape == "ngp") box_.ngp = true;
    else if (shape == "linear") {
        if (P > 1) throw std::runtime_error("beams.xi_shape = linear is not possible with the xi decomposition; use ngp");
        box_.ngp = false;
    } else throw std::runtime_error("beams.xi_shape must be linear or ngp");

    profile_ = DensityProfile(cfg, "plasma");
    auto names = cfg.get_list("plasma.species");
    if (names.empty()) names.push_back("electrons");
    bool has_positive = false;
    for (const auto& n : names) {
        species_.push_back(std::make_unique<PlasmaSpecies>(cfg, n, *grid_, profile_, m1_));
        const auto& sp = *species_.back();   // (neutral gas does not count as a positive species)
        if (sp.charge() > 0 && (!sp.ionizable() || sp.ion_info().P.z0 > 0)) has_positive = true;
    }
    neutralize_ = cfg.get_bool("plasma.neutralize", !has_positive);
    bool any_impact = false;
    for (size_t i = 0; i < species_.size(); ++i) {
        if (!species_[i]->ionizable()) continue;
        const std::string pn = species_[i]->ion_info().product;
        int pi = -1;
        for (size_t j = 0; j < species_.size(); ++j) if (species_[j]->name() == pn) pi = static_cast<int>(j);
        if (pi < 0) throw std::runtime_error(species_[i]->name() + ".ionization.product: no plasma species '" + pn +
                                             "' (add it to plasma.species; '" + pn + ".ppc = 0' gives an initially empty species)");
        if (species_[pi]->ionizable() || species_[pi]->charge() >= 0 || !species_[pi]->mobile())
            throw std::runtime_error(species_[i]->name() + ".ionization.product '" + pn + "' must be a mobile negative species");
        ion_sp_.push_back(static_cast<int>(i));
        ion_prod_.push_back(pi);
        any_impact = any_impact || species_[i]->ion_info().P.impact;
    }
    ion_acc_.assign(ion_sp_.size(), {0, 0, 0, 0});

    std::vector<long> nglobal;
    for (const auto& n : cfg.get_list("beams")) {
        beams_.push_back(std::make_unique<Beam>(cfg, n, *grid_, box_, m1_));
        nglobal.push_back(beams_.back()->num_particles());
        if (P > 1) beams_.back()->restrict_to_local();   // every rank keeps the particles of its slices
    }
    outbox_.resize(beams_.size());

    const int M = grid_->N + 1;
    const int KL = box_.nloc;
    src_ = SliceSources(M);
    fld_ = SliceFields(M);
    bg_rhot_ = MScalar("bg.rhot", M);
    bg_rho_ = MScalar("bg.rho", M);
    gz_ = MVector("gz", M);
    fld = View3D("fld", KL, M, m1_ ? int(F_NC1) : int(F_NC0));
    bsrc = View3D("bsrc", KL, M, m1_ ? int(B_NC1) : int(B_NC0));
    diag_base_ = m1_ ? int(D_NC1) : int(D_NC0);
    diag = View3D("diag", KL, M, diag_base_ + static_cast<int>(ion_sp_.size()));
    if (any_impact) imp_ = View3D("impact", KL, M, 3);
    bguard = View2D("bguard", M, B_NC1);

    if (comm_.root()) {
        std::cout << "QUARZ - Quasistatic Arbitrary-resolution RZ code, Kokkos execution space: " << ExecSpace::name() << "\n";
        if (P > 1)
            std::cout << "MPI: " << P << " ranks, xi decomposition with time pipelining, " << K / P
                      << "-" << (K + P - 1) / P << " slices per rank\n";
        std::cout << "Azimuthal modes: " << (m1_ ? "m = 0 and 1" : "m = 0 (axisymmetric)");
        if (m1_) std::cout << ",  Picard iterations for the chi_1 coupling: " << picard_;
        std::cout << "\n";
        grid_->print_summary(std::cout);
        std::cout << "Box: xi in [" << box_.xi_min << ", " << box_.xi_min + (K - 1) * box_.dxi << "], dxi = "
                  << box_.dxi << ", " << K << " slices;  beam shape in xi: " << (box_.ngp ? "nearest slice" : "linear")
                  << "\n";
        for (const auto& s : species_)
            std::cout << "Plasma species '" << s->name() << "': q = " << s->charge() << ", m = " << s->mass()
                      << (s->frozen() ? ", immobile" : (s->mobile() ? ", mobile" : ", immobile")) << ", "
                      << s->num_particles() << " macro-particles\n";
        for (int i : ion_sp_) std::cout << "Species '" << species_[i]->name() << "'" << species_[i]->ion_info().description() << "\n";
        std::cout << "Neutralising immobile background: " << (neutralize_ ? "yes" : "no") << "\n";
        if (smooth_a_ > 0) {
            std::cout << "Radial smoothing of the plasma sources: a = " << smooth_a_ << "\n";
        }
        for (size_t b = 0; b < beams_.size(); ++b)
            std::cout << "Beam '" << beams_[b]->name() << "': " << nglobal[b] << " macro-particles"
                      << (beams_[b]->rigid() ? " (rigid)" : "") << "\n";
        std::filesystem::create_directories(outdir_);
        grid_->write(outdir_ + "/grid.txt");
    }
    comm_.barrier();   // output directory exists
    if (P == 1) beamlog_.open(outdir_ + "/beams.txt");
    else beamlog_.open(outdir_ + "/beams.part" + std::to_string(r));
    if (P == 1)
        beamlog_ << "# beam step t alive npart charge gamma_mean gamma_rms xi_mean xi_rms r_rms emit_nx"
                    " x_mean y_mean emit_ny sigma_x sigma_y\n";

    // ---- laser envelope
    if (cfg.has("laser.a0")) {
        laser_ = std::make_unique<Laser>(cfg, *grid_, *solver_, box_, t_, dt_);
        fld_.laser = true;
        if (comm_.root()) std::cout << laser_->description() << "\n";
        laserlog_.open(P == 1 ? outdir_ + "/laser.txt" : outdir_ + "/laser.part" + std::to_string(r));
        if (P == 1) laserlog_ << "# step t a_max w_rms xi_centroid L_rms energy (int |a|^2 dV)\n";
        for (int i : ion_sp_) species_[i]->set_laser(laser_->k0(), laser_->circular());
    }
    if (!ion_sp_.empty()) {
        ionlog_.open(P == 1 ? outdir_ + "/ionization.txt" : outdir_ + "/ionization.part" + std::to_string(r));
        if (P == 1) {
            ionlog_ << "# step t";
            for (int i : ion_sp_) {
                const std::string& nm = species_[i]->name();
                ionlog_ << "  " << nm << ":born_weight " << nm << ":born_per_m " << nm << ":born_p2 " << nm << ":z_tail";
            }
            ionlog_ << ION_LOG_NOTE;
        }
    }

    // ---- output format: native binary/text files, openPMD, or both
    const std::string fmt = cfg.get_string("output.format", "native");
    if (fmt != "native" && fmt != "openpmd" && fmt != "both")
        throw std::runtime_error("output.format must be native, openpmd or both");
    out_native_ = (fmt != "openpmd");
    field_files_ = cfg.get_bool("output.field_files", true);
    if (fmt != "native") {
        opmd_ = std::make_unique<OpenPMDWriter>(cfg, outdir_, *grid_, box_, m1_, dt_);
        if (comm_.root()) {
            std::cout << opmd_->description() << "\n";
            if (P > 1)
                std::cout << "  note: openPMD writes are collective; each output step drains the xi pipeline\n";
        }
    }
}

Simulation::~Simulation() = default;

// ---------------------------------------------------------------------------
void Simulation::zero_sources() {
    const int M = grid_->N + 1;
    SliceSources s = src_;
    Kokkos::parallel_for("zero_src", Range(0, M), KOKKOS_LAMBDA(int j) {
        const MScalar* sc[7] = {&s.rhot, &s.drho, &s.jz, &s.chi, &s.rho, &s.ne, &s.ni};
        for (int c = 0; c < 7; ++c) { sc[c]->a0(j) = 0; sc[c]->ar(j) = 0; sc[c]->ai(j) = 0; }
        const MVector* vc[2] = {&s.jp, &s.S};
        for (int c = 0; c < 2; ++c) {
            vc[c]->r0(j) = 0; vc[c]->i0(j) = 0; vc[c]->r1(j) = 0;
            vc[c]->i1(j) = 0; vc[c]->r2(j) = 0; vc[c]->i2(j) = 0;
        }
    });
}

void Simulation::combine_sources(int kl, Real ds) {
    const int M = grid_->N + 1;
    const int KL = box_.nloc;
    const int k = kl;
    SliceSources s = src_;
    MScalar bgt = bg_rhot_, bgr = bg_rho_;
    auto b = bsrc;
    auto gd = bguard;
    const Real idxi = Real(1) / box_.dxi;
    const bool m1 = m1_;
    const bool ngp = box_.ngp;
    const bool sub = ds != Real(0);
    Kokkos::parallel_for("combine", Range(0, M), KOKKOS_LAMBDA(int j) {
        // d(beam source)/dxi.  linear: central difference (serial only);
        // ngp: backward difference, the slice upstream of this rank comes from the guard row
        auto dxi_b = [&](int comp) {
            if (ngp) {
                const Real prev = (k > 0) ? b(k - 1, j, comp) : gd(j, comp);
                return (b(k, j, comp) - prev) * idxi;
            }
            if (KL == 1) return Real(0);
            if (k == 0) return (b(1, j, comp) - b(0, j, comp)) * idxi;
            if (k == KL - 1) return (b(KL - 1, j, comp) - b(KL - 2, j, comp)) * idxi;
            return Real(0.5) * (b(k + 1, j, comp) - b(k - 1, j, comp)) * idxi;
        };
        // beam source at xi_k (+ ds in a sub-slice: linear extrapolation with the same xi-derivative)
        auto bv = [&](int comp) { return sub ? b(k, j, comp) + ds * dxi_b(comp) : b(k, j, comp); };
        s.rhot.a0(j) += bgt.a0(j) + bv(B_RT0);
        s.drho.a0(j) += dxi_b(B_RT0);
        s.jz.a0(j) += bv(B_JZ0);
        s.jp.r1(j) += bv(B_JP1R);
        s.jp.i1(j) += bv(B_JP1I);
        s.rho.a0(j) += bgr.a0(j) + bv(B_RHO0);
        if (bgr.a0(j) > Real(0)) s.ni.a0(j) += bgr.a0(j);
        if (m1) {
            s.rhot.ar(j) += bgt.ar(j) + bv(B_RT1R);
            s.rhot.ai(j) += bgt.ai(j) + bv(B_RT1I);
            s.drho.ar(j) += dxi_b(B_RT1R);
            s.drho.ai(j) += dxi_b(B_RT1I);
            s.jz.ar(j) += bv(B_JZ1R);
            s.jz.ai(j) += bv(B_JZ1I);
            s.jp.r0(j) += bv(B_JP0R);
            s.jp.i0(j) += bv(B_JP0I);
            s.jp.r2(j) += bv(B_JP2R);
            s.jp.i2(j) += bv(B_JP2I);
            s.rho.ar(j) += bgr.ar(j) + bv(B_RHO1R);
            s.rho.ai(j) += bgr.ai(j) + bv(B_RHO1I);
        }
    });
}

void Simulation::set_guard(const double* p) {
    const int M = grid_->N + 1;
    auto h = Kokkos::create_mirror_view(bguard);
    for (int j = 0; j < M; ++j)
        for (int c = 0; c < B_NC1; ++c) h(j, c) = p[B_NC1 * j + c];
    Kokkos::deep_copy(bguard, h);
}

void Simulation::get_last_rhot(std::vector<double>& buf) const {
    const int M = grid_->N + 1;
    auto B = Kokkos::create_mirror_view_and_copy(HostSpace(), bsrc);
    const int k = box_.nloc - 1;
    const int nc = static_cast<int>(B.extent(2));
    for (int j = 0; j < M; ++j)
        for (int c = 0; c < B_NC1; ++c) buf.push_back(c < nc ? B(k, j, c) : 0.0);
}

// W+ = -(d_x + i d_y) psi:  w1 = -psi0',  w2 = -(psi1' - psi1/r),  w0 = -(conj(psi1)' + conj(psi1)/r)
void Simulation::compute_wplus() {
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    solver_->gradient(f.psi.a0, f.t1);
    const bool m1 = m1_;
    if (m1) {
        solver_->gradient_odd(f.psi.ar, f.t2);
        solver_->over_r(f.psi.ar, f.t3);
        solver_->gradient_odd(f.psi.ai, f.t4);
        solver_->over_r(f.psi.ai, f.rhs);
    }
    Kokkos::parallel_for("wplus", Range(0, M), KOKKOS_LAMBDA(int j) {
        f.wp.r1(j) = -f.t1(j);
        f.wp.i1(j) = Real(0);
        if (m1) {
            f.wp.r2(j) = -(f.t2(j) - f.t3(j));
            f.wp.i2(j) = -(f.t4(j) - f.rhs(j));
            f.wp.r0(j) = -(f.t2(j) + f.t3(j));
            f.wp.i0(j) = (f.t4(j) + f.rhs(j));
        }
    });
}

// L B_z = -curl_z J_perp;  curl_z = Im[(d_x - i d_y) J+],  (d_x - i d_y)(j_k e^{ik th}) = (j_k' + k j_k/r) e^{i(k-1) th}
void Simulation::compute_bz() {
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    SliceSources s = src_;
    MVector g = gz_;   // scratch
    solver_->gradient_odd(s.jp.i1, f.t1);
    solver_->over_r(s.jp.i1, f.t2);
    const bool m1 = m1_;
    if (m1) {
        solver_->gradient_odd(s.jp.r2, g.r0);
        solver_->over_r(s.jp.r2, g.i0);
        solver_->gradient_odd(s.jp.i2, g.r1);
        solver_->over_r(s.jp.i2, g.i1);
        solver_->gradient(s.jp.r0, g.r2);
        solver_->gradient(s.jp.i0, g.i2);
    }
    Kokkos::parallel_for("bz_rhs", Range(0, M), KOKKOS_LAMBDA(int j) {
        f.t3(j) = -(f.t1(j) + f.t2(j));                         // -curl_z, mode 0
        if (m1) {
            const Real D1r = g.r0(j) + Real(2) * g.i0(j), D1i = g.r1(j) + Real(2) * g.i1(j);
            const Real Dmr = g.r2(j), Dmi = g.i2(j);
            // c1 = (D_1 - conj(D_-1)) / (2i)
            f.t4(j) = -Real(0.5) * (D1i + Dmi);                 // -Re c1
            f.rhs(j) = Real(0.5) * (D1r - Dmr);                 // -Im c1
        }
    });
    solver_->solve(OpKind::L0, f.t3, f.bz.a0);
    if (m1) {
        solver_->solve(OpKind::L1D, f.t4, f.bz.ar);
        solver_->solve(OpKind::L1D, f.rhs, f.bz.ai);
    }
}

// (L_k - chi0) b_k = i [ ((d_x + i d_y) J_z)_k + S_k ] + chi1 b_{k-1} + conj(chi1) b_{k+1}
void Simulation::solve_bplus(int /*k*/) {
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    SliceSources s = src_;
    MVector g = gz_;
    const bool m1 = m1_;
    // ((d_x + i d_y) J_z):  k=1: J0' ;  k=2: J1' - J1/r ;  k=0: conj(J1)' + conj(J1)/r
    solver_->gradient(s.jz.a0, f.t1);
    if (m1) {
        solver_->gradient_odd(s.jz.ar, f.t2);
        solver_->over_r(s.jz.ar, f.t3);
        solver_->gradient_odd(s.jz.ai, f.t4);
        solver_->over_r(s.jz.ai, f.rhs);
    }
    Kokkos::parallel_for("gz", Range(0, M), KOKKOS_LAMBDA(int j) {
        g.r1(j) = f.t1(j) + s.S.r1(j);
        g.i1(j) = s.S.i1(j);
        if (m1) {
            g.r2(j) = (f.t2(j) - f.t3(j)) + s.S.r2(j);
            g.i2(j) = (f.t4(j) - f.rhs(j)) + s.S.i2(j);
            g.r0(j) = (f.t2(j) + f.t3(j)) + s.S.r0(j);
            g.i0(j) = -(f.t4(j) + f.rhs(j)) + s.S.i0(j);
        }
    });
    // rhs arrays reuse f.bprev (6 arrays)
    MVector rb = f.bprev;
    const int niter = m1 ? picard_ : 1;
    for (int it = 0; it < niter; ++it) {
        Kokkos::parallel_for("bplus_rhs", Range(0, M), KOKKOS_LAMBDA(int j) {
            // i G = (-G_i, G_r)
            Real r1 = -g.i1(j), i1 = g.r1(j);
            if (m1) {
                Real r0 = -g.i0(j), i0 = g.r0(j), r2 = -g.i2(j), i2 = g.r2(j);
                const Real cr = s.chi.ar(j), ci = s.chi.ai(j);   // chi1; conj(chi1) = (cr, -ci)
                const Real b0r = f.bp.r0(j), b0i = f.bp.i0(j), b1r = f.bp.r1(j), b1i = f.bp.i1(j);
                const Real b2r = f.bp.r2(j), b2i = f.bp.i2(j);
                // k=0: conj(chi1) b1
                r0 += cr * b1r + ci * b1i;  i0 += cr * b1i - ci * b1r;
                // k=1: chi1 b0 + conj(chi1) b2
                r1 += cr * b0r - ci * b0i + cr * b2r + ci * b2i;
                i1 += cr * b0i + ci * b0r + cr * b2i - ci * b2r;
                // k=2: chi1 b1
                r2 += cr * b1r - ci * b1i;  i2 += cr * b1i + ci * b1r;
                rb.r0(j) = r0; rb.i0(j) = i0; rb.r2(j) = r2; rb.i2(j) = i2;
            }
            rb.r1(j) = r1; rb.i1(j) = i1;
        });
        solver_->solve(OpKind::L1D, rb.r1, f.bp.r1, s.chi.a0);                 // B_r0
        solver_->solve(OpKind::L1F, rb.i1, f.bp.i1, s.chi.a0, s.jz.a0);       // B_theta0, flux wall
        if (m1) {
            solver_->solve(OpKind::L0, rb.r0, f.bp.r0, s.chi.a0);
            solver_->solve(OpKind::L0, rb.i0, f.bp.i0, s.chi.a0);
            solver_->solve(OpKind::L2D, rb.r2, f.bp.r2, s.chi.a0);
            solver_->solve(OpKind::L2D, rb.i2, f.bp.i2, s.chi.a0);
        }
    }
}

// E+ = W+ - i B+
void Simulation::store_slice(int kl) {
    const int k = kl;
    const int M = grid_->N + 1;
    SliceFields f = fld_;
    SliceSources s = src_;
    auto F = fld;
    auto D = diag;
    const bool m1 = m1_;
    Kokkos::parallel_for("store", Range(0, M), KOKKOS_LAMBDA(int j) {
        F(k, j, F_EP1R) = f.wp.r1(j) + f.bp.i1(j);
        F(k, j, F_EP1I) = f.wp.i1(j) - f.bp.r1(j);
        F(k, j, F_BP1R) = f.bp.r1(j);
        F(k, j, F_BP1I) = f.bp.i1(j);
        F(k, j, F_EZ0) = f.ez.a0(j);
        F(k, j, F_BZ0) = f.bz.a0(j);
        D(k, j, D_PSI0) = f.psi.a0(j);
        D(k, j, D_NE0) = s.ne.a0(j);
        D(k, j, D_NI0) = s.ni.a0(j);
        if (m1) {
            F(k, j, F_EP0R) = f.wp.r0(j) + f.bp.i0(j);
            F(k, j, F_EP0I) = f.wp.i0(j) - f.bp.r0(j);
            F(k, j, F_EP2R) = f.wp.r2(j) + f.bp.i2(j);
            F(k, j, F_EP2I) = f.wp.i2(j) - f.bp.r2(j);
            F(k, j, F_BP0R) = f.bp.r0(j);
            F(k, j, F_BP0I) = f.bp.i0(j);
            F(k, j, F_BP2R) = f.bp.r2(j);
            F(k, j, F_BP2I) = f.bp.i2(j);
            F(k, j, F_EZ1R) = f.ez.ar(j);
            F(k, j, F_EZ1I) = f.ez.ai(j);
            F(k, j, F_BZ1R) = f.bz.ar(j);
            F(k, j, F_BZ1I) = f.bz.ai(j);
            D(k, j, D_PSI1R) = f.psi.ar(j);
            D(k, j, D_PSI1I) = f.psi.ai(j);
            D(k, j, D_NE1R) = s.ne.ar(j);
            D(k, j, D_NE1I) = s.ne.ai(j);
            D(k, j, D_NI1R) = s.ni.ar(j);
            D(k, j, D_NI1I) = s.ni.ai(j);
        }
    });
}

// msg: message from the upstream rank (empty on rank 0), already positioned after the beam part
void Simulation::step_fields(const std::vector<double>& msg, size_t& off) {
    const int K = box_.nxi, k0 = box_.k0, KL = box_.nloc;
    const int M = grid_->N + 1;

    if (laser_) laser_->begin_step();
    // ---- beams -> (local xi, r) arrays
    Kokkos::deep_copy(bsrc, 0.0);
    for (const auto& b : beams_) b->deposit(bsrc);
    if (imp_.extent(0) > 0) {
        Kokkos::deep_copy(imp_, 0.0);
        for (const auto& b : beams_) b->deposit_impact(imp_);
    }
    if (!ion_sp_.empty()) {
        Kokkos::deep_copy(diag, 0.0);
        for (auto& a : ion_acc_) a = {0, 0, 0, 0};
    }

    // ---- fresh plasma at the front of the box, background (every rank: the background is the same)
    const Real z_front = t_ - box_.xi_min;
    for (View1D v : {bg_rhot_.a0, bg_rhot_.ar, bg_rhot_.ai, bg_rho_.a0, bg_rho_.ar, bg_rho_.ai}) Kokkos::deep_copy(v, 0.0);
    unsigned long long sd = seed_;
    for (auto& s : species_) s->load(z_front, sd++);
    for (auto& s : species_)
        if (!s->mobile()) s->deposit_rho(bg_rhot_, bg_rho_);
    if (neutralize_) {
        MScalar a = src_.rhot, b = src_.rho;
        for (View1D v : {a.a0, a.ar, a.ai, b.a0, b.ar, b.ai}) Kokkos::deep_copy(v, 0.0);
        for (auto& s : species_)
            if (s->mobile()) s->deposit_rho(a, b);
        MScalar bgt = bg_rhot_, bgr = bg_rho_;
        Kokkos::parallel_for("neutralize", Range(0, grid_->N + 1), KOKKOS_LAMBDA(int j) {
            bgt.a0(j) -= a.a0(j); bgt.ar(j) -= a.ar(j); bgt.ai(j) -= a.ai(j);
            bgr.a0(j) -= b.a0(j); bgr.ar(j) -= b.ar(j); bgr.ai(j) -= b.ai(j);
        });
    }
    if (solver_->filter_on())   // the background is smoothed like the plasma (neutral plasma edge)
        for (int c = 0; c < 2; ++c) {
            MScalar& m = c == 0 ? bg_rhot_ : bg_rho_;
            solver_->filter(0, m.a0);
            if (m1_) { solver_->filter(1, m.ar); solver_->filter(1, m.ai); }
        }

    if (comm_.rank() == 0) {
        // B+ initial guess for the Picard iteration: zero at the head; no beam upstream of the box
        for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) Kokkos::deep_copy(v, 0.0);
        Kokkos::deep_copy(bguard, 0.0);
    } else {
        // plasma state entering the first local slice, B+ warm start, upstream beam guard slice
        for (auto& sp : species_)
            if (sp->mobile()) off += sp->unpack_state(msg.data() + off);
        for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) {
            auto h = Kokkos::create_mirror_view(v);
            for (int j = 0; j < M; ++j) h(j) = msg[off + j];
            Kokkos::deep_copy(v, h);
            off += M;
        }
        set_guard(msg.data() + off);
        off += static_cast<size_t>(B_NC1) * M;
        if (laser_) {
            laser_->set_guard(msg.data() + off);
            off += Laser::guard_size(M);
        }
    }

    const Real dxi = box_.dxi;
    for (int kl = 0; kl < KL; ++kl) {
        const int k = k0 + kl;   // global slice
        zero_sources();
        if (laser_) laser_->prepare_slice(kl, fld_);   // <a^2> of this slice (time n)
        for (auto& sp : species_)
            if (sp->mobile()) sp->deposit(src_, fld_);
        filter_plasma_sources();
        if (laser_) laser_->advance_slice(kl, src_.chi.a0);   // envelope of this slice -> time n+1
        combine_sources(kl);
        solve_slice_fields();
        store_slice(kl);
        for (size_t s = 0; s < ion_sp_.size(); ++s) {   // ionization with the fields of this slice
            PlasmaSpecies& ion = *species_[ion_sp_[s]];
            const IonizeResult r = ion.ionize(fld_, imp_, kl, dxi, k, step_, *species_[ion_prod_[s]]);
            ion_acc_[s][0] += r.born_w;
            ion_acc_[s][1] += r.born_wp2;
            ion.deposit_charge_state(diag, kl, diag_base_ + static_cast<int>(s));
        }
        if (k < K - 1) {   // the last local push produces the state entering the next rank
            // adaptive sub-slicing: if plasma particles would cross more than max_cells_ radial cells
            // in this step, it is split into nsub sub-slices, each with its own deposit and field
            // solve (beam rho - J_z extrapolated from this slice, laser <a^2> and ionization of this
            // slice); only the regular slices are stored
            const int nsub = subslices(dxi);
            const Real h = dxi / nsub;
            for (int ss = 0; ss < nsub; ++ss) {
                if (ss > 0) {
                    zero_sources();
                    for (auto& sp : species_)
                        if (sp->mobile()) sp->deposit(src_, fld_);
                    filter_plasma_sources();
                    combine_sources(kl, ss * h);
                    solve_slice_fields();
                }
                for (auto& sp : species_)
                    if (sp->mobile()) sp->push(fld_, h);
            }
            nsub_step_ += nsub - 1;
        }
    }
    if (laser_) laser_->end_sweep();
    if (k0 + KL == K)   // the last rank: plasma state at the end of the box
        for (size_t s = 0; s < ion_sp_.size(); ++s) species_[ion_sp_[s]]->charge_sums(ion_acc_[s][2], ion_acc_[s][3]);
    Kokkos::fence();
}

void Simulation::solve_slice_fields() {
    solver_->solve(OpKind::L0, src_.rhot.a0, fld_.psi.a0, View1D(), View1D(), -1.0);
    solver_->solve(OpKind::L0, src_.drho.a0, fld_.ez.a0, View1D(), View1D(), -1.0);
    if (m1_) {
        solver_->solve(OpKind::L1D, src_.rhot.ar, fld_.psi.ar, View1D(), View1D(), -1.0);
        solver_->solve(OpKind::L1D, src_.rhot.ai, fld_.psi.ai, View1D(), View1D(), -1.0);
        solver_->solve(OpKind::L1D, src_.drho.ar, fld_.ez.ar, View1D(), View1D(), -1.0);
        solver_->solve(OpKind::L1D, src_.drho.ai, fld_.ez.ai, View1D(), View1D(), -1.0);
    }
    compute_wplus();
    compute_bz();
    for (auto& sp : species_)
        if (sp->mobile()) sp->deposit_S(fld_, src_.S);
    if (solver_->filter_on()) {
        const MVector& S = src_.S;
        solver_->filter(1, S.r1); solver_->filter(1, S.i1);
        if (m1_) {
            solver_->filter(0, S.r0); solver_->filter(0, S.i0);
            solver_->filter(2, S.r2); solver_->filter(2, S.i2);
        }
    }
    solve_bplus(0);
}

// plasma deposit only (beams and the background are added later in combine_sources; the background
// is filtered once per step). Linear and xi-independent, so E_z = d psi / d xi stays exact.
void Simulation::filter_plasma_sources() {
    if (!solver_->filter_on()) return;
    const SliceSources& s = src_;
    for (const MScalar* m : {&s.rhot, &s.drho, &s.jz, &s.chi, &s.rho, &s.ne, &s.ni}) {
        solver_->filter(0, m->a0);
        if (m1_) { solver_->filter(1, m->ar); solver_->filter(1, m->ai); }
    }
    solver_->filter(1, s.jp.r1); solver_->filter(1, s.jp.i1);
    if (m1_) {
        solver_->filter(0, s.jp.r0); solver_->filter(0, s.jp.i0);
        solver_->filter(2, s.jp.r2); solver_->filter(2, s.jp.i2);
    }
}

int Simulation::subslices(Real dxi) const {
    if (!(max_cells_ > 0)) return 1;
    Real c = 0;
    double hlast = 0;
    for (const auto& sp : species_)
        if (sp->mobile()) { c = std::max(c, sp->max_cells(dxi)); hlast = std::max(hlast, sp->last_step()); }
    int n = c <= max_cells_ ? 1 : static_cast<int>(std::ceil(c / max_cells_));
    // the step may at most double: at least ceil(dxi / (2 h_last)) sub-slices after finer steps
    if (hlast > 0) n = std::max(n, static_cast<int>(std::ceil(dxi / (2 * hlast) - 1e-9)));
    return std::min(substep_max_, n);
}

// message to the downstream rank for step n:
//   [mobile species states (incl. AB history positions)][B+ (6 M)][beam sources of the last local slice (15 M)][beam particles]
std::vector<double> Simulation::make_message(int n) {
    std::vector<double> m;
    m.push_back(static_cast<double>(n));
    for (size_t b = 0; b < beams_.size(); ++b) {
        m.push_back(static_cast<double>(outbox_[b].size() / 7));
        m.insert(m.end(), outbox_[b].begin(), outbox_[b].end());
        outbox_[b].clear();
    }
    for (auto& sp : species_)
        if (sp->mobile()) sp->pack_state(m);
    for (View1D v : {fld_.bp.r0, fld_.bp.i0, fld_.bp.r1, fld_.bp.i1, fld_.bp.r2, fld_.bp.i2}) {
        auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), v);
        m.insert(m.end(), h.data(), h.data() + h.extent(0));
    }
    get_last_rhot(m);
    if (laser_) laser_->get_guard(m);
    return m;
}

// ---------------------------------------------------------------------------
// Output format (version 2):
//   int32 version(=2), int32 M, int32 K, int32 mode1, float64 t, float64 r[M], float64 xi[K],
//   int32 ncomp, then ncomp x { char name[16], float64 data[K][M] }
// mode 0: psi ez er eth br bth bz ne ni rhob
// mode 1 (if present): <name>_c, <name>_s  (cos and sin amplitudes: f = f0 + f_c cos + f_s sin)
std::shared_ptr<const FieldTable> Simulation::field_table() const {
    auto T = std::make_shared<FieldTable>();
    T->KL = box_.nloc;
    T->M = grid_->N + 1;
    // host backends: no copy (the mirror is the array itself); GPU: one host copy
    T->F = Kokkos::create_mirror_view_and_copy(HostSpace(), fld);
    T->D = Kokkos::create_mirror_view_and_copy(HostSpace(), diag);
    T->B = Kokkos::create_mirror_view_and_copy(HostSpace(), bsrc);
    const FieldTable* t = T.get();   // the lambdas below are owned by *t
    auto add = [&](const std::string& name, std::function<double(int, int)> fun) { T->comps.emplace_back(name, fun); };
    // W+ = E+ + i B+ ;  polar m=0: V_r0 = Re v1, V_th0 = Im v1
    add("psi", [t](int k, int j) { return t->D(k, j, D_PSI0); });
    add("ez", [t](int k, int j) { return t->F(k, j, F_EZ0); });
    add("er", [t](int k, int j) { return t->F(k, j, F_EP1R); });
    add("eth", [t](int k, int j) { return t->F(k, j, F_EP1I); });
    add("br", [t](int k, int j) { return t->F(k, j, F_BP1R); });
    add("bth", [t](int k, int j) { return t->F(k, j, F_BP1I); });
    add("bz", [t](int k, int j) { return t->F(k, j, F_BZ0); });
    add("ne", [t](int k, int j) { return t->D(k, j, D_NE0); });
    add("ni", [t](int k, int j) { return t->D(k, j, D_NI0); });
    add("rhob", [t](int k, int j) { return t->B(k, j, B_RHO0); });
    if (m1_) {
        // scalars f1 (complex, f = f0 + 2 Re[f1 e^{i th}]):  f_c = 2 Re f1,  f_s = -2 Im f1
        auto scalD = [&](const std::string& n, int cr, int ci) {
            add(n + "_c", [t, cr](int k, int j) { return 2.0 * t->D(k, j, cr); });
            add(n + "_s", [t, ci](int k, int j) { return -2.0 * t->D(k, j, ci); });
        };
        auto scalF = [&](const std::string& n, int cr, int ci) {
            add(n + "_c", [t, cr](int k, int j) { return 2.0 * t->F(k, j, cr); });
            add(n + "_s", [t, ci](int k, int j) { return -2.0 * t->F(k, j, ci); });
        };
        scalD("psi", D_PSI1R, D_PSI1I);
        scalF("ez", F_EZ1R, F_EZ1I);
        scalF("bz", F_BZ1R, F_BZ1I);
        scalD("ne", D_NE1R, D_NE1I);
        scalD("ni", D_NI1R, D_NI1I);
        add("rhob_c", [t](int k, int j) { return 2.0 * t->B(k, j, B_RHO1R); });
        add("rhob_s", [t](int k, int j) { return -2.0 * t->B(k, j, B_RHO1I); });
        // vectors from V+ modes v0, v2:  V_r1c = Re v0 + Re v2,  V_r1s = Im v0 - Im v2,
        //                                V_th1c = Im v0 + Im v2, V_th1s = Re v2 - Re v0
        auto vec = [&](const std::string& nr, const std::string& nt, int c0r, int c0i, int c2r, int c2i) {
            add(nr + "_c", [t, c0r, c2r](int k, int j) { return t->F(k, j, c0r) + t->F(k, j, c2r); });
            add(nr + "_s", [t, c0i, c2i](int k, int j) { return t->F(k, j, c0i) - t->F(k, j, c2i); });
            add(nt + "_c", [t, c0i, c2i](int k, int j) { return t->F(k, j, c0i) + t->F(k, j, c2i); });
            add(nt + "_s", [t, c0r, c2r](int k, int j) { return t->F(k, j, c2r) - t->F(k, j, c0r); });
        };
        vec("er", "eth", F_EP0R, F_EP0I, F_EP2R, F_EP2I);
        vec("br", "bth", F_BP0R, F_BP0I, F_BP2R, F_BP2I);
    }
    for (size_t s = 0; s < ion_sp_.size(); ++s) {   // ion charge density z n of the ionizable species (m = 0)
        const int c = diag_base_ + static_cast<int>(s);
        add("nz_" + species_[ion_sp_[s]]->name(), [t, c](int k, int j) { return t->D(k, j, c); });
        if (m1_) {
            add("nz_" + species_[ion_sp_[s]]->name() + "_c", [](int, int) { return 0.0; });
            add("nz_" + species_[ion_sp_[s]]->name() + "_s", [](int, int) { return 0.0; });
        }
    }
    if (laser_) {   // laser envelope a^ (m = 0 only)
        const auto& env = laser_->envelope();
        T->L = Kokkos::create_mirror_view_and_copy(HostSpace(), env);
        add("a_re", [t](int k, int j) { return t->L(k, j, 0); });
        add("a_im", [t](int k, int j) { return t->L(k, j, 1); });
        if (m1_)
            for (const char* n : {"a_re_c", "a_re_s", "a_im_c", "a_im_s"}) add(n, [](int, int) { return 0.0; });
    }
    return T;
}

void Simulation::write_fields(const std::string& fn) const {
    const int M = grid_->N + 1, K = box_.nxi;
    const auto T = field_table();
    const auto& out = T->comps;
    // layout: header | ncomp x { name[16], data[K][M] };  every rank writes its rows k0..k0+nloc-1
    const int k0 = box_.k0, KL = box_.nloc;
    const long long hbytes = 4 * 4 + 8 + 8LL * M + 8LL * K + 4;
    const long long cbytes = 16 + 8LL * K * M;
    const long long total = hbytes + static_cast<long long>(out.size()) * cbytes;
    if (comm_.root()) {
        std::vector<char> h(static_cast<size_t>(hbytes));
        char* p = h.data();
        const int32_t hdr[4] = {2, M, K, m1_ ? 1 : 0};
        std::memcpy(p, hdr, sizeof(hdr)); p += sizeof(hdr);
        std::memcpy(p, &t_, 8); p += 8;
        std::memcpy(p, grid_->r.data(), 8 * static_cast<size_t>(M)); p += 8 * static_cast<size_t>(M);
        for (int k = 0; k < K; ++k) { const double x = box_.xi_min + k * box_.dxi; std::memcpy(p, &x, 8); p += 8; }
        const int32_t nc = static_cast<int32_t>(out.size());
        std::memcpy(p, &nc, 4);
        Comm::write_at(fn, 0, h.data(), h.size(), total);
        for (size_t c = 0; c < out.size(); ++c) {
            char name[16];
            std::memset(name, 0, sizeof(name));
            std::strncpy(name, out[c].first.c_str(), 15);
            Comm::write_at(fn, hbytes + static_cast<long long>(c) * cbytes, name, 16);
        }
    }
    std::vector<double> v(static_cast<size_t>(KL) * M);
    for (size_t c = 0; c < out.size(); ++c) {
        for (int k = 0; k < KL; ++k)
            for (int j = 0; j < M; ++j) v[static_cast<size_t>(k) * M + j] = out[c].second(k, j);
        Comm::write_at(fn, hbytes + static_cast<long long>(c) * cbytes + 16 + 8LL * M * k0, v.data(), 8 * v.size());
    }
}

void Simulation::write_axis(const std::string& fn) const {
    // fixed-width text so that every rank can write its own lines at the right offset
    const int K = box_.nxi, k0 = box_.k0, KL = box_.nloc;
    auto F = Kokkos::create_mirror_view_and_copy(HostSpace(), fld);
    auto D = Kokkos::create_mirror_view_and_copy(HostSpace(), diag);
    auto B = Kokkos::create_mirror_view_and_copy(HostSpace(), bsrc);
    const double r1 = grid_->r[1];
    const int ncol = m1_ ? 8 : 6;
    const int lw = 18 * ncol;            // "%+.10e " = 18 characters per column, last blank -> '\n'
    const int hw = 256;
    if (comm_.root()) {
        std::string h = "# t = " + std::to_string(t_);
        h.resize(hw / 2 - 1, ' ');
        h += '\n';
        std::string h2 = "# xi  Ez(0)  psi(0)  ne(0)  rho_beam(0)  (Er-Bth)/r|r1";
        if (m1_) h2 += "  Wx(0)  Wy(0)  [W = E_perp + z x B_perp on the axis, from m=1]";
        h2.resize(hw / 2 - 1, ' ');
        h2 += '\n';
        h += h2;
        Comm::write_at(fn, 0, h.data(), h.size(), hw + static_cast<long long>(lw) * K);
    }
    std::string buf;
    buf.reserve(static_cast<size_t>(lw) * KL);
    char cell[32];
    for (int k = 0; k < KL; ++k) {
        double v[8] = {box_.xi_min + (k0 + k) * box_.dxi, F(k, 0, F_EZ0), D(k, 0, D_PSI0), D(k, 0, D_NE0),
                       B(k, 0, B_RHO0), (F(k, 1, F_EP1R) - F(k, 1, F_BP1I)) / r1, 0, 0};
        if (m1_) { v[6] = F(k, 0, F_EP0R) - F(k, 0, F_BP0I); v[7] = F(k, 0, F_EP0I) + F(k, 0, F_BP0R); }
        for (int c = 0; c < ncol; ++c) {
            std::snprintf(cell, sizeof(cell), "%+.10e%c", v[c], c == ncol - 1 ? '\n' : ' ');
            buf += cell;
        }
    }
    Comm::write_at(fn, hw + static_cast<long long>(lw) * k0, buf.data(), buf.size());
}

void Simulation::write_beam_output(int n) {
    const bool par = comm_.size() > 1;
    const std::string rk = ".part" + std::to_string(comm_.rank());
    char step[16];
    std::snprintf(step, sizeof(step), "%06d", n);
    for (const auto& b : beams_) {
        const auto S = b->sums();
        if (!par) {
            const BeamDiag d = b->from_sums(S, t_);
            beamlog_ << b->name() << " " << n << " " << std::setprecision(10) << d.t << " " << d.alive << " "
                     << d.npart << " " << d.charge << " " << d.gamma_mean << " " << d.gamma_rms << " " << d.xi_mean
                     << " " << d.xi_rms << " " << d.r_rms << " " << d.emit_nx << " " << d.x_mean << " " << d.y_mean
                     << " " << d.emit_ny << " " << d.sigma_x << " " << d.sigma_y << "\n";
        } else {
            beamlog_ << b->name() << " " << n << " " << std::setprecision(17) << t_;
            for (double x : S) beamlog_ << " " << x;
            beamlog_ << "\n";
        }
        if (slice_bins_ > 0 && out_every_ > 0 && n % out_every_ == 0) {
            const auto ss = b->slice_sums(slice_lo_, slice_hi_, slice_bins_);
            const std::string fn = outdir_ + "/slices_" + b->name() + "_" + step + ".txt";
            if (!par) Beam::write_slices_file(fn, t_, b->charge(), slice_lo_, slice_hi_, slice_bins_, ss);
            else {
                std::ofstream o(fn + rk, std::ios::binary);
                o.write(reinterpret_cast<const char*>(&t_), 8);
                o.write(reinterpret_cast<const char*>(ss.data()), 8 * ss.size());
            }
        }
        if (out_native_ && beam_dump_every_ > 0 && n % beam_dump_every_ == 0) {
            const std::string fn = outdir_ + "/beam_" + b->name() + "_" + step + ".bin";
            b->dump(par ? fn + rk : fn);
        }
    }
    beamlog_.flush();
}

// combine the per-rank partial outputs of a parallel run (rank 0, after the run)
void Simulation::merge_partial_outputs() {
    const int P = comm_.size();
    namespace fs = std::filesystem;
    // laser.txt: sums over the ranks of every step
    if (laser_) {
        std::map<int, std::pair<double, std::array<double, 5>>> acc;
        for (int r = 0; r < P; ++r) {
            const std::string fn = outdir_ + "/laser.part" + std::to_string(r);
            std::ifstream in(fn);
            int n;
            double t;
            std::array<double, 5> S;
            while (in >> n >> t >> S[0] >> S[1] >> S[2] >> S[3] >> S[4]) {
                auto it = acc.find(n);
                if (it == acc.end()) { acc[n] = {t, S}; continue; }
                auto& A = it->second.second;
                A[0] += S[0]; A[1] += S[1]; A[2] += S[2]; A[3] = std::max(A[3], S[3]); A[4] += S[4];
            }
            in.close();
            fs::remove(fn);
        }
        std::ofstream o(outdir_ + "/laser.txt");
        o << "# step t a_max w_rms xi_centroid L_rms energy (int |a|^2 dV)\n";
        for (const auto& e : acc) {
            const auto& S = e.second.second;
            const double E = S[0] > 0 ? S[0] : 1;
            const double xc = S[1] / E, L = std::sqrt(std::max(0.0, S[4] / E - xc * xc)), w = std::sqrt(2 * S[2] / E);
            o << e.first << " " << std::setprecision(10) << e.second.first << " " << S[3] << " " << w << " " << xc << " "
              << L << " " << S[0] << "\n";
        }
    }
    // ionization.txt: sums over the ranks
    if (!ion_sp_.empty()) {
        const size_t ns = ion_sp_.size();
        std::map<int, std::pair<double, std::vector<std::array<double, 4>>>> acc;
        for (int r = 0; r < P; ++r) {
            const std::string fn = outdir_ + "/ionization.part" + std::to_string(r);
            std::ifstream in(fn);
            int n;
            double t;
            while (in >> n >> t) {
                std::vector<std::array<double, 4>> A(ns);
                for (auto& a : A) for (double& x : a) in >> x;
                auto it = acc.find(n);
                if (it == acc.end()) { acc[n] = {t, A}; continue; }
                for (size_t s = 0; s < ns; ++s) for (int c = 0; c < 4; ++c) it->second.second[s][c] += A[s][c];
            }
            in.close();
            fs::remove(fn);
        }
        std::ofstream o(outdir_ + "/ionization.txt");
        o << "# step t";
        for (int i : ion_sp_) {
            const std::string& nm = species_[i]->name();
            o << "  " << nm << ":born_weight " << nm << ":born_per_m " << nm << ":born_p2 " << nm << ":z_tail";
        }
        o << ION_LOG_NOTE;
        const double n0 = cfg_.get_double("units.n0_cm3") * 1e6;
        const double kpi = 2.99792458e8 / (5.64146e4 * std::sqrt(n0 * 1e-6));
        for (const auto& e : acc) ion_line(o, e.first, e.second.first, e.second.second, n0 * kpi * kpi);
    }
    // beams.txt: sum the partial moments of every (beam, step)
    {
        std::map<std::pair<std::string, int>, std::pair<double, std::array<double, 17>>> acc;
        std::vector<std::pair<std::string, int>> order;
        for (int r = 0; r < P; ++r) {
            std::ifstream in(outdir_ + "/beams.part" + std::to_string(r));
            std::string name;
            int n;
            double t;
            while (in >> name >> n >> t) {
                std::array<double, 17> S;
                for (auto& x : S) in >> x;
                auto key = std::make_pair(name, n);
                auto it = acc.find(key);
                if (it == acc.end()) { acc[key] = {t, S}; order.push_back(key); }
                else for (int c = 0; c < 17; ++c) it->second.second[c] += S[c];
            }
            fs::remove(outdir_ + "/beams.part" + std::to_string(r));
        }
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second < b.second : a.first < b.first; });
        std::ofstream o(outdir_ + "/beams.txt");
        o << "# beam step t alive npart charge gamma_mean gamma_rms xi_mean xi_rms r_rms emit_nx"
             " x_mean y_mean emit_ny sigma_x sigma_y\n";
        std::map<std::string, const Beam*> beam;
        for (const auto& b : beams_) beam[b->name()] = b.get();
        for (const auto& key : order) {
            const auto& e = acc[key];
            const BeamDiag d = beam[key.first]->from_sums(e.second, e.first);
            o << key.first << " " << key.second << " " << std::setprecision(10) << d.t << " " << d.alive << " "
              << d.npart << " " << d.charge << " " << d.gamma_mean << " " << d.gamma_rms << " " << d.xi_mean << " "
              << d.xi_rms << " " << d.r_rms << " " << d.emit_nx << " " << d.x_mean << " " << d.y_mean << " "
              << d.emit_ny << " " << d.sigma_x << " " << d.sigma_y << "\n";
        }
    }
    // slices and particle dumps
    for (const auto& entry : fs::directory_iterator(outdir_)) {
        const std::string fn = entry.path().string();
        if (fn.size() < 7 || fn.substr(fn.size() - 6) != ".part0") continue;
        const std::string base = fn.substr(0, fn.size() - 6);
        const std::string leaf = entry.path().filename().string();
        if (leaf.rfind("slices_", 0) == 0) {
            std::vector<double> sum;
            double t = 0;
            for (int r = 0; r < P; ++r) {
                std::ifstream in(base + ".part" + std::to_string(r), std::ios::binary);
                in.seekg(0, std::ios::end);
                const size_t nb = static_cast<size_t>(in.tellg());
                in.seekg(0);
                std::vector<double> v(nb / 8);
                in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(nb));
                t = v[0];
                if (sum.empty()) sum.assign(v.begin() + 1, v.end());
                else for (size_t i = 1; i < v.size(); ++i) sum[i - 1] += v[i];
                fs::remove(base + ".part" + std::to_string(r));
            }
            std::string bname = leaf.substr(7);
            bname = bname.substr(0, bname.rfind('_'));
            Real q = -1;
            for (const auto& b : beams_) if (b->name() == bname) q = b->charge();
            Beam::write_slices_file(base, t, q, slice_lo_, slice_hi_, slice_bins_, sum);
        } else if (leaf.rfind("beam_", 0) == 0) {
            std::vector<double> all;
            for (int r = 0; r < P; ++r) {
                std::ifstream in(base + ".part" + std::to_string(r), std::ios::binary);
                int32_t n = 0;
                in.read(reinterpret_cast<char*>(&n), 4);
                std::vector<double> v(7 * static_cast<size_t>(n));
                in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(8 * v.size()));
                all.insert(all.end(), v.begin(), v.end());
                fs::remove(base + ".part" + std::to_string(r));
            }
            std::ofstream o(base, std::ios::binary);
            const int32_t n = static_cast<int32_t>(all.size() / 7);
            o.write(reinterpret_cast<const char*>(&n), 4);
            o.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(8 * all.size()));
        }
    }
}

void Simulation::write_laser_diag(int n) {
    const auto S = laser_->sums();
    if (comm_.size() == 1) {
        const double e = S[0] > 0 ? S[0] : 1;
        const double xc = S[1] / e, L = std::sqrt(std::max(0.0, S[4] / e - xc * xc)), w = std::sqrt(2 * S[2] / e);
        laserlog_ << n << " " << std::setprecision(10) << t_ << " " << S[3] << " " << w << " " << xc << " " << L << " "
                  << S[0] << "\n";
    } else {
        laserlog_ << n << " " << std::setprecision(17) << t_;
        for (double x : S) laserlog_ << " " << x;
        laserlog_ << "\n";
    }
    laserlog_.flush();
}

void Simulation::write_ion_diag(int n) {
    if (comm_.size() == 1) {
        const double n0 = cfg_.get_double("units.n0_cm3") * 1e6;                 // m^-3
        const double kpi = 2.99792458e8 / (5.64146e4 * std::sqrt(n0 * 1e-6));    // c/omega_p, m
        ion_line(ionlog_, n, t_, ion_acc_, n0 * kpi * kpi);
    } else {
        ionlog_ << n << " " << std::setprecision(17) << t_;
        for (const auto& a : ion_acc_) for (double x : a) ionlog_ << " " << x;
        ionlog_ << "\n";
    }
    ionlog_.flush();
}

void Simulation::run() {
    Kokkos::Timer total;
    const int P = comm_.size(), rank = comm_.rank();
    for (int n = 0; n <= nsteps_; ++n) {
        Kokkos::Timer timer;
        // ---- message from the upstream rank: beam particles, plasma state, B+, guard
        std::vector<double> msg;
        size_t off = 0;
        if (rank > 0) {
            msg = comm_.recv(rank - 1, n % 30000);
            if (static_cast<int>(msg[0]) != n) throw std::runtime_error("pipeline message out of order");
            off = 1;
            for (auto& b : beams_) {
                const size_t np = static_cast<size_t>(msg[off]);
                ++off;
                b->append(msg.data() + off, np);
                off += 7 * np;
            }
        }
        step_ = n;
        step_fields(msg, off);
        const double tf = timer.seconds();
        if (rank < P - 1) comm_.isend(rank + 1, n % 30000, make_message(n));

        long lost = 0;
        for (auto& s : species_) if (s->mobile()) lost += s->lost_count();
        const double nsub = nsub_step_;
        nsub_step_ = 0;
        lost_total_ += lost;

        write_beam_output(n);
        if (laser_) write_laser_diag(n);
        if (!ion_sp_.empty()) write_ion_diag(n);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%06d", n);
        const bool field_out = out_every_ > 0 && n % out_every_ == 0;
        const bool beam_out = beam_dump_every_ > 0 && n % beam_dump_every_ == 0;
        if (field_out) {
            if (out_native_ && field_files_) write_fields(outdir_ + "/fields_" + buf + ".bin");
            write_axis(outdir_ + "/axis_" + buf + ".txt");
        }
        if (opmd_ && (field_out || beam_out)) {
            static const std::vector<std::unique_ptr<Beam>> none;
            const auto table = field_out ? field_table() : nullptr;
            opmd_->write(n, t_, table.get(), beam_out ? beams_ : none);
        }

        if (comm_.root()) {
            std::cout << "step " << std::setw(6) << n << "  t = " << std::setw(10) << t_ << "   sweep " << std::fixed
                      << std::setprecision(3) << tf << " s" << std::defaultfloat;
            if (P > 1) std::cout << " (rank 0 of " << P << ", pipelined)";
            if (lost) std::cout << "   (" << lost << " plasma particles removed as trapped)";
            if (nsub > 0) std::cout << "   (" << static_cast<long>(nsub) << " extra sub-slices)";
            std::cout << std::endl;
        } else if (lost) {
            std::cout << "rank " << rank << " step " << n << ": " << lost
                      << " plasma particles removed as trapped" << std::endl;
        }

        if (n < nsteps_) {
            for (size_t b = 0; b < beams_.size(); ++b) {
                beams_[b]->push(fld, dt_, laser_ ? laser_->ponderomotive() : View3D());
                if (P > 1) {
                    auto out = beams_[b]->extract_outgoing();
                    if (rank < P - 1) outbox_[b].insert(outbox_[b].end(), out.begin(), out.end());
                }
            }
            t_ += dt_;
        }
    }
    comm_.wait_send();
    Kokkos::fence();
    if (opmd_) opmd_->close();
    beamlog_.close();
    laserlog_.close();
    ionlog_.close();
    const double tmax = comm_.allreduce_max(total.seconds());
    if (P > 1 && comm_.root()) merge_partial_outputs();
    comm_.barrier();
    if (comm_.root()) std::cout << "Total run time " << tmax << " s\n";
}

} // namespace quarz
