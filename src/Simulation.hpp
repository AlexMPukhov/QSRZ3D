// Time loop: for every time step the plasma is swept through the box slice
// by slice (xi = xi_min ... xi_max), the fields are stored on the (xi, r)
// grid in azimuthal modes (m = 0, optionally m = 1), then the beams are
// advanced with a large time step dt.
//
// Parallelisation (MPI): domain decomposition along xi with time pipelining.
// Rank r owns the slices [k0, k0 + nloc).  For time step n it
//   1. receives from rank r-1 the plasma state entering its first slice (particles,
//      Adams-Bashforth history, B+ warm start), the beam rho - J_z of the slice just
//      upstream, and beam particles that slipped into its slices;
//   2. deposits its beam particles, sweeps its slices, and sends the same kind of
//      message to rank r+1;
//   3. writes its part of the output and pushes its beam particles.
// While rank r works on step n, rank r-1 already works on step n+1: with P ranks and
// many time steps the speed-up approaches P.  Beam particles are owned by nearest
// slice and use nearest-slice (NGP) deposit/gather in xi, which keeps every rank's
// work local to its slices; serial runs can use the same scheme (beams.xi_shape = ngp)
// and then give the same results as parallel runs.
#pragma once

#include "Beam.hpp"
#include "Config.hpp"
#include "FieldSolver.hpp"
#include "FieldStore.hpp"
#include "FieldTable.hpp"
#include "Laser.hpp"
#include "OpenPMDWriter.hpp"
#include "Parallel.hpp"
#include "PlasmaSpecies.hpp"
#include "Profiles.hpp"
#include "RadialGrid.hpp"
#include "SliceData.hpp"

#include <fstream>
#include <functional>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace qsrz {

// diagnostics-only arrays stored per slice
enum DiagComp : int { D_PSI0 = 0, D_NE0, D_NI0, D_PSI1R, D_PSI1I, D_NE1R, D_NE1I, D_NI1R, D_NI1I, D_NC0 = 3, D_NC1 = 9 };

class Simulation {
public:
    explicit Simulation(const Config& cfg);
    ~Simulation();

    void run();

    bool mode1() const { return m1_; }
    const RadialGrid& grid() const { return *grid_; }
    const BeamGrid& box() const { return box_; }
    Real time() const { return t_; }

    View3D fld;    // (local xi, r, FieldComp): fields for the beams
    View3D bsrc;   // (local xi, r, BeamComp): beam sources
    View3D diag;   // (local xi, r, DiagComp)
    View2D bguard; // (r, 3): beam rho - J_z (modes 0, 1r, 1i) of the slice just upstream of this rank

    void write_fields(const std::string& filename) const;
    void write_axis(const std::string& filename) const;
    std::shared_ptr<const FieldTable> field_table() const;   // host copy of the local fields, by name

    // slice-level kernels (public: they contain device lambdas, CUDA restriction)
    void zero_sources();
    void combine_sources(int kl);
    void compute_wplus();
    void compute_bz();
    void solve_bplus(int k);
    void store_slice(int kl);
    void set_guard(const double* p);
    void get_last_rhot(std::vector<double>& buf) const;
    void step_fields(const std::vector<double>& msg, size_t& off);   // one quasi-static sweep of the local slices
    std::vector<double> make_message(int n);                         // message for the downstream rank

private:
    void write_beam_output(int n);
    void merge_partial_outputs();

    const Config& cfg_;
    Comm& comm_;
    bool m1_ = false;
    int picard_ = 1;
    std::unique_ptr<RadialGrid> grid_;
    std::unique_ptr<FieldSolver> solver_;
    DensityProfile profile_;
    std::vector<std::unique_ptr<PlasmaSpecies>> species_;
    std::vector<std::unique_ptr<Beam>> beams_;
    std::vector<std::vector<double>> outbox_;   // per beam: particles for the downstream rank
    SliceSources src_;
    SliceFields fld_;
    MScalar bg_rhot_, bg_rho_;
    MVector gz_;                 // i (d_x + i d_y) J_z  work array
    bool neutralize_ = true;
    BeamGrid box_;
    Real dt_ = 0, t_ = 0;
    int nsteps_ = 0;
    int out_every_ = 1, beam_dump_every_ = 0, slice_bins_ = 0;
    double slice_lo_ = 0, slice_hi_ = 0;
    std::string outdir_;
    unsigned long long seed_ = 1;
    std::ofstream beamlog_;
    long lost_total_ = 0;
    bool out_native_ = true;                  // output.format = native | openpmd | both
    std::unique_ptr<OpenPMDWriter> opmd_;
    std::unique_ptr<Laser> laser_;          // laser envelope (laser.a0 given)
    std::ofstream laserlog_;
    void write_laser_diag(int n);
    // ---- ionization
    std::vector<int> ion_sp_, ion_prod_;     // ionizable species and their electron product species (indices)
    View3D imp_;                             // (local xi, r, 3): beam impact-ionization sources
    int diag_base_ = 0;                      // first diag component of the ion charge densities
    int step_ = 0;
    std::ofstream ionlog_;
    std::vector<std::array<double, 4>> ion_acc_;   // per ionizable species: born w, born w p^2, tail z w, tail w
    void write_ion_diag(int n);
};

} // namespace qsrz
