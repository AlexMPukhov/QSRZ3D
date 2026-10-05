// Non-uniform radial grid  0 = r_0 < r_1 < ... < r_N = R.
//
// All grid quantities live on the nodes r_j.  Particles are weighted to the
// nodes with linear (hat) functions W_j(r).  The node volume is the exact
// volume of the hat function,
//
//     V_j = 2*pi * Int W_j(r) r dr ,
//
// (Verboncoeur's volume-weighting correction generalised to non-uniform
// cells).  With this choice a uniform plasma deposits an exactly uniform
// density, also at the axis, and the Laplacian built from the same hat
// functions is the lumped-mass linear finite-element operator, so the
// discrete Gauss law  Sum_j V_j (L psi)_j = boundary flux  holds exactly.
#pragma once

#include "Types.hpp"
#include <iosfwd>
#include <string>
#include <vector>

namespace qsrz {

class Config;

// Device-side, trivially copyable view of the grid for use inside kernels.
struct GridD {
    View1D r;      // nodes, N+1
    View1D h;      // cell widths h_j = r_{j+1}-r_j, N
    View1D inv_h;  // 1/h_j, N
    View1D inv_V;  // 1/V_j, N+1
    int N = 0;
    Real R = 0;

    // cell index j with r_j <= x < r_{j+1}; x is clamped into [0,R]
    KOKKOS_INLINE_FUNCTION int locate(Real x) const {
        if (x <= Real(0)) return 0;
        if (x >= R) return N - 1;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (r(mid) <= x) lo = mid; else hi = mid;
        }
        return lo;
    }
    // cell index and fractional position t in [0,1]
    KOKKOS_INLINE_FUNCTION void weights(Real x, int& j, Real& t) const {
        j = locate(x);
        t = (x - r(j)) * inv_h(j);
        t = t < Real(0) ? Real(0) : (t > Real(1) ? Real(1) : t);
    }
    // linear interpolation of a node field
    KOKKOS_INLINE_FUNCTION Real interp(const View1D& f, int j, Real t) const {
        return (Real(1) - t) * f(j) + t * f(j + 1);
    }
};

class RadialGrid {
public:
    explicit RadialGrid(const Config& cfg);
    explicit RadialGrid(const std::vector<Real>& nodes);   // explicit node list

    int  N = 0;            // number of cells (N+1 nodes)
    Real R = 0;            // outer radius
    std::vector<Real> r;     // nodes
    std::vector<Real> h;     // cell widths
    std::vector<Real> rmid;  // cell mid points r_{j+1/2}
    std::vector<Real> V;     // node volumes (hat-function volumes, include 2 pi)

    GridD d;               // device copy

    Real hmin() const;
    Real hmax() const;
    Real max_ratio() const;  // max |h_{j+1}/h_j| (>=1)
    void print_summary(std::ostream& os) const;
    void write(const std::string& filename) const;

    // helpers used by the generators (public for testing)
    static std::vector<Real> make_uniform(Real R, Real dr);
    static std::vector<Real> make_regions(Real R, const std::vector<std::pair<double, double>>& regions,
                                          Real max_ratio);
    static std::vector<Real> read_file(const std::string& filename);

private:
    void finalize(const std::vector<Real>& nodes);
};

} // namespace qsrz
