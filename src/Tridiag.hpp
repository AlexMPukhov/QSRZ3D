// Tridiagonal solver  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i,  i = 0..M-1
//
// Two device implementations:
//   * Thomas   - sequential, one thread; optimal on CPU backends
//   * PCR      - parallel cyclic reduction executed by one team (one GPU
//                thread block); log2(M) steps of O(M) parallel work.
//   * Partition - Thomas/PCR hybrid (Laszlo, Giles & Appleyard, ACM TOMS 42, 31 (2016)):
//                the system is cut into P chunks of >= 8 rows; each team thread
//                eliminates the interior of its chunks (sequential, O(M/P)), the
//                2P chunk-end unknowns form a tridiagonal system solved by PCR in
//                team scratch memory, then the interior is recovered. P depends on
//                M only, so the result does not depend on the thread count.
//   Auto = Thomas on a host backend, Partition on a GPU. Partition may also pay off on
//   a CPU with many threads (solver.tridiag = partition); not measured yet.
// All leave a, b, c, d untouched.
#pragma once

#include "Types.hpp"

namespace quarz {

enum class TridiagMethod { Auto, Thomas, PCR, Partition };

class TridiagSolver {
public:
    TridiagSolver() = default;
    TridiagSolver(int M, TridiagMethod method);

    void solve(const View1D& a, const View1D& b, const View1D& c, const View1D& d, const View1D& x) const;

    int size() const { return M_; }
    TridiagMethod method() const { return method_; }

private:
    int M_ = 0;
    TridiagMethod method_ = TridiagMethod::Thomas;
    View1D cp_, dp_;   // Thomas scratch
    View2D pa_, pb_, pc_, pd_;  // PCR ping-pong buffers (2 x M)
    View1D qa_, qc_, qd_;       // Partition: modified coefficients per row
    int P_ = 0;                 // Partition: number of chunks
};

} // namespace quarz
