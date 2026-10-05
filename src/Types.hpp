// QSRZ - quasi-static axisymmetric (r, xi) PIC code with non-uniform radial grid
// Common types and constants.
#pragma once

#include <Kokkos_Core.hpp>

namespace qsrz {

using Real = double;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
using HostSpace = Kokkos::HostSpace;

using View1D  = Kokkos::View<Real*, MemSpace>;
using View2D  = Kokkos::View<Real**, Kokkos::LayoutRight, MemSpace>;
using View3D  = Kokkos::View<Real***, Kokkos::LayoutRight, MemSpace>;
using IView1D = Kokkos::View<int*, MemSpace>;

using Range = Kokkos::RangePolicy<ExecSpace>;

constexpr Real PI = 3.14159265358979323846;

// true if the default execution space runs on the host (Serial/OpenMP/Threads)
inline constexpr bool exec_is_host() {
    return Kokkos::SpaceAccessibility<HostSpace, MemSpace>::accessible;
}

} // namespace qsrz
