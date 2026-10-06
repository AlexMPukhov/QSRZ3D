// Host copy of the fields of the local slices, presented as named components on the
// native (xi, r) nodes:  value(k, j) with k = local slice, j = radial node.
//
// m = 0 names:  psi ez er eth br bth bz ne ni rhob
// m = 1 adds the same names with "_c" / "_s":  f = f0 + f_c cos(theta) + f_s sin(theta)
// laser runs add a_re, a_im: the complex envelope a^ (m = 0)
//
// Used by the native binary writer and by the openPMD writer.
#pragma once

#include "Types.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace quarz {

struct FieldTable {
    using HostView3D = Kokkos::View<Real***, Kokkos::LayoutRight, HostSpace>;
    HostView3D F, D, B;   // fields, diagnostics, beam sources (local slices)
    HostView3D L;         // laser envelope (Re, Im), if any
    int KL = 0, M = 0;    // local slices, radial nodes
    std::vector<std::pair<std::string, std::function<double(int, int)>>> comps;

    const std::function<double(int, int)>* find(const std::string& name) const {
        for (const auto& c : comps)
            if (c.first == name) return &c.second;
        return nullptr;
    }
};

} // namespace quarz
