// Per-slice source and field arrays on the N+1 radial nodes, decomposed in
// azimuthal modes.
//
// Real scalars (rho, psi, E_z, B_z, ...):
//     f(r,theta) = f0(r) + 2 Re[ f1(r) e^{i theta} ]  = f0 + f1c cos + f1s sin,
//     f1 = f1r + i f1i = (f1c - i f1s)/2 ,        deposit:  f1 = < f e^{-i theta} >
// Transverse vectors (J_perp, B_perp, E_perp, W_perp) are stored through the complex
// combination V+ = V_x + i V_y :
//     V+(r,theta) = v0(r) + v1(r) e^{i theta} + v2(r) e^{2 i theta},   deposit: v_k = < V+ e^{-ik theta} >
// v1 = V_r0 + i V_theta0 is the axisymmetric (m=0) part, v0 and v2 carry m=1.
// Each v_k obeys a scalar equation with the operator L_k = (1/r)(r f')' - k^2 f / r^2.
#pragma once

#include "Types.hpp"
#include <string>

namespace qsrz {

struct MScalar {
    View1D a0, ar, ai;   // mode 0 ; mode 1 real/imag parts of f1
    MScalar() = default;
    MScalar(const std::string& n, int M) : a0(n + ".0", M), ar(n + ".1r", M), ai(n + ".1i", M) {}
};

struct MVector {
    View1D r0, i0, r1, i1, r2, i2;   // v_k = r_k + i i_k , k = 0,1,2
    MVector() = default;
    MVector(const std::string& n, int M)
        : r0(n + ".r0", M), i0(n + ".i0", M), r1(n + ".r1", M), i1(n + ".i1", M), r2(n + ".r2", M), i2(n + ".i2", M) {}
};

struct SliceSources {
    MScalar rhot;   // rho - J_z   (plasma: q w per particle, conserved)
    MScalar drho;   // d(rho - J_z)/dxi   (exact xi-derivative of the deposit)
    MScalar jz;     // J_z
    MScalar chi;    // sum q^2 w / (m (gamma - p_z)) / V    (implicit part of dJ_perp/dxi)
    MScalar rho;    // charge density
    MScalar ne;     // density of negative plasma species
    MScalar ni;     // density of positive plasma species (incl. immobile background)
    MVector jp;     // J+ = J_x + i J_y
    MVector S;      // dJ+/dxi + i chi B+   (explicit remainder)

    SliceSources() = default;
    explicit SliceSources(int M)
        : rhot("src.rhot", M), drho("src.drho", M), jz("src.jz", M), chi("src.chi", M), rho("src.rho", M),
          ne("src.ne", M), ni("src.ni", M), jp("src.jp", M), S("src.S", M) {}
};

struct SliceFields {
    MScalar psi;    // psi = phi - A_z
    MScalar ez;     // E_z = d psi / d xi
    MScalar bz;     // B_z
    MVector wp;     // W+ = (E_x - B_y) + i (E_y + B_x) = -(d_x + i d_y) psi
    MVector bp;     // B+ = B_x + i B_y
    MVector bprev;  // previous Picard iterate of B+
    View1D aa, daa; // laser: <a^2> (time-averaged, m = 0) and d<a^2>/dr; used when laser = true
    bool laser = false;
    // scratch
    View1D t1, t2, t3, t4, rhs;

    SliceFields() = default;
    explicit SliceFields(int M)
        : psi("f.psi", M), ez("f.ez", M), bz("f.bz", M), wp("f.wp", M), bp("f.bp", M), bprev("f.bprev", M),
          aa("f.aa", M), daa("f.daa", M),
          t1("f.t1", M), t2("f.t2", M), t3("f.t3", M), t4("f.t4", M), rhs("f.rhs", M) {}
};

// ---------------------------------------------------------------------------
// evaluation of mode sums at a particle (cos, sin of its azimuth given)
KOKKOS_INLINE_FUNCTION Real eval_scalar(Real f0, Real fr, Real fi, Real c, Real s, bool m1) {
    return m1 ? f0 + Real(2) * (fr * c - fi * s) : f0;
}
// V+ = v0 + v1 e^{i th} + v2 e^{2 i th}
KOKKOS_INLINE_FUNCTION void eval_vector(Real r0, Real i0, Real r1, Real i1, Real r2, Real i2, Real c, Real s,
                                        bool m1, Real& vx, Real& vy) {
    vx = r1 * c - i1 * s;
    vy = r1 * s + i1 * c;
    if (m1) {
        const Real c2 = c * c - s * s, s2 = Real(2) * c * s;
        vx += r0 + r2 * c2 - i2 * s2;
        vy += i0 + r2 * s2 + i2 * c2;
    }
}

} // namespace qsrz
