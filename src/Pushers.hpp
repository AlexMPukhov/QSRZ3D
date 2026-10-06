// Relativistic momentum pushers  u^{n-1/2} -> u^{n+1/2}  for given E, B at the particle.
// u = p/(m c) ; eps = (q/m) dt/2 in normalised units.  All are header-only and
// usable inside Kokkos kernels.
//
//   boris   Boris (1970): E/2 - rotation with gamma(u^-) - E/2
//   vay     Vay (2008):   u^{n+1/2} - u^{n-1/2} = 2 eps [E + (v^{n+1/2} + v^{n-1/2})/2 x B]
//   hc      Higuera & Cary (2017), as written in their paper (E/2 - rotation - E/2)
//   imp     Pukhov note, eqs. (19)-(36): implicit midpoint in u,
//           u - u0 = 2 eps [E + (ubar/gammabar) x B],  ubar = (u+u0)/2
//   imp_rr  Pukhov note, eqs. (2)-(18): trapezoidal velocity average plus an implicit
//           radiation-reaction damping -nu u (nu from u0, Landau-Lifshitz leading term):
//           u - u0 = 2 eps [E + (v + v0)/2 x B] - dt nu u
//
// Algebraic identities, checked numerically in tests/test_pushers.cpp:
//   imp          == hc   (to round-off)
//   imp_rr(nu=0) == vay  (to round-off)
#pragma once

#include "Types.hpp"

namespace quarz {
namespace push {

struct V3 { Real x, y, z; };

KOKKOS_INLINE_FUNCTION V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
KOKKOS_INLINE_FUNCTION V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
KOKKOS_INLINE_FUNCTION V3 operator*(Real s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
KOKKOS_INLINE_FUNCTION Real dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
KOKKOS_INLINE_FUNCTION V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// positive root of  g^4 - 2 x g^2 - c = 0 ,  g^2 = x + sqrt(x^2 + c),  c >= 0,
// written without cancellation when x < 0 (strong B, weak u)
KOKKOS_INLINE_FUNCTION Real gamma2_root(Real x, Real c) {
    const Real s = Kokkos::sqrt(x * x + c);
    return x >= Real(0) ? x + s : (s > -x ? c / (s - x) : Real(1));
}

// Solve  p = a + (p / gamma(p)) x b  exactly (the common kernel of Vay, HC and the note):
// gamma^2 = (1+a^2-b^2)/2 + sqrt(((1+a^2-b^2)/2)^2 + b^2 + (a.b)^2)       [note eq. (15),(32)]
// p = [gamma^2 a + gamma a x b + b (a.b)] / (gamma^2 + b^2)               [note eq. (18),(35)]
KOKKOS_INLINE_FUNCTION V3 solve_implicit(V3 a, V3 b) {
    const Real b2 = dot(b, b), ab = dot(a, b);
    const Real g2 = gamma2_root(Real(0.5) * (Real(1) + dot(a, a) - b2), b2 + ab * ab);
    const Real g = Kokkos::sqrt(g2);
    const V3 axb = cross(a, b);
    const Real inv = Real(1) / (g2 + b2);
    return {inv * (g2 * a.x + g * axb.x + ab * b.x), inv * (g2 * a.y + g * axb.y + ab * b.y),
            inv * (g2 * a.z + g * axb.z + ab * b.z)};
}

KOKKOS_INLINE_FUNCTION V3 boris(V3 u, V3 E, V3 B, Real eps) {
    const V3 um = u + eps * E;
    const Real g = Kokkos::sqrt(Real(1) + dot(um, um));
    const V3 t = (eps / g) * B;
    const V3 s = (Real(2) / (Real(1) + dot(t, t))) * t;
    const V3 up = um + cross(um + cross(um, t), s);
    return up + eps * E;
}

KOKKOS_INLINE_FUNCTION V3 vay(V3 u, V3 E, V3 B, Real eps) {
    const Real g0 = Kokkos::sqrt(Real(1) + dot(u, u));
    // u' = u^{n-1/2} + eps (E + v^{n-1/2} x B) + eps E ;  then u = u' + (u/gamma) x (eps B)
    const V3 up = u + (Real(2) * eps) * E + (eps / g0) * cross(u, B);
    return solve_implicit(up, eps * B);
}

KOKKOS_INLINE_FUNCTION V3 higuera_cary(V3 u, V3 E, V3 B, Real eps) {
    const V3 um = u + eps * E;
    const V3 tau = eps * B;
    const Real ustar = dot(um, tau);
    const Real sigma = Real(1) + dot(um, um) - dot(tau, tau);
    const Real gp = Kokkos::sqrt(Real(0.5) * (sigma + Kokkos::sqrt(sigma * sigma + Real(4) * (dot(tau, tau) + ustar * ustar))));
    const V3 t = (Real(1) / gp) * tau;
    const Real s = Real(1) / (Real(1) + dot(t, t));
    const V3 upl = s * (um + dot(um, t) * t + cross(um, t));
    return upl + eps * E + cross(upl, t);
}

// note eqs. (19)-(36):  ubar = a + (ubar/gammabar) x b,  a = u0 + eps E,  b = eps B,  u = 2 ubar - u0
KOKKOS_INLINE_FUNCTION V3 imp(V3 u, V3 E, V3 B, Real eps) {
    const V3 ubar = solve_implicit(u + eps * E, eps * B);
    return Real(2) * ubar - u;
}

// note eqs. (2)-(18):  dt = 2 eps/qm is not needed explicitly; nu_dt = nu * dt.
//   a = [2 u0 + 4 eps E + 2 eps v0 x B] / (2 (1 + nu dt)),  b = eps B / (1 + nu dt)
KOKKOS_INLINE_FUNCTION V3 imp_rr(V3 u, V3 E, V3 B, Real eps, Real nu_dt) {
    const Real g0 = Kokkos::sqrt(Real(1) + dot(u, u));
    const Real f = Real(1) / (Real(1) + nu_dt);
    const V3 a = f * (u + (Real(2) * eps) * E + (eps / g0) * cross(u, B));
    return solve_implicit(a, (f * eps) * B);
}

// Landau-Lifshitz leading-order damping rate (normalised units, t in 1/omega_p, charge Z e,
// mass M m_e, qm = Z/M):
//   du/dt|_RR = -(2/3) r_e k_p (Z^4/M^3) gamma^2 [ (E + v x B)^2 - (v.E)^2 ] v  ==  -nu u
//   nu = rr * qm^2 * gamma [ ... ],   rr = (2/3) r_e k_p Z^2/M   (the caller supplies rr)
KOKKOS_INLINE_FUNCTION Real rr_rate(V3 u, V3 E, V3 B, Real qm, Real rr) {
    const Real g = Kokkos::sqrt(Real(1) + dot(u, u));
    const V3 v = (Real(1) / g) * u;
    const V3 F = E + cross(v, B);
    const Real ve = dot(v, E);
    Real w = dot(F, F) - ve * ve;
    if (w < Real(0)) w = Real(0);
    return rr * qm * qm * g * w;
}

} // namespace push
} // namespace quarz
