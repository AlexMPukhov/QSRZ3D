// Single-particle comparison of the momentum pushers in src/Pushers.hpp:
//   Boris, Vay, Higuera-Cary, and the two schemes of A. Pukhov's note
//   "Implicit particle pusher including radiation reaction":
//     imp    = eqs. (19)-(36)  implicit midpoint in u
//     imp_rr = eqs. (2)-(18)   trapezoidal velocity average + implicit RR damping
#include "Pushers.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace qsrz;
using namespace qsrz::push;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}
static double norm(V3 a) { return std::sqrt(dot(a, a)); }

enum Kind { BORIS, VAY, HC, IMP, IMPRR0 };
static const char* NAMES[] = {"Boris ", "Vay   ", "HC    ", "IMP   ", "IMP_RR"};
static V3 step(Kind k, V3 u, V3 E, V3 B, double eps) {
    switch (k) {
        case BORIS: return boris(u, E, B, eps);
        case VAY: return vay(u, E, B, eps);
        case HC: return higuera_cary(u, E, B, eps);
        case IMP: return imp(u, E, B, eps);
        default: return imp_rr(u, E, B, eps, 0.0);
    }
}

// 1. algebraic identities on random data spanning gamma ~ 1 .. 1e6, fields up to 1e4
static void identities() {
    std::printf("1. Algebraic identities (1e6 random u, E, B)\n");
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(-1, 1), L(-3, 6);
    double d_hc = 0, d_vay = 0;
    for (int n = 0; n < 1000000; ++n) {
        const double su = std::pow(10.0, L(rng)), sf = std::pow(10.0, L(rng) - 2);
        V3 u{su * U(rng), su * U(rng), su * U(rng)};
        V3 E{sf * U(rng), sf * U(rng), sf * U(rng)}, B{sf * U(rng), sf * U(rng), sf * U(rng)};
        const double eps = 0.5 * std::pow(10.0, L(rng) - 3);
        const V3 a = imp(u, E, B, eps), b = higuera_cary(u, E, B, eps);
        const V3 c = imp_rr(u, E, B, eps, 0.0), d = vay(u, E, B, eps);
        const double sc = 1.0 + norm(u) + 2 * eps * (norm(E) + norm(B));
        d_hc = std::max(d_hc, norm(a - b) / sc);
        d_vay = std::max(d_vay, norm(c - d) / sc);
    }
    char m[160];
    std::snprintf(m, sizeof m, "note eqs.(19)-(36) == Higuera-Cary        max rel. diff %.1e", d_hc);
    check(d_hc < 1e-12, m);
    std::snprintf(m, sizeof m, "note eqs.(2)-(18) with nu=0 == Vay (2008) max rel. diff %.1e", d_vay);
    check(d_vay < 1e-12, m);
}

// 2. uniform B: gyration phase error and |u| conservation
static void gyration() {
    std::printf("2. Gyration in uniform B (gamma = 10, 64 steps per exact period, 100 periods)\n");
    const double Bz = 1.0, g = 10.0, uperp = std::sqrt(g * g - 1);
    const double omega = Bz / g, T = 2 * M_PI / omega, dt = T / 64;
    for (Kind k : {BORIS, VAY, HC, IMP}) {
        V3 u{uperp, 0, 0};
        const int N = 64 * 100;
        for (int n = 0; n < N; ++n) u = step(k, u, {0, 0, 0}, {0, 0, Bz}, -0.5 * dt);   // electron q=-1
        const double ph = std::atan2(u.y, u.x);
        std::printf("    %s  phase error after 100 periods %+.4f rad,  | |u|-|u0| |/|u0| = %.1e\n", NAMES[k], ph,
                    std::abs(norm(u) - uperp) / uperp);
    }
}

// 3. force-free motion E + v x B = 0 (relativistic E x B drift / beam in its own field)
static void force_free() {
    std::printf("3. Force-free particle, E = v_d B (gamma_d = 2e4, B = 60, dt = 20: tau = B dt/gamma ~ 0.06)\n");
    const double gd = 2e4, vd = std::sqrt(1 - 1 / (gd * gd)), Bm = 60.0, dt = 20.0;
    const V3 E{vd * Bm, 0, 0}, B{0, Bm, 0};  // E_x = v_z B_y  ->  E + v x B = 0 for v = vd z
    for (Kind k : {BORIS, VAY, HC, IMP, IMPRR0}) {
        V3 u{0, 0, gd * vd};
        for (int n = 0; n < 1000; ++n) u = step(k, u, E, B, -0.5 * dt);
        std::printf("    %s  spurious transverse momentum after 1000 steps: %.2e  (gamma change %.2e)\n", NAMES[k],
                    std::abs(u.x), std::sqrt(1 + dot(u, u)) - gd);
        if (k != BORIS) check(std::abs(u.x) < 1e-6, "  force-free drift preserved");
    }
}

// 4. betatron oscillation of a gamma = 2e4 electron in an ion channel plus a huge
//    self-field pair E_x = S x, B_y = S x (cancels to 1/(2 gamma^2)), as for a pinched witness
static void betatron_selffield() {
    std::printf("4. Betatron in ion channel + self-field pair (gamma = 2e4, S = 1e4), 20 periods\n");
    const double g0 = 2e4, S = 1e4, x0 = 0.01;
    for (double spp : {64.0, 16.0}) {
        for (Kind k : {BORIS, VAY, HC, IMP}) {
            // exact: x'' = -(1/2 + S(1-vz)) x / gamma ~ -(1/(2 gamma)) x (S term ~1e-5 relative)
            const double om = std::sqrt((0.5 + S / (2 * g0 * g0)) / g0);
            const double dt = 2 * M_PI / om / spp;
            V3 u{0, 0, std::sqrt(g0 * g0 - 1)};
            double x = x0;
            // leapfrog start: u at -dt/2
            { const V3 E{0.5 * x + S * x, 0, 0}, B{0, S * x, 0};
              const double g = std::sqrt(1 + dot(u, u));
              u = u + (-1.0 * (-0.5 * dt)) * (E + cross((1 / g) * u, B)); }
            const int N = static_cast<int>(20 * spp);
            double xmax_last = 0;
            for (int n = 0; n < N; ++n) {
                const V3 E{0.5 * x + S * x, 0, 0}, B{0, S * x, 0};
                u = step(k, u, E, B, -0.5 * dt);
                x += u.x / std::sqrt(1 + dot(u, u)) * dt;
                if (n >= N - spp) xmax_last = std::max(xmax_last, std::abs(x));
            }
            // exact after 20 periods: x = x0
            std::printf("    %3.0f steps/period  %s  x(20 T)/x0 = %+.5f   amplitude in last period / x0 = %.5f\n", spp,
                        NAMES[k], x / x0, xmax_last / x0);
        }
    }
}

// 5. radiation reaction (imp_rr, eqs. (2)-(18)): energy loss of a betatron-oscillating electron
static void radiation_reaction() {
    std::printf("5. Radiation reaction, imp_rr: betatron electron gamma0 = 1e4, r_beta = 0.1, rr = 2e-7\n");
    const double g0 = 1e4, xb = 0.1, rr = 2e-7;
    const double om = 1 / std::sqrt(2 * g0), T = 2 * M_PI / om;
    for (double spp : {32.0, 8.0}) {
        const double dt = T / spp;
        V3 u{0, 0, std::sqrt(g0 * g0 - 1)};
        double x = xb, loss_expected = 0;
        auto H = [&](double xx, V3 uu) { return std::sqrt(1 + dot(uu, uu)) + 0.25 * xx * xx; };  // conserved w/o RR
        const double H0 = H(x, u);
        const int N = static_cast<int>(10 * spp);   // 10 periods
        for (int n = 0; n < N; ++n) {
            const V3 E{0.5 * x, 0, 0}, B{0, 0, 0};
            const double nu = rr_rate(u, E, B, -1.0, rr);
            const double g = std::sqrt(1 + dot(u, u));
            loss_expected += nu * (dot(u, u) / g) * dt;          // -dgamma/dt = nu u.v
            u = imp_rr(u, E, B, 0.5 * (-1.0) * dt, nu * dt);
            x += u.x / std::sqrt(1 + dot(u, u)) * dt;
        }
        const double loss = H0 - H(x, u);
        const double analytic = rr * g0 * g0 * xb * xb / 8.0 * N * dt;   // <dgamma/dt> = rr gamma^2 r_b^2/8
        std::printf("    %2.0f steps/period: Delta H = %.4f,  sum nu u.v dt = %.4f,  analytic (1/12) r_e k_p g^2 r_b^2 t = %.4f\n",
                    spp, loss, loss_expected, analytic);
        char m[128];
        std::snprintf(m, sizeof m, "  energy loss consistent with LL rate (%.1f%%)", 100 * std::abs(loss / loss_expected - 1));
        check(std::abs(loss / loss_expected - 1) < 0.02, m);
    }
}

int main() {
    identities();
    gyration();
    force_free();
    betatron_selffield();
    radiation_reaction();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
