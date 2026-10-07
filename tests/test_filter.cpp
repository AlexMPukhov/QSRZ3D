// Radial smoothing filter f = (1 - a^2 Laplacian)^{-1} s (plasma.smooth_length), on a stretched grid:
//  1. manufactured solution f = r^n exp(-r^2/w^2), n = 0, 1, 2: second-order convergence;
//  2. n = 0: total charge sum V_j f_j conserved, constants preserved;
//  3. n = 0: point charge on the axis -> 2D Green's function K_0(r/a) / (2 pi a^2).
#include "FieldSolver.hpp"
#include "RadialGrid.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace quarz;

// stretched grid h0 -> hmax with growth factor q (a refinement halves h0, hmax and takes sqrt(q))
static std::vector<Real> nodes(Real h0, Real hmax, Real R, Real q = 1.05) {
    std::vector<Real> r = {0};
    Real h = h0;
    while (r.back() < R - 1e-12) { r.push_back(std::min(R, r.back() + h)); h = std::min(hmax, h * q); }
    return r;
}

static void to_dev(const std::vector<Real>& v, View1D& d) {
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
}
static std::vector<Real> to_host(const View1D& d) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    return std::vector<Real>(h.data(), h.data() + h.extent(0));
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    int fails = 0;
    const Real a = 0.05, w = 0.5, R = 4.0;
    // 1. manufactured solutions
    for (int n = 0; n <= 2; ++n) {
        double err[2];
        for (int lev = 0; lev < 2; ++lev) {
            const Real s = lev ? 0.5 : 1.0;
            RadialGrid g(nodes(0.004 * s, 0.04 * s, R, lev ? std::sqrt(1.1) : 1.1));
            FieldSolver fs(g, TridiagMethod::Thomas);
            fs.set_filter(std::vector<Real>(g.N, a * a));
            std::vector<Real> src(g.N + 1), ex(g.N + 1);
            for (int j = 0; j <= g.N; ++j) {
                const Real r = g.r[j], gg = std::exp(-r * r / (w * w)), f = std::pow(r, n) * gg;
                ex[j] = f;
                src[j] = f - a * a * f * (4 * r * r / (w * w * w * w) - 4.0 * (n + 1) / (w * w));
            }
            View1D d("f", g.N + 1);
            to_dev(src, d);
            fs.filter(n, d);
            auto f = to_host(d);
            double e = 0;
            for (int j = 0; j <= g.N; ++j) e = std::max(e, std::abs(f[j] - ex[j]));
            err[lev] = e;
        }
        const double order = std::log2(err[0] / err[1]);
        const bool ok = order > 1.8 && err[1] < 1e-4;
        std::printf("n = %d: max error %.2e -> %.2e, order %.2f  %s\n", n, err[0], err[1], order, ok ? "" : "FAIL");
        fails += !ok;
    }
    RadialGrid g(nodes(0.0005, 0.02, R));
    FieldSolver fs(g, TridiagMethod::Thomas);
    fs.set_filter(std::vector<Real>(g.N, a * a));
    // 2. conservation and constants
    {
        std::mt19937 rng(1);
        std::uniform_real_distribution<double> u(-1, 1);
        std::vector<Real> s(g.N + 1), one(g.N + 1, 1.0);
        for (auto& v : s) v = u(rng);
        View1D d("s", g.N + 1), c("c", g.N + 1);
        to_dev(s, d); to_dev(one, c);
        fs.filter(0, d); fs.filter(0, c);
        auto f = to_host(d), fc = to_host(c);
        double q0 = 0, q1 = 0, ce = 0;
        for (int j = 0; j <= g.N; ++j) { q0 += g.V[j] * s[j]; q1 += g.V[j] * f[j]; ce = std::max(ce, std::abs(fc[j] - 1)); }
        const double e = std::abs(q1 - q0) / std::abs(q0);
        const bool ok = e < 1e-12 && ce < 1e-12;
        std::printf("n = 0: charge conserved to %.1e, constant preserved to %.1e  %s\n", e, ce, ok ? "" : "FAIL");
        fails += !ok;
    }
    // 3. Green's function
    {
        std::vector<Real> s(g.N + 1, 0.0);
        s[0] = 1.0 / g.V[0];
        View1D d("s", g.N + 1);
        to_dev(s, d);
        fs.filter(0, d);
        auto f = to_host(d);
        double e = 0;
        for (int j = 0; j <= g.N; ++j) {
            const Real r = g.r[j];
            if (r < a || r > 8 * a) continue;
            const double G = std::cyl_bessel_k(0.0, r / a) / (2 * PI * a * a);
            e = std::max(e, std::abs(f[j] - G) / G);
        }
        const bool ok = e < 0.01;
        std::printf("n = 0: point charge vs K0(r/a)/(2 pi a^2), a <= r <= 8a: max rel. error %.2e  %s\n", e, ok ? "" : "FAIL");
        fails += !ok;
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
