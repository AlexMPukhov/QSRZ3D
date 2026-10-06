// Unit tests: tridiagonal solvers, Laplacian and vector-Laplacian operators
// on uniform and non-uniform grids (manufactured solutions), deposition of a
// uniform plasma.
#include "Config.hpp"
#include "FieldSolver.hpp"
#include "PlasmaSpecies.hpp"
#include "RadialGrid.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace quarz;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static View1D to_dev(const std::vector<Real>& v) {
    View1D d("v", v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
}
static std::vector<Real> to_host(const View1D& d) {
    auto h = Kokkos::create_mirror_view_and_copy(HostSpace(), d);
    return std::vector<Real>(h.data(), h.data() + h.extent(0));
}

static void test_tridiag() {
    std::printf("Tridiagonal solvers\n");
    const int M = 1537;
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<Real> a(M), b(M), c(M), x(M), d(M);
    for (int i = 0; i < M; ++i) { a[i] = i ? U(rng) : 0; c[i] = i < M - 1 ? U(rng) : 0; b[i] = 2.5 + U(rng); x[i] = U(rng); }
    for (int i = 0; i < M; ++i) d[i] = b[i] * x[i] + (i ? a[i] * x[i - 1] : 0) + (i < M - 1 ? c[i] * x[i + 1] : 0);
    auto A = to_dev(a), B = to_dev(b), C = to_dev(c), D = to_dev(d);
    for (auto m : {TridiagMethod::Thomas, TridiagMethod::PCR}) {
        TridiagSolver ts(M, m);
        View1D X("x", M);
        ts.solve(A, B, C, D, X);
        auto xs = to_host(X);
        double err = 0;
        for (int i = 0; i < M; ++i) err = std::max(err, std::abs(xs[i] - x[i]));
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s max error %.2e", m == TridiagMethod::PCR ? "PCR   " : "Thomas", err);
        check(err < 1e-12, msg);
    }
}

// manufactured solutions (R = 4):
//   L0 : psi = exp(-r^2) - exp(-R^2)                     L0 psi = (4 r^2 - 4) e^{-r^2}
//   L1F: B = r exp(-r^2), chi = 1 + r^2, flux wall        (L1 - chi) B = (4 r^3 - 8 r) e^{-r^2} - chi B
//   Ln (n=1,2, Dirichlet): f = r^n (e^{-r^2} - e^{-R^2}), chi = 1 + r^2
//                          L_n f = r^n e^{-r^2} (4 r^2 - 4 n - 4)
static void errors(const RadialGrid& g, double e[4]) {
    FieldSolver fs(g, TridiagMethod::Auto);
    const int M = g.N + 1;
    const double R = g.R;
    std::vector<Real> rhs(M), chi(M), rhsB(M), jz(M, 0), rhs1(M), rhs2(M);
    for (int j = 0; j < M; ++j) {
        const double r = g.r[j], ex = std::exp(-r * r);
        rhs[j] = (4 * r * r - 4) * ex;
        chi[j] = 1 + r * r;
        rhsB[j] = (4 * r * r * r - 8 * r) * ex - chi[j] * r * ex;
        rhs1[j] = r * ex * (4 * r * r - 8) - chi[j] * r * (ex - std::exp(-R * R));
        rhs2[j] = r * r * ex * (4 * r * r - 12) - chi[j] * r * r * (ex - std::exp(-R * R));
    }
    jz[M - 1] = (2 - 2 * R * R) * std::exp(-R * R);   // (1/r) d(rB)/dr at the wall
    View1D psi("psi", M), B("B", M), f1("f1", M), f2("f2", M);
    fs.solve(OpKind::L0, to_dev(rhs), psi);
    fs.solve(OpKind::L1F, to_dev(rhsB), B, to_dev(chi), to_dev(jz));
    fs.solve(OpKind::L1D, to_dev(rhs1), f1, to_dev(chi));
    fs.solve(OpKind::L2D, to_dev(rhs2), f2, to_dev(chi));
    auto hp = to_host(psi), hb = to_host(B), h1 = to_host(f1), h2 = to_host(f2);
    for (int c = 0; c < 4; ++c) e[c] = 0;
    for (int j = 0; j < M; ++j) {
        const double r = g.r[j], ex = std::exp(-r * r), eR = std::exp(-R * R);
        e[0] = std::max(e[0], std::abs(hp[j] - (ex - eR)));
        e[1] = std::max(e[1], std::abs(hb[j] - r * ex));
        e[2] = std::max(e[2], std::abs(h1[j] - r * (ex - eR)));
        e[3] = std::max(e[3], std::abs(h2[j] - r * r * (ex - eR)));
    }
}

static void test_operators() {
    std::printf("Field operators (manufactured solutions, R = 4): L0 | L1-chi (flux wall) | L1-chi | L2-chi\n");
    const char* nm[4] = {"L0", "L1F", "L1D", "L2D"};
    for (int pass = 0; pass < 2; ++pass) {
        std::printf(pass == 0 ? "  uniform grids:\n" : "  stretched grids (fine at axis, ratio 1.05):\n");
        double prev[4] = {0, 0, 0, 0};
        for (int f : {1, 2, 4, 8}) {
            std::vector<Real> nodes;
            if (pass == 0) nodes = RadialGrid::make_uniform(4.0, 0.04 / f);
            else nodes = RadialGrid::make_regions(4.0, {{0.3, 0.01 / f}, {1.5, 0.05 / f}, {4.0, 0.2 / f}},
                                                  std::pow(1.05, 1.0 / f));
            RadialGrid g(nodes);
            double e[4];
            errors(g, e);
            std::printf("    N=%5d ", g.N);
            for (int c = 0; c < 4; ++c) std::printf(" %s %.2e", nm[c], e[c]);
            if (prev[0] > 0) {
                std::printf("  orders");
                for (int c = 0; c < 4; ++c) std::printf(" %.2f", std::log2(prev[c] / e[c]));
            }
            std::printf("\n");
            if (f == 8)
                for (int c = 0; c < 4; ++c) {
                    char msg[96];
                    std::snprintf(msg, sizeof msg, "%s second order on %s grid (%.2f)", nm[c],
                                  pass ? "stretched" : "uniform", std::log2(prev[c] / e[c]));
                    check(std::log2(prev[c] / e[c]) > 1.7, msg);
                }
            for (int c = 0; c < 4; ++c) prev[c] = e[c];
        }
    }
}

static void test_uniform_deposit() {
    std::printf("Deposition of a uniform plasma on a stretched grid\n");
    Config cfg;
    cfg.set("electrons.ppc", "2");
    auto nodes = RadialGrid::make_regions(3.0, {{0.2, 0.002}, {3.0, 0.05}}, 1.08);
    RadialGrid g(nodes);
    DensityProfile prof;
    PlasmaSpecies e(cfg, "electrons", g, prof, false);
    e.load(0.0, 1);
    MScalar rhot("rhot", g.N + 1), rho("rho", g.N + 1);
    e.deposit_rho(rhot, rho);
    auto h = to_host(rhot.a0);
    double err = 0;
    for (int j = 0; j < g.N; ++j) err = std::max(err, std::abs(h[j] + 1.0));  // interior nodes (wall node is a half cell)
    char msg[128];
    std::snprintf(msg, sizeof(msg), "uniform density reproduced incl. axis node, max |n-1| = %.2e", err);
    check(err < 2e-3, msg);
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);
    test_tridiag();
    test_operators();
    test_uniform_deposit();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
