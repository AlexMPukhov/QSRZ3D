// Variable-step Adams-Bashforth coefficients (adaptive sub-slicing of the plasma push):
//  1. uniform history points reproduce the classical coefficients of make_ab;
//  2. for non-uniform points the step integrates a polynomial of degree order-1 exactly;
//  3. the coefficients sum to 1 (constant right-hand side -> Euler step).
#include "PlasmaSpecies.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

using namespace quarz;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    int fails = 0;
    double worst = 0;
    for (int p = 1; p <= 5; ++p) {
        const double h = 0.01;
        double t[5];
        for (int m = 0; m < p; ++m) t[m] = 3.0 - m * h;
        const ABCoeffs u = make_ab(p), v = make_ab_variable(p, t, h);
        for (int m = 0; m < p; ++m) worst = std::max(worst, std::abs(u.c[m] - v.c[m]));
        // non-uniform history (steps h, 3h/2, h/4, ...), new step 0.7 h: integrate f(t) = sum a_k t^k
        double tn[5] = {1.0, 1.0 - h, 1.0 - 2.5 * h, 1.0 - 2.75 * h, 1.0 - 3.5 * h};
        const double hn = 0.7 * h;
        const ABCoeffs w = make_ab_variable(p, tn, hn);
        double sum = 0, num = 0, exact = 0;
        const double a[5] = {0.3, -1.2, 2.0, 0.7, -0.4};
        for (int m = 0; m < p; ++m) {
            sum += w.c[m];
            double f = 0;
            for (int k = 0; k < p; ++k) f += a[k] * std::pow(tn[m], k);
            num += hn * w.c[m] * f;
        }
        for (int k = 0; k < p; ++k) exact += a[k] * (std::pow(tn[0] + hn, k + 1) - std::pow(tn[0], k + 1)) / (k + 1);
        const double e = std::abs(num - exact) / std::abs(exact);
        if (e > 1e-11 || std::abs(sum - 1) > 1e-12) {
            std::printf("order %d: polynomial error %.2e, sum of coefficients - 1 = %.2e  FAIL\n", p, e, sum - 1);
            ++fails;
        } else {
            std::printf("order %d: polynomial of degree %d integrated to %.1e, sum - 1 = %.1e\n", p, p - 1, e, sum - 1);
        }
    }
    std::printf("uniform points vs classical coefficients: max difference %.2e\n", worst);
    if (worst > 1e-11) ++fails;   // round-off of the Lagrange products (differences ~h)
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
