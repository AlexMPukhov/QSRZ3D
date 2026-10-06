#include "FieldSolver.hpp"

#include <cmath>
#include <vector>

namespace quarz {

namespace {
View1D to_device(const char* name, const std::vector<Real>& v) {
    View1D d(std::string(name), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
}
} // namespace

FieldSolver::FieldSolver(const RadialGrid& g, TridiagMethod method) : grid_(g), tri_(g.N + 1, method) {
    const int N = g.N;
    const int M = N + 1;

    // ---- n = 0: lumped-mass FEM, rows 0..N-1, Dirichlet at N
    {
        std::vector<Real> a(M, 0), b(M, 0), c(M, 0);
        for (int j = 0; j < N; ++j) {
            const Real invV = 2 * PI / g.V[j];
            a[j] = (j > 0) ? g.rmid[j - 1] / g.h[j - 1] * invV : 0.0;
            c[j] = g.rmid[j] / g.h[j] * invV;
            b[j] = -(a[j] + c[j]);
        }
        b[N] = 1.0;
        L0_.a = to_device("L0.a", a); L0_.b = to_device("L0.b", b); L0_.c = to_device("L0.c", c);
        L0_.axis_regular = true;
    }

    // ---- n >= 1: finite volume  L_n f = r^{n-1} (D_{j+1/2} - D_{j-1/2}) / dV,
    //      D_{j+1/2} = (r_{j+1}^n f_{j+1} - r_j^n f_j) / (h_j r_{j+1/2}^{2n-1})
    auto build_fv = [&](int n, bool flux, Op& op, const char* tag) {
        std::vector<Real> a(M, 0), b(M, 0), c(M, 0);
        b[0] = 1.0;   // f(0) = 0
        auto kp = [&](int j) { return 1.0 / (g.h[j] * std::pow(g.rmid[j], 2 * n - 1)); };
        for (int j = 1; j < N; ++j) {
            const Real dV = 0.5 * (g.h[j] + g.h[j - 1]);
            const Real pre = std::pow(g.r[j], n - 1) / dV;
            c[j] = pre * std::pow(g.r[j + 1], n) * kp(j);
            a[j] = pre * std::pow(g.r[j - 1], n) * kp(j - 1);
            b[j] = -pre * std::pow(g.r[j], n) * (kp(j) + kp(j - 1));
        }
        if (flux) {   // (D_R - D_{N-1/2}) / (h_{N-1}/2) = rhs,  D_R = g  (n = 1 only)
            const Real dV = 0.5 * g.h[N - 1];
            const Real k = kp(N - 1) / dV;
            a[N] = g.r[N - 1] * k;
            b[N] = -g.r[N] * k;
            flux_coef_ = 1.0 / dV;
        } else {
            b[N] = 1.0;
        }
        std::string s(tag);
        op.a = to_device((s + ".a").c_str(), a);
        op.b = to_device((s + ".b").c_str(), b);
        op.c = to_device((s + ".c").c_str(), c);
        op.flux = flux;
        op.axis_regular = false;
    };
    build_fv(1, false, L1D_, "L1D");
    build_fv(1, true, L1F_, "L1F");
    build_fv(2, false, L2D_, "L2D");

    // ---- gradient stencils
    std::vector<Real> gm(M, 0), g0(M, 0), gp(M, 0), ir(M, 0);
    for (int j = 1; j < N; ++j) {
        const Real hm = g.h[j - 1], hp = g.h[j];
        const Real den = hm * hp * (hm + hp);
        gp[j] = hm * hm / den;
        gm[j] = -hp * hp / den;
        g0[j] = (hp * hp - hm * hm) / den;
    }
    {
        const Real h1 = g.h[N - 1], h2 = (N >= 2) ? g.h[N - 2] : g.h[N - 1];
        const Real s = h1 + h2;
        g0[N] = (2 * h1 + h2) / (h1 * s);
        gm[N] = -s / (h1 * h2);
        gp[N] = h1 / (h2 * s);
    }
    for (int j = 1; j <= N; ++j) ir[j] = 1.0 / g.r[j];
    ir[0] = 1.0 / g.r[1];
    gc_m_ = to_device("fs.gm", gm); gc_0_ = to_device("fs.g0", g0); gc_p_ = to_device("fs.gp", gp);
    inv_r_ = to_device("fs.ir", ir);
    bw_ = View1D("fs.bw", M);
    dw_ = View1D("fs.dw", M);
}

const FieldSolver::Op& FieldSolver::op(OpKind k) const {
    switch (k) {
        case OpKind::L0: return L0_;
        case OpKind::L1D: return L1D_;
        case OpKind::L1F: return L1F_;
        default: return L2D_;
    }
}

void FieldSolver::solve(OpKind k, const View1D& rhs, const View1D& f, const View1D& chi, const View1D& wall,
                        Real scale) const {
    const Op& o = op(k);
    const int N = grid_.N;
    auto b0 = o.b;
    auto bw = bw_;
    auto dw = dw_;
    const bool has_chi = chi.extent(0) > 0;
    const bool flux = o.flux;
    const bool regular = o.axis_regular;
    const Real fc = flux_coef_;
    Kokkos::parallel_for("fs.prep", Range(0, N + 1), KOKKOS_LAMBDA(int j) {
        const bool dirichlet_row = (j == N && !flux) || (j == 0 && !regular);
        if (dirichlet_row) { bw(j) = b0(j); dw(j) = Real(0); return; }
        bw(j) = has_chi ? b0(j) - chi(j) : b0(j);
        dw(j) = scale * rhs(j);
        if (j == N && flux) dw(j) -= fc * wall(N);
    });
    tri_.solve(o.a, bw, o.c, dw, f);
}

void FieldSolver::gradient(const View1D& f, const View1D& df) const {
    const int N = grid_.N;
    auto gm = gc_m_; auto g0 = gc_0_; auto gp = gc_p_;
    Kokkos::parallel_for("fs.grad", Range(0, N + 1), KOKKOS_LAMBDA(int j) {
        if (j == 0) { df(0) = Real(0); return; }
        if (j == N) { df(N) = g0(N) * f(N) + gm(N) * f(N - 1) + gp(N) * f(N - 2); return; }
        df(j) = gm(j) * f(j - 1) + g0(j) * f(j) + gp(j) * f(j + 1);
    });
}

void FieldSolver::gradient_odd(const View1D& f, const View1D& df) const {
    const int N = grid_.N;
    auto gm = gc_m_; auto g0 = gc_0_; auto gp = gc_p_; auto ir = inv_r_;
    Kokkos::parallel_for("fs.grad_odd", Range(0, N + 1), KOKKOS_LAMBDA(int j) {
        if (j == 0) { df(0) = f(1) * ir(0); return; }
        if (j == N) { df(N) = g0(N) * f(N) + gm(N) * f(N - 1) + gp(N) * f(N - 2); return; }
        df(j) = gm(j) * f(j - 1) + g0(j) * f(j) + gp(j) * f(j + 1);
    });
}

void FieldSolver::over_r(const View1D& f, const View1D& out) const {
    const int N = grid_.N;
    auto ir = inv_r_;
    Kokkos::parallel_for("fs.over_r", Range(0, N + 1), KOKKOS_LAMBDA(int j) {
        out(j) = (j == 0 ? f(1) : f(j)) * ir(j);
    });
}

} // namespace quarz
