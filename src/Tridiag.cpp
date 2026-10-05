#include "Tridiag.hpp"

namespace qsrz {

TridiagSolver::TridiagSolver(int M, TridiagMethod method) : M_(M) {
    if (method == TridiagMethod::Auto) method = exec_is_host() ? TridiagMethod::Thomas : TridiagMethod::PCR;
    method_ = method;
    if (method_ == TridiagMethod::Thomas) {
        cp_ = View1D("tridiag.cp", M);
        dp_ = View1D("tridiag.dp", M);
    } else {
        pa_ = View2D("tridiag.pa", 2, M);
        pb_ = View2D("tridiag.pb", 2, M);
        pc_ = View2D("tridiag.pc", 2, M);
        pd_ = View2D("tridiag.pd", 2, M);
    }
}

void TridiagSolver::solve(const View1D& a, const View1D& b, const View1D& c, const View1D& d,
                          const View1D& x) const {
    const int M = M_;
    if (method_ == TridiagMethod::Thomas) {
        auto cp = cp_;
        auto dp = dp_;
        Kokkos::parallel_for("tridiag_thomas", Range(0, 1), KOKKOS_LAMBDA(int) {
            cp(0) = c(0) / b(0);
            dp(0) = d(0) / b(0);
            for (int i = 1; i < M; ++i) {
                const Real m = Real(1) / (b(i) - a(i) * cp(i - 1));
                cp(i) = c(i) * m;
                dp(i) = (d(i) - a(i) * dp(i - 1)) * m;
            }
            x(M - 1) = dp(M - 1);
            for (int i = M - 2; i >= 0; --i) x(i) = dp(i) - cp(i) * x(i + 1);
        });
        return;
    }

    // ---- parallel cyclic reduction, one team ----
    auto pa = pa_;
    auto pb = pb_;
    auto pc = pc_;
    auto pd = pd_;
    using Team = Kokkos::TeamPolicy<ExecSpace>;
    using Member = Team::member_type;
    Kokkos::parallel_for("tridiag_pcr", Team(1, Kokkos::AUTO), KOKKOS_LAMBDA(const Member& team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](int i) {
            pa(0, i) = a(i);
            pb(0, i) = b(i);
            pc(0, i) = c(i);
            pd(0, i) = d(i);
        });
        team.team_barrier();
        int src = 0;
        for (int s = 1; s < M; s *= 2) {
            const int dst = 1 - src;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](int i) {
                const bool lo = (i - s >= 0), hi = (i + s < M);
                const Real alpha = lo ? -pa(src, i) / pb(src, i - s) : Real(0);
                const Real gamma = hi ? -pc(src, i) / pb(src, i + s) : Real(0);
                pa(dst, i) = lo ? alpha * pa(src, i - s) : Real(0);
                pc(dst, i) = hi ? gamma * pc(src, i + s) : Real(0);
                pb(dst, i) = pb(src, i) + (lo ? alpha * pc(src, i - s) : Real(0)) +
                             (hi ? gamma * pa(src, i + s) : Real(0));
                pd(dst, i) = pd(src, i) + (lo ? alpha * pd(src, i - s) : Real(0)) +
                             (hi ? gamma * pd(src, i + s) : Real(0));
            });
            team.team_barrier();
            src = dst;
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](int i) { x(i) = pd(src, i) / pb(src, i); });
    });
}

} // namespace qsrz
