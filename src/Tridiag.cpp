#include "Tridiag.hpp"

#include <algorithm>

namespace quarz {

TridiagSolver::TridiagSolver(int M, TridiagMethod method) : M_(M) {
    if (method == TridiagMethod::Auto)
        method = exec_is_host() ? TridiagMethod::Thomas : TridiagMethod::Partition;
    if (method == TridiagMethod::Partition && M < 16) method = TridiagMethod::Thomas;
    method_ = method;
    if (method_ == TridiagMethod::Partition) {
        P_ = std::min(256, M / 8);   // chunks of >= 8 rows; reduced system of 2P <= 512 rows
        qa_ = View1D("tridiag.qa", M);
        qc_ = View1D("tridiag.qc", M);
        qd_ = View1D("tridiag.qd", M);
    } else if (method_ == TridiagMethod::Thomas) {
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
    if (method_ == TridiagMethod::Partition) {
        auto qa = qa_;
        auto qc = qc_;
        auto qd = qd_;
        const int P = P_, R = 2 * P_;
        using Team = Kokkos::TeamPolicy<ExecSpace>;
        using Member = Team::member_type;
        using Scratch = Kokkos::View<Real*, ExecSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
        const size_t bytes = 8 * Scratch::shmem_size(R);
        const int ts = exec_is_host() ? std::min(ExecSpace().concurrency(), P) : std::min(256, P);
        Team pol(1, ts);
        pol = pol.set_scratch_size(0, Kokkos::PerTeam(bytes));
        Kokkos::parallel_for("tridiag_partition", pol, KOKKOS_LAMBDA(const Member& team) {
            Scratch ra0(team.team_scratch(0), R), rb0(team.team_scratch(0), R), rc0(team.team_scratch(0), R),
                rd0(team.team_scratch(0), R), ra1(team.team_scratch(0), R), rb1(team.team_scratch(0), R),
                rc1(team.team_scratch(0), R), rd1(team.team_scratch(0), R);
            // 1. per chunk [s, e): every row s < i < e becomes  x_i + qa_i x_s + qc_i x_{e-1} = qd_i
            //    (row e-1: x_{e-1} + qa x_s + qc x_e = qd); the end rows go to the reduced system
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, P), [&](int t) {
                const int s = int((long)t * M / P), e = int((long)(t + 1) * M / P);
                Real r = Real(1) / b(s + 1);
                qa(s + 1) = a(s + 1) * r;
                qc(s + 1) = c(s + 1) * r;
                qd(s + 1) = d(s + 1) * r;
                for (int i = s + 2; i < e; ++i) {
                    const Real ci = (i == M - 1) ? Real(0) : c(i);
                    r = Real(1) / (b(i) - a(i) * qc(i - 1));
                    qd(i) = r * (d(i) - a(i) * qd(i - 1));
                    qa(i) = -r * a(i) * qa(i - 1);
                    qc(i) = r * ci;
                }
                for (int i = e - 3; i > s; --i) {
                    const Real cn = qc(i);
                    qd(i) -= cn * qd(i + 1);
                    qa(i) -= cn * qa(i + 1);
                    qc(i) = -cn * qc(i + 1);
                }
                const Real as = (s == 0) ? Real(0) : a(s);
                ra0(2 * t) = as;
                rb0(2 * t) = b(s) - c(s) * qa(s + 1);
                rc0(2 * t) = -c(s) * qc(s + 1);
                rd0(2 * t) = d(s) - c(s) * qd(s + 1);
                ra0(2 * t + 1) = qa(e - 1);
                rb0(2 * t + 1) = Real(1);
                rc0(2 * t + 1) = qc(e - 1);
                rd0(2 * t + 1) = qd(e - 1);
            });
            team.team_barrier();
            // 2. PCR on the 2P chunk-end unknowns
            bool flip = false;
            for (int st = 1; st < R; st *= 2) {
                const Scratch& sa = flip ? ra1 : ra0; const Scratch& sb = flip ? rb1 : rb0;
                const Scratch& sc = flip ? rc1 : rc0; const Scratch& sd = flip ? rd1 : rd0;
                const Scratch& da = flip ? ra0 : ra1; const Scratch& db = flip ? rb0 : rb1;
                const Scratch& dc = flip ? rc0 : rc1; const Scratch& dd = flip ? rd0 : rd1;
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, R), [&](int i) {
                    const bool lo = (i - st >= 0), hi = (i + st < R);
                    const Real alpha = lo ? -sa(i) / sb(i - st) : Real(0);
                    const Real gamma = hi ? -sc(i) / sb(i + st) : Real(0);
                    da(i) = lo ? alpha * sa(i - st) : Real(0);
                    dc(i) = hi ? gamma * sc(i + st) : Real(0);
                    db(i) = sb(i) + (lo ? alpha * sc(i - st) : Real(0)) + (hi ? gamma * sa(i + st) : Real(0));
                    dd(i) = sd(i) + (lo ? alpha * sd(i - st) : Real(0)) + (hi ? gamma * sd(i + st) : Real(0));
                });
                team.team_barrier();
                flip = !flip;
            }
            const Scratch& fb = flip ? rb1 : rb0;
            const Scratch& fd = flip ? rd1 : rd0;
            // 3. recover the interior of every chunk
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, P), [&](int t) {
                const int s = int((long)t * M / P), e = int((long)(t + 1) * M / P);
                const Real xs = fd(2 * t) / fb(2 * t), xe = fd(2 * t + 1) / fb(2 * t + 1);
                x(s) = xs;
                x(e - 1) = xe;
                for (int i = s + 1; i < e - 1; ++i) x(i) = qd(i) - qa(i) * xs - qc(i) * xe;
            });
        });
        return;
    }
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

} // namespace quarz
