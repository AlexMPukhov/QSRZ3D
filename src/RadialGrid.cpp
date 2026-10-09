#include "RadialGrid.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace quarz {

// ---------------------------------------------------------------------------
std::vector<Real> RadialGrid::make_uniform(Real R, Real dr) {
    if (R <= 0 || dr <= 0) throw std::runtime_error("grid: R and dr must be positive");
    const int n = std::max(1, static_cast<int>(std::lround(R / dr)));
    std::vector<Real> nodes(n + 1);
    for (int j = 0; j <= n; ++j) nodes[j] = R * Real(j) / Real(n);
    return nodes;
}

// Region-based grid: regions = {(r_end_0, h_0), (r_end_1, h_1), ...}
// Target spacing h_i is used for r < r_end_i. Between regions the spacing
// changes geometrically with ratio <= max_ratio per cell. If an inner region
// is coarser than a following one, the grid starts refining early enough so
// that the fine spacing is reached at the region boundary.
std::vector<Real> RadialGrid::make_regions(Real R, const std::vector<std::pair<double, double>>& regions,
                                           Real max_ratio) {
    if (regions.empty()) throw std::runtime_error("grid: grid.regions is empty");
    if (max_ratio < 1.0) throw std::runtime_error("grid: grid.max_ratio must be >= 1");
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].second <= 0) throw std::runtime_error("grid: region spacing must be > 0");
        if (i > 0 && regions[i].first <= regions[i - 1].first)
            throw std::runtime_error("grid: region end radii must increase");
    }
    const Real g = max_ratio - 1.0;

    auto target = [&](Real x) {
        Real ht = regions.back().second;
        for (const auto& reg : regions)
            if (x < reg.first) { ht = reg.second; break; }
        // look ahead: do not arrive at a finer region with too coarse a spacing
        Real start = 0;
        for (const auto& reg : regions) {
            if (start > x && g > 0) ht = std::min(ht, Real(reg.second + g * (start - x)));
            if (start > x && g == 0) ht = std::min(ht, Real(reg.second));
            start = reg.first;
        }
        return ht;
    };

    std::vector<Real> nodes{0.0};
    Real x = 0, hprev = target(0);
    while (x < R * (1 - 1e-12)) {
        Real ht = target(x + 0.5 * hprev);
        Real hnew = std::clamp(ht, hprev / max_ratio, hprev * max_ratio);
        if (nodes.size() == 1) hnew = target(0);
        x += hnew;
        nodes.push_back(x);
        hprev = hnew;
        if (nodes.size() > 10000000) throw std::runtime_error("grid: too many nodes");
    }
    // end exactly at R: keep the node closest to R and rescale the whole grid by the
    // (tiny) factor R/r_last; this preserves all neighbour ratios.
    const size_t n = nodes.size();
    if (n >= 3 && (nodes[n - 1] - R) > (R - nodes[n - 2])) nodes.pop_back();
    const Real scale = R / nodes.back();
    for (auto& x : nodes) x *= scale;
    nodes.back() = R;
    return nodes;
}

std::vector<Real> RadialGrid::read_file(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("grid: cannot open grid file '" + filename + "'");
    std::vector<Real> nodes;
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        nodes.push_back(std::stod(line));
    }
    return nodes;
}

// ---------------------------------------------------------------------------
RadialGrid::RadialGrid(const Config& cfg) {
    const std::string type = cfg.get_string("grid.type", "uniform");
    std::vector<Real> nodes;
    if (type == "uniform") {
        nodes = make_uniform(cfg.get_double("grid.rmax"), cfg.get_double("grid.dr"));
    } else if (type == "regions" || type == "stretched") {   // "stretched": alias
        nodes = make_regions(cfg.get_double("grid.rmax"), cfg.get_pairs("grid.regions"),
                             cfg.get_double("grid.max_ratio", 1.05));
    } else if (type == "file") {
        nodes = read_file(cfg.get_string("grid.file"));
    } else {
        throw std::runtime_error("grid: unknown grid.type '" + type + "' (uniform|regions|file)");
    }
    finalize(nodes);
}

RadialGrid::RadialGrid(const std::vector<Real>& nodes) { finalize(nodes); }

void RadialGrid::finalize(const std::vector<Real>& nodes) {
    if (nodes.size() < 3) throw std::runtime_error("grid: need at least 2 cells");
    if (std::abs(nodes[0]) > 0) throw std::runtime_error("grid: first node must be r=0");
    for (size_t j = 1; j < nodes.size(); ++j)
        if (!(nodes[j] > nodes[j - 1])) throw std::runtime_error("grid: nodes must be strictly increasing");

    r = nodes;
    N = static_cast<int>(r.size()) - 1;
    R = r[N];
    h.resize(N);
    rmid.resize(N);
    for (int j = 0; j < N; ++j) {
        h[j] = r[j + 1] - r[j];
        rmid[j] = 0.5 * (r[j + 1] + r[j]);
    }
    // hat-function volumes: left part  Int_{r_{j-1}}^{r_j} (r-r_{j-1})/h r dr = r_{j-1} h/2 + h^2/3
    //                       right part Int_{r_j}^{r_{j+1}} (r_{j+1}-r)/h r dr = r_j h/2 + h^2/6
    V.assign(N + 1, 0.0);
    for (int j = 0; j <= N; ++j) {
        Real v = 0;
        if (j > 0)  v += r[j - 1] * h[j - 1] / 2 + h[j - 1] * h[j - 1] / 3;
        if (j < N)  v += r[j] * h[j] / 2 + h[j] * h[j] / 6;
        V[j] = 2 * PI * v;
    }

    // device copies
    d.N = N;
    d.R = R;
    d.r = View1D("grid.r", N + 1);
    d.h = View1D("grid.h", N);
    d.inv_h = View1D("grid.inv_h", N);
    d.inv_V = View1D("grid.inv_V", N + 1);
    auto hr = Kokkos::create_mirror_view(d.r);
    auto hh = Kokkos::create_mirror_view(d.h);
    auto hih = Kokkos::create_mirror_view(d.inv_h);
    auto hiv = Kokkos::create_mirror_view(d.inv_V);
    for (int j = 0; j <= N; ++j) { hr(j) = r[j]; hiv(j) = 1.0 / V[j]; }
    for (int j = 0; j < N; ++j) { hh(j) = h[j]; hih(j) = 1.0 / h[j]; }
    Kokkos::deep_copy(d.r, hr);
    Kokkos::deep_copy(d.h, hh);
    Kokkos::deep_copy(d.inv_h, hih);
    Kokkos::deep_copy(d.inv_V, hiv);
}

Real RadialGrid::hmin() const { return *std::min_element(h.begin(), h.end()); }
Real RadialGrid::hmax() const { return *std::max_element(h.begin(), h.end()); }
Real RadialGrid::max_ratio() const {
    Real m = 1;
    for (int j = 1; j < N; ++j) m = std::max(m, std::max(h[j] / h[j - 1], h[j - 1] / h[j]));
    return m;
}

void RadialGrid::print_summary(std::ostream& os) const {
    os << "Radial grid: N = " << N << " cells, R = " << R << ", h_min = " << hmin() << " (at axis h_0 = "
       << h[0] << "), h_max = " << hmax() << ", max neighbour ratio = " << max_ratio() << "\n";
    if (max_ratio() > 1.25)
        os << "  WARNING: neighbouring cells differ by more than 25%; accuracy degrades to first order "
              "at such jumps. Use grid.max_ratio <= 1.1.\n";
}

void RadialGrid::write(const std::string& filename) const {
    std::ofstream out(filename);
    out << "# j  r_j  h_j  V_j\n" << std::setprecision(16);
    for (int j = 0; j <= N; ++j)
        out << j << " " << r[j] << " " << (j < N ? h[j] : 0.0) << " " << V[j] << "\n";
}

} // namespace quarz
