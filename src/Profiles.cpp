#include "Profiles.hpp"
#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quarz {

Table1D::Table1D(std::vector<std::pair<double, double>> pts) : pts_(std::move(pts)) {
    std::sort(pts_.begin(), pts_.end());
}

double Table1D::operator()(double x) const {
    if (pts_.empty()) return 1.0;
    if (x <= pts_.front().first) return pts_.front().second;
    if (x >= pts_.back().first) return pts_.back().second;
    auto it = std::upper_bound(pts_.begin(), pts_.end(), std::make_pair(x, -1e300),
                               [](const auto& a, const auto& b) { return a.first < b.first; });
    const auto& p1 = *it;
    const auto& p0 = *(it - 1);
    const double t = (x - p0.first) / (p1.first - p0.first);
    return (1 - t) * p0.second + t * p1.second;
}

DensityProfile::DensityProfile(const Config& cfg, const std::string& p) {
    if (read_function(cfg, p + ".density", 3, parser_, {"x", "y", "z"})) {
        parsed_ = true;     // built-in keys (radial, rmax, channel_*, tables) are then not read
        return;
    }
    type_ = cfg.get_string(p + ".radial", "uniform");
    rmax_ = cfg.get_double(p + ".rmax", 1e300);
    if (type_ == "uniform") {
    } else if (type_ == "channel") {
        depth_ = cfg.get_double(p + ".channel_depth");
        rc_ = cfg.get_double(p + ".channel_radius");
    } else if (type_ == "table") {
        rtab_ = Table1D(cfg.get_pairs(p + ".radial_table"));
    } else {
        throw std::runtime_error("profile: unknown " + p + ".radial '" + type_ + "' (uniform|channel|table)");
    }
    if (cfg.has(p + ".z_table")) ztab_ = Table1D(cfg.get_pairs(p + ".z_table"));
}

double DensityProfile::eval_avg(double r, double z) const {
    constexpr int NQ = 16;
    double s = 0;
    for (int q = 0; q < NQ; ++q) s += eval(r * std::cos(2 * PI * q / NQ), r * std::sin(2 * PI * q / NQ), z);
    return s / NQ;
}

double DensityProfile::radial(double r) const {
    if (r > rmax_) return 0.0;
    if (type_ == "channel") return 1.0 + depth_ * (r / rc_) * (r / rc_);
    if (type_ == "table") return std::max(0.0, rtab_(r));
    return 1.0;
}

double DensityProfile::longitudinal(double z) const {
    return ztab_.empty() ? 1.0 : std::max(0.0, ztab_(z));
}

} // namespace quarz
