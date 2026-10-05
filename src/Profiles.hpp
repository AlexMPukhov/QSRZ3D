// Plasma density profiles n(r) * f(z)  (host-side, evaluated at loading).
#pragma once

#include "Parser.hpp"
#include "Types.hpp"
#include <string>
#include <vector>
#include <utility>

namespace qsrz {

class Config;

// piecewise-linear table; constant extrapolation outside
class Table1D {
public:
    Table1D() = default;
    explicit Table1D(std::vector<std::pair<double, double>> pts);
    double operator()(double x) const;
    bool empty() const { return pts_.empty(); }
private:
    std::vector<std::pair<double, double>> pts_;
};

class DensityProfile {
public:
    DensityProfile() = default;
    // reads keys  <prefix>.radial, <prefix>.rmax, <prefix>.channel_*, <prefix>.radial_table, <prefix>.z_table
    DensityProfile(const Config& cfg, const std::string& prefix);

    // parsed profile  <prefix>.density(x,y,z) = "..."  (z = lab position); replaces the built-ins
    bool parsed() const { return parsed_; }
    double eval(double x, double y, double z) const { const double n = parser_(x, y, z); return n > 0 ? n : 0.0; }
    // azimuthal average at radius r (m = 0 part, used by axisymmetric runs)
    double eval_avg(double r, double z) const;

    double radial(double r) const;    // n(r) / n0 at the entrance
    double longitudinal(double z) const;  // f(z)
    double operator()(double r, double z) const { return radial(r) * longitudinal(z); }

private:
    std::string type_ = "uniform";
    double rmax_ = 1e300;       // plasma column radius
    double depth_ = 0;          // channel: n = 1 + depth * (r/rc)^2
    double rc_ = 1;
    Table1D rtab_, ztab_;
    bool parsed_ = false;
    Parser parser_;
};

} // namespace qsrz
