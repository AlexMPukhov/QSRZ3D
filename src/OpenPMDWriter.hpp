// openPMD output (https://github.com/openPMD/openPMD-standard, version 1.1.0) through
// openPMD-api.  Optional: compiled in when CMake finds openPMD-api (QUARZ_USE_OPENPMD);
// otherwise requesting it in the input file is an error.
//
// Fields: geometry "thetaMode", axes (r, z), data order C, shape [modes][r][z] with
//   modes = (f0)  or  (f0, f_c, f_s)  where f = f0 + f_c cos(theta) + f_s sin(theta)
//   (the convention read by openPMD-viewer; geometryParameters "m=1;imag=+" or "m=2;imag=+").
//   The longitudinal axis is the lab-frame z = t - xi, stored in increasing z.
//   openPMD requires a uniform grid spacing.  By default the fields are therefore
//   interpolated (piecewise linear, as the hat functions of the code) onto a uniform radial
//   grid  r = 0, dr, 2dr, ... <= output.openpmd_rmax.  With output.openpmd_grid = native they
//   are written on the native non-uniform nodes, with geometry "other" and the node
//   positions in the mesh attribute "quarz_r_nodes".
// Particles: every beam is a species with position (x, y, z = t - xi), positionOffset (0),
//   momentum, weighting, charge and mass.
// Units: SI (unitSI attributes) when the plasma density is given (units.n0_cm3); without it
//   the data are in the code's normalised units and every unitSI is 1.
//
// MPI: the writes are collective.  In the xi pipeline the ranks reach a given step at
// different times, so an openPMD output step waits until the last rank has finished that
// step (the pipeline drains: about P - 1 block sweeps per output).  For frequent output with
// many ranks the native format, which needs no synchronisation, is cheaper.
#pragma once

#include "FieldTable.hpp"

#include <memory>
#include <string>
#include <vector>

namespace quarz {

class Beam;
class Config;
class RadialGrid;
struct BeamGrid;

class OpenPMDWriter {
public:
    OpenPMDWriter(const Config& cfg, const std::string& outdir, const RadialGrid& grid, const BeamGrid& box, bool mode1,
                  double dt);
    ~OpenPMDWriter();
    static bool available();   // compiled with openPMD-api

    // collective over all ranks; every rank passes its local slices and particles
    void write(int step, double t, const FieldTable* fields, const std::vector<std::unique_ptr<Beam>>& beams);
    void close();
    std::string description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace quarz
