#!/bin/bash
# Reproduces the validation results quoted in README.md.
# usage: cd validation && ./run_validation.sh [path/to/qsrz]   (needs python3 + numpy + scipy)
set -e
QS=${1:-../build/qsrz}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}

echo "== 1. linear wake vs Green's function (n_b = 0.001) =="
$QS linear.in driver.n0=0.001 output.dir=out_lin_u > /dev/null
python3 check_linear.py out_lin_u "uniform dr=0.01         " 0.001 2>/dev/null
$QS linear.in driver.n0=0.001 output.dir=out_lin_s grid.type=regions "grid.regions=0.3:0.002 2:0.02 8:0.05" \
    grid.max_ratio=1.05 > /dev/null
python3 check_linear.py out_lin_s "stretched 0.002 -> 0.05 " 0.001 2>/dev/null

echo "== 2. blowout: resolution study =="
run() { name=$1; shift; $QS blowout.in output.dir=out_b_$name xi.max=10 "$@" > /dev/null; python3 bsum.py out_b_$name $name; }
run uniform_dr0.01_dxi0.01
run uniform_dr0.005_dxi0.005 grid.dr=0.005 xi.step=0.005
run uniform_dr0.0025_dxi0.0025 grid.dr=0.0025 xi.step=0.0025
run stretched_dxi0.005 xi.step=0.005 grid.type=regions "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" grid.max_ratio=1.05
run stretched_dxi0.005_PCR xi.step=0.005 grid.type=regions "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" \
    grid.max_ratio=1.05 solver.tridiag=pcr

echo "== 3. betatron oscillation in an ion channel =="
$QS betatron.in > /dev/null
python3 -c "
import sys; sys.path.insert(0,'../tools')
from qsrz_read import read_beamlog
import numpy as np
b=read_beamlog('out_betatron/beams.txt')['witness']; t=b['t']; rr=b['r_rms']; w=1/np.sqrt(2*1000)
print('max |r_rms - r0 |cos(w_b t)|| / r0 = %.2e' % (abs(rr-rr[0]*abs(np.cos(w*t))).max()/rr[0]))"

echo "== 4. witness energy gain in a rigid-driver blowout =="
$QS gain.in > /dev/null
python3 -c "
import sys; sys.path.insert(0,'../tools')
from qsrz_read import read_beamlog
import numpy as np
b=read_beamlog('out_gain/beams.txt')['witness']
print('d<gamma>/dt = %.4f   (-Ez at xi=7 from the blowout run: 0.4471)' % np.polyfit(b['t'],b['gamma_mean'],1)[0])"

echo "== 5. pinched witness + ion motion: grid comparison (reference takes ~1 min) =="
im() { name=$1; shift; $QS ../examples/pwfa_ion_motion.in time.steps=0 output.dir=out_im_$name "$@" > /dev/null; }
im stretched
im u001 grid.type=uniform grid.dr=0.01
im u0005 grid.type=uniform grid.dr=0.005
im u0001 grid.type=uniform grid.dr=0.00025 xi.max=7.3
python3 im_compare.py

echo "== 6. azimuthal mode 1: displaced driver vs translated axisymmetric solution =="
$QS linear.in xi.max=12 driver.analytic=1 driver.n0=0.01 output.dir=out_m1_ref > /dev/null
$QS linear.in xi.max=12 driver.analytic=1 driver.n0=0.01 modes=1 driver.x0=0.01 output.dir=out_m1_shift > /dev/null
echo " linear regime (d = 0.01):"
python3 check_shift2.py out_m1_ref out_m1_shift 0.01 3 -1e9 1e9 0.15 2>/dev/null
B="blowout.in xi.max=9 xi.step=0.005 grid.type=regions grid.max_ratio=1.05 driver.analytic=1"
$QS $B "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" output.dir=out_m1_bref > /dev/null
$QS $B "grid.regions=0.5:0.0025 2.5:0.01 8:0.05" modes=1 driver.x0=0.002 output.dir=out_m1_bshift > /dev/null
echo " blowout, inside the first bucket (d = 0.002):"
python3 check_shift2.py out_m1_bref out_m1_bshift 0.002 1.0 0 8 0.15 2>/dev/null
rm -rf out_m1_ref out_m1_shift out_m1_bref out_m1_bshift   # large files

echo "== 7. azimuthal mode 1: centroid betatron oscillation of an offset beam =="
$QS betatron.in modes=1 witness.x0=0.05 witness.nsym=4 output.every=0 output.dir=out_m1_cbt > /dev/null
python3 -c "
import sys; sys.path.insert(0,'../tools')
from qsrz_read import read_beamlog
import numpy as np
b=read_beamlog('out_m1_cbt/beams.txt')['witness']; t=b['t']; w=1/np.sqrt(2*1000); xc=b['x_mean']
print('max |x_c - x0 cos(w_b t)|/x0 = %.2e,  max |y_c|/x0 = %.1e' % (abs(xc-xc[0]*np.cos(w*t)).max()/xc[0], abs(b['y_mean']).max()/xc[0]))"

echo "== 8. function parser: parsed profiles vs built-in ones (must agree to round-off) =="
B="blowout.in xi.max=9 driver.analytic=1"
$QS $B output.dir=out_p_ref > /dev/null
$QS $B "plasma.density(x,y,z)=1" output.dir=out_p_1 > /dev/null
python3 cmp_fields.py out_p_ref out_p_1 'uniform: built-in vs parsed "1"' 8.5
$QS $B plasma.radial=channel plasma.channel_depth=0.3 plasma.channel_radius=1.5 output.dir=out_p_c1 > /dev/null
$QS $B my_constants.dn=0.3 my_constants.rc=1.5 "plasma.density(x,y,z)=1 + dn*(x^2+y^2)/rc^2" output.dir=out_p_c2 > /dev/null
python3 cmp_fields.py out_p_c1 out_p_c2 "channel: built-in vs parsed" 8.5
$QS $B driver.cut=99 driver.profile=parsed "driver.density(x,y,xi)=10*exp(-(x^2+y^2)/(2*0.3^2)-(xi-3)^2/2)" output.dir=out_p_b > /dev/null
python3 cmp_fields.py out_p_ref out_p_b "analytic driver: built-in vs parsed" 8.5
rm -rf out_p_*

echo "== 9. MPI (decomposition along xi) vs serial with beams.xi_shape = ngp =="
if command -v mpirun > /dev/null 2>&1; then
    MPIRUN="mpirun --oversubscribe -np"
    [ "$(id -u)" = "0" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe -np"
    export OMP_NUM_THREADS=1
    G="gain.in time.steps=20 output.every=10 output.beam_every=10 output.beam_slices=50"
    $QS $G beams.xi_shape=ngp output.dir=out_mpi_s > /dev/null
    $MPIRUN 2 $QS $G output.dir=out_mpi_p2 > /dev/null
    echo " witness gain, 2 ranks (expected: bit-identical, 0):"
    python3 cmp_runs.py out_mpi_s out_mpi_p2 0 || true
    $QS mpi_slip.in beams.xi_shape=ngp output.dir=out_mpi_s2 > /dev/null
    $MPIRUN 3 $QS mpi_slip.in output.dir=out_mpi_p3 > /dev/null
    echo " slipping beam + mobile ions + m = 1, 3 ranks (expected: round-off from particle order):"
    python3 cmp_runs.py out_mpi_s2 out_mpi_p3 1e-6 || true
    rm -rf out_mpi_*
else
    echo " (mpirun not found - skipped)"
fi

echo "== 10. openPMD output (needs openPMD-api in the build and the openpmd-api Python module) =="
if python3 -c "import openpmd_api" 2> /dev/null; then
    export OMP_NUM_THREADS=1   # deterministic deposits: bit-identical comparisons
    H="../examples/hosing.in time.steps=4 output.every=2 output.beam_every=2 driver.nparticles=40000 xi.max=6 beams.xi_shape=ngp output.format=both units.n0_cm3=1e17"
    if $QS $H output.openpmd_grid=native output.dir=out_opmd_nat > out_opmd.log 2>&1; then
        echo " native-node openPMD vs native files (expected 0):"; python3 check_openpmd.py out_opmd_nat native | tail -1
        $QS $H output.dir=out_opmd_uni > /dev/null 2>&1
        echo " uniform openPMD grid vs interpolated native data (expected round-off):"; python3 check_openpmd.py out_opmd_uni uniform | tail -1
        if command -v mpirun > /dev/null 2>&1; then
            MPIRUN="mpirun --oversubscribe -np"; [ "$(id -u)" = "0" ] && MPIRUN="mpirun --allow-run-as-root --oversubscribe -np"
            OMP_NUM_THREADS=1 $MPIRUN 2 $QS $H output.format=openpmd output.dir=out_opmd_p2 > /dev/null 2>&1
            echo " 2 MPI ranks vs serial (expected 0):"; python3 cmp_openpmd.py out_opmd_uni out_opmd_p2
        fi
    else
        echo " (this build has no openPMD support - skipped)"
    fi
    rm -rf out_opmd_* out_opmd.log
else
    echo " (python module openpmd_api not found - skipped)"
fi

echo "== 11. laser envelope solver =="
$QS laser_vacuum.in > /dev/null
echo " diffraction in vacuum (k0 = 20, w0 = 2, through the focus):"; python3 laser_check.py vacuum out_laser_vac 20 2 40 0.1
$QS laser_linear.in > /dev/null
echo " linear wake (a0 = 0.05) vs (d^2/dxi^2 + 1) psi = <a^2>/2 (difference = O(a0^2) nonlinearity):"; python3 laser_check.py linear out_laser_lin 0.05 3 1.4142135623730951 4
$QS laser_channel.in time.dt=2.5 time.steps=120 > /dev/null
echo " matched parabolic channel (stretched grid):"; python3 laser_check.py channel out_laser_ch 10 3
S="laser_selffocus.in grid.rmax=90 grid.regions=40:0.2 90:0.5 xi.step=0.1 time.dt=100 time.steps=40 laser.w0=20"
$QS $S laser.a0=0.4 electrons.density=1e-9 output.dir=out_laser_sf0 > /dev/null
$QS $S laser.a0=0.2 output.dir=out_laser_sf05 > /dev/null
$QS $S laser.a0=0.4 output.dir=out_laser_sf2 > /dev/null
echo " relativistic self-focusing, P/Pc = 0 (vacuum: 1 .97 .894 .8 .707), 0.5, 2:"
python3 laser_check.py selffocus out_laser_sf0 0.4 out_laser_sf05 0.2 out_laser_sf2 0.4
rm -rf out_laser_*

echo "== 12. ionization =="
$QS ion_adk.in > /dev/null
echo " field (ADK) ionization of H and He by a beam field: code vs independent ADK (Markov chain of the per-slice probabilities):"
python3 ion_check.py adk out_ion_adk 1e17 16 0.01 gas:He:1e-4 hyd:H:1e-4
$QS ion_laser.in > /dev/null
echo " laser ionization of N (up to N5+), a0 = 0.1, 0.8 um at 1e17 cm^-3, wide spot: code vs cycle-resolved ADK and test particles:"
python3 ion_check.py laser out_ion_laser 1e17 132 0.1 3 10 N 1e-6 5 2>/dev/null
$QS ion_impact.in > /dev/null
echo " impact ionization of Ar by a rigid 400 GeV proton beam (Bethe, M^2 = 4.22, C = 37.93; nsplit = 2000):"
python3 ion_check.py impact out_ion_impact 1e15 4.22:37.93 10 1 15 50 427 1
rm -rf out_ion_*
