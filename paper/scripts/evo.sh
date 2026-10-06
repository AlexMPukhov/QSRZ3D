#!/bin/bash
export OMP_PROC_BIND=false OMP_NUM_THREADS=2
R=$(cd "$(dirname "$0")/../.." && pwd); Q=${QUARZ:-$R/build/quarz}; I=$R/examples/pwfa_ion_motion.in
C="output.every=0 output.beam_every=0 driver.analytic=1"
$Q $I $C output.dir=evo_s1 > evo_s1.log 2>&1
$Q $I $C grid.type=uniform grid.dr=0.01 output.dir=evo_u0.01 > evo_u0.01.log 2>&1
$Q $I $C "grid.regions=0.05:0.00025 2.5:0.005 8:0.025" output.dir=evo_s0.5 > evo_s0.5.log 2>&1
$Q $I $C grid.type=uniform grid.dr=0.005 output.dir=evo_u0.005 > evo_u0.005.log 2>&1
echo done > evo.done
