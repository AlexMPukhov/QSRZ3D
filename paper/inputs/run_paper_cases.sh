#!/bin/bash
# Run the paper cases and print the time per sweep (one time step through the box).
#   QUARZ=/path/to/quarz  LAUNCH="mpirun -np 4"  bash run_paper_cases.sh [group ...]
# groups: conv  ref  maps  evo  bench  mpi     (default: conv maps evo bench)
# Each run writes out_<case>/ and log_<case>.txt in the current directory.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
QUARZ=${QUARZ:-$HERE/../../build/quarz}
LAUNCH=${LAUNCH:-}
GROUPS_=${*:-conv maps evo bench}

run() {   # name, deck, extra overrides...
    local name=$1 deck=$2; shift 2
    rm -rf "out_$name"
    local t0=$(date +%s.%N)
    $LAUNCH "$QUARZ" "$deck" output.dir="out_$name" "$@" > "log_$name.txt" 2>&1
    local rc=$? t1=$(date +%s.%N)
    local sw=$(grep -o 'sweep *[0-9.]* s' "log_$name.txt" | awk '{s+=$2; n++} END {if (n) printf "%.3f", s/n; else print "-"}')
    printf "%-22s rc=%d  wall %8.1f s   mean sweep %s s\n" "$name" $rc "$(echo "$t1 - $t0" | bc)" "$sw"
}

for g in $GROUPS_; do
  case $g in
    conv)  for c in u0.02 u0.01 u0.005 u0.0025 u0.00125 s4 s2 s1 s0.5 s0.25 s0.125 s0.0625; do
               run conv_$c "$HERE/conv_$c.in"; done ;;
    ref)   run conv_ref_s0.03125 "$HERE/conv_ref_s0.03125.in"
           run conv_ref_u0.00025 "$HERE/conv_ref_u0.00025.in" ;;
    maps)  run maps_s1 "$HERE/maps_s1.in" ;;
    evo)   for c in s1 s0.5 u0.01 u0.005; do run evo_$c "$HERE/evo_$c.in"; done ;;
    bench) run bench_particles_s1 "$HERE/bench_particles_s1.in" ;;
    mpi)   # the parallel-efficiency cases of the paper (run once with LAUNCH="", once with mpirun)
           run mpi_gain "$HERE/../../validation/gain.in" time.steps=20 output.every=0
           run mpi_hosing "$HERE/../../examples/hosing.in" time.steps=12 output.every=0 output.beam_every=0
           run mpi_pwfa "$HERE/../../examples/pwfa_ion_motion.in" time.steps=10 output.every=0 output.beam_every=0 ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done
