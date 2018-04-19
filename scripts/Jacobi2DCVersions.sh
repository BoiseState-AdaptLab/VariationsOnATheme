# !/bin/bash
# This script is for Jacobi2D-Versions
cd themes/Jacobi2D

# To run this script, "cd .." to go parent directory
# and enter "scripts/Jacobi2DCVersions.sh"

# Using the variables below, the loops can be modified 
# for OpenMP and CUDA benchmarks

#---------------OpenMP--------------
# The list below can be extented to run different Jacobi2D OpenMP variants.
# Variants need to be seperated by space in the list.
VARIANTS_OMP="Jacobi2D-NaiveParallel-OMP_dyn Jacobi2D-NaiveParallel-OMP_static"

# The list below are tau values for Jacobi2D OpenMP Diamond Tiling variants
TAUS="276"

#The time steps (-T) loop variables for OMP
TimeStartOMP=1 TimeEndOMP=6 TimeIncrementOMP=1
#If we use list for times steps the list is below
TIME="100"

#The problem Size (-p) loop variables for OMP
PStartOMP=2046 PEndOMP=2046 PIncrementOMP=1

#The number of threads (--num_threads) list  variables for OMP
THREADS="1 2 3 4 6 12 24"

#The number of runs variable
RUNS=10

for Variants in $VARIANTS_OMP
do
  echo
  echo $Variants
  echo
  for Time in $TIME
  do
    for ((P=$PStartOMP; P<=$PEndOMP; P+=$PIncrementOMP))
    do
      for num_threads in $THREADS 
      do
        for run in `seq 1 $RUNS`
        do
          echo "$Variants -p $P -T $Time -c $num_threads"
          ./$Variants -p $P -T $Time -c $num_threads
        done;
      done;
    done;
  done;
done;

