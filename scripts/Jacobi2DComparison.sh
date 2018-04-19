# !/bin/bash
# This script is for Jacobi2D-Versions
cd themes/Jacobi2D

# To run this script, "cd .." to go parent directory
# and enter "scripts/Jacobi2DComparison.sh"
# PBS -l nodes=1:ppn=1,walltime=48:00:00

# Setting the environment variable 
# The environment variable can be selected as: 
# "static" or "dynamic" or "guided"
export OMP_SCHEDULE="dynamic"
echo
echo "OMP_SCHEDULE is $OMP_SCHEDULE" 

# Using the variables below, the loops can be modified 
# for OpenMP and CUDA benchmarks

#---------------OpenMP--------------
# The list below can be extented to run different Jacobi2D OpenMP variants.
# Variants need to be seperated by space in the list.
VARIANTS_OMP="Jacobi2D-NaiveParallel-OMP Jacobi2D-NaiveParallel3DArray-OMP"

# The list below are tau values for Jacobi2D OpenMP Diamond Tiling variants
TAUS="276"

#The time steps (-T) loop variables for OMP
TimeStartOMP=1 TimeEndOMP=6 TimeIncrementOMP=1
#If we use list for times steps the list is below
TIME="100"

#The problem Size (--Nx) loop variables for OMP
NxStartOMP=4096 NxEndOMP=4096 NxIncrementOMP=1

#The problem Size (--Ny) loop variables for OMP
NyStartOMP=4096 NyEndOMP=4096 NyIncrementOMP=1

#The number of threads (--num_threads) list  variables for OMP
THREADS="1 2 4 6 8 10 12"

#The number of runs variable
RUNS=10

for Variants in $VARIANTS_OMP
do
  echo
  echo $Variants
  echo
  for Time in $TIME
  do
    for ((Nx=$NxStartOMP; Nx<=$NxEndOMP; Nx+=$NxIncrementOMP))
    do
      for ((Ny=$NyStartOMP; Ny<=$NyEndOMP; Ny+=$NyIncrementOMP))
      do
        for num_threads in $THREADS 
        do
          if [ `echo $Variants | grep -c "Diamond" ` -gt 0 ]
          then
            for taus in $TAUS 
            do    
              for run in `seq 1 $RUNS`
              do                  
                echo "$Variants --Nx $Nx --Ny $Ny -T $Time --tau $taus --num_threads $num_threads"
                ./$Variants --Nx $Nx --Ny $Ny -T $Time --tau $taus --num_threads $num_threads
              done;
              echo
            done;
          else
            for run in `seq 1 $RUNS`
            do
              echo "$Variants --Nx $Nx --Ny $Ny -T $Time --num_threads $num_threads"
              ./$Variants --Nx $Nx --Ny $Ny -T $Time --num_threads $num_threads 
            done;
            echo
          fi
        done;
      done;
    done;
  done;
done;
  
