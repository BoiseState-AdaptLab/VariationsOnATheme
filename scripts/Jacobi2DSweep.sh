# !/bin/bash
# This script is for Jacobi2D-Versions
cd themes/Jacobi2D

# To run this script, "cd .." to go parent directory
# and enter "scripts/Jacobi2DSweep.sh"
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
TAUS="165 174 183 189 192 195 198 201 204 207 210 213 219 291 294 297 300 303 306 312 315 318 321 390 402 513 1002 2001"

#The time steps (-T) loop variables for OMP
TimeStartOMP=1 TimeEndOMP=6 TimeIncrementOMP=1

#The problem Size (--Nx) loop variables for OMP
NxStartOMP=1000 NxEndOMP=1500 NxIncrementOMP=500

#The problem Size (--Ny) loop variables for OMP
NyStartOMP=1000 NyEndOMP=1500 NyIncrementOMP=500

#The number of threads (--num_threads) list  variables for OMP
THREADS="1 2 3 4"

for Variants in $VARIANTS_OMP
do
  echo
  echo $Variants
  for ((Time=$TimeStartOMP; Time<=$TimeEndOMP; Time+=$TimeIncrementOMP))
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
              echo
              echo "$Variants --Nx $Nx --Ny $Ny -T $Time --tau $taus --num_threads $num_threads -v"
              ./$Variants --Nx $Nx --Ny $Ny -T $Time --tau $taus --num_threads $num_threads -v
            done;
          else
            echo
            echo "$Variants --Nx $Nx --Ny $Ny -T $Time --num_threads $num_threads -v"
            ./$Variants --Nx $Nx --Ny $Ny -T $Time --num_threads $num_threads -v
          fi
        done;
        echo
        echo
      done;
    done;
  done;
done;
