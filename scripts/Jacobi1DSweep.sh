# !/bin/bash
# This script is for Jacobi1D-Versions
cd themes/Jacobi1D

# To run this script, "cd .." to go parent directory
# and enter "scripts/Jacobi1DSweep.sh"
# PBS -l nodes=1:ppn=1,walltime=48:00:00

# Using the variables below, the loops can be modified 
# for OpenMP and CUDA benchmarks

#---------------OpenMP--------------
# The list below can be extented to run different Jacobi1D OpenMP variants.
# Variants need to be seperated by space in the list.
VARIANTS_OMP="Jacobi1D-NaiveParallel-OMP"

#The time steps (-T) loop variables for OMP
TimeStartOMP=1 TimeEndOMP=6 TimeIncrementOMP=1

#The problem Size (--Nx) loop variables for OMP
NxStartOMP=25000 NxEndOMP=100000 NxIncrementOMP=25000

#The number of threads (--num_threads) loop variables for OMP
num_threadsStartOMP=1 num_threadsEndOMP=4 num_threadsIncrementOMP=1

#---------------CUDA--------------
# The list below can be extented to run different Jacobi1D CUDA variants.
# Variants need to be seperated by space in the list.
VARIANTS_CUDA="Jacobi1D-NaiveParallelGlobal-CUDA"

#The time steps (-T) loop variables for CUDA
TimeStartCUDA=1 TimeEndCUDA=6 TimeIncrementCUDA=1

#The problem Size (--Nx) loop variables for CUDA
NxStartCUDA=25000 NxEndCUDA=100000 NxIncrementCUDA=25000

#The block size (--bx) loop variables for CUDA
blockSizeStartCUDA=128 blockSizeEndCUDA=1024 blockSizeIncrementCUDA=128

for Variants in $VARIANTS_OMP
do
  echo
  echo $Variants
  for ((Time=$TimeStartOMP; Time<=$TimeEndOMP; Time+=$TimeIncrementOMP))
  do
    for ((Nx=$NxStartOMP; Nx<=$NxEndOMP; Nx+=$NxIncrementOMP))
    do
      for ((num_threads=$num_threadsStartOMP; 
            num_threads<=$num_threadsEndOMP; 
            num_threads+=$num_threadsIncrementOMP))
      do
        echo
	echo "$Variants --Nx $Nx -T $Time --num_threads $num_threads -v"
	./$Variants --Nx $Nx -T $Time --num_threads $num_threads -v
      done;
      echo
      echo
    done;
  done;
done

for Variants in $VARIANTS_CUDA 
do
  echo
  echo $Variants
  for ((Time=$TimeStartCUDA; Time<=$TimeEndCUDA; Time+=$TimeIncrementCUDA))
  do
    for ((Nx=$NxStartCUDA; Nx<=$NxEndCUDA; Nx+=$NxIncrementCUDA))        
    do
      for ((blockSize=$blockSizeStartCUDA; 
            blockSize<=$blockSizeEndCUDA; 
            blockSize+=$blockSizeIncrementCUDA))
      do
        echo
        echo "$Variants --Nx $Nx -T $Time --bx $blockSize -v"
        ./$Variants  --Nx $Nx -T $Time --bx $blockSize -v
      done;
      echo
      echo
    done;
  done;
done
