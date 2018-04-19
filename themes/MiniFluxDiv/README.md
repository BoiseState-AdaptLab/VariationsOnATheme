# MiniFluxDiv

The original version of MiniFluxDiv was developed by Stephen Guzik of the
Mechanical Engineering department at CSU. The original benchmark was then
rewritten in order to use code generation by Catherine Olschanowsky and
Michelle Strout.

This file tracks the different variations of the benchmark that are available
in this directory.

The original benchmark was written using portions of the Chombo framework.
This dependency has been removed in most of the versions here, but the
development patterns remain. The equations and workflow of the benchmark are
covered in the miniFluxdiv-explain-serial.cpp file. This file will discuss
only the execution schedules.

The original execution schedule can be described as

1. Component loop on the outside
2. Series of loop nests
3. C array indexing was done using an FArrayBox which is an object within the
Chombo framework

An example loop nest includes the following:

```
for (int iC=0; iC != nComp; ++iC){
  int D_DECL6(i0, i1, i2, i3, i4, i5);
  for (i2=0; i2 < n2; ++i2)
    for (i1=0; i1 < n1; ++i1)
      for (i0=0; i0 < n0; ++i0){
        flux[iC][fluxs2+i2][fluxs1+i1][fluxs0+i0] = factor1*(
        phi[iC][phis2+i2-2*ii2][phis1+i1-2*ii1][phis0+i0-2*ii0]  +
        7*(phi[iC][phis2+i2-  ii2][phis1+i1-  ii1][phis0+i0-  ii0]  +
        phi[iC][phis2+i2      ][phis1+i1      ][phis0+i0      ]) +
        phi[iC][phis2+i2+  ii2][phis1+i1+  ii1][phis0+i0+  ii0]);
      }
    }
  }
}
```

nComp is the number of components i2,i1 and i0 are the z, y, and x directions
respectively.

All of schedules make use of the loop chain pragmas to generate code for the loop structures,
with the exception of __explain-serial__ and __explain-benchmark__. The pragmas can be found
in the __explainTripleCache-lc__ schedule for reference. The generated code is in C/C++.

### Schedules

The following is a brief description of each schedule implemented within this repository.

__explain-serial__

This is the serial version of the code, designed for explanatory purposes and well commented.

__explain-baseline__

The baseline version of the code, similar to the serial version, but with per-box parallelism using an OpenMP pragma around the box loop. Temporary storage caches are optimized for space efficiency. "Storage Optimized" in Figure 21.

__explainTripleCache-lc__

A modification of the baseline version that is not storage optimized, with three separate velocity caches for each of the _x_, _y_, and _z_-axes to enable loop chain optimizations that are not legal schedules in the storage optimized baseline version. Also contains the initial loop chain pragmas for reference. This schedule is referred to as "Baseline" in Figure 21 of the paper.

__explainTripleCache-fuse__

A version of the previous triple cache code, with the loop nests shifted then fused using the automatic shift determination feature of the loop chain tool. This schedule is referred to as "Shift and Fuse" in Figure 21 of the paper.

__explainTripleCache-tile888__

Similar to the previous entry, this version is shifted, fused, then tiled by the loop chain tool with  tile dimensions of 8x8x8. This schedule is referred to as "Tiled 8x8x8" in Figure 21 of the paper.

__explainTripleCache-tile161616__

Same as the previous version, but with a tile size of 16x16x16. This schedule is referred to as "Tiled 16x16x16" in Figure 21 of the paper.

__explainTripleCache-tile323232__

Same as the previous two versions, except the tile size is 32x32x32. This schedule is referred to as "Tiled 32x32x32" in Figure 21 of the paper.

### Running

The 'scripts' folder contains the necessary scripts to run the schedules. The primary script is
a Python program called __runMiniFluxDiv.py__ that can be invoked with a '-h' argument to view the help message.

```
$ ./runMiniFluxDiv.py -h
usage: runMiniFluxDiv [-h] [-e EXEC] [-l LEGEND] [-b BOXES] [-c CELLS]
                      [-p THREADS] [-n RUNS] [-d DELIM] [-v VERIFY]

Execute miniFluxDiv test cases.

optional arguments:
  -h, --help            show this help message and exit
  -e EXEC, --exec EXEC  Executable to run.
  -l LEGEND, --legend LEGEND
                        Chart legend for output table.
  -b BOXES, --boxes BOXES
                        List of box sizes.
  -c CELLS, --cells CELLS
                        List of cell sizes.
  -p THREADS, --threads THREADS
                        List of thread counts.
  -n RUNS, --runs RUNS  Number of runs.
  -d DELIM, --delim DELIM
                        List delimiter.
  -v VERIFY, --verify VERIFY
                        Perform verification.
```

This script will run the provided executable, the specified number of times, with the desired box,
cell, and thread configurations, and report the run time data in CSV format.
Example calls can be found in the accompany shell scripts:

1. __runMiniFluxDiv.sh__ will run a single executable with the conditions specified in the paper.

2. __runMiniFluxDiv-all.sh__ will run a all of the above executable scheduels with the conditions specified in the paper, redirecting the output to a single CSV file.

Here is an example output:

| Program                      	| Legend            	| nBoxes 	| nCells 	| nThreads 	| Time0    	| Time1    	| Time2    	| Time3    	| Time4    	| MeanTime 	| MedianTime 	| StdevTime 	|
|------------------------------	|-------------------	|--------	|--------	|----------	|----------	|----------	|----------	|----------	|----------	|----------	|------------	|-----------	|
| miniFluxdiv-explain-baseline 	| Storage Optimized 	| 28     	| 128    	| 1        	| 3.666721 	| 3.66713  	| 3.647525 	| 3.657901 	| 3.65188  	| 3.658231 	| 3.657901   	| 0.008751  	|
| miniFluxdiv-explain-baseline 	| Storage Optimized 	| 224    	| 64     	| 1        	| 3.028484 	| 3.028278 	| 3.025327 	| 3.024791 	| 3.029452 	| 3.027266 	| 3.028278   	| 0.002072  	|
| miniFluxdiv-explain-baseline 	| Storage Optimized 	| 1792   	| 32     	| 1        	| 2.961895 	| 2.961715 	| 2.961422 	| 2.970956 	| 2.970116 	| 2.965221 	| 2.961895   	| 0.004864  	|
| miniFluxdiv-explain-baseline 	| Storage Optimized 	| 14336  	| 16     	| 1        	| 2.629569 	| 2.632029 	| 2.630971 	| 2.629619 	| 2.627903 	| 2.630018 	| 2.629619   	| 0.001564  	|
