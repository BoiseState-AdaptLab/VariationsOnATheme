# MiniFluxDiv in M<sup>2</sup>DFGs
## CGO 2018

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

### Building

Each code variant (execution schedule) consists of a configuration file (.cfg) and an ISCC input file (.in).
The generated code is in C/C++. An accompanying Python script called ``islcodgen.py`` in the ``scripts`` directory
is responsible for generating the code from ISCC and composing it with the configuration file to produce the output
source file (.h) and compile it into an executable binary.

For that reason, ISL/ISCC are a dependence of this project. ISL is developed and maintained by [INRIA](http://isl.gforge.inria.fr)
and can be obtained from their site or from the [git repository](http://repo.or.cz/w/isl.git). ISCC is included in the
associated [barvinok](http://barvinok.gforge.inria.fr) package. After installing, please ensure that ``iscc`` can be
found within your ``$PATH`` environmental variable for ``islcodegen.py`` to find.

Finally, the code can be built with the following command:

``make -f Makefile.cgo``

### Schedules

The following is a brief description of each schedule implemented within this repository. The labels correspond to
those given in the legend of __Figure 6__ in the paper.

__Series of Loops (SA)__

This is the original, series of loops version of the code, in single assignment form (SA), and with the
component loop (_p,e,u,v,w_) loops on the outside (CLO). The MMDFG for this variant is in __Figure 3__ of the paper.
This is the starting point of each of the transformations.

__Series of Loops (Reduced)__

This is the same of the previous variant, but with the data reduction optimization applied to
minimize temporary storage. The relative cost from the cost model is _30N<sup>2</sup> + 56N_.

__Fuse Among Directions__

In this variant, read reduce fusions are applied to each of the operations in each of the directions, i.e., the
_Fx1_ and _Fy1_ operations are fused together, as re the _Fx2_ and _Fy2_, and finally the _Dx_ and _Dy_ operations.
This corresponds to the MMDFG in __Figure 7__. The relative cost is _22N<sup>2</sup> + 46N_. The single assignment
version is the only one for this variant, as no significant storage reductions are possible. This is the best
performing variant for the small box (_C_=16) case.

__Fuse Within Directions (SA)__

This variant begins with a __reschedule__ operation for each of the velocity components (_u,v_) for the _Fx1_ and _Fy1_
operations, respectively to satisfy the data dependencies. Then producer-consumer fusion is performed on the remaining
operations in each direction, e.g., _Fx1_, _Fx2_, _Dx_, and _Fy1_, _Fy2_, _Dy_. The MMDFG for this variant is given in
__Figure 8__ of the paper. The relative cost is _16N<sup>2</sup> + 46N + 14_. This is the single assignment version.

__Fuse Within Directions (Reduced)__

This is the temporary storage optimized (data reduced) version of the fuse within directions variant.

__Fuse All Levels (SA)__

A combination of the previous two variants, with reschedule operations, followed by producer-consumer fusions, and
completed with read reduction fusions. The MMDFG for this variant is given in __Figure 9__ of the paper. The relative
cost from the cost model is given as _14N<sup>2</sup> + 44N + 11_.

__Fuse All Levels (Reduce)__

This is the temporary storage optimized (data reduced) version of the fuse all levels variant. This is the most
performant variant at full thread count (_p_=28) for the large box case (_C_=128).

__Overlapped Tiling (Fuse All Levels)__

Finally, the shift-and-fuse overlapped tiling technique as described in __Figure 5__ of the paper is applied to the
data reduced Fuse All Levels variant. The experimental results for this variant are represented by the lavender line
in __Figure 6b__ of the paper.

### Running

The ``scripts`` folder contains the necessary scripts to run the schedules. The primary script is
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

2. __runMiniFluxDiv-cgo-all.sh__ will run all of the above executable schedules with the conditions specified in the paper.
The script is intended to be run as part of a SLURM or PBS script, so results are written to ``stdout``.

Here is an example output:

| Program                  	| Legend            	| nBoxes 	| nCells 	| nThreads 	| Time0    	| Time1    	| Time2    	| Time3    	| Time4    	| MeanTime 	| MedianTime 	| StdevTime 	|
|--------------------------	|-------------------	|--------	|--------	|----------	|----------	|----------	|----------	|----------	|----------	|----------	|------------	|-----------	|
| miniFluxdiv-seriesSSACLO 	| Series SSA CLO 	| 28     	| 128    	| 1        	| 3.666721 	| 3.66713  	| 3.647525 	| 3.657901 	| 3.65188  	| 3.658231 	| 3.657901   	| 0.008751  	|
| miniFluxdiv-seriesSSACLO 	| Series SSA CLO  	| 224    	| 64     	| 1        	| 3.028484 	| 3.028278 	| 3.025327 	| 3.024791 	| 3.029452 	| 3.027266 	| 3.028278   	| 0.002072  	|
| miniFluxdiv-seriesSSACLO 	| Series SSA CLO  	| 1792   	| 32     	| 1        	| 2.961895 	| 2.961715 	| 2.961422 	| 2.970956 	| 2.970116 	| 2.965221 	| 2.961895   	| 0.004864  	|
| miniFluxdiv-seriesSSACLO  | Series SSA CLO  	| 14336  	| 16     	| 1        	| 2.629569 	| 2.632029 	| 2.630971 	| 2.629619 	| 2.627903 	| 2.630018 	| 2.629619   	| 0.001564  	|
