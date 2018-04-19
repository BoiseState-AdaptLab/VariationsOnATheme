#!/bin/bash
#
#   This file is part of MiniFluxDiv.

#   MiniFluxDiv is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.

#   MiniFluxDiv is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with MiniFluxDiv. If not, see <http://www.gnu.org/licenses/>.
#

python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverlapTS8PB -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -l  "FuseAll-CLO-Redux-Overlap-TS8-PB"
python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverlapTS16PB -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "FuseAll-CLO-Redux-Overlap-TS16-PB"
python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverlapTS32PB -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "FuseAll-CLO-Redux-Overlap-TS32-PB"
python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverlapTS8PT -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "FuseAll-CLO-Redux-Overlap-TS8-PT"
python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverlapTS16PT -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "FuseAll-CLO-Redux-Overlap-TS16-PT"
python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverlapTS32PT -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "FuseAll-CLO-Redux-Overlap-TS32-PT"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-seriesCLOReduxOverlapTS8PB -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l  "Series-CLO-Redux-Overlap-TS8-PB"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-seriesCLOReduxOverlapTS16PB -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l  "Series-CLO-Redux-Overlap-TS16-PB"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-seriesCLOReduxOverlapTS32PB -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l  "Series-CLO-Redux-Overlap-TS32-PB"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-seriesCLOReduxOverlapTS8PT -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l  "Series-CLO-Redux-Overlap-TS8-PT"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-seriesCLOReduxOverlapTS16PT -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l  "Series-CLO-Redux-Overlap-TS16-PT"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-seriesCLOReduxOverlapTS32PT -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l  "Series-CLO-Redux-Overlap-TS32-PT"

#python3 runMiniFluxDiv.py ../themes/MiniFluxDiv/miniFluxdiv-baseline-OMP 32 32,64,128
#python3 runMiniFluxDiv.py -e $1 -b 16384,2048,256,32 -c 16,32,64,128 -n 5 -p 1,2,4,7,8,14,16,21,24,28
#python3 runMiniFluxDiv.py -e $1 -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,4,7,8,14,16,21,24,28 -v 1 -l $2
#python3 runMiniFluxDiv.py -e $1 -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -v 1 -l $2
# 2x1
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N2x1 -b 28 -c 128 -n 5 -p 2 -v 1 -H 1 -l "FuseAll-CLO-Redux-Overlap-TS8-N2x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N2x1 -b 28 -c 128 -n 5 -p 2 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N2x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N2x1 -b 28 -c 128 -n 5 -p 2 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N2x1"
# 2x2
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N2x2 -b 28 -c 128 -n 5 -p 4 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N2x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N2x2 -b 28 -c 128 -n 5 -p 4 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N2x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N2x2 -b 28 -c 128 -n 5 -p 4 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N2x2"
# 4x1
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N4x1 -b 28 -c 128 -n 5 -p 4 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N4x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N4x1 -b 28 -c 128 -n 5 -p 4 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N4x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N4x1 -b 28 -c 128 -n 5 -p 4 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N4x1"
# 2x4
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N2x4 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N2x4"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N2x4 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N2x4"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N2x4 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N2x4"
# 4x2
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N4x2 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N4x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N4x2 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N4x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N4x2 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N4x2"
# 8x1
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N4x2 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N8x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N4x2 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N8x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N4x2 -b 28 -c 128 -n 5 -p 8 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N8x1"
# 2x7
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N2x7 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N2x7"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N2x7 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N2x7"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N2x7 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N2x7"
# 7x2
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N7x2 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N7x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N7x2 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N7x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N7x2 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N7x2"
# 14x1
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N14x1 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N14x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N14x1 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N14x1"
#ython3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N14x1 -b 28 -c 128 -n 5 -p 14 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N14x1"
# 2x8
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N2x8 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N2x8"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N2x8 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N2x8"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N2x8 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N2x8"
# 8x2
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N8x2 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N8x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N8x2 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N8x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N8x2 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N8x2"
# 16x1
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N16x1 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N16x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N16x1 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N16x1"
#ython3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N16x1 -b 28 -c 128 -n 5 -p 16 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N16x1"
# 2x14
#Qpython3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N2x14 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N2x14"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N2x14 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N2x14"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N2x14 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N2x14"
# 14x2
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N14x2 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N14x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N14x2 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N14x2"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N14x2 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N14x2"
# 7x4
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N7x4 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N7x4"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N7x4 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N7x4"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N7x4 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N7x4"
# 4x7
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N4x7 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N4x7"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N4x7 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N4x7"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N4x7 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N4x7"
# 28x1
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS8N28x1 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS8-N28x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS16N28x1 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS16-N28x1"
#python3 runMiniFluxDiv.py -e ../themes/Benchmark/miniFluxDiv-fuseAllCLOReduxOverTS32N28x1 -b 28 -c 128 -n 5 -p 28 -v 1 -H 0 -l "FuseAll-CLO-Redux-Overlap-TS32-N28x1"
