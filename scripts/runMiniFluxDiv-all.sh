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

cd ../themes/MiniFluxDiv
make
#
cd ../../scripts
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxdiv-explain-baseline	-l "StorageOptimized" -b 24,192,1536,12288 -c 128,64,32,16 -n 5 -p 1,2,4,8,12,16,20,24 -v 1 >> minifluxdiv-data-all.csv
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxdiv-explainTripleCache-lc -l "Baseline" -b 24,192,1536,12288 -c 128,64,32,16 -n 5 -p 1,2,4,8,12,16,20,24 -v 1 >> minifluxdiv-data-all.csv
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxdiv-explainTripleCache-fuse -l "ShiftAndFuse" -b 24,192,1536,12288 -c 128,64,32,16 -n 5 -p 1,2,4,8,12,16,20,24 -v 1 >> minifluxdiv-data-all.csv
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxdiv-explainTripleCache-tile888 -l "Tiled8x8x8" -b 24,192,1536,12288 -c 128,64,32,16 -n 5 -p 1,2,4,8,12,16,20,24 -v 1 >> minifluxdiv-data-all.csv
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxdiv-explainTripleCache-tile161616 -l "Tiled16x16x16" -b 24,192,1536,12288 -c 128,64,32,16 -n 5 -p 1,2,4,8,12,16,20,24 -v 1 >> minifluxdiv-data-all.csv
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxdiv-explainTripleCache--tile323232 -l "Tiled32x32x32" -b 24,192,1536,12288 -c 128,64,32,16 -n 5 -p 1,2,4,8,12,16,20,24 -v 1 >> minifluxdiv-data-all.csv
