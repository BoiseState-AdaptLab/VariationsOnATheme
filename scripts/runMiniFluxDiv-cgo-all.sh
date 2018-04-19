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

python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-seriesSSACLO -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -l  "Series-SSA-CLO"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-seriesReduceCLO -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "Series-Reduce-CLO"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-fuseAmongDirsSSA -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "Fuse-Among-Dirs-SSA"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-fuseAllCLOSSA -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "Fuse-All-SSA-CLO"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-fuseAllCLOREDUX -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "Fuse-All-Redux-CLO"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-fuseWithinDirsCLOSSA -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "Fuse-Within-Dirs-SSA-CLO"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-fuseWithinDirsCLOREDUX -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "Fuse-Within-Dirs-Redux-CLO"
python3 runMiniFluxDiv.py -e ../themes/MiniFluxDiv/miniFluxDiv-fuseAllCLOReduxOverlap -b 28,224,1792,14336 -c 128,64,32,16 -n 5 -p 1,2,8,14,16,28 -H 0 -l  "FuseAll-CLO-Redux-Overlap-TS8"

