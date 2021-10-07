/**
    This file is part of MiniFluxDiv.

    MiniFluxDiv is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    MiniFluxDiv is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MiniFluxDiv. If not, see <http://www.gnu.org/licenses/>.
 **/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include <ctype.h>
#include <omp.h>

#include "../common/Configuration.h"
#include "../common/Measurements.h"
#include "util.h"

/****************************************************************************
 * This file is meant to explain the mini-application as it was developed.
 * Please see the comments below for a full explaination.
 *
 ****************************************************************************
 *
 * Modified on 08 July 2016 by Sarah Willer
 * Changes to param options:
 *  >> replaced call to MFD_Configuration constructor with call to
 *     Configuration constructor
 *  >> "help" 'h' and "verify" 'v' are still included in the Configuration
 *     constructor. The other common command line options are listed in
 *     Configuration.txt
 *  >> nGhost set to NGHOST defined in Configuration.h
 *  >> numComp set to NCOMP defined in Configuration.h
 *  >> added explicit addParam calls for numCell and numBox
 *
****************************************************************************/

Real** mini_flux_div_lc(Real** old_boxes,Real** new_boxes,
                        Configuration& config, Measurements& measurements);

using namespace std;

int main(int argc, char **argv) {

    int verify=0;
    int flux_totalSize;
    int idx,iz,iy,ix;
    int index;
    int v, c;
    Real **result_data;
    struct timeval tv1,tv2;

    // Constructor for parsing the command line arguments
    Configuration config;

    /**************************************************************************
    **  Parameter options. Help and verify are constructor defaults.  *********
    **************************************************************************/
    config.addParamInt("numCell", 'C' , 128 ,
                       "--numCell, the number of cells in a single dimension of a"
                       " single box");

    config.addParamInt("numBox",'B', 32 ,
                       "--numBox, the number of independent boxes to process");

    config.addParamInt("num_threads",'p',1,
                       "-p <num_threads>, number of cores");

    //Constructor for Measurements
    Measurements measurements;

    config.parse(argc,argv);

    int numCell= config.getInt("numCell");
    int numBox= config.getInt("numBox");
    int nGhost = NGHOST;
    int numComp= NCOMP;

    // t = config.getInt("tests");
    // v = config.getInt("verify");

    if(numCell < 0) {
        fprintf(stderr,"The value of numCell has to be a positive number %d\n",
                numCell);
        exit(-1);
    }

    if(numBox < 0) {
        fprintf(stderr,"The value of numBox has to be a positive number %d\n",
                numBox);
        exit(-1);
    }

    setNumThreads(config.getInt("num_threads"));

    Real **old_boxes;
    Real **new_boxes;

    //allocate call
    allocateAndInitialize(&old_boxes,&new_boxes,config);

    // Calling Benchmark Function
    mini_flux_div_lc(old_boxes,new_boxes,config,measurements);

    //Calling Verification from Util
    if(config.getBool("v")) {
        if (verifyResult(new_boxes, config)) {
            measurements.setField("verification", "SUCCESS");
        } else {
            measurements.setField("verification", "FAILURE");
        }
    }

    //get results from measurements
    string result = measurements.toLDAPString();
    string config_in = config.toLDAPString();
    cout << config_in << result << endl;

    return 0;
}

/*******************************************************************************
 * The following function is meant to be explainatory code. This implementation
 * is meant purely to explain the control flow and algorithm. Other
 * implementations include optimizations that sometimes obfuscate the
 * algorithm.
 *
 *  Processing the boxes means that we will be reading the data from
 *  old_boxes and writing them to new_boxes
 *  The following are the equations for this calculation
 *
 *  There are 5 components: p, e, u, v, w (density, energy, velocity (3D))
 *  Each of these components is represented as a 3D array (initialized
 *  above).
 * p_{t+1}(z,y,x) = h(p_t,z,y,x) + h'(p_t,z,y,x) + h"(p_t,z,y,x)
 * e_{t+1}(z,y,x) = h(e_t,z,y,x) + h'(e_t,z,y,x) + h"(e_t,z,y,x)
 * u_{t+1}(z,y,x) = h(u_t,z,y,x) + h'(u_t,z,y,x) + h"(u_t,z,y,x)
 * v_{t+1}(z,y,x) = h(v_t,z,y,x) + h'(v_t,z,y,x) + h"(v_t,z,y,x)
 * w_{t+1}(z,y,x) = h(w_t,z,y,x) + h'(w_t,z,y,x) + h"(w_t,z,y,x)
 *
 *  Computing face-centered, flux values based on cell-centered values
 *  g(component,z,y,x) is a stencil operation that looks like the following:
 *  g(c,z,y,x) = factor1*
 *        (c[z][y][x-2]+7*(c[z][y][x-1]+c[z][y][x])+c[z][y][x+1])
 *  similarly for g' and g"
 *  g'(c,z,y,x) = factor1*
 *        (c[z][y-2][x]+7*(c[z][y-1][x]+c[z][y][x])+c[z][y+1][x])
 *  g"(c,z,y,x) = factor1*
 *        (c[z-2][y][x]+7*(c[z-1][y][x]+c[z][y][x])+c[z+1][y][x])
 *
 *  Computing cell-centered values based on face-centered flux values
 *  h(component,z,y,x) is a stencil operation that looks like the following:
 *  h(c,z,y,x)  = factor2*
 *                  (g(c,z,y,x+1)*g(u_t,z,y,x+1)-g(c,z,y,x)*g(u_t,z,y,x))
 *  h'(c,z,y,x) = factor2*
 *                  (g'(c,z,y+1,x)*g'(v_t,z,y+1,x)-g'(c,z,y,x)*g'(v_t,z,y,x))
 *  h"(c,z,y,x) = factor2*
 *                  (g"(c,z+1,y,x)*g"(w_t,z+1,y,x)-g"(c,z,y,x)*g"(w_t,z,y,x))
 *
 *  in this example code we omit some space and time saving optimizations
 *  in order to make the code easy to learn.  FIXME: are these correct?
 *  Step 1 is to calculate all of the g() values
 *  Step 2 multiplies the values together for the first column in the
 *         equations above
 *  Step 3 Return to Step 1 for g' and then for g"
 *
 *  The following is a table describing how the notation above
 *  maps to the storage in code that we are using below
 *  value      name | variable name
 *  ----------------------------------
 *  p_{t+1}         | new_data[0]
 *  e_{t+1}         | new_data[1]
 *  u_{t+1}         | new_data[2]
 *  v_{t+1}         | new_data[3]
 *  w_{t+1}         | new_data[4]
 *  g(p_t)          | g_cache[0]
 *  g(e_t)          | g_cache[1]
 *  ... same pattern for u,v,w
 *  g'(p_t)         | g_cache[0]
 *  ... continue pattern
 *  g"(p_t)         | g_cache[0]
 *
 *  g_cache can be reused because we accumulate into new_data between
 *  iterations
 ****************************************************************************/

Real** mini_flux_div_lc(Real** old_boxes,Real** new_boxes,
                        Configuration& config,Measurements& measurements) {
    double time_spent;
    struct timeval  tv1, tv2;
    int idx,ix,iy,iz;
    int numCell= config.getInt("numCell");
    int numBox= config.getInt("numBox");
    int nGhost = NGHOST;
    int numComp= NCOMP;
    // The size of the 3D data is (numCell+2*nGhost)^3
    int full_numCell = numCell+2*nGhost;
    int full_numCell2 = full_numCell*full_numCell;
    int full_numCell3 = full_numCell*full_numCell*full_numCell;
    //int totalCells = full_numCell*full_numCell*full_numCell;
    int flux_totalSize = numCell*numCell*(numCell+1);

    // allocate the g cache -- to save all of the g() calculations
    // There is one value per face in the box. (numCell+1) is the
    // number of faces in each direction.

    // DATA Usage:
    // in this code we are reusing the g_cache to store 2 values per variable
    // (component).
    // Additionally we are accumulating the result into [p,e,u,v,w]_DATA

    // iterate over all of the boxes
    gettimeofday(&tv1, NULL);
    Real* g_cache = NULL;

    #pragma omp parallel for default(shared) private(idx) firstprivate(g_cache)
    for (idx=0; idx < numBox; idx++) {
        Real* old_box = old_boxes[idx];
        Real* new_box = new_boxes[idx];
        if(g_cache == NULL){
          g_cache = (Real*)malloc(sizeof(Real)*numCell*numCell*(numCell+1)*numComp);
        }

        //----------------------  x-direction

        // Iterate over faces and calculate g()
        //---------------------- x-direction
        #pragma omp-lc loopchain schedule(fuse)
        {
            // FLUX1X
            #pragma omp-lc for domain(0:numCell-1,0:numCell-1,0:numCell) with (iz,iy,ix) write CACHEX {(iz,iy,ix)}, \
            read OLD {(iz,iy,ix-2),(iz,iy,ix-1),(iz,iy,ix),(iz,iy,ix+1)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix <= numCell; ix++) {
                        p_CACHE_x(iz, iy, ix) = factor1 *
                                                (p_DATA_old(iz, iy, ix - 2) +
                                                 7 * (p_DATA_old(iz, iy, ix - 1) + p_DATA_old(iz, iy, ix)) +
                                                 p_DATA_old(iz, iy, ix + 1));
                        e_CACHE_x(iz, iy, ix) = factor1 *
                                                (e_DATA_old(iz, iy, ix - 2) +
                                                 7 * (e_DATA_old(iz, iy, ix - 1) + e_DATA_old(iz, iy, ix)) +
                                                 e_DATA_old(iz, iy, ix + 1));
                        u_CACHE_x(iz, iy, ix) = factor1 *
                                                (u_DATA_old(iz, iy, ix - 2) +
                                                 7 * (u_DATA_old(iz, iy, ix - 1) + u_DATA_old(iz, iy, ix)) +
                                                 u_DATA_old(iz, iy, ix + 1));
                        v_CACHE_x(iz, iy, ix) = factor1 *
                                                (v_DATA_old(iz, iy, ix - 2) +
                                                 7 * (v_DATA_old(iz, iy, ix - 1) + v_DATA_old(iz, iy, ix)) +
                                                 v_DATA_old(iz, iy, ix + 1));
                        w_CACHE_x(iz, iy, ix) = factor1 *
                                                (w_DATA_old(iz, iy, ix - 2) +
                                                 7 * (w_DATA_old(iz, iy, ix - 1) + w_DATA_old(iz, iy, ix)) +
                                                 w_DATA_old(iz, iy, ix + 1));
                    }
                }
            }

            // compute part of h() and reuse space for g()
            // u_CACHE_x(iz,iy,ix) must be written to last for reuse to work.
            // FLUX2X
            #pragma omp-lc for domain(0:numCell-1,0:numCell-1,0:numCell) with (iz,iy,ix) write CACHEX {(iz,iy,ix)}, \
            read CACHEX {(iz,iy,ix)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix <= numCell; ix++) {
                        p_CACHE_x(iz, iy, ix) *= factor2 * u_CACHE_x(iz, iy, ix);
                        e_CACHE_x(iz, iy, ix) *= factor2 * u_CACHE_x(iz, iy, ix);
                        v_CACHE_x(iz, iy, ix) *= factor2 * u_CACHE_x(iz, iy, ix);
                        w_CACHE_x(iz, iy, ix) *= factor2 * u_CACHE_x(iz, iy, ix);
                        u_CACHE_x(iz, iy, ix) *= factor2 * u_CACHE_x(iz, iy, ix);
                    }
                }
            }

            // finish h()
            // iterate over cells
            // and save the difference of the two adjacent faces into the cell data
            // DIFFX
            #pragma omp-lc for domain(0:numCell-1,0:numCell-1,0:numCell-1) with (iz,iy,ix) write NEW {(iz,iy,ix)}, \
            read NEW {(iz,iy,ix)}, read CACHEX {(iz,iy,ix+1),(iz,iy,ix)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_DATA_new(iz, iy, ix) += p_CACHE_x(iz, iy, ix + 1) - p_CACHE_x(iz, iy, ix);
                        e_DATA_new(iz, iy, ix) += e_CACHE_x(iz, iy, ix + 1) - e_CACHE_x(iz, iy, ix);
                        u_DATA_new(iz, iy, ix) += u_CACHE_x(iz, iy, ix + 1) - u_CACHE_x(iz, iy, ix);
                        v_DATA_new(iz, iy, ix) += v_CACHE_x(iz, iy, ix + 1) - v_CACHE_x(iz, iy, ix);
                        w_DATA_new(iz, iy, ix) += w_CACHE_x(iz, iy, ix + 1) - w_CACHE_x(iz, iy, ix);
                    }
                }
            }

            //---------------------- y-direction
            // Iterate over faces and calculate g'()
            // FLUX1Y
            #pragma omp-lc for domain(0:numCell-1,0:numCell,0:numCell-1) with (iz,iy,ix) write CACHEY {(iz,iy,ix)}, \
            read OLD {(iz,iy-2,ix),(iz,iy-1,ix),(iz,iy,ix),(iz,iy+1,ix)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy <= numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_CACHE_y(iz, iy, ix) = factor1 *
                                                (p_DATA_old(iz, iy - 2, ix) +
                                                 7 * (p_DATA_old(iz, iy - 1, ix) + p_DATA_old(iz, iy, ix)) +
                                                 p_DATA_old(iz, iy + 1, ix));
                        e_CACHE_y(iz, iy, ix) = factor1 *
                                                (e_DATA_old(iz, iy - 2, ix) +
                                                 7 * (e_DATA_old(iz, iy - 1, ix) + e_DATA_old(iz, iy, ix)) +
                                                 e_DATA_old(iz, iy + 1, ix));
                        u_CACHE_y(iz, iy, ix) = factor1 *
                                                (u_DATA_old(iz, iy - 2, ix) +
                                                 7 * (u_DATA_old(iz, iy - 1, ix) + u_DATA_old(iz, iy, ix)) +
                                                 u_DATA_old(iz, iy + 1, ix));
                        v_CACHE_y(iz, iy, ix) = factor1 *
                                                (v_DATA_old(iz, iy - 2, ix) +
                                                 7 * (v_DATA_old(iz, iy - 1, ix) + v_DATA_old(iz, iy, ix)) +
                                                 v_DATA_old(iz, iy + 1, ix));
                        w_CACHE_y(iz, iy, ix) = factor1 *
                                                (w_DATA_old(iz, iy - 2, ix) +
                                                 7 * (w_DATA_old(iz, iy - 1, ix) + w_DATA_old(iz, iy, ix)) +
                                                 w_DATA_old(iz, iy + 1, ix));
                    }
                }
            }

            // compute part of h'() and reuse space for g'()
            // v_CACHE_x(iz,iy,ix) must be written to last for reuse to work.
            // FLUX2Y
            #pragma omp-lc for domain(0:numCell-1,0:numCell,0:numCell-1) with (iz,iy,ix) write CACHEY {(iz,iy,ix)}, \
            read CACHEY {(iz,iy,ix)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy <= numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_CACHE_y(iz, iy, ix) = factor2 * p_CACHE_y(iz, iy, ix) * v_CACHE_y(iz, iy, ix);
                        e_CACHE_y(iz, iy, ix) = factor2 * e_CACHE_y(iz, iy, ix) * v_CACHE_y(iz, iy, ix);
                        u_CACHE_y(iz, iy, ix) = factor2 * u_CACHE_y(iz, iy, ix) * v_CACHE_y(iz, iy, ix);
                        w_CACHE_y(iz, iy, ix) = factor2 * w_CACHE_y(iz, iy, ix) * v_CACHE_y(iz, iy, ix);
                        v_CACHE_y(iz, iy, ix) = factor2 * v_CACHE_y(iz, iy, ix) * v_CACHE_y(iz, iy, ix);
                    }
                }
            }

            // finish h'()
            // iterate over cells
            // and save the difference of the two adjacent faces into the cell data
            // DIFFY
            #pragma omp-lc for domain(0:numCell-1,0:numCell-1,0:numCell-1) with (iz,iy,ix) write NEW {(iz,iy,ix)}, \
            read NEW {(iz,iy,ix)}, read CACHEY {(iz,iy+1,ix),(iz,iy,ix)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_DATA_new(iz, iy, ix) += p_CACHE_y(iz, iy + 1, ix) - p_CACHE_y(iz, iy, ix);
                        e_DATA_new(iz, iy, ix) += e_CACHE_y(iz, iy + 1, ix) - e_CACHE_y(iz, iy, ix);
                        u_DATA_new(iz, iy, ix) += u_CACHE_y(iz, iy + 1, ix) - u_CACHE_y(iz, iy, ix);
                        v_DATA_new(iz, iy, ix) += v_CACHE_y(iz, iy + 1, ix) - v_CACHE_y(iz, iy, ix);
                        w_DATA_new(iz, iy, ix) += w_CACHE_y(iz, iy + 1, ix) - w_CACHE_y(iz, iy, ix);
                    }
                }
            }

            //----------------------  z-direction
            // Iterate over faces and calculate g"()
            // FLUX1Z
            #pragma omp-lc for domain(0:numCell,0:numCell-1,0:numCell-1) with (iz,iy,ix) write CACHEZ {(iz,iy,ix)}, \
            read OLD {(iz-2,iy,ix),(iz-1,iy,ix),(iz,iy,ix),(iz+1,iy,ix)}
            for (iz = 0; iz <= numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_CACHE_z(iz, iy, ix) = factor1 *
                                                (p_DATA_old(iz - 2, iy, ix) +
                                                 7 * (p_DATA_old(iz - 1, iy, ix) + p_DATA_old(iz, iy, ix)) +
                                                 p_DATA_old(iz + 1, iy, ix));
                        e_CACHE_z(iz, iy, ix) = factor1 *
                                                (e_DATA_old(iz - 2, iy, ix) +
                                                 7 * (e_DATA_old(iz - 1, iy, ix) + e_DATA_old(iz, iy, ix)) +
                                                 e_DATA_old(iz + 1, iy, ix));
                        u_CACHE_z(iz, iy, ix) = factor1 *
                                                (u_DATA_old(iz - 2, iy, ix) +
                                                 7 * (u_DATA_old(iz - 1, iy, ix) + u_DATA_old(iz, iy, ix)) +
                                                 u_DATA_old(iz + 1, iy, ix));
                        v_CACHE_z(iz, iy, ix) = factor1 *
                                                (v_DATA_old(iz - 2, iy, ix) +
                                                 7 * (v_DATA_old(iz - 1, iy, ix) + v_DATA_old(iz, iy, ix)) +
                                                 v_DATA_old(iz + 1, iy, ix));
                        w_CACHE_z(iz, iy, ix) = factor1 *
                                                (w_DATA_old(iz - 2, iy, ix) +
                                                 7 * (w_DATA_old(iz - 1, iy, ix) + w_DATA_old(iz, iy, ix)) +
                                                 w_DATA_old(iz + 1, iy, ix));
                    }
                }
            }

            // compute part of h"() and reuse space for g"()
            // w_CACHE_x(iz,iy,ix) must be written to last for reuse to work.
            // FLUX2Y
            #pragma omp-lc for domain(0:numCell,0:numCell-1,0:numCell-1) with (iz,iy,ix) write CACHEZ {(iz,iy,ix)}, \
            read CACHEZ {(iz,iy,ix)}
            for (iz = 0; iz <= numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_CACHE_z(iz, iy, ix) = factor2 * p_CACHE_z(iz, iy, ix) * w_CACHE_z(iz, iy, ix);
                        e_CACHE_z(iz, iy, ix) = factor2 * e_CACHE_z(iz, iy, ix) * w_CACHE_z(iz, iy, ix);
                        u_CACHE_z(iz, iy, ix) = factor2 * u_CACHE_z(iz, iy, ix) * w_CACHE_z(iz, iy, ix);
                        v_CACHE_z(iz, iy, ix) = factor2 * v_CACHE_z(iz, iy, ix) * w_CACHE_z(iz, iy, ix);
                        w_CACHE_z(iz, iy, ix) = factor2 * w_CACHE_z(iz, iy, ix) * w_CACHE_z(iz, iy, ix);
                    }
                }
            }

            // finish h"()
            // iterate over cells
            // and save the difference of the two adjacent faces into the cell data
            // DIFFY
            #pragma omp-lc for domain(0:numCell-1,0:numCell-1,0:numCell-1) with (iz,iy,ix) write NEW {(iz,iy,ix)}, \
            read NEW {(iz,iy,ix)}, read CACHEZ {(iz+1,iy,ix),(iz,iy,ix)}
            for (iz = 0; iz < numCell; iz++) {
                for (iy = 0; iy < numCell; iy++) {
                    for (ix = 0; ix < numCell; ix++) {
                        p_DATA_new(iz, iy, ix) += p_CACHE_z(iz + 1, iy, ix) - p_CACHE_z(iz, iy, ix);
                        e_DATA_new(iz, iy, ix) += e_CACHE_z(iz + 1, iy, ix) - e_CACHE_z(iz, iy, ix);
                        u_DATA_new(iz, iy, ix) += u_CACHE_z(iz + 1, iy, ix) - u_CACHE_z(iz, iy, ix);
                        v_DATA_new(iz, iy, ix) += v_CACHE_z(iz + 1, iy, ix) - v_CACHE_z(iz, iy, ix);
                        w_DATA_new(iz, iy, ix) += w_CACHE_z(iz + 1, iy, ix) - w_CACHE_z(iz, iy, ix);
                    }
                }
            }
        }
    }

    // Print out computed values
    for (int idx = 0; idx < numBox; idx++) {
        Real *hope_box = new_boxes[idx];
        for (int iz = 0; iz < numCell; iz++) {
            for (int iy = 0; iy < numCell; iy++) {
                for (int ix = 0; ix < numCell; ix++) {
                    for (int c = 0; c < numComp; c++) {
                        Real hope_val = *(GET_VAL_PTR(hope_box, c, iz, iy, ix));
                        printf("%d,%d,%d,%d,%f\n", c, iz, iy, ix, hope_val);
                    }
                }
            }
        }
    }

    gettimeofday(&tv2, NULL);
    double time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
                  (double) (tv2.tv_sec - tv1.tv_sec);
    measurements.setField("RunTime",time);

    return new_boxes;
}
