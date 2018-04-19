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
Real **mini_flux_div_lc(Real **old_boxes,Real **new_boxes,class Configuration &config,class Measurements &measurements);
using namespace std;

int main(int argc,char **argv)
{
  int verify = 0;
  int flux_totalSize;
  int idx;
  int iz;
  int iy;
  int ix;
  int index;
  int v;
  int c;
  Real **result_data;
  struct timeval tv1;
  struct timeval tv2;
// Constructor for parsing the command line arguments
  class Configuration config;
/**************************************************************************
    **  Parameter options. Help and verify are constructor defaults.  *********
    **************************************************************************/
  config .  addParamInt ("numCell",'C',128,"--numCell, the number of cells in a single dimension of a single box");
  config .  addParamInt ("numBox",'B',32,"--numBox, the number of independent boxes to process");
  config .  addParamInt ("num_threads",'p',1,"-p <num_threads>, number of cores");
//Constructor for Measurements
  class Measurements measurements;
  config .  parse (argc,argv);
  int numCell = config .  getInt ("numCell");
  int numBox = config .  getInt ("numBox");
  int nGhost = 2;
  int numComp = 5;
// t = config.getInt("tests");
// v = config.getInt("verify");
  if (numCell < 0) {
    fprintf(stderr,"The value of numCell has to be a positive number %d\n",numCell);
    exit(- 1);
  }
  if (numBox < 0) {
    fprintf(stderr,"The value of numBox has to be a positive number %d\n",numBox);
    exit(- 1);
  }
  setNumThreads((config .  getInt ("num_threads")));
  Real **old_boxes;
  Real **new_boxes;
//allocate call
  allocateAndInitialize(&old_boxes,&new_boxes,config);
// Calling Benchmark Function
  mini_flux_div_lc(old_boxes,new_boxes,config,measurements);
//Calling Verification from Util
  if (config .  getBool ("v")) {
    if (verifyResult(new_boxes,config)) {
      measurements .  setField ("verification","SUCCESS");
    }
     else {
      measurements .  setField ("verification","FAILURE");
    }
  }
//get results from measurements
  std::string result = measurements .  toLDAPString ();
  std::string config_in = config .  toLDAPString ();
  ((cout<<config_in)<<result) << std::endl;
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

Real **mini_flux_div_lc(Real **old_boxes,Real **new_boxes,class Configuration &config,class Measurements &measurements)
{
  double time_spent;
  struct timeval tv1;
  struct timeval tv2;
  int idx;
  int ix;
  int iy;
  int iz;
  int numCell = config .  getInt ("numCell");
  int numBox = config .  getInt ("numBox");
  int nGhost = 2;
  int numComp = 5;
// The size of the 3D data is (numCell+2*nGhost)^3
  int full_numCell = numCell + 2 * nGhost;
  int full_numCell2 = full_numCell * full_numCell;
  int full_numCell3 = full_numCell * full_numCell * full_numCell;
  //int totalCells = full_numCell * full_numCell * full_numCell;
  int flux_totalSize = numCell * numCell * (numCell + 1);
// allocate the g cache -- to save all of the g() calculations
// There is one value per face in the box. (numCell+1) is the
// number of faces in each direction.
  Real *gx_cache = 0L;
  Real *gy_cache = 0L;
  Real *gz_cache = 0L;
// DATA Usage:
// in this code we are reusing the g_cache to store 2 values per variable
// (component).
// Additionally we are accumulating the result into [p,e,u,v,w]_DATA
// iterate over all of the boxes
  gettimeofday(&tv1,0L);
  
#pragma omp parallel for default(shared) private(idx) firstprivate(gx_cache,gy_cache,gz_cache)
  for (idx = 0; idx < numBox; idx++) {
    Real *old_box = old_boxes[idx];
    Real *new_box = new_boxes[idx];
    if (gx_cache == 0L) {
      gx_cache = ((Real *)(malloc(sizeof(Real ) * numCell * numCell * (numCell + 1) * numComp)));
      gy_cache = ((Real *)(malloc(sizeof(Real ) * numCell * numCell * (numCell + 1) * numComp)));
      gz_cache = ((Real *)(malloc(sizeof(Real ) * numCell * numCell * (numCell + 1) * numComp)));
    }
//----------------------  x-direction
// Iterate over faces and calculate g()
//---------------------- x-direction
//#pragma omplc loopchain schedule(fuse((0,0,0), (0,0,0), (0,0,1), (0,0,0), (0,0,0), (0,1,1), (0,0,0), (0,0,0), (1,1,1)))
//#pragma omplc loopchain schedule(fuse((0,0,0), (0,0,0), (0,0,1), (0,0,0), (0,0,0), (0,1,0), (0,0,0), (0,0,0), (1,0,0)))
    
#pragma omplc loopchain schedule( fuse() )
{
{
        for (int omplc_gen_iter_2 = 1; omplc_gen_iter_2 <= numCell; omplc_gen_iter_2 = omplc_gen_iter_2 + 1) {
          for (int omplc_gen_iter_3 = 1; omplc_gen_iter_3 <= numCell; omplc_gen_iter_3 = omplc_gen_iter_3 + 1) {{
              int ix = omplc_gen_iter_3 - 1;
              int iy = omplc_gen_iter_2 - 1;
              int iz = 0;
               *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 0 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 0 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 0 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
               *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 1 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 1 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 1 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
               *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 2 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 2 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 2 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
               *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 3 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 3 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 3 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
               *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 4 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 4 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 4 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
            }
{
              int ix = omplc_gen_iter_3 - 1;
              int iy = omplc_gen_iter_2 - 1;
              int iz = 0;
               *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
               *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
               *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
               *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
               *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
            }
          }
        }
        for (int omplc_gen_iter_1 = 1; omplc_gen_iter_1 <= numCell; omplc_gen_iter_1 = omplc_gen_iter_1 + 1) {
          for (int omplc_gen_iter_3 = 1; omplc_gen_iter_3 <= numCell; omplc_gen_iter_3 = omplc_gen_iter_3 + 1) {{
              int ix = omplc_gen_iter_3 - 1;
              int iy = 0;
              int iz = omplc_gen_iter_1 - 1;
               *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
               *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
               *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
               *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
               *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
            }
{
              int ix = omplc_gen_iter_3 - 1;
              int iy = 0;
              int iz = omplc_gen_iter_1 - 1;
               *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
               *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
               *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
               *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
               *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
            }
          }
          for (int omplc_gen_iter_2 = 1; omplc_gen_iter_2 <= numCell; omplc_gen_iter_2 = omplc_gen_iter_2 + 1) {{{
                int ix = 0;
                int iy = omplc_gen_iter_2 - 1;
                int iz = omplc_gen_iter_1 - 1;
                 *(gx_cache + 0 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                 *(gx_cache + 1 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                 *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                 *(gx_cache + 3 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                 *(gx_cache + 4 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
              }
{
                int ix = 0;
                int iy = omplc_gen_iter_2 - 1;
                int iz = omplc_gen_iter_1 - 1;
                 *(gx_cache + 0 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                 *(gx_cache + 1 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                 *(gx_cache + 3 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                 *(gx_cache + 4 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                 *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
              }
            }
            for (int omplc_gen_iter_3 = 1; omplc_gen_iter_3 <= numCell; omplc_gen_iter_3 = omplc_gen_iter_3 + 1) {{{{{{{{{
                              int ix = omplc_gen_iter_3;
                              int iy = omplc_gen_iter_2 - 1;
                              int iz = omplc_gen_iter_1 - 1;
                               *(gx_cache + 0 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                               *(gx_cache + 1 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                               *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                               *(gx_cache + 3 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                               *(gx_cache + 4 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) = 1. / 12. * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 2 + nGhost)) + ((double )7) * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix - 1 + nGhost)) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + 1 + nGhost)));
                            }
{
                              int ix = omplc_gen_iter_3;
                              int iy = omplc_gen_iter_2 - 1;
                              int iz = omplc_gen_iter_1 - 1;
                               *(gx_cache + 0 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                               *(gx_cache + 1 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                               *(gx_cache + 3 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                               *(gx_cache + 4 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                               *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix) *= 2. *  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                            }
                          }
{
                            int ix = omplc_gen_iter_3 - 1;
                            int iy = omplc_gen_iter_2 - 1;
                            int iz = omplc_gen_iter_1 - 1;
                             *(new_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gx_cache + 0 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + (ix + 1)) -  *(gx_cache + 0 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                             *(new_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gx_cache + 1 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + (ix + 1)) -  *(gx_cache + 1 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                             *(new_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + (ix + 1)) -  *(gx_cache + 2 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                             *(new_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gx_cache + 3 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + (ix + 1)) -  *(gx_cache + 3 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                             *(new_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gx_cache + 4 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + (ix + 1)) -  *(gx_cache + 4 * (numCell + ((int )(0 == 2))) * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iz * (numCell + ((int )(0 == 1))) * (numCell + ((int )(0 == 0))) + iy * (numCell + ((int )(0 == 0))) + ix);
                          }
                        }
{
                          int ix = omplc_gen_iter_3 - 1;
                          int iy = omplc_gen_iter_2;
                          int iz = omplc_gen_iter_1 - 1;
                           *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
                           *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
                           *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
                           *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
                           *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 1. / 12. * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 2 + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy - 1 + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + 1 + nGhost) * full_numCell + (ix + nGhost)));
                        }
                      }
{
                        int ix = omplc_gen_iter_3 - 1;
                        int iy = omplc_gen_iter_2;
                        int iz = omplc_gen_iter_1 - 1;
                         *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                         *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                         *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                         *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                         *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) = 2. *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix) *  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                      }
                    }
{
                      int ix = omplc_gen_iter_3 - 1;
                      int iy = omplc_gen_iter_2 - 1;
                      int iz = omplc_gen_iter_1 - 1;
                       *(new_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + (iy + 1) * (numCell + ((int )(1 == 0))) + ix) -  *(gy_cache + 0 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                       *(new_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + (iy + 1) * (numCell + ((int )(1 == 0))) + ix) -  *(gy_cache + 1 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                       *(new_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + (iy + 1) * (numCell + ((int )(1 == 0))) + ix) -  *(gy_cache + 2 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                       *(new_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + (iy + 1) * (numCell + ((int )(1 == 0))) + ix) -  *(gy_cache + 3 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                       *(new_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + (iy + 1) * (numCell + ((int )(1 == 0))) + ix) -  *(gy_cache + 4 * (numCell + ((int )(1 == 2))) * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iz * (numCell + ((int )(1 == 1))) * (numCell + ((int )(1 == 0))) + iy * (numCell + ((int )(1 == 0))) + ix);
                    }
                  }
{
                    int ix = omplc_gen_iter_3 - 1;
                    int iy = omplc_gen_iter_2 - 1;
                    int iz = omplc_gen_iter_1;
                     *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 0 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 0 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 0 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
                     *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 1 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 1 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 1 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
                     *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 2 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 2 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 2 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
                     *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 3 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 3 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 3 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
                     *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 1. / 12. * ( *(old_box + 4 * full_numCell3 + (iz - 2 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) + ((double )7) * ( *(old_box + 4 * full_numCell3 + (iz - 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +  *(old_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost))) +  *(old_box + 4 * full_numCell3 + (iz + 1 + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)));
                  }
                }
{
                  int ix = omplc_gen_iter_3 - 1;
                  int iy = omplc_gen_iter_2 - 1;
                  int iz = omplc_gen_iter_1;
                   *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                   *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                   *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                   *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                   *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) = 2. *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) *  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                }
              }
{
                int ix = omplc_gen_iter_3 - 1;
                int iy = omplc_gen_iter_2 - 1;
                int iz = omplc_gen_iter_1 - 1;
                 *(new_box + 0 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + (iz + 1) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) -  *(gz_cache + 0 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                 *(new_box + 1 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + (iz + 1) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) -  *(gz_cache + 1 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                 *(new_box + 2 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + (iz + 1) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) -  *(gz_cache + 2 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                 *(new_box + 3 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + (iz + 1) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) -  *(gz_cache + 3 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
                 *(new_box + 4 * full_numCell3 + (iz + nGhost) * full_numCell2 + (iy + nGhost) * full_numCell + (ix + nGhost)) +=  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + (iz + 1) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix) -  *(gz_cache + 4 * (numCell + ((int )(2 == 2))) * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iz * (numCell + ((int )(2 == 1))) * (numCell + ((int )(2 == 0))) + iy * (numCell + ((int )(2 == 0))) + ix);
              }
            }
          }
        }
      }
    }
  }
  gettimeofday(&tv2,0L);
  double time = ((double )(tv2 . tv_usec - tv1 . tv_usec)) / 1000000 + ((double )(tv2 . tv_sec - tv1 . tv_sec));
  measurements .  setField ("RunTime",time);
}
