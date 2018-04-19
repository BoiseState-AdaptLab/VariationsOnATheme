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

#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "../common/Configuration.h"


typedef double Real;
#define DIMS 3
#define NCOMP 5
#define NGHOST 2

//#define floord(x, y)  (int)floor((x),(y))
#if ! defined ASSUME_POSITIVE_INTMOD
#define ASSUME_POSITIVE_INTMOD 0
#endif

#if ASSUME_POSITIVE_INTMOD
#define intDiv(x,y) (eassert(((x)%(y)) >= 0), ((x)/(y)))
#define intMod(x,y) (eassert(((x)%(y)) >= 0), ((x)%(y)))
#else
#define intDiv_(x,y)  ((((x)%(y))>=0) ? ((x)/(y)) : (((x)/(y)) -1))
#define intMod_(x,y)  ((((x)%(y))>=0) ? ((x)%(y)) : (((x)%(y)) +y))
#define checkIntDiv(x,y) (eassert((y) > 0 && intMod_((x),(y)) >= 0 && intMod_((x),(y)) <= (y) && x==((y)*intDiv_((x),(y)) + intMod_((x),(y)))))
#define intDiv(x,y) (checkIntDiv((x),(y)), intDiv_((x),(y)))
#define intMod(x,y) (checkIntDiv((x),(y)), intMod_((x),(y)))
#endif

#if defined CUDA
#undef intDiv
#undef intMod
//#define intDiv(x,y) ((((x)%(y))>=0) ? ((x)/(y)) : (((x)/(y)) -1))
//#define intMod(x,y) ((((x)%(y))>=0) ? ((x)%(y)) : (((x)%(y)) +y))
#define intDiv(x,y) ( ((x)/(y)))
#define intMod(x,y) ( ((x)%(y)))
#endif


#define ceild(n, d) intDiv_((n), (d)) + ((intMod_((n),(d))>0)?1:0)
#define floord(n, d)  intDiv_((n), (d))

#define dx 0.5
#define factor1  (1./12.)
#define factor2  2.
#define compMultiplier  1.

#define GET_OFFSET(c,z,y,x) (c)*full_numCell3 + ((z)+nGhost)*full_numCell2 + ((y)+nGhost)*full_numCell + ((x)+nGhost)

#undef GET_VAL_PTR
#define  GET_VAL_PTR(b,c,z,y,x) (b)+\
                                (c)*full_numCell3+\
                                ((z)+nGhost)*full_numCell2+\
                                ((y)+nGhost)*full_numCell+\
                                ((x)+nGhost)

#define GET_TILED_FACE_VAL_PTR(d,b,c,z,y,x) (b)+\
  (c)*(cellsPerTile+((d)==2))*(cellsPerTile+((d)==1))*(cellsPerTile+((d)==0)) +\
        (z)*(cellsPerTile+((d)==1))*(cellsPerTile+((d)==0))+\
        (y)*(cellsPerTile+((d)==0))+\
        (x)  

#define GET_FACE_VAL_PTR(d,b,c,z,y,x) (b)+\
        (c)*(numCell+((d)==2))*(numCell+((d)==1))*(numCell+((d)==0)) +\
        (z)*(numCell+((d)==1))*(numCell+((d)==0))+\
        (y)*(numCell+((d)==0))+\
        (x) 

#define GET_VEL_VAL_PTR(d,v,z,y,x) (v) +\
        (z)*(numCell+((d)==1))*(numCell+((d)==0))+\
        (y)*(numCell+((d)==0))+\
        (x) 



/*************************************************************************************
 * defines for mini_flux_div_basic code -- this makes for easier reading and debugging
 *************************************************************************************/
#define p_DATA_old(z,y,x) *(GET_VAL_PTR(old_box,0,z,y,x))
#define e_DATA_old(z,y,x) *(GET_VAL_PTR(old_box,1,z,y,x))
#define u_DATA_old(z,y,x) *(GET_VAL_PTR(old_box,2,z,y,x))
#define v_DATA_old(z,y,x) *(GET_VAL_PTR(old_box,3,z,y,x))
#define w_DATA_old(z,y,x) *(GET_VAL_PTR(old_box,4,z,y,x))

#define p_DATA_new(z,y,x) *(GET_VAL_PTR(new_box,0,z,y,x))
#define e_DATA_new(z,y,x) *(GET_VAL_PTR(new_box,1,z,y,x))
#define u_DATA_new(z,y,x) *(GET_VAL_PTR(new_box,2,z,y,x))
#define v_DATA_new(z,y,x) *(GET_VAL_PTR(new_box,3,z,y,x))
#define w_DATA_new(z,y,x) *(GET_VAL_PTR(new_box,4,z,y,x))

#define p_CACHE_x(z,y,x) *(GET_FACE_VAL_PTR(0,g_cache,0,z,y,x))
#define e_CACHE_x(z,y,x) *(GET_FACE_VAL_PTR(0,g_cache,1,z,y,x))
#define u_CACHE_x(z,y,x) *(GET_FACE_VAL_PTR(0,g_cache,2,z,y,x))
#define v_CACHE_x(z,y,x) *(GET_FACE_VAL_PTR(0,g_cache,3,z,y,x))
#define w_CACHE_x(z,y,x) *(GET_FACE_VAL_PTR(0,g_cache,4,z,y,x))

#define p_CACHE_y(z,y,x) *(GET_FACE_VAL_PTR(1,g_cache,0,z,y,x))
#define e_CACHE_y(z,y,x) *(GET_FACE_VAL_PTR(1,g_cache,1,z,y,x))
#define u_CACHE_y(z,y,x) *(GET_FACE_VAL_PTR(1,g_cache,2,z,y,x))
#define v_CACHE_y(z,y,x) *(GET_FACE_VAL_PTR(1,g_cache,3,z,y,x))
#define w_CACHE_y(z,y,x) *(GET_FACE_VAL_PTR(1,g_cache,4,z,y,x))

#define p_CACHE_z(z,y,x) *(GET_FACE_VAL_PTR(2,g_cache,0,z,y,x))
#define e_CACHE_z(z,y,x) *(GET_FACE_VAL_PTR(2,g_cache,1,z,y,x))
#define u_CACHE_z(z,y,x) *(GET_FACE_VAL_PTR(2,g_cache,2,z,y,x))
#define v_CACHE_z(z,y,x) *(GET_FACE_VAL_PTR(2,g_cache,3,z,y,x))
#define w_CACHE_z(z,y,x) *(GET_FACE_VAL_PTR(2,g_cache,4,z,y,x))

#define p_CACHE3_x(z,y,x) *(GET_FACE_VAL_PTR(0,gx_cache,0,z,y,x))
#define e_CACHE3_x(z,y,x) *(GET_FACE_VAL_PTR(0,gx_cache,1,z,y,x))
#define u_CACHE3_x(z,y,x) *(GET_FACE_VAL_PTR(0,gx_cache,2,z,y,x))
#define v_CACHE3_x(z,y,x) *(GET_FACE_VAL_PTR(0,gx_cache,3,z,y,x))
#define w_CACHE3_x(z,y,x) *(GET_FACE_VAL_PTR(0,gx_cache,4,z,y,x))

#define p_CACHE3_y(z,y,x) *(GET_FACE_VAL_PTR(1,gy_cache,0,z,y,x))
#define e_CACHE3_y(z,y,x) *(GET_FACE_VAL_PTR(1,gy_cache,1,z,y,x))
#define u_CACHE3_y(z,y,x) *(GET_FACE_VAL_PTR(1,gy_cache,2,z,y,x))
#define v_CACHE3_y(z,y,x) *(GET_FACE_VAL_PTR(1,gy_cache,3,z,y,x))
#define w_CACHE3_y(z,y,x) *(GET_FACE_VAL_PTR(1,gy_cache,4,z,y,x))

#define p_CACHE3_z(z,y,x) *(GET_FACE_VAL_PTR(2,gz_cache,0,z,y,x))
#define e_CACHE3_z(z,y,x) *(GET_FACE_VAL_PTR(2,gz_cache,1,z,y,x))
#define u_CACHE3_z(z,y,x) *(GET_FACE_VAL_PTR(2,gz_cache,2,z,y,x))
#define v_CACHE3_z(z,y,x) *(GET_FACE_VAL_PTR(2,gz_cache,3,z,y,x))
#define w_CACHE3_z(z,y,x) *(GET_FACE_VAL_PTR(2,gz_cache,4,z,y,x))

bool verifyResult(Real** result_data,Configuration& config);

void toCSV(const char * csvFile, Real** result_data,Configuration& config);

Real** mini_flux_div_truth(Real** old_boxes, Real** new_boxes,
                           Configuration& config);


bool allocateAndInitialize(Real*** old_boxes_in,Real *** new_boxes_in,
                           Configuration& config);

bool setNumThreads(int nThreads);
bool freeMemory(Real*** old_boxes_in,Real *** new_boxes_in,
                           Configuration& config);
