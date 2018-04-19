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

#ifndef _MINIFLUXUTIL_H_
#define _MINIFLUXUTIL_H_

#include "ISLFunctions.h"

#define ABS(x)  ((x)<0)?-(x):(x)
#define EPS .000001
#define N_A  1E20

#define dx 0.5
#define factor1 (1.0/12.0)
#define factor2 2.0

#define NBOXES 32
#define NCELLS 128
#define NTHREADS 1
#define NDIMS 3
#define NCOMP 5
#define NGHOST 2

#define GET_OFFSET(c,z,y,x) (c)*full_numCell3 + ((z)+nGhost)*full_numCell2 + ((y)+nGhost)*full_numCell + ((x)+nGhost)

#undef GET_VAL_PTR
#define GET_VAL_PTR(b,c,z,y,x) (b)+ (c)*full_numCell3 + ((z)+nGhost) * full_numCell2 +\
                    ((y)+nGhost)*full_numCell+((x)+nGhost)

#define GET_FACE_VAL_X(b,c,z,y,x) (b)+(c)*(numCell)*(numCell)*(numCell+1) +\
        (z)*(numCell)*(numCell+1)+(y)*(numCell+1)+(x)

#define GET_FACE_VAL_Y(b,c,z,y,x) (b)+(c)*(numCell)*(numCell+1)*(numCell) +\
        (z)*(numCell+1)*(numCell)+(y)*(numCell)+(x)

#define GET_FACE_VAL_Z(b,c,z,y,x) (b)+(c)*(numCell+1)*(numCell)*(numCell) +\
        (z)*(numCell)*(numCell)+(y)*(numCell)+(x)

#define PHI_IN(c,z,y,x) *(GET_VAL_PTR(old_box,(c),(z),(y),(x)))
#define PHI_OUT(c,z,y,x) *(GET_VAL_PTR(new_box,(c),(z),(y),(x)))
#define PHI_PTR(c,z,y,x) (GET_VAL_PTR(new_box,(c),(z),(y),(x)))
#define PHI_REF(c,z,y,x) *(GET_VAL_PTR(ref_box,(c),(z),(y),(x)))

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

#define p_DATA_ref(z,y,x) *(GET_VAL_PTR(ref_box,0,z,y,x))
#define e_DATA_ref(z,y,x) *(GET_VAL_PTR(ref_box,1,z,y,x))
#define u_DATA_ref(z,y,x) *(GET_VAL_PTR(ref_box,2,z,y,x))
#define v_DATA_ref(z,y,x) *(GET_VAL_PTR(ref_box,3,z,y,x))
#define w_DATA_ref(z,y,x) *(GET_VAL_PTR(ref_box,4,z,y,x))

static int _printCounter = 0;

#define PRINT(d,c,z,y,x,v1,v2,v3,v4,v5) {\
    switch (d) {\
        case 0: fprintf(stderr, "FLUX1X"); break;\
        case 1: fprintf(stderr, "FLUX2X"); break;\
        case 2: fprintf(stderr, "DIFFX"); break;\
        case 3: fprintf(stderr, "FLUX1Y"); break;\
        case 4: fprintf(stderr, "FLUX2Y"); break;\
        case 5: fprintf(stderr, "DIFFY"); break;\
        case 6: fprintf(stderr, "FLUX1Z"); break;\
        case 7: fprintf(stderr, "FLUX2Z"); break;\
        case 8: fprintf(stderr, "DIFFZ"); break;\
    }\
    fprintf(stderr, "\t%d\t%d\t%d\t%d\t%g\t%g\t%g", c,z,y,x,v1,v2,v3);\
    if (v4 != N_A) fprintf(stderr, "\t%g", v4);\
    if (v5 != N_A) fprintf(stderr, "\t%g", v5);\
    fprintf(stderr, "\t%d\n", _printCounter);\
    _printCounter += 1;\
}

#endif
