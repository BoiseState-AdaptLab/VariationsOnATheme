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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <ctype.h>
#include <assert.h>
#include <iostream>
#include "util.h"

bool freeMemory(Real*** old_boxes_in,Real *** new_boxes_in,
                           Configuration& config) {
  int numBox = config.getInt("numBox");
  Real** old_boxes = *old_boxes_in;
  Real** new_boxes = *new_boxes_in;

  for(int idx=0;idx<numBox;idx++){
    free(old_boxes[idx]);
    free(new_boxes[idx]);
  }
  free(*old_boxes_in);
  free(*new_boxes_in);
}

bool allocateAndInitialize(Real*** old_boxes_in,Real *** new_boxes_in,
                           Configuration& config) {

   int numCell= config.getInt("numCell");
   int numBox = config.getInt("numBox");
   int nGhost = NGHOST;
   int numComp= NCOMP;

   char buf[1024];

   Real** old_boxes = *old_boxes_in;
   old_boxes = (Real**)malloc(sizeof(Real*)*numBox);
   *old_boxes_in = old_boxes;
   if(old_boxes == NULL){
     return false;
   }
   Real** new_boxes = *new_boxes_in;
   new_boxes = (Real**)malloc(sizeof(Real*)*numBox); 
   *new_boxes_in = new_boxes;
   if(new_boxes == NULL){
     return false;
   }

   int full_numCell= numCell+2*nGhost;
   int full_numCell2= full_numCell*full_numCell;
   //int full_numCell3= full_numCell*full_numCell+full_numCell;
   int full_numCell3= full_numCell*full_numCell*full_numCell;
   int total_Cells= full_numCell *full_numCell*full_numCell;
   int flux_total_Size= numCell*numCell*(numCell+1);
   int idx;
   int iz,iy,ix;

   for(idx=0;idx<numBox;idx++){
       old_boxes[idx] = (Real*)malloc(sizeof(Real)*total_Cells*numComp);
       if (old_boxes[idx] == NULL) {
           sprintf(buf, "old_boxes[%d] ERROR: ", idx);
           perror(buf);
           return false;
       }

       new_boxes[idx] = (Real*)malloc(sizeof(Real)*total_Cells*numComp);
       if (new_boxes[idx] == NULL) {
           perror("new_boxes[idx] ERROR: ");
           return false;
       }
   }

   for(idx=0;idx < numBox;idx++){
      Real* old_box = old_boxes[idx];
      Real* new_box = new_boxes[idx];
      for(iz=-nGhost;iz<(full_numCell-nGhost);iz++){
         for(iy=-nGhost;iy<(full_numCell-nGhost);iy++){
            for(ix=-nGhost;ix<(full_numCell-nGhost);ix++){
               p_DATA_new(iz,iy,ix) = dx*(iz+iy+ix);
               e_DATA_new(iz,iy,ix) = 1.+dx*(iz+iy+ix);
               u_DATA_new(iz,iy,ix) = 2.+dx*(iz+iy+ix);
               v_DATA_new(iz,iy,ix) = 3.+dx*(iz+iy+ix);
               w_DATA_new(iz,iy,ix) = 4.+dx*(iz+iy+ix);

               p_DATA_old(iz,iy,ix) = dx*(iz+iy+ix);
               e_DATA_old(iz,iy,ix) = 1.+dx*(iz+iy+ix);
               u_DATA_old(iz,iy,ix) = 2.+dx*(iz+iy+ix);
               v_DATA_old(iz,iy,ix) = 3.+dx*(iz+iy+ix);
               w_DATA_old(iz,iy,ix) = 4.+dx*(iz+iy+ix);
            }
        }
     }
  }
  return true;
}

bool verifyResult(Real** result_data,Configuration& config){

   int numCell= config.getInt("numCell");
   int numBox= config.getInt("numBox");
   int nGhost = NGHOST;
   int numComp= NCOMP;
   int full_numCell= numCell+2*nGhost;
   int full_numCell2= full_numCell*full_numCell;
   //int full_numCell3= full_numCell*full_numCell+full_numCell;
   int full_numCell3= full_numCell*full_numCell*full_numCell;
   Real **old_boxes;
   Real **truth_boxes;
   Real e = .000001;

   allocateAndInitialize(&old_boxes,&truth_boxes,config);

   mini_flux_div_truth(old_boxes,truth_boxes,config);

   for(int idx=0;idx < numBox;idx++){
       Real* truth = truth_boxes[idx];
       Real* hope = result_data[idx];
       for(int iz=0;iz<numCell;iz++){
          for(int iy=0;iy<numCell;iy++){
             for(int ix=0;ix<numCell;ix++){
               for(int c=0;c<numComp;c++){
                  if(*(GET_VAL_PTR(hope,c,iz,iy,ix)) >
                      (*(GET_VAL_PTR(truth,c,iz,iy,ix))+e) ||
                     *(GET_VAL_PTR(hope,c,iz,iy,ix)) <
                      (*(GET_VAL_PTR(truth,c,iz,iy,ix))-e)){
                    std::cout << "Found(in box " << idx <<"): "
                              << *(GET_VAL_PTR(hope,c,iz,iy,ix)) 
                              << ", Expected "<<*(GET_VAL_PTR(truth,c,iz,iy,ix))
                              << ", position " << "(" << c << "," << iz << ","
                              << iy << "," << ix << ")";
                    printf("%A:%A\n",*(GET_VAL_PTR(hope,c,iz,iy,ix)),
                           *(GET_VAL_PTR(truth,c,iz,iy,ix)));
                    freeMemory(&old_boxes,&truth_boxes,config);
                    return false;
                  }
                }
             }
          }
       }
    }
    freeMemory(&old_boxes,&truth_boxes,config);
    return true;
}


void toCSV(const char * csvFile, Real** result_data,Configuration& config) {
    int numCell= config.getInt("numCell");
    int numBox= config.getInt("numBox");
    int nGhost = NGHOST;
    int numComp= NCOMP;
    int full_numCell= numCell+2*nGhost;
    int full_numCell2= full_numCell*full_numCell;
    //int full_numCell3= full_numCell*full_numCell+full_numCell;
    int full_numCell3= full_numCell*full_numCell*full_numCell;
    Real **old_boxes;
    Real **truth_boxes;

    allocateAndInitialize(&old_boxes,&truth_boxes,config);
    mini_flux_div_truth(old_boxes,truth_boxes,config);

    FILE *file = fopen(csvFile, "w");
    if (file != NULL) {
        fprintf(file, "z,y,x,c,expected,actual,diff,exp-hex,act-hex\n");
        for (int idx = 0; idx < numBox; idx++) {
            Real *truth_box = truth_boxes[idx];
            Real *hope_box = result_data[idx];
            for (int iz = 0; iz < numCell; iz++) {
                for (int iy = 0; iy < numCell; iy++) {
                    for (int ix = 0; ix < numCell; ix++) {
                        for (int c = 0; c < numComp; c++) {
                            Real hope_val = *(GET_VAL_PTR(hope_box, c, iz, iy, ix));
                            Real truth_val = *(GET_VAL_PTR(truth_box, c, iz, iy, ix));
                            fprintf(file, "%d,%d,%d,%d,%f,%f,%f,%A,%A\n", c, iz, iy, ix, hope_val,
                                    truth_val, truth_val - hope_val, hope_val, truth_val);
                        }
                    }
                }
            }
        }

        fclose(file);
    }
}


Real** mini_flux_div_truth(Real ** old_boxes, Real** new_boxes,
                           Configuration& config){

  int idx;
  int ic,iz,iy,ix;
  int numCell= config.getInt("numCell");
  int numBox= config.getInt("numBox");
  int nGhost = NGHOST;
  int numComp = NCOMP;
  int full_numCell= numCell+2*nGhost;
  int full_numCell2= full_numCell*full_numCell;
  //int full_numCell3= full_numCell*full_numCell+full_numCell;
  int full_numCell3= full_numCell*full_numCell*full_numCell;
  //int total_Cells= full_numCell *full_numCell*full_numCell;
  int flux_total_Size= numCell*numCell*(numCell+1);

  // loop bounds
  int phi_comp_mult = ((numCell+2*nGhost)*(numCell+2*nGhost)
                                   *(numCell+2*nGhost));
  const int phi_pencil_size = (numCell+2*nGhost);
  const int flux_comp_mult = ((numCell)*(numCell)*(numCell+1));

  //fprintf(stderr, "BEGIN mini_flux_div_truth\n");

  // process each of the boxes one at a time
  for(idx=0;idx < numBox;idx++){
      Real* old_box = old_boxes[idx];
      Real* new_box = new_boxes[idx];
      int f_xu,f_yu,f_zu;
      int flux_pencil_x;
      int flux_pencil_y;
      int iDir,ic,iz,iy,ix;
      int phiOffset1,phiOffset2,fluxOffset1;

      // the flux cache
      Real* fluxCache = (Real*)malloc(sizeof(Real)*numCell*numCell*(numCell+1)*numComp);
      // Allocate the space for the velocity cache
       // This is only a single component
      Real* velCache = (Real*)malloc(sizeof(Real)*numCell*numCell*(numCell+1));

      // compute the fluxes on the faces in each direction
      for(iDir=0;iDir<DIMS;iDir++){
         // x-direction
         //std::string axis = "";
         if(iDir == 0){
            //axis = "X";
            f_zu = numCell;
            f_yu = numCell;
            f_xu = numCell+1;
            flux_pencil_x = numCell+1;
            flux_pencil_y = numCell;
            phiOffset1 = 1;
            phiOffset2 = 2;
            fluxOffset1 = 1;
         }else if(iDir == 1) {
            //axis = "Y";
            f_zu = numCell;
            f_yu = numCell+1;
            f_xu = numCell;
            flux_pencil_x = numCell;
            flux_pencil_y = numCell+1;
            phiOffset1 = phi_pencil_size;
            phiOffset2 = phi_pencil_size*2;
            fluxOffset1 = numCell;
         }else if(iDir == 2) {
            //axis = "Z";
            f_zu = numCell+1;
            f_yu = numCell;
            f_xu = numCell;
            flux_pencil_x = numCell;
            flux_pencil_y = numCell;
            phiOffset1 = phi_pencil_size*phi_pencil_size;
            phiOffset2 = phi_pencil_size*phi_pencil_size*2;
            fluxOffset1 = numCell*numCell;
         }
         // the upper bounds are determined by direction info above
         // Iterate over faces and calculate g()
          for(iz=0;iz<f_zu;iz++){
            for(iy=0;iy<f_yu;iy++){
               for(ic=0;ic<numComp;ic++){
                  Real* phip = GET_VAL_PTR(old_box,ic,iz,iy,0);
                  Real* fluxp = GET_FACE_VAL_PTR(iDir,fluxCache,ic,iz,iy,0);
                  for(ix=0;ix<f_xu;ix++){
                     //std::string action = "FLUX1" + axis;
                     //fprintf(stderr, "%s: %d,%d,%d,%d (%d,%d)\n", action.c_str(),ic,iz,iy,ix,0,0);

                     *fluxp = factor1*((*(phip - phiOffset2)) +7*((*(phip - phiOffset1)) + (*(phip))) +(*(phip + phiOffset1)));
                     ++phip;
                     ++fluxp;
                  }
               }
            }
         }

        // cache the velocity component for the next half of the calculation
        memcpy(velCache,(fluxCache+(iDir+2)*((numCell+1)*numCell*numCell)),sizeof(Real)*numCell*numCell*(numCell+1));

        for(iz=0;iz<f_zu;iz++){
            for(iy=0;iy<f_yu;iy++){
               for(ic=0;ic<numComp;ic++){
                  //pointer arithmetic
                  Real* velp = velCache + iz*flux_pencil_y*flux_pencil_x+ iy*flux_pencil_x;
                  Real* fluxp = GET_FACE_VAL_PTR(iDir,fluxCache,ic,iz,iy,0);
                  // inner loop
                  for(ix=0;ix<f_xu;ix++){
                      //std::string action = "FLUX2" + axis;
                      //fprintf(stderr, "%s: %d,%d,%d,%d (%d,%d)\n", action.c_str(),ic,iz,iy,ix,0,0);

                     *fluxp *= factor2*(*velp);
                     ++fluxp;
                     ++velp;
                  }
               }
            }
        }
       // compute the second half of the flux calculation
        // accumulate the differences into the new data box
       for(iz=0;iz<numCell;iz++){
          for(iy=0;iy<numCell;iy++){
             for(ic=0;ic<numComp;ic++){
                // pointer arithmetic
                Real* phip = GET_VAL_PTR(new_box,ic,iz,iy,0);
                Real* fluxp = GET_FACE_VAL_PTR(iDir,fluxCache,ic,iz,iy,0);
                for(ix=0;ix<numCell;ix++){
                   //std::string action = "DIFF" + axis;
                   //fprintf(stderr, "%s: %d,%d,%d,%d (%d,%d)\n", action.c_str(),ic,iz,iy,ix,0,0);

                   *phip += (*(fluxp + fluxOffset1)) - (*fluxp);
                   ++phip;
                   ++fluxp;
                }
             }
          }
       }
    } // direction loop

    free(fluxCache);
    free(velCache);
   } // box loop
   return new_boxes;
}

bool setNumThreads(int nThreads) {
    if(nThreads > omp_get_max_threads()){
        printf("--num_threads cannot be more than %d\n",omp_get_max_threads());
        return false;
    }else if (nThreads < 1){
        printf("--num_threads cannot be less than %d\n",1);
        return false;
    }else{
        omp_set_num_threads(nThreads);
        return true;
    }
}
