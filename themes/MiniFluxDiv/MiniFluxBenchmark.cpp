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

#include "MiniFluxBenchmark.h"

MiniFluxBenchmark::MiniFluxBenchmark() {
}

MiniFluxBenchmark::MiniFluxBenchmark(int argc, char *argv[]) {
    _config.addParamInt("numCell", 'C', NCELLS, "--numCell, the number of cells in a single dimension of a single box");
    _config.addParamInt("numBox",'B', NBOXES, "--numBox, the number of independent boxes to process");
    _config.addParamInt("num_threads", 'p', NTHREADS, "-p <num_threads>, number of cores");

    _config.parse(argc, argv);
}

MiniFluxBenchmark::~MiniFluxBenchmark() {
}

void MiniFluxBenchmark::init() {
    char buf[1024];

    int numCell= _config.getInt("numCell");
    int numBox = _config.getInt("numBox");
    int nGhost = NGHOST;
    int numComp= NCOMP;

    _execFxn = mini_flux_div_lc ;       // Set exec function...
    _evalFxn = mini_flux_div_truth;     // Set eval function...
    _compFxn = mini_flux_div_comp;

    _verify = _config.getBool("v");     // Get verify flag
    _name = "MiniFluxdiv";
    _status = false;

    Real **old_boxes = (Real**) malloc(sizeof(Real*) * numBox);
    _inputData = old_boxes;

    if (old_boxes == NULL) {
        strcpy(buf, "old_boxes ERROR: ");
        perror(buf);
        return;
    }

    Real **new_boxes = (Real**) malloc(sizeof(Real*) * numBox);
    _outputData = new_boxes;

    if (new_boxes == NULL) {
        strcpy(buf, "new_boxes ERROR: ");
        perror(buf);
        return;
    }

    Real **ref_boxes;
    if (_verify) {
        ref_boxes = (Real**) malloc(sizeof(Real*) * numBox);
        _verifyData = ref_boxes;

        if (ref_boxes == NULL) {
            strcpy(buf, "ref_boxes ERROR: ");
            perror(buf);
            return;
        }
    }

    int full_numCell= numCell+2*nGhost;
    int full_numCell2= full_numCell*full_numCell;
    //int full_numCell3= full_numCell*full_numCell+full_numCell;
    int full_numCell3= full_numCell*full_numCell*full_numCell;
    int total_Cells= full_numCell *full_numCell*full_numCell;
    //int flux_total_Size= numCell*numCell*(numCell+1);
    int idx;
    int iz,iy,ix;

    int boxSize = total_Cells * numComp;

    for(idx=0;idx<numBox;idx++){
        old_boxes[idx] = (Real*) malloc(sizeof(Real)*boxSize);
        if (old_boxes[idx] == NULL) {
            sprintf(buf, "old_boxes[%d] ERROR: ", idx);
            perror(buf);
            return;
        }

        new_boxes[idx] = (Real*) malloc(sizeof(Real)*boxSize);
        if (new_boxes[idx] == NULL) {
            sprintf(buf, "new_boxes[%d] ERROR: ", idx);
            perror(buf);
            return;
        }

        if (_verify) {
            ref_boxes[idx] = (Real*) malloc(sizeof(Real)*boxSize);
            if (ref_boxes[idx] == NULL) {
                sprintf(buf, "ref_boxes[%d] ERROR: ", idx);
                perror(buf);
                return;
            }
        }
    }

    for(idx=0;idx < numBox;idx++){
        Real* old_box = old_boxes[idx];
        Real* new_box = new_boxes[idx];
        Real* ref_box;
        if (_verify) {
            ref_box = ref_boxes[idx];
        }

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

                    if (_verify) {
                        p_DATA_ref(iz,iy,ix) = dx*(iz+iy+ix);
                        e_DATA_ref(iz,iy,ix) = 1.+dx*(iz+iy+ix);
                        u_DATA_ref(iz,iy,ix) = 2.+dx*(iz+iy+ix);
                        v_DATA_ref(iz,iy,ix) = 3.+dx*(iz+iy+ix);
                        w_DATA_ref(iz,iy,ix) = 4.+dx*(iz+iy+ix);
                    }
                }
            }
        }
    }

    _nRows = numBox;
    _nCols = boxSize;

    _status = true;
}

void MiniFluxBenchmark::finish() {
    int numBox = _config.getInt("numBox");
    Real** old_boxes = _inputData;
    Real** new_boxes = _outputData;
    Real** ref_boxes = _verifyData;

    for(int idx=0;idx<numBox;idx++){
        free(old_boxes[idx]);
        free(new_boxes[idx]);

        if (_verify) {
            free(ref_boxes[idx]);
        }
    }

    free(_inputData);
    free(_outputData);
    if (_verify) {
        free(_verifyData);
    }
}

string MiniFluxBenchmark::error() {
//    char buff[1024];
//    ostringstream oss;
//
//    if (_errorLoc.size() > 0) {
//        int idx = _errorLoc[0];
//        int c = _errorLoc[1];
//        int iz = _errorLoc[2];
//        int iy = _errorLoc[3];
//        int ix = _errorLoc[4];
//
//        int nGhost = NGHOST;
//        int numComp = NCOMP;
//        int numCell= _config.getInt("numCell");
//        int full_numCell= numCell+2*nGhost;
//        int full_numCell2= full_numCell*full_numCell;
//        int full_numCell3= full_numCell*full_numCell*full_numCell;
//
//        DType *hope = _verifyData[idx];
//        DType *truth = _outputData[idx];
//
//        oss << "Found(in box " << idx << "): "
//            << *(GET_VAL_PTR(hope, c, iz, iy, ix))
//            << ", Expected " << *(GET_VAL_PTR(truth, c, iz, iy, ix))
//            << ", position " << "(" << c << "," << iz << ","
//            << iy << "," << ix << ") ";
//
//        sprintf(buff, "%A:%A", *(GET_VAL_PTR(hope, c, iz, iy, ix)),
//                               *(GET_VAL_PTR(truth, c, iz, iy, ix)));
//
//        oss << "[" << buff << "]";
//    }
//
//    return oss.str();
    return "";
}

void MiniFluxBenchmark::toCSV() {
    int numCell= _config.getInt("numCell");
    int numBox= _config.getInt("numBox");
    int nGhost = NGHOST;
    int numComp= NCOMP;
    int full_numCell= numCell+2*nGhost;
    int full_numCell2= full_numCell*full_numCell;
    int full_numCell3= full_numCell*full_numCell*full_numCell;
    Real **result_data = _outputData;
    Real **truth_boxes = _verifyData;

    string csvFile = _name + ".csv";
    FILE *file = fopen(csvFile.c_str(), "w");

    if (file != NULL) {
        fprintf(file, "b,z,y,x,c,expected,actual,diff,exp-hex,act-hex\n");
        for (int idx = 0; idx < numBox; idx++) {
            Real *truth_box = truth_boxes[idx];
            Real *hope_box = result_data[idx];
            for (int iz = 0; iz < numCell; iz++) {
                for (int iy = 0; iy < numCell; iy++) {
                    for (int ix = 0; ix < numCell; ix++) {
                        for (int c = 0; c < numComp; c++) {
                            Real hope_val = *(GET_VAL_PTR(hope_box, c, iz, iy, ix));
                            Real truth_val = *(GET_VAL_PTR(truth_box, c, iz, iy, ix));
                            fprintf(file, "%d,%d,%d,%d,%d,%f,%f,%f,%A,%A\n", idx, c, iz, iy, ix, hope_val,
                                    truth_val, truth_val - hope_val, hope_val, truth_val);
                        }
                    }
                }
            }
        }

        fclose(file);
    }

}

MiniFluxDivData* mini_flux_div_init(Configuration& config) {
    MiniFluxDivData *data = (MiniFluxDivData*) malloc(sizeof(MiniFluxDivData));
    data->numCell= config.getInt("numCell");
    data->numBox= config.getInt("numBox");
    data->nThreads = config.getInt("num_threads");
    data->numComp = NCOMP;
    data->nGhost = NGHOST;
    // The size of the 3D data is (numCell+2*nGhost)^3
    data->fullNumCell = data->numCell + 2 * data->nGhost;
    data->fullNumCell2 = data->fullNumCell * data->fullNumCell;
    // Is this it?
    //int full_numCell3 = full_numCell*full_numCell+full_numCell;
    data->fullNumCell3 = data->fullNumCell * data->fullNumCell * data->fullNumCell;
    //int totalCells = full_numCell*full_numCell*full_numCell;
    //int flux_totalSize = numCell*numCell*(numCell+1);
    //data->g_caches = NULL;

    return data;
}    // miniFluxDiv_init

void mini_flux_div_truth(DType ** old_boxes, DType** new_boxes, Real *timer, Configuration& config) {
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
    //int phi_comp_mult = ((numCell+2*nGhost)*(numCell+2*nGhost)*(numCell+2*nGhost));
    //const int flux_comp_mult = ((numCell)*(numCell)*(numCell+1));
    const int phi_pencil_size = (numCell+2*nGhost);

    #undef GET_VAL_PTR
    #define GET_VAL_PTR(b,c,z,y,x) (b)+ (c)*full_numCell3 + ((z)+nGhost) * full_numCell2 +\
                    ((y)+nGhost)*full_numCell+((x)+nGhost)

    #undef GET_FACE_VAL_PTR
    #define GET_FACE_VAL_PTR(d,b,c,z,y,x) (b)+\
        (c)*(numCell+((d)==2))*(numCell+((d)==1))*(numCell+((d)==0)) +\
        (z)*(numCell+((d)==1))*(numCell+((d)==0))+\
        (y)*(numCell+((d)==0))+(x)

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
        for(iDir=0;iDir<NDIMS;iDir++){
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
                        for (ix=0;ix<f_xu;ix++) {
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

    //return new_boxes;
}

void mini_flux_div_lc(Real** old_boxes,Real** new_boxes, Real *timer, Configuration& config) {
    double tbegin, tend;
    MiniFluxDivData *mfdData;
    struct timeval tval;

    // Initialize data structure...
    mfdData = mini_flux_div_init(config);

    // Allocate data...
    miniFluxDiv_alloc(mfdData);

    // Start timer before executing kernel...
    gettimeofday(&tval, NULL);
    tbegin = (double) tval.tv_sec + (((double) tval.tv_usec) / 1000000);

    // Iterate over all of the boxes
    miniFluxDiv_kernel(old_boxes, new_boxes, mfdData);

    // Stop timer after executing kernel...
    gettimeofday(&tval, NULL);
    tend = (double) tval.tv_sec + (((double) tval.tv_usec) / 1000000);
    //*timer = tend - tbegin;
    if (mfdData->innerTime > 0.0) {
        *timer = mfdData->innerTime;
    }

    // Free data memory...
    miniFluxDiv_free(mfdData);

    // Free structure...
    free(mfdData);
}

bool mini_flux_div_comp(Real** new_boxes, Real** ref_boxes, Configuration& config, vector<int>& loc) {
    char buff[1024];
    int numCell= config.getInt("numCell");
    int numBox= config.getInt("numBox");
    int nGhost = NGHOST;
    int numComp= NCOMP;
    int full_numCell= numCell+2*nGhost;
    int full_numCell2= full_numCell*full_numCell;
    //int full_numCell3= full_numCell*full_numCell+full_numCell;
    int full_numCell3= full_numCell*full_numCell*full_numCell;
    Real e = EPS;

    loc.clear();

    bool status = true;
    for(int idx=0;idx < numBox;idx++){
        Real* truth = ref_boxes[idx];
        Real* hope = new_boxes[idx];

        for(int iz=0;iz<numCell;iz++){
            for(int iy=0;iy<numCell;iy++){
                for(int ix=0;ix<numCell;ix++){
                    for(int c=0;c<numComp;c++){
                        Real hope_val = *(GET_VAL_PTR(hope,c,iz,iy,ix));
                        Real truth_val = *(GET_VAL_PTR(truth,c,iz,iy,ix));

                        int ndx = (c)*full_numCell3 + ((iz)+nGhost) * full_numCell2 + ((iy)+nGhost)*full_numCell+((ix)+nGhost);
                        if (hope_val > (truth_val+e) ||
                            hope_val < (truth_val-e)) {

//                            cout << "Found(in box " << idx << "): " << hope_val
//                                 << ", Expected " << truth_val
//                                 << ", position " << "(" << c << "," << iz << ","
//                                 << iy << "," << ix << ") ";
//
//                            sprintf(buff, "%A:%A", *(GET_VAL_PTR(hope, c, iz, iy, ix)),
//                                    *(GET_VAL_PTR(truth, c, iz, iy, ix)));
//
//                            cout << "[" << buff << "]";

                            loc.push_back(idx);
                            loc.push_back(c);
                            loc.push_back(iz);
                            loc.push_back(iy);
                            loc.push_back(ix);

                            status = false;
                            //return status;
                        }
                    }
                }
            }
        }
    }

    return status;
}

