#include<stdio.h>
#include "util.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file,
                      int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),file, line);
      if (abort) exit(code);
   }
}

extern "C" void cuda_init_rand_wrapper(unsigned long long seed,
                                       curandState_t *state,
                                       int cols, dim3 grid, dim3 block);


extern "C" void cuda_segment_update_model(unsigned char* map,pixel* frame,
                                          pixel* init_model,int time_sample,
                                          int rows,int cols,int samples,
                                          int radius,int match_threshold,
                                          curandState_t *state,dim3 grid,
                                          dim3 block);

extern "C" void cuda_median_blur(unsigned char*map,unsigned char *output_map,
                                 int rows,int cols,
                                 int kernel_size,size_t mem_size,
                                 dim3 grid,dim3 block);
