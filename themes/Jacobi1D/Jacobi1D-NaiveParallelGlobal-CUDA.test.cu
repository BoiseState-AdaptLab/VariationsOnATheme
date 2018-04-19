/*****************************************************************************
 * Jacobi1D benchmark
 * Basic parallelisation with CUDA using global memory
 *
 * Usage:
 *  make cuda
 *  ./Jacobi1D-NaiveParallelGlobal-CUDA --Nx 5000 -T 100 --bx 256
 *
 * To see possible options:
 *   ./Jacobi1D-NaiveParallelGlobal-CUDA --help 
*****************************************************************************/
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string>
#include <assert.h>
#include <sstream>

#include "util.h"
#include "../common/Measurements.h"
#include "../common/Configuration.h"
#include "../common/CUDA_util.h"

#define gdata(t,i)  data[( (t) & 1 ) * (Nx+2)  + (i) ]
#define stencil(t,i) gdata(t,i) =  (  gdata(t-1, i-1)  +  \
                                      gdata(t-1, i  )  +  \
                                      gdata(t-1, i+1) ) / 3

using namespace std;

/*****************************************************************************
 *
 * Jacobi1DKernel Function
 *
 * This function is the implementation of Jacobi 1D in global memory.
 * It is naive version, so there is no tiling operations.
*****************************************************************************/
__global__ void Jacobi1DKernel( 
                        dataType * data,    // pointer to the data in global
                                            // memory
                        int Nx,              // problem size
                        int t)              // time steps      
{
  // Computation space
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Eliminate boundary threads
  if( 1 <= i && i <= Nx ){
    // Jacobi 1D Stencil
    stencil(t,i);
  }
}

int main(int argc, char* argv[]) {

  // Rather than calling fflush
  setbuf(stdout, NULL);
  
  // 1 - Command line parsing 
  Measurements  measurements;
  Configuration configuration;

  /**************************************************************************
  **  Parameter options. Help and verify are constructor defaults  **********
  **************************************************************************/
  configuration.addParamInt("Nx",'k',100,
                            "--Nx <problem-size> for x-axis in elements (N)");

  configuration.addParamInt("Ny",'l',100,
                            "--Ny <problem-size> for y-axis in elements (N)");

  configuration.addParamInt("T",'T',100,
                            "-T <time-steps>, the number of time steps");

  configuration.addParamInt("num_threads",'p',1,
                            "-p <num_threads>, number of cores");

  configuration.addParamInt("global_seed", 'g', 1524, 
                            "--global_seed <global_seed>, seed for rng");

  configuration.addParamBool("n",'n', false, "-n do not print time");

  configuration.parse(argc,argv);
  
  // End Command line parsing
  
  // 2 - Data allocation and initialization
  int gridX;
  int blockX;
  int upper_bound_T = configuration.getInt("T");  
  int lower_bound_i = 1;
  int upper_bound_i = lower_bound_i + configuration.getInt("Nx") - 1; 
  dataType * space[2] = { NULL, NULL };

  // Get block size 
  blockX = configuration.getInt("bx");
  dim3 blockSize(blockX,1,1);

  // Calculate gridX with respect to active threads in a blocks
  gridX  = ceil((dataType)(configuration.getInt("Nx") + 2) / blockX);
  dim3 gridSize(gridX,1,1);
  
  // Allocate HOST arrays on CPU
  if( !allocateSpace(space,configuration.getInt("Nx")) ){
    printf( "Could not allocate space array\n" );
    return 1; 
  }
  // Initialize HOST arrays 
  initializeSpace(space,lower_bound_i,upper_bound_i,upper_bound_T,
                                          configuration.getInt("global_seed"));
  // Allocate DEVICE arrays on GPU
  int d_space_size = (2 * (configuration.getInt("Nx") + 2) * sizeof(dataType));
  dataType * d_space;
  gpuErrchk(cudaMalloc( (void**) &d_space, d_space_size));

  // End data allocation and initialization

  // 3 - Jacobi 1D timed
  // Begin timed test
  // Define the time variables
  float elapsedTimeTotalCUDA, elapsedTimeSendToGPU, elapsedTimeSendToCPU;

  // Define the event variables
  cudaEvent_t startTotalCUDA, stopTotalCUDA, startSendToCPU, stopSendToCPU,
              startSendToGPU, stopSendToGPU;

  // Create two events, startTotalCUDA and stopTotalCUDA, to
  // measure total elapsed time that includes data transfer time
  // (CPU -> GPU and GPU -> CPU) and kernels' execution time
  cudaEventCreate(&startTotalCUDA);
  cudaEventCreate(&stopTotalCUDA);
  //Trigger startTotalCUDA event
  cudaEventRecord(startTotalCUDA,0);

  // Create two events startSendToGPU and stopSendToGPU
  // to measure elapsed time that is data transfer
  // time from the CPU to the GPU
  cudaEventCreate(&startSendToGPU);
  cudaEventCreate(&stopSendToGPU);
  //Trigger "startSendToGPU" event
  cudaEventRecord(startSendToGPU,0);

  // Copy data from HOST to DEVICE           
  gpuErrchk(cudaMemcpy( d_space, space[0],(configuration.getInt("Nx") + 2) * 
           sizeof(dataType),cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy( &d_space[configuration.getInt("Nx") + 2], space[1],
            (configuration.getInt("Nx") + 2) * sizeof(dataType),
            cudaMemcpyHostToDevice));

  // Trigger "stopSendToGPU" event
  cudaEventRecord(stopSendToGPU,0) ;
  // Sync "stopSendToGPU" event
  cudaEventSynchronize(stopSendToGPU);

  COUNT_MACRO_INIT();
  
  // Kernel Calls
  for(int t = 1; t <= upper_bound_T; ++t) {  
    Jacobi1DKernel<<<gridSize,blockSize>>>(d_space,
                                          configuration.getInt("Nx"),
                                          t);
    gpuErrchkSync(cudaDeviceSynchronize());
    
    COUNT_MACRO_KERNEL_CALL();
    COUNT_MACRO_RESOURCE_ITERS(gridX*blockX);
    COUNT_MACRO_BYTES_FROM_GLOBAL(gridX*blockX*sizeof(dataType));
    COUNT_MACRO_BYTES_TO_GLOBAL((gridX*blockX-2)*sizeof(dataType));
  }
  gpuErrchkSync(cudaPeekAtLastError());

  // Create two events, startSendToCPU and stopSendToCPU,
  // to measure elapsed time that is data transfer
  // time from the GPU to the CPU
  cudaEventCreate(&startSendToCPU);
  cudaEventCreate(&stopSendToCPU);

  //Trigger "startSendToCPU" event
  cudaEventRecord(startSendToCPU,0);
  // Copy data from DEVICE to HOST  
  gpuErrchk(cudaMemcpy(space[upper_bound_T & 1], 
                       &d_space[((upper_bound_T) & 1) * 
                       (configuration.getInt("Nx") + 2)], 
                       (configuration.getInt("Nx") + 2) * sizeof(dataType), 
                       cudaMemcpyDeviceToHost));                     
  // Trigger "stopSendToCPU" event
  cudaEventRecord(stopSendToCPU,0);
  // Sync "stopSendToCPU" event
  cudaEventSynchronize(stopSendToCPU);

  // Trigger "stopTotalCUDA" event
  cudaEventRecord(stopTotalCUDA,0);
  // Sync "stopTotalCUDA" event
  cudaEventSynchronize(stopTotalCUDA);

  // Calculate data transfer time from the GPU to the CPU
  cudaEventElapsedTime(&elapsedTimeSendToGPU, startSendToGPU, stopSendToGPU);

  // Calculate data transfer time from the CPU to the GPU
  cudaEventElapsedTime(&elapsedTimeSendToCPU, startSendToCPU, stopSendToCPU);

  // Calculate elapsed time in Kernel including the transfer times
  cudaEventElapsedTime(&elapsedTimeTotalCUDA, startTotalCUDA, stopTotalCUDA);

  // Free DEVICE arrays
  cudaFree( d_space );

  measurements.setField("elapsedTimeSendToGPU",elapsedTimeSendToGPU*0.001);
  measurements.setField("elapsedTimeSendToCPU",elapsedTimeSendToCPU*0.001);
  measurements.setField("elapsedTimeTotalCUDA",elapsedTimeTotalCUDA*0.001); 
  measurements.setField("elapsedTimeProcessingCUDA",(elapsedTimeTotalCUDA - 
                                                     elapsedTimeSendToGPU -
                                                     elapsedTimeSendToCPU)
                                                     *0.0001);
  
  // End Jacobi 1D timed

  // 4 - Verification and output (optional)
  // Verification
  if( configuration.getBool("v") ){
    if( verifyResultJacobi1DCuda(space[upper_bound_T & 1],
                                 configuration.getInt("Nx"),
                                 configuration.getInt("global_seed"),
                                 configuration.getInt("T"),
                                 measurements) ){
       measurements.setField("verification","SUCCESS");
    }else{
       measurements.setField("verification","FAILURE");
    }
  } 
  // Output
  if( !configuration.getBool("n") ){
    string ldap = configuration.toLDAPString();
    ldap += measurements.toLDAPString();  
    cout<<ldap;
    COUNT_MACRO_PRINT();
    cout<<endl;
  }

  // End verification and output 
  
  // Free the space in memory  
  free ( space[0] );
  free ( space[1] ); 
  
  return 0;
}
