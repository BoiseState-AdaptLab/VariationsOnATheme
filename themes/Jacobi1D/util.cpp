/*****************************************************************************
 * util.cpp Implementation file
 *
 *  It contains space allocation, space initialization, verification
 *  for Jacobi1D implementations, and command line options checking
 *
 *  All that verification does is to run the exact same computation in a 
 *  serial manner
 *
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <ctype.h>
#include <assert.h>

#include "util.h"

/*****************************************************************************
 * verifyResultJacobi1D() is a verification function for Jacobi1D
 * implementations
 *
 * if verification is successful
 *   return true
 * else
 *  return false
 *
*****************************************************************************/
bool verifyResultJacobi1D(real* result, Configuration& configuration){

  int  upper_bound_T = configuration.getInt("T");
  int  Nx = configuration.getInt("Nx");
  int  lower_bound_i = 1;
  int  upper_bound_i = lower_bound_i + Nx - 1;
  real* data;
  bool success = true;

  if( !allocateSpace(&data,configuration) ){
    printf( "Could not allocate space array (for verification)\n" );
    return false;
  }
  initializeSpace(data,configuration);
  
  // run serial Jacobi 1D
  for( int t = 1; t <= upper_bound_T; ++t ){
    for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
      stencil(data,t,i);
    }
  }
  for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
    if( space(data,upper_bound_T,i) != space(result,upper_bound_T,i) ){
      fprintf(stderr,"Position: %d, values: expected %f, found %f\n",
              i,space(data,upper_bound_T,i),space(result,upper_bound_T,i));

      success = false;
      break;
    }
  }
  return success;
}
/*****************************************************************************
 * verifyResultJacobi1DCuda() is a verification function for Jacobi1D CUDA 
 * implementations
 *
 * if verification is successful
 *   return true

 *
*****************************************************************************/
bool verifyResultJacobi1DCuda(real* result, Configuration& configuration){

  int  upper_bound_T = configuration.getInt("T");
  int  Nx = configuration.getInt("Nx");
  int  lower_bound_i = 1;
  int  upper_bound_i = lower_bound_i + Nx - 1;
  real* data;
  bool success = true;

  if( !allocateSpace(&data,configuration) ){
    printf( "Could not allocate space array (for verification)\n" );
    return false;
  }
  initializeSpace(data,configuration);

  // run serial Jacobi 1D
  for( int t = 1; t <= upper_bound_T; ++t ){
    for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
      stencil(data,t,i);
    }
  }
  for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
    if( space(data,upper_bound_T,i) != space(result,upper_bound_T,i) ){

      fprintf(stderr,"Position: %d, values: expected %f, found %f\n",
              i,space(data,upper_bound_T,i),space(result,upper_bound_T,i));
      success = false;
      break;
    }
  }
  return success;
}
/*****************************************************************************
 * allocateSpace() function allocates memory space for Jacobi1D benchmarks
 *
 * if allocation is successful
 *  return true
 * else
 *  return false
 *
*****************************************************************************/
bool allocateSpace(real** data, Configuration& configuration){

  // Allocate time-steps 0 and 1
  int requiredSpace = sizeof(real) * (configuration.getInt("Nx") + 2) * 2;
  *data = (real*) malloc(requiredSpace);
  if( *data == NULL ){
    return false;
  }
  return true;
}
/*****************************************************************************
 * initializeSpace() function initializes memory space for Jacobi1D benchmarks
 *
 *
*****************************************************************************/
void initializeSpace(real* data, Configuration& configuration){

  int  Nx = configuration.getInt("Nx");
  int  lower_bound_i = 1;
  int  upper_bound_i = lower_bound_i + Nx - 1;

  // Use global seed to seed the random number generator (will be constant)
  srand(configuration.getInt("global_seed"));

  // seed the space.
  for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
    space(data,0,i) = rand() / (real)rand();
  }

  // Set halo values (sanity)
  space(data,0,0) = 0;
  space(data,1,0) = 0;

  space(data,0,Nx+1) = 0;
  space(data,1,Nx+1) = 0;
}
/*****************************************************************************
 * checkCommandLineOptions() function checks the command line options
 *  for Jacobi1D benchmarks
 *
 * if checking is successful
 *  return true
 * else
 *  return false
 *
*****************************************************************************/
bool checkCommandLineOptions(Configuration &configuration)
{
  // Check the number of threads
  if(configuration.getInt("num_threads") > omp_get_max_threads()){
    printf("--num_threads cannot be more than %d\n",omp_get_max_threads());
    return false;
  }else if (configuration.getInt("num_threads") < 1){
    printf("--num_threads cannot be less than %d\n",1);
    return false;
  }else{
    omp_set_num_threads(configuration.getInt("num_threads"));
  }
  //Check the number of time steps
  if(configuration.getInt("T") < 1){
    printf("-T cannot be less than %d\n",1);
    return false;
  }
  //Check problem size
  if(configuration.getInt("Nx") < 1){
    printf("--Nx cannot be less than %d\n",1);
    return false;
  }
  return true;
}
