/*****************************************************************************
 * util.cpp Implementation file
 *
 *  It contains space allocation, space initialization, verification
 *  for Jacobi2D implementations, and command line options checking
 *  
 *  All that verification does is to run the exact same computation in a 
 *  serial manner
 *  
 *  See util.h for usage of the functions
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
 * verifyResultJacobi2D() is a verification function for Jacobi2D  
 * implementations
 *  
 * if verification is successful
 *   return true
 * else
 *   return false
 *
*****************************************************************************/
bool verifyResultJacobi2D(real* result, Configuration& configuration){

  int  upper_bound_T = configuration.getInt("T");
  int  Nx = configuration.getInt("Nx");
  int  Ny = configuration.getInt("Ny");  
  int  lower_bound_i = 1;
  int  lower_bound_j = 1;  
  int  upper_bound_i = lower_bound_i + Ny - 1;
  int  upper_bound_j = lower_bound_j + Nx - 1; 
  real* data;
  bool  success = true;
    
  if( !allocateSpace(&data,configuration) ){
    printf( "Could not allocate space array (for verification)\n" );
    return false;  
  }
  initializeSpace(data,configuration);

  // run serial Jacobi 2D
  for( int t = 1; t <= upper_bound_T; ++t ){
    for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
      for( int j = lower_bound_j; j <= upper_bound_j; ++j ){
        stencil(data,t,i,j); 
      }    
    }
  }
  real epsilon = 0.002;
  for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
    for( int j = lower_bound_j; j <= upper_bound_j; ++j ){
      if(space(data,upper_bound_T,i,j) > space(result,upper_bound_T,i,j) + epsilon || 
space(data,upper_bound_T,i,j) < space(result,upper_bound_T,i,j)-epsilon ){

        fprintf(stderr,"Position: (%d,%d) Expected: %f, Found: %f\n",i,j,
                space(data,upper_bound_T,i,j),space(result,upper_bound_T,i,j));                
        fprintf(stderr,"Position: (%d,%d) Expected: %A, Found: %A\n",i,j,
                space(data,upper_bound_T,i,j),space(result,upper_bound_T,i,j));                
        success = false;
        //break;
      }
    }
    //if (!success) break;
  }
  return success;
}
/*****************************************************************************
 * verifyResultJacobi2DCuda() is a verification function for Jacobi2D CUDA 
 * implementations
 *   
 * if verification is successful
 *   return true
 * else
 *  return false 
 *
*****************************************************************************/
bool verifyResultJacobi2DCuda(real* result, Configuration& configuration){

}
/*****************************************************************************
 * allocateSpace() function allocates memory space for Jacobi2D benchmarks
 * 
 * if allocation is successful
 *  return true
 * else
 *  return false 
 *
*****************************************************************************/
bool allocateSpace(real** data, Configuration& configuration){

  // Allocate time-steps 0 and 1
  int requiredSpace = sizeof(real) * (configuration.getInt("Nx") + 2) * 
                                     (configuration.getInt("Ny") + 2) * 2;
  *data = (real*) malloc(requiredSpace);
  if( *data == NULL ){
    return false;
  }
  return true;
}
/*****************************************************************************
 * initializeSpace() function initializes memory space for Jacobi2D benchmarks
 *
 *
*****************************************************************************/
void initializeSpace(real* data, Configuration& configuration){
 
  int  Nx = configuration.getInt("Nx");
  int  Ny = configuration.getInt("Ny");
  int  lower_bound_i = 1;
  int  lower_bound_j = 1;
  int  upper_bound_i = lower_bound_i + Ny - 1;
  int  upper_bound_j = lower_bound_j + Nx - 1;
 
  // Use global seed to seed the random number gen (will be constant)
  srand(configuration.getInt("global_seed"));
  
  // Seed the space
  for( int i = lower_bound_i; i <= upper_bound_i; ++i ){
    for( int j = lower_bound_j; j <= upper_bound_j; ++j ){
      space(data,0,i,j) = rand() / (real)rand();
    }
  }

  // Set halo values (sanity)
  for( int i = 0; i < upper_bound_i + 2; ++i ){
    space(data,0,i,0) = 0;
    space(data,1,i,0) = 0;

    space(data,0,i,Nx+1) = 0;
    space(data,1,i,Nx+1) = 0;
  }
  for( int j = 0; j < upper_bound_j + 2; ++j ){
    space(data,0,0,j) = 0;
    space(data,1,0,j) = 0;

    space(data,0,Ny+1,j) = 0;
    space(data,1,Ny+1,j) = 0;
  }
}
/*****************************************************************************
 * checkCommandLineOptions() function checks the command line options
 *  for Jacobi2D benchmarks
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
  //Check problem size
  if(configuration.getInt("Ny") < 1){
    printf("--Ny cannot be less than %d\n",1);
    return false;
  }
  //Check tau size
  if(configuration.getInt("tau") < 1){
    printf("--tau cannot be less than %d\n",1);
    return false;
  }
             
  return true;
}
