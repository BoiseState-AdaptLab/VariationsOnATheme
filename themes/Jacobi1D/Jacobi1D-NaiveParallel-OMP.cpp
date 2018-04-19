/*****************************************************************************
 * Jacobi1D benchmark
 * Basic parallelisation with OpenMP using a static scheduling for the loop
 * over the spatial dimension.
 *
 * Usage:
 *  make omp
 *  ./Jacobi1D-NaiveParallel-OMP --Nx 5000000 -T 50 --num_threads 8
 * For a run on 8 threads
 *
 * To see possible options:
 *  ./Jacobi1D-NaiveParallel-OMP --help 
 *
****************************************************************************/
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <ctype.h>
#include <stdbool.h>
#include <assert.h>
#include <string>
#include <sstream>

#include "util.h"
#include "../common/Measurements.h"
#include "../common/Configuration.h"

using namespace std;

// main
// Stages
// 1 - command line parsing
// 2 - data allocation and initialization
// 3 - jacobi 1D timed within an openmp loop
// 4 - output and optional verification

int main( int argc, char* argv[] ){

  // rather than calling fflush    
  setbuf(stdout, NULL); 

  // 1 - Command line parsing and checking 
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
  
  // Checking command line options
  if( !checkCommandLineOptions(configuration) ){
    return 1;
  }
  
  // End Command line parsing and checking

  // 2 - data allocation and initialization
  int i;
  int Nx = configuration.getInt("Nx");
  int upper_bound_T = configuration.getInt("T");  
  int lower_bound_i = 1;
  int upper_bound_i = lower_bound_i + Nx - 1;
  real* data = NULL; 

  // Allocate time-steps 0 and 1
  if( !allocateSpace(&data,configuration) ){
    printf( "Could not allocate space array\n" );
    return 1;
  }
  // Initialize arrays
  // First touch for openmp 
  #pragma omp parallel for private( i ) 
  for( i = lower_bound_i; i <= upper_bound_i; ++i ){
    space(data,0,i) = 0;
  } 
  initializeSpace(data,configuration);

  // End data allocation and initialization

  // 3 - Jacobi 1D timed within an openmp loop
  // Begin timed test  
  double start_time = omp_get_wtime();
    
  for( int t = 1; t <= upper_bound_T; ++t ){
    #pragma omp parallel for private( i ) schedule(runtime)
    for( i = lower_bound_i; i <= upper_bound_i; ++i ){
      // stencil(data,t,i) macro is defined at util.h file
      stencil(data,t,i);
    }
  }
  double end_time = omp_get_wtime();
  double time =  end_time - start_time;
  
  measurements.setField("elapsedTime",time);
  
  // End timed test

  // 4 - Verification and output (optional)
  // Verification
  if( configuration.getBool("v") ){
    if( verifyResultJacobi1D(data,configuration) ){
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
  free ( data );

  return 0; 
}
