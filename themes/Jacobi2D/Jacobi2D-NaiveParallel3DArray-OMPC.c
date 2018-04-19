/*****************************************************************************
 * Jacobi2D benchmark
 * Basic parallelisation with OpenMP using a static scheduling for the loop
 * over the spatial dimension.
 *
 * Usage:
 *  make omp
 *  ./Jacobi2D-NaiveParallel-OMP --Nx 5000 --Ny 5000 -T  50 --num_threads 8
 * For a run on 8 threads
 *
 * To see possible options:
 *   Jacobi2D-NaiveParallel-OMP --help 
*****************************************************************************/ 
#include <stdio.h>
//#include <iostream>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <ctype.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
//#include <sstream>

#include "util3DArray-C.h"
//#include "../common/Measurements.h"
//#include "../common/Configuration.h"

//using namespace std;

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
  if (argc < 2 || strstr(argv[1], "-h")) {
      usage(argv[0]);
      return -1;
  }

  Measurements  measurements;
  Configuration configuration;

  /**************************************************************************
  **  Parameter options. Help and verify are constructor defaults  **********
  **************************************************************************/
    parseArgs(argc, argv, &configuration);

    // Init meas
    measurementsInit(&measurements);

//    configurationToLDAP(&configuration, stdout);
//    measurementsToLDAP(&measurements, stdout);
//    printf("\n");
//    return 0;

  // Checking command line options 
  /*if( !checkCommandLineOptions(configuration) ){
    return 1;
  }  */

  // End Command line parsing and checking 

  // 2 - Data allocation and initialization
  int i;
  int j;
  int Nx = configuration.Nx;
  int Ny = configuration.Ny;
  int upper_bound_T = configuration.T;
  int lower_bound_i = 1;
  int lower_bound_j = 1;  
  int upper_bound_i = lower_bound_i + Ny - 1;
  int upper_bound_j = lower_bound_j + Nx - 1; 
  real ** data[2];

  // Allocate time-steps 0 and 1
  if( !allocateSpace(data,&configuration) ){
    printf( "Could not allocate space for array\n" );
    return 1;
  }

  // Initialize arrays 
  // First touch for openmp	
  #pragma omp parallel for private(i,j) collapse(2) schedule(runtime)
  for( i = lower_bound_i; i <= upper_bound_i; ++i ){
    for( j = lower_bound_j; j <= upper_bound_j; ++j ){
      data[0][i][j] = 0;
    }
  }

  initializeSpace(data,&configuration);
 
  // End data allocation and initialization
  
  // 3 - jacobi 2D timed within an openmp loop
  double start_time = omp_get_wtime();

  for( int t = 1; t <= upper_bound_T; ++t ){ 
    #pragma omp parallel for private(i,j) collapse(2) schedule(runtime)
    for( i = lower_bound_i; i <= upper_bound_i; ++i ){
      for( j = lower_bound_j; j <= upper_bound_j; ++j ){
        stencil(t,i,j);
      }    
    }
  }
  double end_time = omp_get_wtime();
  double time =  (end_time - start_time);

  measurements.elapsedTime = time;

  // End timed test
  
  // 4 - Verification and output (optional)
  // Verification
  if( configuration.verify ){
    if( verifyResultJacobi2D(data[(upper_bound_T) & 1],&configuration) ){
       //measurements.setField("verification","SUCCESS");
        strcpy(measurements.verification, "SUCCESS");
    }else{
       //measurements.setField("verification","FAILURE");
        strcpy(measurements.verification, "FAILURE");
    }
  }
  // Output
  if( !configuration.no_time ){
      //string ldap = configuration.toLDAPString();
      configurationToLDAP(&configuration, stdout);
    //ldap += measurements.toLDAPString();
      measurementsToLDAP(&measurements, stdout);
    //cout<<ldap;
    //cout<<endl;
      printf("\n");
  }
  
  // End verification and output
  
  // Free the space in memory
   for( int i = 0; i < Ny + 2; ++i ){
     free( data[0][i] );
     free( data[1][i] );
   }

   free( data[0] );
   free( data[1] );
 
  return 0;   
}
