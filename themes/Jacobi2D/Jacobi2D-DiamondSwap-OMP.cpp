/******************************************************************************
 * Jacobi2D benchmark
 * Diamond tiling parameterized by hand.
 *
 * Modified Jacobi2D-DiamondSlabISCCParam-OMP-test.c to get similar format
 * to other drivers and then put in parameterized loop bounds.
 *
 * Look for the notes titled "Parameterizing diamond tiling by hand" in
 * ProjectNotes/chapel-diamond-MMS-log.txt for the details of how this was 
 * done.
 *
 * Usage:
 *  make omp
 *  ./Jacobi2D-DiamondSwap-OMP --Nx 5000 --Ny 5000 -T 50 --num_threads 8
 * For a run on 8 threads
 *
 * To see possible options:
 *  ./Jacobi2D-DiamondSwap-OMP --help
 *
******************************************************************************/
/****************************************************************************
 *
 * Modified on 08 July 2016 by Sarah Willer
 * Changes to param options:
 *  >> "help" 'h' and "verify" 'v' are still included in the Configuration 
 *     constructor. The other common command line options are listed in 
 *     Configuration.txt
 *  >> added explicit addParam calls for Nx, Ny, T, tau, num_threads,
 *     global_seed, n
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
#include <math.h>

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

  configuration.addParamInt("tau",'t',30,"--tau <tau>, distance between tiling"
                                "hyperplanes (all diamond(slab))");

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

  // 2 - Data allocation and initialization
  int i;
  int j;
  int t;
  int thyme;
  int k1;
  int k2;
  int Nx = configuration.getInt("Nx");
  int Ny = configuration.getInt("Ny");
  int tau = configuration.getInt("tau");
  int upper_bound_T = configuration.getInt("T");
  int lower_bound_i = 1;
  int lower_bound_j = 1;
  int upper_bound_i = lower_bound_i + Ny - 1;
  int upper_bound_j = lower_bound_j + Nx - 1;
  real* data;

  // Allocate time-steps 0 and 1
  if( !allocateSpace(&data,configuration) ){
    printf( "Could not allocate space for array\n" );
    return 1;
  }
  // Initialize arrays
  // First touch for openmp
  #pragma omp parallel for private(i,j) collapse(2) schedule(runtime)
  for( i = lower_bound_i; i <= upper_bound_i; ++i ){
    for( j = lower_bound_j; j <= upper_bound_j; ++j ){
      space(data,0,i,j) = 0;
    }
  }
  initializeSpace(data,configuration);

  // End data allocation and initialization

  // 3 - Jacobi 2D timed within an openmp loop
  double start_time = omp_get_wtime();

  // Loop over tile wavefronts.
  for (thyme=ceild(3,tau)-3; thyme<=floord(3*upper_bound_T,tau); thyme++){

    // The next two loops iterate within a tile wavefront.
    int k1_lb = ceild(3*lower_bound_j+2+(thyme-2)*tau,tau*3);
    int k1_ub = floord(3*upper_bound_j+(thyme+2)*tau,tau*3);

    int k2_lb = floord((2*thyme-2)*tau-3*upper_bound_i+2,tau*3);
    int k2_ub = floord((2+2*thyme)*tau-2-3*lower_bound_i,tau*3);

    #pragma omp parallel for shared(start_time, lower_bound_i, lower_bound_j, upper_bound_i, upper_bound_j ) private(k1,k2,t,i,j) schedule(runtime) collapse(2)
    for( k1 = k1_lb; k1 <= k1_ub; k1++ ){
      for( int x = k2_lb; x <= k2_ub; x++ ){
        k2 = x-k1;

        // Loop over time within a tile
        for(t = max(1,floord(thyme*tau-1, 3) + 1);
          t < min(upper_bound_T+1, tau + floord(thyme*tau, 3)); t+=1){

          // Loops over spatial dimensions within tile
          for( i = max(lower_bound_i, max(-2*tau-k1*tau-k2*tau+2*t+2,
            (thyme-k1-k2)*tau-t)); i <= min(upper_bound_i,
            min(tau+(thyme-k1-k2)*tau-t-1, -k1*tau-k2*tau+2*t)); i+=1) {
            for( j = max(lower_bound_j,max(k1*tau-t, -tau-k2*tau+t-i+1));
              j <= min(upper_bound_j,min(tau+k1*tau-t-1, -k2*tau+t-i)); j+=1){
              stencil(data,t,i,j);
            } // for j
          } // for i
        } // for t
      } // for k2
    } // for k1
  } // for thyme

  double end_time = omp_get_wtime();
  double time =  (end_time - start_time);

  measurements.setField("elapsedTime",time);

  // End timed test

  // 4 - Verification and output (optional)
  // Verification
  if( configuration.getBool("v") ){
    if( verifyResultJacobi2D(data,configuration) ){
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
    cout<<endl;
  }

  // End verification and output

  // Free the space in memory
  free( data );

  return 0;
}
