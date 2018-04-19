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

#include "util3DArray.h"
#include "../common/Measurements.h"
#include "../common/Configuration.h"

#define OTS    1024
#define ITS    1024
#define ITSOTS (ITS * OTS)

using namespace std;

double run_schedule(const string& schedule, real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots);

// #1
void schedule_baseline(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2);
// #2
void schedule_unrolled(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2);
// #3
void schedule_fuse_0_0_1_1_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2);
// #4
void schedule_tile_inner_inner_serial_serial_fuse_0_0_1_1_(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its);
// #5
void schedule_fuse_0_0_1_1_tile_inner_inner_serial_serial_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its );
// #6
void schedule_tile_inner_inner_serial_tile_outer_outer_fuse_0_0_1_1_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots );
// #7
void schedule_tile_inner_inner_tile_outer_outer_serial_fuse_0_0_1_1_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots );
// #8
void schedule_tile_inner_inner_tile_10_10_parallel_serial_serial_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots );
// #9
void schedule_tile_outer_outer_tile_parallel_serial_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int ots );

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

  configuration.addParamString("schedule", 's', "baseline",
                               "-s <schedule>, scheduling method");

  configuration.addParamInt("inner-tile-size", 'i', ITS,
                            "-i <inner-tile-size>, inner-tile-size");

  configuration.addParamInt("outer-tile-size", 'o', OTS,
                            "-i <outer-tile-size>, outer-tile-size");

  configuration.parse(argc,argv);

  if(configuration.getInt("num_threads") > omp_get_max_threads()){
    printf("--num_threads cannot be more than %d\n",omp_get_max_threads());
    return false;
  }else if (configuration.getInt("num_threads") < 1){
    printf("--num_threads cannot be less than %d\n",1);
    return false;
  }else{
    omp_set_num_threads(configuration.getInt("num_threads"));
    printf("Checking %d vs %d threads\n",omp_get_max_threads(),configuration.getInt("num_threads"));
  } 

  // End Command line parsing and checking 

  // 2 - Data allocation and initialization
  int i;
  int j;
  int Nx = configuration.getInt("Nx");
  int Ny = configuration.getInt("Ny");
  int upper_bound_T = configuration.getInt("T");  
  int lower_bound_i = 1;
  int lower_bound_j = 1;  
  int upper_bound_i = lower_bound_i + Ny - 1;
  int upper_bound_j = lower_bound_j + Nx - 1;
  int inner_tile_size = configuration.getInt("inner-tile-size");
  int outer_tile_size = configuration.getInt("outer-tile-size");

  real **data[2];

  // Allocate time-steps 0 and 1
  if( !allocateSpace(data,configuration) ){
    printf( "Could not allocate space for array\n" );
    return 1;
  }
  // Initialize arrays 
  // First touch for openmp	
  for( i = lower_bound_i; i <= upper_bound_i; ++i ){
    for( j = lower_bound_j; j <= upper_bound_j; ++j ){
      data[0][i][j] = 0;
    }
  }
  initializeSpace(data,configuration);
 
  // End data allocation and initialization

  // 3 - Perform benchmark on given schedule...

  string schedule = configuration.getString("schedule");
  double time = run_schedule(schedule, data, upper_bound_T, lower_bound_i, upper_bound_i, lower_bound_j, upper_bound_j, inner_tile_size, outer_tile_size);
  measurements.setField("elapsedTime",time);

  // End timed test
  
  // 4 - Verification and output (optional)
  // Verification
  if( configuration.getBool("v") ){
    if( verifyResultJacobi2D(data[(upper_bound_T) & 1],configuration) ){
       measurements.setField("verification","SUCCESS");    
    }else{
       measurements.setField("verification","FAILURE");
    }
  }
  // Output
  if( !configuration.getBool("n") ){
    string ldap = configuration.toLDAPString();
    ldap += measurements.toLDAPString();
    cout << ldap;

    if (schedule.size() > 0) {
        cout << "schedule:" << schedule << ",";
    }

    cout << endl;
  }

  // End verification and output
  
  // Free the space in memory
   for( int i = 0; i < configuration.getInt("Ny") + 2; ++i ){
     free( data[0][i] );
     free( data[1][i] );
   } 
   free( data[0] );
   free( data[1] );
 
  return 0;   
}

double run_schedule(const string& schedule, real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots) {
    double time = omp_get_wtime();

    if (schedule.find("base") != string::npos) {                                                            // 1
        schedule_baseline(data, ubT, lb1, ub1, lb2, ub2);
    } else if (schedule.find("unroll") != string::npos) {                                                   // 2
        schedule_unrolled(data, ubT, lb1, ub1, lb2, ub2);
    } else if (schedule.find("tile_inner_inner_serial_serial_fuse_0_0_1_1") != string::npos) {              // 4
        schedule_tile_inner_inner_serial_serial_fuse_0_0_1_1_(data, ubT, lb1, ub1, lb2, ub2, its);
    } else if (schedule.find("fuse_0_0_1_1_tile_inner_inner_serial_serial") != string::npos) {              // 5
        schedule_fuse_0_0_1_1_tile_inner_inner_serial_serial_(data, ubT, lb1, ub1, lb2, ub2, its);
    } else if (schedule.find("tile_inner_inner_serial_tile_outer_outer_fuse_0_0_1_1") != string::npos) {    // 6
        schedule_tile_inner_inner_serial_tile_outer_outer_fuse_0_0_1_1_(data, ubT, lb1, ub1, lb2, ub2, its, ots);
    } else if (schedule.find("tile_inner_inner_tile_outer_outer_serial_fuse_0_0_1_1") != string::npos) {    // 7
        schedule_tile_inner_inner_tile_outer_outer_serial_fuse_0_0_1_1_(data, ubT, lb1, ub1, lb2, ub2, its, ots);
    } else if (schedule.find("tile_inner_inner_tile_10_10_parallel_serial_serial") != string::npos) {       // 8
        schedule_tile_inner_inner_tile_10_10_parallel_serial_serial_(data, ubT, lb1, ub1, lb2, ub2, its, ots);
    } else if (schedule.find("tile_outer_outer_tile_parallel_serial") != string::npos) {       // 9
        schedule_tile_outer_outer_tile_parallel_serial_(data, ubT, lb1, ub1, lb2, ub2, its);
    } else if (schedule.find("fuse_0_0_1_1") != string::npos) {                                             // 3
        schedule_fuse_0_0_1_1_(data, ubT, lb1, ub1, lb2, ub2);
    } else {
        cerr << "Unrecognized schedule: '" << schedule << "'" << endl;
    }

    return  (omp_get_wtime() - time);
}
// 1
void schedule_baseline(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2) {
    int i, j;
    for( int t = 1; t <= ubT; ++t ){
        #pragma omp parallel for private(i,j) schedule(runtime)
        for( i = lb1; i <= ub1; ++i ){
            for( j = lb2; j <= ub2; ++j ){
                stencil(t,i,j);
            }
        }
    }
}
// 2
void schedule_unrolled(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2) {
    int i, j;

    bool is_odd_T = (ubT % 2 != 0);
    for( int t = 1; t <= ubT; t += 2 ){
         #pragma omp-lc loopchain schedule(fuse((0,0),(1,1)))
        {
           #pragma omp-lc for domain(lb1:ub1,lb2:ub2)\
	       with (i,j) write A {(i,j)}, \
	                  read  B {(i-1,j),(i,j),(i+1,j),\
				   (i,j-1),(i,j+1)}
            for( i = lb1; i <= ub1; ++i ) {
                for (j = lb2; j <= ub2; ++j) {
                    stencil(t, i, j);
                }
            }

           #pragma omp-lc for domain(lb1:ub1,lb2:ub2)\
	       with (i,j) write B {(i,j)}, \
	                  read  A {(i-1,j),(i,j),(i+1,j),\
				   (i,j-1),(i,j+1)}
            for( i = lb1; i <= ub1; ++i ) {
                for (j = lb2; j <= ub2; ++j) {
                    stencil(t+1, i, j);
                }
            }
        }// end of loop chain
    } // end of time loop

    if (is_odd_T) {
        for (i = lb1; i <= ub1; ++i) {
            for (j = lb2; j <= ub2; ++j) {
                stencil(ubT, i, j);
            }
        }
    }
}

// Imported from generated_code.cpp..

// 3  ==========================================================
//schedule(fuse((0,0),(1,1)))
// ==========================================================
void schedule_fuse_0_0_1_1_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2 ){

if (ub1 >= lb1 && ub2 >= lb2) {
for( int t = 1; t <= ubT; t += 2 ){
  for (int c2 = lb2; c2 <= ub2; c2 += 1)
    stencil(t,lb1, c2);
  for (int c1 = lb1 + 1; c1 <= ub1; c1 += 1) {
    stencil(t, c1, lb2);
    for (int c2 = lb2 + 1; c2 <= ub2; c2 += 1) {
      stencil(t, c1, c2);
      stencil(t+1, c1 - 1, c2 - 1);
    }
    stencil(t+1,c1 - 1, ub2);
  }
  for (int c2 = lb2 + 1; c2 <= ub2 + 1; c2 += 1)
    stencil(t+1,ub1, c2 - 1);
}}
}
// 4  ==========================================================
//schedule(tile((its,its),serial,serial),fuse((0,0),(1,1)))
// ==========================================================
/*void schedule_tile_inner_inner_serial_serial_fuse_0_0_1_1_(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its){
    for( int t = 1; t <= ubT; t += 2 ){
        for (int c1 = floord(lb1, its); c1 <= floord(ub1, its); c1 += 1){
            for (int c2 = floord(lb2, its); c2 <= floord(ub2, its); c2 += 1) {
                for (int c4 = max(lb1, its * c1); c4 <= min(ub1, its * c1 + (its - 1)); c4 += 1){
                    for (int c5 = max(lb2, its * c2); c5 <= min(ub2, its * c2 + (its - 1)); c5 += 1){
                        stencil(t, c4, c5);
                    }
                }
                for (int c4 = max(lb1 + 1, its * c1 + 1); c4 <= min(ub1 + 1, its * c1 + its); c4 += 1){
                    for (int c5 = max(lb2 + 1, its * c2 + 1); c5 <= min(ub2 + 1, its * c2 + its); c5 += 1){
                        stencil(t + 1, c4 - 1, c5 - 1);
                    }
                }
            }
        }
    }
} */
#define ITS 64
void schedule_tile_inner_inner_serial_serial_fuse_0_0_1_1_(real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its){
fprintf(stderr,"I am in the right place\n");
    for( int t = 1; t <= ubT; t += 2 ){
        for (int c1 = floord(lb1, ITS); c1 <= floord(ub1, ITS); c1 += 1)
            for (int c2 = floord(lb2, ITS); c2 <= floord(ub2, ITS); c2 += 1) {
                for (int c4 = max(lb1, ITS * c1); c4 <= min(ub1, ITS * c1 + (ITS - 1)); c4 += 1)
                    for (int c5 = max(lb2, ITS * c2); c5 <= min(ub2, ITS * c2 + (ITS - 1)); c5 += 1)
                        stencil(t, c4, c5);
                for (int c4 = max(lb1 + 1, ITS * c1 + 1); c4 <= min(ub1 + 1, ITS * c1 + ITS); c4 += 1)
                    for (int c5 = max(lb2 + 1, ITS * c2 + 1); c5 <= min(ub2 + 1, ITS * c2 + ITS); c5 += 1)
                        stencil(t + 1, c4 - 1, c5 - 1);
            }
    }
}
// 5 ==========================================================
//schedule(fuse((0,0),(1,1)),tile((its,its),serial,serial))
// ==========================================================
void schedule_fuse_0_0_1_1_tile_inner_inner_serial_serial_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its) {
    if (ub1 >= lb1 && ub2 >= lb2)
        for( int t = 1; t <= ubT; t += 2 ){
            for (int c1 = floord(lb1, its); c1 <= floord(ub1 + 1, its); c1 += 1)
                for (int c2 = max(floord(lb2, its), c1 + floord(lb2 - ub1, its));
                     c2 <= min(c1 + floord(-lb1 + ub2 - 1, its) + 1, floord(ub2 + 1, its)); c2 += 1)
                    for (int c4 = max(max(lb1, its * c1), lb1 - ub2 + its * c2);
                         c4 <= min(min(ub1 + 1, its * c1 + (its - 1)), -lb2 + ub1 + its * c2 + (its - 1)); c4 += 1)
                        for (int c5 = max(max(lb2, its * c2), lb2 - ub1 + c4);
                             c5 <= min(min(ub2 + 1, its * c2 + (its - 1)), -lb1 + ub2 + c4); c5 += 1) {
                            if (ub1 >= c4 && ub2 >= c5)
                                stencil(t, c4, c5);
                            if (c4 >= lb1 + 1 && c5 >= lb2 + 1)
                                stencil(t + 1, c4 - 1, c5 - 1);
                        }
        }
}
// 6  ==========================================================
//schedule(tile((ITS,ITS),serial,tile((OTS,OTS))),fuse((0,0),(1,1)))
// ==========================================================
void schedule_tile_inner_inner_serial_tile_outer_outer_fuse_0_0_1_1_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots ){
    for( int t = 1; t <= ubT; t += 2 ){
        for (int c1 = floord(lb1, ots); c1 <= floord(ub1, ots); c1 += 1)
            for (int c2 = floord(lb2, ots); c2 <= floord(ub2, ots); c2 += 1) {
                for (int c4 = max(floord(lb1, its), c1 + floord(c1, 8));
                     c4 <= min(floord(ub1, its), c1 + floord(c1, 8) + 1); c4 += 1)
                    for (int c5 = max(floord(lb2, its), c2 + floord(c2, 8));
                         c5 <= min(floord(ub2, its), c2 + floord(c2, 8) + 1); c5 += 1)
                        for (int c7 = max(max(lb1, ots * c1), its * c4);
                             c7 <= min(min(ub1, ots * c1 + (ots - 1)), its * c4 + (its - 1)); c7 += 1)
                            for (int c8 = max(max(lb2, ots * c2), its * c5);
                                 c8 <= min(min(ub2, ots * c2 + (ots - 1)), its * c5 + (its - 1)); c8 += 1)
                                stencil(t, c7, c8);
                for (int c4 = max(floord(lb1, its), c1 + floord(c1, 8));
                     c4 <= min(floord(ub1, its), c1 + floord(c1, 8) + 1); c4 += 1)
                    for (int c5 = max(floord(lb2, its), c2 + floord(c2, 8));
                         c5 <= min(floord(ub2, its), c2 + floord(c2, 8) + 1); c5 += 1)
                        for (int c7 = max(max(lb1 + 1, ots * c1 + 1), its * c4 + 1);
                             c7 <= min(min(ub1 + 1, ots * c1 + ots), its * c4 + its); c7 += 1)
                            for (int c8 = max(max(lb2 + 1, ots * c2 + 1), its * c5 + 1);
                                 c8 <= min(min(ub2 + 1, ots * c2 + ots), its * c5 + its); c8 += 1)
                                stencil(t + 1, c7 - 1, c8 - 1);
            }
    }
}
// 7  ==========================================================
//schedule(tile((ITS,ITS),tile((OTS,OTS)),serial),fuse((0,0),(1,1)))
// ==========================================================
void schedule_tile_inner_inner_tile_outer_outer_serial_fuse_0_0_1_1_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots ){
    long itsots = its * ots;
    for( int t = 1; t <= ubT; t += 2 ){
        for (int c1 = floord(lb1, itsots); c1 <= floord(ub1, itsots); c1 += 1)
            for (int c2 = floord(lb2, itsots); c2 <= floord(ub2, itsots); c2 += 1) {
                for (int c4 = max(its * c1, floord(lb1, ots));
                     c4 <= min(its * c1 + (its - 1), floord(ub1, ots)); c4 += 1)
                    for (int c5 = max(its * c2, floord(lb2, ots));
                         c5 <= min(its * c2 + (its - 1), floord(ub2, ots)); c5 += 1)
                        for (int c7 = max(lb1, ots * c4); c7 <= min(ub1, ots * c4 + (ots - 1)); c7 += 1)
                            for (int c8 = max(lb2, ots * c5); c8 <= min(ub2, ots * c5 + (ots - 1)); c8 += 1)
                                stencil(t, c7, c8);
                for (int c4 = max(its * c1, floord(lb1, ots));
                     c4 <= min(its * c1 + (its - 1), floord(ub1, ots)); c4 += 1)
                    for (int c5 = max(its * c2, floord(lb2, ots));
                         c5 <= min(its * c2 + (its - 1), floord(ub2, ots)); c5 += 1)
                        for (int c7 = max(lb1 + 1, ots * c4 + 1); c7 <= min(ub1 + 1, ots * c4 + ots); c7 += 1)
                            for (int c8 = max(lb2 + 1, ots * c5 + 1); c8 <= min(ub2 + 1, ots * c5 + ots); c8 += 1)
                                stencil(t + 1, c7 - 1, c8 - 1);
            }
    }
}
// 8 ==========================================================
//schedule(tile((OTS,OTS),tile((ITS,ITS), parallel, serial),serial))
// ==========================================================
void schedule_tile_inner_inner_tile_10_10_parallel_serial_serial_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int its, int ots ){
    for( int t = 1; t <= ubT; t += 2 ){
        for (int c1 = floord(lb1, (ots * its)); c1 <= floord(ub1, (ots * its)); c1 += 1)
            for (int c2 = floord(lb2, (ots * its)); c2 <= floord(ub2, (ots * its)); c2 += 1)
                    #pragma omp parallel for
                    for (int c4 = max(its * c1, floord(lb1, ots));
                         c4 <= min(its * c1 + 63, floord(ub1, ots)); c4 += 1)
                        for (int c5 = max(its * c2, floord(lb2, ots));
                             c5 <= min(its * c2 + 63, floord(ub2, ots)); c5 += 1)
                            for (int c7 = max(lb1, ots * c4); c7 <= min(ub1, ots * c4 + (ots - 1)); c7 += 1)
                                for (int c8 = max(lb2, ots * c5); c8 <= min(ub2, ots * c5 + (ots - 1)); c8 += 1)
                                    stencil(t, c7, c8);
        for (int c1 = floord(lb1, (ots * its)); c1 <= floord(ub1, (ots * its)); c1 += 1)
            for (int c2 = floord(lb2, (ots * its)); c2 <= floord(ub2, (ots * its)); c2 += 1)
                     #pragma omp parallel for
                    for (int c4 = max(its * c1, floord(lb1, ots));
                         c4 <= min(its * c1 + 63, floord(ub1, ots)); c4 += 1)
                        for (int c5 = max(its * c2, floord(lb2, ots));
                             c5 <= min(its * c2 + 63, floord(ub2, ots)); c5 += 1)
                            for (int c7 = max(lb1, ots * c4); c7 <= min(ub1, ots * c4 + (ots - 1)); c7 += 1)
                                for (int c8 = max(lb2, ots * c5); c8 <= min(ub2, ots * c5 + (ots - 1)); c8 += 1)
                                    stencil(t + 1, c7, c8);
    }
}

// 9 ==========================================================
//schedule(tile((OTS,OTS), parallel, serial)
// ==========================================================
void schedule_tile_outer_outer_tile_parallel_serial_( real **data[2], int ubT, int lb1, int ub1, int lb2, int ub2, int ots ) {
    for( int t = 1; t <= ubT; t += 2 ) {
        #pragma omp parallel for
        for (int c1 = floord(lb1, ots); c1 <= floord(ub1, ots); c1 += 1)
            for (int c2 = floord(lb2, ots); c2 <= floord(ub2, ots); c2 += 1)
                for (int c4 = max(lb1, ots * c1); c4 <= min(ub1, ots * c1 + (ots - 1)); c4 += 1)
                    for (int c5 = max(lb2, ots * c2); c5 <= min(ub2, ots * c2 + (ots - 1)); c5 += 1)
                        stencil(t, c4, c5);

        #pragma omp parallel for
        for (int c1 = floord(lb1, ots); c1 <= floord(ub1, ots); c1 += 1)
            for (int c2 = floord(lb2, ots); c2 <= floord(ub2, ots); c2 += 1)
                for (int c4 = max(lb1, ots * c1); c4 <= min(ub1, ots * c1 + (ots - 1)); c4 += 1)
                    for (int c5 = max(lb2, ots * c2); c5 <= min(ub2, ots * c2 + (ots - 1)); c5 += 1)
                        stencil(t + 1, c4, c5);
    }
}
