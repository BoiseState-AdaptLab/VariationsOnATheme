#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cmath>
#include "timer.h"
#include "../common/Configuration.cpp"
#include "../common/Measurements.h"
using namespace std;



void printFullMatrix(double *data, int W, int H);
void printMatrix(double *data, int W, int H);



int validate(double *result, int width, int height){
  // int i,j;
  //
  //
  //
  // double *B  = (double *) malloc( sizeof(double) * N * N );
  //
  // // Initialization
  // for (i=0; i < N; i++ ) {
  //   for (j=0; j < N; j++){
  //     B(i, j) = 0.0;
  //   }
  // }
  //
  // // S1 operation:
  // // Calculate the average of the surrounding values in the A matrix, the store the average in the current cell
  // // of the B matrix.
  // #pragma omp parallel for
  // for (i=1; i < N-1; i++) {
  //   for (j=1;j < N-1; j++) {
  //     B(i,j) = ((A(i-1,j-1)+A(i-1,j)+A(i-1,j+1)+A(i,j-1)+A(i,j+1)+A(i+1,j-1)+A(i+1,j)+A(i+1,j+1)))/8;
  //   }
  // }
  //
  // // S2 operation
  // // Calculate -> ABS(cell - southwestNeighbor) + ABS(southNeighbor - westNeighbor)
  // #pragma omp parallel for
  // for (i=1; i<N-2; i++) {
  //   for (j=2; j<N-1; j++){
  //     A(i,j) = fabs(B(i,j)-B(i+1,j-1)) + fabs(B(i+1,j)-B(i,j-1));
  //   }
  // }
  //
  // // Validate each element in the result matrix and the verification matrix
  // for ( i=0 ; i < N ; i++ ) {
  //   for ( j=0 ; j < N ; j++) {
  //     if (A[i*N+j]!=result[i*N+j]){
  //       return 0;
  //     }
  //   }
  // }
  // return 1;
}
