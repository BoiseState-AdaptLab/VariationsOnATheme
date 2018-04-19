/*
 * util.c - util functions
 */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <math.h>
#include "util.h"
// #include "../common/Configuration.h"
#define   A(i,j)           A[(i)*N+(j)]
#define   B(i,j)           B[(i)*N+(j)]

void printMatrix(double *data, int size);
void printFullMatrix(double *data, int size);

int validate(double *data, int size) {

	int     N = size;
	int     t;
	double  *A, *B;



	//~~~~~Step 1. Command Line Parsing~~~~~
	
	//How config will hopefully work:
	//	Configuration config = new Configuration(argc,argv)
	//	int N = config.getParam("Nx") 
	//Potentially:
	//	get_NX();
	
	//Configuration config;
	//config.parseCmdLine(argc,argv);
	//N = config.cmdLineArgs->Nx;
	//printf("Here is our N: %d",N);

	// Timer
	double  time;

	// temporary variables
	int     i,j;
	double  *temp;

	// Check commandline args.
	if ( argc > 1 ) {
		N = atoi(argv[1]);
	} else {
		printf("Usage : %s [N]\n", argv[0]);
		exit(1);
	}
	

	//~~~~~Step 2. Allocate & Initialize use Srand w/ global seed (when available)~~~~~ 
	
	// Initiallization and Allocation should be in a utils file
	// Should return a pointer to the initialized data
	//
	// Memory allocation for data array.
	A  = (double *) malloc( sizeof(double) * N * N );
	B  = (double *) malloc( sizeof(double) * N * N );
	if ( A == NULL || B == NULL ) {
		printf("[ERROR] : Fail to allocate memory.\n");
		exit(1);
	}

	// Initialization
	for (i=0; i < N; i++ ) {
		for (j=0; j < N; j++){
			A(i, j) = (double)(rand() % (10 + 1 - 0) + 0);
			B(i, j) = 0.0;
		}
	}
	FILE *f = fopen("file.txt", "w");
	//~~~~~Step 3. Benchmard (Timed Solution)~~~~~
	

	// change to OpenMP timer (include in the library) beware of overhead associated with using 1 proccessor
	initialize_timer();
	start_timer();

	// S1 operation:
	// Calculate the average of the surrounding values in the A matrix, the store the average in the current cell
	// of the B matrix.
	for (i=1; i < N-1; i++) {
		for (j=1;j < N-1; j++) {
			B(i,j) = ((A(i-1,j-1)+A(i-1,j)+A(i-1,j+1)+A(i,j-1)+A(i,j+1)+A(i+1,j-1)+A(i+1,j)+A(i+1,j+1)))/8;
		}
	}

	// S2 operation
	// Calculate -> ABS(cell - southwestNeighbor) + ABS(southNeighbor - westNeighbor)
	//
	// Look for a way to get a faster absolute value (bit operations)
	// 	-> compare to the current
	//
	for (i=1; i<N-2; i++) {
		for (j=2; j<N-1; j++){
			A(i,j) = fabs(B(i,j)-B(i+1,j-1)) + fabs(B(i+1,j)-B(i,j-1));
		}
	}

	stop_timer();
	time = elapsed_time();

	printf("Data : %d by %d, Iterations : %d , Time : %lf sec\n", N, N, t, time);



}

//~~~~~Step 4. Printing results & validation~~~~~

void printFullMatrix(double *data, int size) {
	int i,j;
	/* print a portion of the matrix */
	for ( i=0 ; i < size ; i++ ) {
		for ( j=0 ; j < size ; j++) {
			printf("%lf ", data[i*size+j]);
		}
		printf("\n");
	}

	return;
}

void printMatrix(double *data, int size) {
	int i,j;
	/* print a portion of the matrix */
	for ( i= size/10 ; i < size/2 ; i+=size/10 ) {
		for ( j=size/10 ; j < size/2 ; j+=size/10 ) {
			printf("%lf ", data[i*size+j]);
		}
		printf("\n");
	}

	return;
}