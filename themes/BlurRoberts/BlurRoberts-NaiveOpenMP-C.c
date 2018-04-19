#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include "timer.h"
#include <math.h>
#include "tempValidate.h"
#define   A(i,j)           A[(i)*N+(j)]
#define   B(i,j)           B[(i)*N+(j)]
#define   initialData(i,j)	initialData[(i)*N +(j)]
regex_t regex;
int reti;

void printMatrix(double *data, int size);
void printFullMatrix(double *data, int size);




int main(int argc, char **argv) {

	int     N;
	int     t;
	double  *A, *B, *initialData;



	//~~~~~Step 1. Command Line Parsing~~~~~


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

	// Memory allocation for data array.
	A  = (double *) malloc( sizeof(double) * N * N );
	B  = (double *) malloc( sizeof(double) * N * N );
	initialData = (double *) malloc( sizeof(double) * N * N );
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
	for (i=0; i < N; i++ ) {
		for (j=0; j < N; j++){
			initialData(i, j) = A(i, j);
		}
	}

	//~~~~~Step 3. Benchmard (Timed Solution)~~~~~

	initialize_timer();
	start_timer();

	// S1 operation:
	// Calculate the average of the surrounding values in the A matrix, the store the average in the current cell
	// of the B matrix.
	#pragma omp parallel for
	for (i=1; i < N-1; i++) {
		for (j=1;j < N-1; j++) {
			B(i,j) = ((A(i-1,j-1)+A(i-1,j)+A(i-1,j+1)+A(i,j-1)+A(i,j+1)+A(i+1,j-1)+A(i+1,j)+A(i+1,j+1)))/8;
		}
	}

	// S2 operation
	// Calculate -> ABS(cell - southwestNeighbor) + ABS(southNeighbor - westNeighbor)
	#pragma omp parallel for
	for (i=1; i<N-2; i++) {
		for (j=2; j<N-1; j++){
			A(i,j) = fabs(B(i,j)-B(i+1,j-1)) + fabs(B(i+1,j)-B(i,j-1));
		}
	}

	stop_timer();
	time = elapsed_time();

	printf("Data : %d by %d, Iterations : %d , Time : %lf sec\n", N, N, t, time);
	printMatrix(A,N);
	char* validationStatus = validate(N, initialData, A);
	printf("%s\n",validationStatus);



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
