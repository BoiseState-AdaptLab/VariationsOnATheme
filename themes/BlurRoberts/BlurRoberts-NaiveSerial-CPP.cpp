#include "util.h"
#define   A(i,j)           A[(i)*Nx+(j)]
#define   B(i,j)           B[(i)*Nx+(j)]




int main(int argc, char **argv) {

  // Step 1. Command Line Parsing

  int Nx;
  int Ny;
  int t = 1;
  if (argc<3){
    cout<<"Error in parsing command line args"<<endl;
    cout<<"Usage: "<<argv[0]<<" -<param> <value"<<endl;
    exit(EXIT_FAILURE);
  }


  Configuration config(argc,argv);

  // If not defined, will be set to 100
  Nx = config.getInt("Nx");

  // If not defined, will be set to 100
  Ny = config.getInt("Ny");

  // ******************** Running into some errors with this, view below  ********************
  Measurements m();




  // Step 2. Allocate and Initialize, use Srand w/ global seed
  int size = Nx*Ny;

  // Allocate the size of the array on the heap
  double *A  = new double[( sizeof(double) * size )];
  double *B  = new double[( sizeof(double) * size )];

  if (A==NULL || B==NULL){
    cout<<"[ERROR] : Fail to allocate memory."<<endl;
    exit(1);
  }

  for (int i=0; i<Ny; i++){
    for (int j=0; j<Nx; j++){

      // ******************** Assuming fixing the makefile will get this to work? Is this how accessing srands will work? ********************
      // Need to include this by working on the Makefile
      // A(i, j) = srand(config.getInt("global_seed")) % (max + min -0) +0;

      // Temporary
      A(i, j) = (double)(rand() % (10 + 1 - 0) + 0);

      // Final
      B(i, j) = 0.0;

    }
  }


	// Step 3. Benchmard (Timed Solution)

	double start_time = omp_get_wtime();

	// S1 operation:
	// Calculate the average of the surrounding values in the A matrix, the store the average in the current cell
	// of the B matrix.
	for (int i=1; i < Ny-1; i++) {
		for (int j=1;j < Nx-1; j++) {
			B(i,j) = ((A(i-1,j-1)+A(i-1,j)+A(i-1,j+1)+A(i,j-1)+A(i,j+1)+A(i+1,j-1)+A(i+1,j)+A(i+1,j+1)))/8;
		}
  }



	// S2 operation
	// Calculate -> ABS(cell - southwestNeighbor) + ABS(southNeighbor - westNeighbor)
	for (int i=1; i<Ny-2; i++) {
		for (int j=2; j<Nx-1; j++){
			A(i,j) = abs(B(i,j)-B(i+1,j-1)) + abs(B(i+1,j)-B(i,j-1));
		}
	}

  double time = omp_get_wtime()-start_time;

  // Display the time it took to complete the operation and the problem size
	printf("Data : %d by %d , Time : %lf sec\n", Nx, Ny, time);

  // ------------------------------------------------------------------------------
  // INSTRUCTIONS IN NOTES:
  // if (config.getBool('validate')){
  //   call validate (this would be util.h's validate function)
  //   if (true){
  //     m.setFeild("validation",success)
  //   }
  //   else{
  //     m.setField("Validation",failure)
  //   }
  // }

  // cout<<config.toLDAPString()<<endl;
  // ------------------------------------------------------------------------------

  // error: request for member ‘getLDAPString’ in ‘m’, which is of non-class type ‘Measurements()’
  m.getLDAPString();

  // clean up memory so there are no leaks
  delete [] A;
  delete [] B;

}

// Step 4. Printing results & validation

void printFullMatrix(double *data, int width, int height) {
	int i,j;
	/* print a portion of the matrix */
	for ( i=0 ; i < height ; i++ ) {
		for ( j=0 ; j < width ; j++) {
			printf("%lf ", data[i*width+j]);
		}
		printf("\n");
	}

	return;
}

void printMatrix(double *data, int width, int height) {
	int i,j;
	/* print a portion of the matrix */
	for ( i=height/10 ; i < height/2 ; i+=height/10 ) {
		for ( j=width/10 ; j < width/2 ; j+=width/10 ) {
			printf("%lf ", data[i*width+j]);
		}
		printf("\n");
	}

	return;
}
