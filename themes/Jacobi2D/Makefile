CC=gcc
CXX=g++
COMMON_DIR=../common
OPTIONS_OMP=  -ffp-contract=off -g -O0 -fopenmp

CPPVARIANTS = Jacobi2D-NaiveParallel-OMP\
	Jacobi2D-DiamondSwap-OMP\
  Jacobi2D-NaiveParallel-LoopChain\
	Jacobi2D-Diamond-OMP

CPPVARIANT3DARRAY = Jacobi2D-NaiveParallel3DArray-LoopChain\
		    Jacobi2D-NaiveParallel3DArray-OMP

CVARIANTS = Jacobi2D-NaiveParallel-OMP_dyn\
	Jacobi2D-NaiveParallel-OMP_static\
	
CVARIANTS3DARRAY = Jacobi2D-NaiveParallel3DArray-OMPC

all: $(CPPVARIANTS) $(CPPVARIANT3DARRAY) $(CVARIANTS)

omp: $(CPPVARIANTS)

test3DArray: $(CPPVARIANT3DARRAY)

svn: $(CVARIANTS) 

$(CPPVARIANTS): % : %.cpp util.cpp $(COMMON_DIR)/Measurements.cpp $(COMMON_DIR)/Configuration.cpp
	$(CXX)  $^ -o $@ $(OPTIONS_OMP)

$(CPPVARIANT3DARRAY): % : %.cpp util.cpp $(COMMON_DIR)/Measurements.cpp $(COMMON_DIR)/Configuration.cpp
	$(CXX) $(OPTIONS_OMP)  $^ -o $@ 

$(CVARIANTS3DARRAY): % : %.c util3DArray.c
	$(CC)  $^ -o $@ $(OPTIONS_OMP)

$(CVARIANTS): % : %.test.c
	$(CC)  $^ -o $@ $(OPTIONS_OMP)

clean:
	if [ -e $(all) ]; then rm  *OMP; rm Jacobi2D-NaiveParallel-OMP_dyn; rm Jacobi2D-NaiveParallel-OMP_static; fi; 
 
