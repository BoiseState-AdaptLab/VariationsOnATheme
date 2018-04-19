CC=gcc
CXX=g++
NVCC=nvcc
COMMON_DIR=../common

#CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

OPTIONS_CUDA   := -O4 -ftz=false  $(GENCODE_FLAGS) -DCUDA -DCUDADEBUG -DCOUNT -DDATATYPEDOUBLE
OPTIONS_OMP    := -O4 -fopenmp -DDATATYPEDOUBLE

CUDAVARIANTS = Jacobi1D-NaiveParallelGlobal-CUDA\

CVARIANTS = Jacobi1D-NaiveParallel-OMP\

all: $(CUDAVARIANTS) $(CVARIANTS)

cuda: $(CUDAVARIANTS)

omp: $(CVARIANTS)

$(CUDAVARIANTS): % : %.test.cu util.cpp $(COMMON_DIR)/Measurements.cpp $(COMMON_DIR)/Configuration.cpp $(COMMON_DIR)/CUDA_util.cpp
		$(NVCC) $^ -o  $@ $(OPTIONS_CUDA)

$(CVARIANTS): % : %.cpp util.cpp $(COMMON_DIR)/Measurements.cpp $(COMMON_DIR)/Configuration.cpp
	$(CXX)  $^ -o $@ $(OPTIONS_OMP)

clean:
	if [ -a *CUDA ]; then rm  *CUDA; fi;
	if [ -a *OMP ]; then rm  *OMP; fi; 	
