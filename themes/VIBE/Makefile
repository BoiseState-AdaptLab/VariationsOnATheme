#!/bin/bash

SRC = .
NVCC = nvcc
# fix the CXX path, so that it points to location where your g++ is located
CXX = /usr/lib/gcc/5.3.0/bin/g++ 

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart `pkg-config --libs opencv`
# tells CUDA which g++ compiler to use (if omitted will use the default one)
CUDA_FLAGS= --compiler-bindir $(CXX)

CPUOBJS = Measurements.o Configuration.o util.o verification.o
GPUOBJS = Configuration.o util.o Measurements.o verification.o

CUDAVARIANT_1 = VIBE-NaiveParallel-CUDA
CUDAVARIANT_2 = VIBE-NaiveParallelBlur-CUDA	
CUDAVARIANT_3 = VIBE-FusedParallelBlur_Overlap-CUDA
CUDAVARIANT_4 = VIBE-NaiveParallel_Overlap-CUDA	
	       
CVARIANTS = VIBE-NaiveSerial-CPP\
            VIBE-NaiveSerial_memcpy-CPP\
            VIBE-NaiveSerial_nocopy-CPP\
            vibe_generate_map\
            vibe_generate_histogram\
            generate_video\
                 
 
$(CUDAVARIANT_1): % : %-driver.cpp %-kernel.cu %-kernel.h util.cpp util.h verification.cpp verification.h
	$(CXX) $(CFLAGS) -std=c++11 -c VIBE-NaiveParallel-CUDA-driver.cpp -o VIBE-NaiveParallel-CUDA-driver.o
	$(CXX) -c -O3 util.cpp -o util.o
	$(CXX) -c -O3 verification.cpp -o verification.o
	$(CXX) -c -O3 $(SRC)/../common/Configuration.cpp -o Configuration.o
	$(CXX) -c -O3 $(SRC)/../common/Measurements.cpp -o Measurements.o
	$(NVCC) $(CUDA_FLAGS) -G -c -O3 VIBE-NaiveParallel-CUDA-kernel.cu -o VIBE-NaiveParallel-CUDA-kernel.o
	$(CXX) -o $(CUDAVARIANT_1) VIBE-NaiveParallel-CUDA-driver.o $(CPUOBJS) VIBE-NaiveParallel-CUDA-kernel.o -O3 $(LDFLAGS)

$(CUDAVARIANT_2): % : %-driver.cpp %-kernel.cu %-kernel.h util.cpp util.h verification.cpp verification.h
	$(CXX) $(CFLAGS) -std=c++11 -c VIBE-NaiveParallelBlur-CUDA-driver.cpp -o VIBE-NaiveParallelBlur-CUDA-driver.o
	$(CXX) -c -O3 util.cpp -o util.o
	$(CXX) -c -O3 verification.cpp -o verification.o
	$(CXX) -c -O3 $(SRC)/../common/Configuration.cpp -o Configuration.o
	$(CXX) -c -O3 $(SRC)/../common/Measurements.cpp -o Measurements.o
	$(NVCC) $(CUDA_FLAGS) -G -c -O3 VIBE-NaiveParallelBlur-CUDA-kernel.cu -o VIBE-NaiveParallelBlur-CUDA-kernel.o
	$(CXX) -o $(CUDAVARIANT_2) VIBE-NaiveParallelBlur-CUDA-driver.o $(CPUOBJS) VIBE-NaiveParallelBlur-CUDA-kernel.o -O3 $(LDFLAGS)

$(CUDAVARIANT_4): % : %-driver.cpp %-kernel.cu %-kernel.h util.cpp util.h verification.cpp verification.h
	$(CXX) $(CFLAGS) -std=c++11 -c VIBE-NaiveParallel_Overlap-CUDA-driver.cpp -o VIBE-NaiveParallel_Overlap-CUDA-driver.o
	$(CXX) -c -O3 util.cpp -o util.o
	$(CXX) -c -O3 verification.cpp -o verification.o
	$(CXX) -c -O3 $(SRC)/../common/Configuration.cpp -o Configuration.o
	$(CXX) -c -O3 $(SRC)/../common/Measurements.cpp -o Measurements.o
	$(NVCC) $(CUDA_FLAGS) -G -c -O3 VIBE-NaiveParallel_Overlap-CUDA-kernel.cu -o VIBE-NaiveParallel_Overlap-CUDA-kernel.o
	$(CXX) -o $(CUDAVARIANT_4) VIBE-NaiveParallel_Overlap-CUDA-driver.o $(CPUOBJS) VIBE-NaiveParallel_Overlap-CUDA-kernel.o -O3 $(LDFLAGS)


$(CUDAVARIANT_3): % : %-driver.cpp %-kernel.cu %-kernel.h util.cpp util.h verification.cpp verification.h
	$(CXX) $(CFLAGS) -std=c++11 -c VIBE-FusedParallelBlur_Overlap-CUDA-driver.cpp -o VIBE-FusedParallelBlur_Overlap-CUDA-driver.o
	$(CXX) -c -O3 util.cpp -o util.o
	$(CXX) -c -O3 verification.cpp -o verification.o
	$(CXX) -c -O3 $(SRC)/../common/Configuration.cpp -o Configuration.o
	$(CXX) -c -O3 $(SRC)/../common/Measurements.cpp -o Measurements.o
	$(NVCC) $(CUDA_FLAGS) -G -c -O3 VIBE-FusedParallelBlur_Overlap-CUDA-kernel.cu -o VIBE-FusedParallelBlur_Overlap-CUDA-kernel.o
	$(CXX) -o $(CUDAVARIANT_3) VIBE-FusedParallelBlur_Overlap-CUDA-driver.o $(CPUOBJS) VIBE-FusedParallelBlur_Overlap-CUDA-kernel.o -O3 $(LDFLAGS)

median_test_driver: % : %.cpp  util.cpp util.h verification.cpp verification.h
	$(CXX) $(CFLAGS) -std=c++11 -c median_test_driver.cpp -o median_test_driver.o
	$(CXX) -c -O3 util.cpp -o util.o
	$(CXX) -c -O3 verification.cpp -o verification.o
	$(CXX) -c -O3 $(SRC)/../common/Configuration.cpp -o Configuration.o
	$(CXX) -c -O3 $(SRC)/../common/Measurements.cpp -o Measurements.o
	$(NVCC) $(CUDA_FLAGS) -G -c -O3 VIBE-FusedParallelBlur-CUDA-kernel.cu -o VIBE-FusedParallelBlur-CUDA-kernel.o
	$(CXX) -o median_test_driver median_test_driver.o $(CPUOBJS) VIBE-FusedParallelBlur-CUDA-kernel.o -O3 $(LDFLAGS)


$(CVARIANTS): % : %.cpp $(SRC)/../common/Configuration.cpp $(SRC)/../common/Measurements.cpp $(SRC)/util.cpp $(SRC)/verification.cpp 
	$(CXX) $^ -pg -g -std=c++11 -O3 -o $@ $(LDFLAGS) 
	
clean:
	rm -f *o gmon.out
	rm -f $(CVARIANTS) $(CUDAVARIANT_1) $(CUDAVARIANT_2) $(CUDAVARIANT_3) $(CUDAVARIANT_4) $(CUDAVARIANT_5)

