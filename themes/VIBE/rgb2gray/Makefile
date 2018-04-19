SRC=src
BIN=bin
OBJ=obj

NVCC = nvcc
CXX = g++
#CFLAGS = -I/usr/include -I/usr/local/include -I/usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/include -I/s/parsons/h/proj/vision/usr/duttar/OpenCV-2.4.5/include
#LFLAGS = -L/usr/lib64 -L/s/parsons/h/proj/vision/usr/duttar/OpenCV-2.4.5/lib -L/usr/local/lib -L/usr/local/cuda-6.5/targets/armv7-linux-gnueabihf/lib -lopencv_highgui -lopencv_imgproc -lopencv_core  
CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib -lcudart `pkg-config --libs opencv`


GPUOBJS = $(OBJ)/rgb2gray-image-CUDA.o $(OBJ)/timer.o 
GPUOBJS1 = $(OBJ)/rgb2gray-image-CUDA_int.o $(OBJ)/timer.o
GPUOBJS2 = $(OBJ)/rgb2gray-image-CUDA-1D_int.o $(OBJ)/timer.o
GPUOBJS3 = $(OBJ)/rgb2gray-image-CUDA-1D-memorycoelacing_int.o $(OBJ)/timer.o


rgb2gray-image-CUDA:$(GPUOBJS)
	$(NVCC) -g -o $(BIN)/rgb2gray-image-CUDA $(GPUOBJS) $(LDFLAGS)

$(OBJ)/rgb2gray-image-CUDA.o:$(SRC)/rgb2gray-image-CUDA.cu
	$(NVCC) $(CFLAGS) -g -G -c $(SRC)/rgb2gray-image-CUDA.cu -o $(OBJ)/rgb2gray-image-CUDA.o

$(OBJ)/timer.o:$(SRC)/timer.cu
	$(NVCC) $(CFLAGS) -g -G -c $(SRC)/timer.cu -o $(OBJ)/timer.o

rgb2gray-image-CUDA_int:$(GPUOBJS1)
	$(NVCC) -g -G -o $(BIN)/rgb2gray-image-CUDA_int $(GPUOBJS1) $(LDFLAGS)

$(OBJ)/rgb2gray-image-CUDA_int.o:$(SRC)/rgb2gray-image-CUDA_int.cu
	$(NVCC) $(CFLAGS) -g -G -c $(SRC)/rgb2gray-image-CUDA_int.cu -o $(OBJ)/rgb2gray-image-CUDA_int.o

rgb2gray-image-CUDA-1D_int:$(GPUOBJS2)
	$(NVCC) $(LFLAGS) -g -G -o $(BIN)/rgb2gray-image-CUDA-1D_int $(GPUOBJS2)

$(OBJ)/rgb2gray-image-CUDA-1D_int.o:$(SRC)/rgb2gray-image-CUDA-1D_int.cu
	$(NVCC) $(CFLAGS) -g -G -c $(SRC)/rgb2gray-image-CUDA-1D_int.cu -o $(OBJ)/rgb2gray-image-CUDA-1D_int.o

rgb2gray-image-CUDA-1D-memorycoelacing_int:$(GPUOBJS3)
	$(NVCC) $(LFLAGS) -g -G -o $(BIN)/rgb2gray-image-CUDA-1D-memorycoelacing_int $(GPUOBJS3)

$(OBJ)/rgb2gray-image-CUDA-1D-memorycoelacing_int.o:$(SRC)/rgb2gray-image-CUDA-1D-memorycoelacing_int.cu
	$(NVCC) $(CFLAGS) -g -G -c $(SRC)/rgb2gray-image-CUDA-1D-memorycoelacing_int.cu -o $(OBJ)/rgb2gray-image-CUDA-1D-memorycoelacing_int.o

test: $(OBJ)/test.o
	$(CXX) -g -o $(BIN)/tester $(OBJ)/test.o

$(OBJ)/test.o: $(SRC)/test.c
	$(CXX) -g -o $(OBJ)/test.o  -c $(SRC)/test.c

trial:$(OBJ)/trial.o
	$(CXX) $(LFLAGS) -o $(BIN)/trial $(OBJ)/trial.o 

$(OBJ)/trial.o: $(SRC)/trial.cpp
	$(CXX) $(CFLAGS) -g -o $(OBJ)/trial.o -c $(SRC)/trial.cpp 

clean:
	rm -f $(OBJ)/*.o
	rm -f bin/tester
	rm -f $(BIN)/rgb2gray-image-CUDA
	rm -f $(BIN)/trial
	rm -f $(BIN)/rgb2gray-image-CUDA_int
	rm -f $(BIN)/rgb2gray-image-CUDA-1D_int
	rm -f $(BIN)/rgb2gray-image-CUDA-1D-memorycoelacing_int
