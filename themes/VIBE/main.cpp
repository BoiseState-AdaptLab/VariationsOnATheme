#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>

#include "opencv2/opencv.hpp"

#include<stdlib.h>
#include<cstdlib>
#include<time.h>

#include "util.h"
#include "VIBE-NaiveParallel-CUDA.h"
#include "timer.cu"
#include "../common/Configuration.h"

#include "cuda_runtime.h"


using namespace std;
using namespace cv;

int iter = 0 ;

#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255

#define image(n,i,j,k) image[(n*(width*height) + i*(width) + j)*3 + k]
#define segmap(i,j)  segmap[(i*width) + j]

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
   
int main(int argc,char* argv[])
{   


  std::string filename;
  int N;
  int R;
  int time_sample;
  int match_count;

//timer to calculate the initialization of the model
  GpuTimer timer1;

//timer to calculate the time taken to segment 
//the frames
  GpuTimer timer2;

//timer taken to calculate the time taken to update
// the model
  GpuTimer timer3;

//timer taken to calculate the total time
  GpuTimer timer4;
  
  float time1 = 0;
  float time2 = 0;
  float time3 = 0;
  float time4 = 0;

   // Constructor for parsing command line arguments
    Configuration config;
  
   /**************************************************************************
    ** Parameter options. Help and verify are constructor defaults.          *
    **************************************************************************/ 

    config.addParamString("filename", 'f', argv[1], "--The filename of the video");
    // changed the 'o' to 'n' (conflict with "output")
    config.addParamInt("numberofSamples", 's' , 20 , "--The number of samples in model for each pixel");
    config.addParamInt("radiusofSphere", 'r' , 16 , "--The radius of euclidean sphere");
    config.addParamInt("timeSampling",'i', 16 , "--Sampling in time factor");
    config.addParamInt("matchingSamples", 'c' , 2 , "--Number of samples within the radius");
    config.addParamString("output", 'o',"NULL", " --The filename of the result");
    // added the following addParam's from vibe_naive: common/Configuration.cpp 
    config.addParamInt("Nx",'k',100,"--Nx <problem-size> for x-axis in elements (N)");
    config.addParamInt("Ny",'l',100,"--Ny <problem-size> for y-axis in elements (N)");
    config.addParamInt("Nz",'m',100,"--Nz <problem-size> for z-axis in elements (N)");
    config.addParamInt("bx",'x',1,"--bx <block-size> for x-axis in elements");
    config.addParamInt("by",'y',1,"--by <block-size> for y-axis in elements");
    config.addParamInt("bz",'z',1,"--bz <block-size> for z-axis in elements");
    config.addParamInt("tilex",'a',128,"--tilex <tile-size> for x-axis in elements");
    config.addParamInt("tiley",'b',128,"--tiley <tile-size> for y-axis in elements");
    // changed c to q (conflict with "matchingSamples" 
    config.addParamInt("tilez",'q',128,"--tilez <tile-size> for z-axis in elements");
    config.addParamInt("T",'T',100,"-T <time-steps>, the number of time steps");
    // changed h to q (conflict with "help")
    config.addParamInt("height",'j',-1,"height <#>, number of time steps in tile");
    config.addParamInt("tau",'t',30,"--tau <tau>, distance between tiling"
                                "hyperplanes (all diamond(slab))");
    config.addParamInt("num_threads",'p',1,"-p <num_threads>, number of cores");
    config.addParamInt("global_seed", 'g', 1524, "--global_seed " 
                                           "<global_seed>, seed for rng"); 
    config.addParamBool("n",'n', false, "-n do not print time");

    config.parse(argc, argv);

  filename = config.getString("filename");
 
  if(filename == "NULL")
  {
    cout << " unable to find file" << endl;
    exit(-1);
  }
  
  N = config.getInt("numberofSamples");
  R = config.getInt("radiusofSphere");
  time_sample = config.getInt("timeSampling");
  match_count = config.getInt("matchingSamples");

//checking for command line arguments
  if(N < 0)
 {
   fprintf(stderr,"The value of N has to be a positive number %d\n",N);
   exit(-1);
 }

 if(R < 0)
 {
   fprintf(stderr,"The value of R has to be a positive number %d\n",R);
    exit(-1);
 } 

  if(time_sample < 0)
  {
   fprintf(stderr,"The value of time_sample has to be a positive number %d\n",time_sample);
   exit(-1);
  }

  if(match_count < 0)
  {
   fprintf(stderr,"The value of match_count has to be a positive number %d\n",match_count);
   exit(-1);
  }

//2 Initializiing pointers

  pixel *h_image = NULL;
  pixel *h_start_image = NULL;
  unsigned char *h_segmap = NULL;
  pixel *h_model = NULL;

  pixel *d_image = NULL;
  unsigned char *d_segmap = NULL;
  pixel *d_model = NULL;
  pixel *d_start_image = NULL;

// Parameters of the image
 
  int height;
  int width;
  int numofFrames;

// Initializing variables

  int frameCount = 0;
  int x_block,y_block; 

  x_block = config.getInt("bx");
  y_block = config.getInt("by"); 

// Getting frame info
  string& fp = filename;
  VideoCapture capture(fp);

//get frame info
  numofFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
  width  = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
  height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
  cout << height << endl;
  cout << width << endl;
 
  
//Allocating memory to pixel
  size_t sizeModel = width*height*N;
  size_t sizeRGB = width*height;

  h_model = new pixel[sizeModel];
  h_image = new pixel[sizeRGB];
  h_start_image = h_image;


  const dim3 blockSize(x_block,y_block,1);
  const dim3 gridSize(1,1,1);
  //size_t dyn_mem_size = x_block*y_block*sizeof(pixel);

//Allocating memory to device memory
  gpuErrchk(cudaMalloc((void**)&d_image,sizeRGB*sizeof(pixel)));
  gpuErrchk(cudaMalloc((void**)&d_model,sizeModel*sizeof(pixel)));

//copying memory to device for model  
  gpuErrchk(cudaMemcpy(d_model,h_model,sizeModel*sizeof(pixel),cudaMemcpyHostToDevice));

// to create different random numbers every time
  srand(time(NULL));
   
// Reading all the frames and initializing the model 
  Mat frame;
  capture >> frame;

  timer4.Start();
  while(!frame.empty())
  {
 
// allocate memory for segmented map for each frame
    h_segmap = new unsigned char[width*height];

//allocate memory to device 
    gpuErrchk(cudaMalloc((void**)&d_segmap,sizeRGB*sizeof(unsigned char))); 

//copy memory from host to device
    gpuErrchk(cudaMemcpy(d_segmap,h_segmap,sizeRGB*sizeof(unsigned char),cudaMemcpyHostToDevice));

// point image to start of space allocated    
    h_image = h_start_image;

// reading in data into image from capture instance frame
    for(int j=0;j < height*width;j++)
    {
       pixel pm ;
       pm.b = frame.data[3*j];
       pm.g = frame.data[3*j+1];
       pm.r = frame.data[3*j+2];
       h_image[j] = pm;
       
    }
// copy memory of image from host to device

    gpuErrchk(cudaMemcpy(d_image,h_image,sizeRGB*sizeof(pixel),cudaMemcpyHostToDevice));
    
// initialize pixel model and time 
    
    if(frameCount ==0){ 
     
      timer1.Start();
      initialize_model(h_image,h_model,height,width,N); 
      timer1.Stop();
      time1 = time1 = timer1.Elapsed(); 
    
    }

 
// segment the frame and time
   
   // timer2.Start();
    segment_frames<<<gridSize, blockSize>>>(d_segmap,d_image,
                                            d_model,height,
                                            width,N,R,
                                            match_count);

    timer2.Stop();
    time2 = time2 + timer2.Elapsed();
    
// update the model and its neighbour and time
  /*  if(frameCount > 0)
    {
      timer3.Start();      
      update_model(segmap,image,model,height,width,
                 time_sample,N,frameCount);
      timer3.Stop();
      time3  = time3 + timer3.Elapsed();    
    }*/

    gpuErrchk(cudaMemcpy(h_model,d_model,sizeModel*sizeof(pixel),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_segmap,d_segmap,sizeRGB*sizeof(unsigned char),cudaMemcpyDeviceToHost));

    cudaFree(d_segmap);
   
    frameCount++;
    cout <<"The frameCount is:" << frameCount <<  endl;
    
    delete[] h_segmap;
    
    capture >> frame;
   
  }
  timer4.Stop();
  time4 = time4 + timer4.Elapsed();
 
//total time for all frames


  cudaFree(d_image);
  cudaFree(d_model);

//delete the allocated arrays
  delete[] h_start_image;
  delete[] h_model;

//release the cpture instance
  capture.release();

//print the string configuration
  string lda = config.toLDAPString();
  cout << lda << endl;

  return 0;
}

