#include "VIBE-FusedParallelBlur-CUDA-kernel.h"

#include "../common/Configuration.h"
#include "../common/Measurements.h"
#include "verification.h"
#include "opencv2/opencv.hpp"
#include "timer.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <time.h>

#define VIDEO
using namespace std;
using namespace cv;

int iter = 0 ;

#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255

int main(int argc,char* argv[]) 
{

  std::string filename;
  int N;
  int R;
  int time_sample;
  int match_count;
  unsigned long globalseed;
  bool verify;

// timer prep
  GpuTimer timer1; // init 
  GpuTimer timer2; // segment
  GpuTimer timer3; // update
  GpuTimer timer4; // memory transfer from CPU to GPU
  GpuTimer timer5;  // memory transfer from GPU to CPU
  GpuTimer timer6; // total
  GpuTimer timer7; // curand intialization
  GpuTimer timer8; // capture_time
  GpuTimer timer9;
  float time2;
  float time3;
  float time4;
  float time5;
  float time6;
  float time7;
  float time8;
  float time9 = 0.0;
  float initialize_time = 0.0;
  float segment_time = 0.0;
  float update_time = 0.0;
  float total_time = 0.0;
  float memcpy_GPU_TO_CPU = 0.0;
  float memcpy_CPU_TO_GPU = 0.0;
  float capture_time = 0.0;

  // parsing command line arguments
  int size = 3;
   
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
  if(filename == "NULL") {
    cerr << " unable to find file" << endl;
    return 0;
  }

  N = config.getInt("numberofSamples");
  R = config.getInt("radiusofSphere");
  time_sample = config.getInt("timeSampling");
  match_count = config.getInt("matchingSamples");
  verify = config.getBool("v");

  //checking for command line arguments
  if(N < 0) {
   fprintf(stderr,"The value of N has to be a positive number %d\n",N);
   return 0;
  }

  if(R < 0) {
   fprintf(stderr,"The value of R has to be a positive number %d\n",R);
    return 0;
  }

  if(time_sample < 0) {
   fprintf(stderr,"The value of time_sample has to be a positive number %d\n",
                   time_sample);
   return 0;
  }

  if(match_count < 0) {
   fprintf(stderr,"The value of match_count has to be a positive number %d\n",
                  match_count);
   return 0;
  }

  //2 Initializiing pointers
  pixel *h_image = NULL;
  pixel *h_start_image = NULL;

  unsigned char *h_segmap = NULL;
  unsigned char *blur_map = NULL;
  unsigned char* h_segmap_gndtruth = NULL;

  pixel *h_model = NULL;
  pixel *h_model_gndtruth = NULL;

  pixel *d_image = NULL;
  unsigned char *d_segmap = NULL;
  unsigned char *d_blur_map = NULL;
  pixel *d_model = NULL;
  pixel *d_start_image = NULL;
  curandState_t *d_state;

  // Parameters of the image
  int height;
  int width;
  int numofFrames;

  // Initializing variables
  int frameCount = 0;
  int map_count = 0;
  int verification_threshold;
                                                                                               
   int x_block,y_block;

  x_block = config.getInt("bx");
  y_block = config.getInt("by");

  // Getting frame info
  string& fp = filename;
  VideoCapture capture(fp);

//get frame info
#ifdef VIDEO
  numofFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
  width  = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
  height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);

  //Allocating memory to pixel
  size_t sizeModel = width*height*N;
  size_t sizeRGB = width*height;
  verification_threshold = 0.20*sizeRGB;

  h_model = new pixel[sizeModel];
  h_model_gndtruth = new pixel[sizeModel];

  h_image = new pixel[sizeRGB];
  h_start_image = h_image;

  //Allocating memory to device memory
  gpuErrchk(cudaMalloc((void**)&d_image,sizeRGB*sizeof(pixel)));
  gpuErrchk(cudaMalloc((void**)&d_model,sizeModel*sizeof(pixel)));
  gpuErrchk(cudaMalloc((void**)&d_state,sizeof(curandState_t)*sizeRGB));

  //allocating block and grid sizes
  const dim3 blockSize(x_block,y_block,1);
  const dim3 gridSize(ceil(width/x_block),ceil(height/y_block),1);
#endif


  // Reading all the frames and initializing the model
  // to create different random numbers every time

  globalseed = config.getInt("global_seed");
  srand(globalseed);  // host-side random numbers


  // Reading all the frames and initializing the model 
  Mat frame;
  capture >> frame;

  
  while(!frame.empty() && frameCount < 30){

    timer6.Start();
    #ifdef FRAME
    height = frame.rows;
    width = frame.cols;

    const dim3 blockSize(x_block,y_block,1);
    const dim3 gridSize(ceil(width/x_block),ceil(height/y_block),1);
  
    size_t sizeModel = height*width*N;
    size_t sizeRGB = height*width;
    verification_threshold = 0.20*sizeRGB;
    // allocate memory for segmented map for each frame
    h_image = new pixel[height*width];

    gpuErrchk(cudaMalloc((void**)&d_image,sizeRGB*sizeof(pixel)));
    gpuErrchk(cudaMalloc((void**)&d_model,sizeModel*sizeof(pixel)));
    #endif

    if(x_block > width || y_block > height )
    {
      cout << "The block size is inappropriate for image size" << endl;
      return 0;
    }

    // initialize random number generator
    timer7.Start();
    if(frameCount == 0) {
      gpuErrchk(cudaMalloc((void**)&d_state,sizeof(curandState_t)*sizeRGB));
      cuda_init_rand_wrapper(globalseed,d_state,width,gridSize,blockSize);
      //gpuErrchk(cudaDeviceSynchronize());
    }
    timer7.Stop();
    time7=timer7.Elapsed();

    h_segmap = new unsigned char[sizeRGB];
    blur_map = new unsigned char[sizeRGB];
    h_segmap_gndtruth = new unsigned char[sizeRGB];

    //allocate memory to device 
    gpuErrchk(cudaMalloc((void**)&d_segmap,sizeRGB*sizeof(unsigned char)));
    gpuErrchk(cudaMalloc((void**)&d_blur_map,sizeRGB*sizeof(unsigned char)));

    //copy memory from host to device
    timer4.Start();
    gpuErrchk(cudaMemcpy(d_segmap,h_segmap,sizeRGB*sizeof(unsigned char),
              cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_blur_map,blur_map,sizeRGB*sizeof(unsigned char),
              cudaMemcpyHostToDevice));
    timer4.Stop();
    time4 = timer4.Elapsed()/1000;
    memcpy_GPU_TO_CPU += time4;

    // point image to start of space allocated    
    #ifdef VIDEO
    h_image = h_start_image;
    #endif


    pixel* frame_pixel = (pixel*)frame.data;
    memcpy(h_image,frame_pixel,height*width*sizeof(pixel));

    // copy memory of image from host to device
    timer4.Start();
    gpuErrchk(cudaMemcpy(d_image,h_image,sizeRGB*sizeof(pixel),
              cudaMemcpyHostToDevice));
    timer4.Stop();
    time4 = timer4.Elapsed()/1000;
    memcpy_GPU_TO_CPU += time4;

    // initialize pixel model and time 

    if(frameCount ==0){

      timer1.Start();                                                                                                    
                                                                                                       

    #ifdef FRAME
      h_model = new pixel[sizeModel];
      h_model_gndtruth = new pixel[sizeModel];
      #endif
      // initial model for gpu version

      initialize_model(h_image,h_model,height,width,N);

      timer1.Stop();
      initialize_time = timer1.Elapsed()/1000;

     /* if(verify){
        // initialize for the serial verion-groundtruth
        initialize_model(h_image,h_model_gndtruth,height,width,N);
      }*/

    

      timer4.Start();
      gpuErrchk(cudaMemcpy(d_model,h_model,sizeModel*sizeof(pixel),
              cudaMemcpyHostToDevice));
      timer4.Stop();
      time4 = timer4.Elapsed()/1000;
      memcpy_GPU_TO_CPU += time4;
    }
    // segment the frame and time
    timer2.Start();
    cuda_segment_update_model(d_segmap,d_image,
                              d_model,time_sample,
                              height,width,N,R,
                              match_count,d_state,
                              gridSize,blockSize);
    timer2.Stop();
    time2 = timer2.Elapsed()/1000;
    segment_time += time2;


    size_t memsize = (x_block+2)*(y_block+2)*sizeof(unsigned char);
    timer9.Start();
    cuda_median_blur(d_segmap,d_blur_map,height,width,size,memsize,gridSize,blockSize);
    timer9.Stop();
    time9 += timer9.Elapsed()/1000;
    gpuErrchk(cudaDeviceSynchronize()); 
    //frame segmentation for serial version-groundtruth
    /*if(verify){
      segment_frame(h_segmap_gndtruth,h_image,
                    h_model_gndtruth,height,
                    width,N,R,match_count);
     }*/

    timer5.Start();                                                                                                             
    //gpuErrchk(cudaMemcpy(h_model,d_model,sizeModel*sizeof(pixel),
      //        cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_segmap,d_segmap,sizeRGB*sizeof(unsigned char),
              cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(blur_map,d_blur_map,sizeRGB*sizeof(unsigned char),
              cudaMemcpyDeviceToHost));
    timer5.Stop();
    time5 = timer5.Elapsed()/1000;
    memcpy_CPU_TO_GPU += time5;

    //median blur the map as post processing
    Mat output_map = Mat(height,width,CV_8UC1);
    output_map.data = blur_map;
  
    if(frameCount == 19)
    {
      string output = config.getString("output");
      imwrite(output,output_map);
    }
    // Free device segmentation map memory
    cudaFree(d_segmap);
    cudaFree(d_blur_map);
    frameCount++;
   /* if(verify){
     
      int diff_count = collect_verification_statistics(h_segmap,
                                                       h_segmap_gndtruth,
                                                       height,width); 

      if(diff_count < verification_threshold)
      {
         map_count  +=1;
        
      }
    }*/

    delete[] h_segmap;
    delete[] blur_map;
    #ifdef FRAME
    delete[] h_image;
    #endif
    timer8.Start();
    capture >> frame;
    timer8.Stop();
    time8 = timer8.Elapsed()/1000;
    capture_time += time8;

    timer6.Stop();
    time6 = timer6.Elapsed()/1000;
    total_time += time6;
                                                                                                                     
  }                                                                                                                              

  //total time for all frames
  cudaFree(d_state);
  cudaFree(d_image);
  cudaFree(d_model);

  //delete the allocated arrays
#ifdef VIDEO
  delete[] h_start_image;
#endif
  delete[] h_model;

  //release the cpture instance
  capture.release();

  //print time measurements
  Measurements measure;
  measure.setField("Memory transfer from CPU to Gpu",memcpy_GPU_TO_CPU);
  measure.setField("Memory transfer from GPU to CPU",memcpy_CPU_TO_GPU); 
  measure.setField("Initialize_model",initialize_time);
  measure.setField("Segment_update_frames",segment_time);
  measure.setField("Median Blur",time9);
  measure.setField("Curand_Initialization",time7);
  measure.setField("Total_time",total_time);
  measure.setField("capture time",capture_time);
  cout << measure.toLDAPString() << endl;


  //print the string configuration
  string lda = config.toLDAPString();
  cout << lda << endl;

  return 0;
}

