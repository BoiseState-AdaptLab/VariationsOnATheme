#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<stdlib.h>
#include<time.h>

#include "opencv2/opencv.hpp"

#include "../common/Configuration.h"
#include "../common/Measurements.h"
#include "util.h"

#define VIDEO

using namespace std;
using namespace cv;

int iter = 0 ;
#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255


#define image(n,i,j,k) image[(n*(width*height) + i*(width) + j)*3 + k]
#define segmap(i,j)  segmap[(i*width) + j]
  
void segment_frames(unsigned char* map,unsigned char* frame,pixel* init_model,
                    int rows,int cols,int samples,int radius,
                    int match_threshold)
{
   bool match = false;
   int frame_index=0;
   int model_index=0;
   unsigned char* pix_pointer;
   
//  
   for(int k = 0;k < rows;k++)
   {
     for(int l = 0; l < cols; l++)
     {  
        
        //index of each pixel
        frame_index = 3*(k*cols + l);
        pix_pointer = &frame[frame_index];
       
        //index of model
        model_index = k*samples*cols + l*samples;
       
        // calculate the samples that lie within the radius threshold 
        match = Count_samples(pix_pointer,init_model,match_threshold,
                              samples,radius,model_index);   
        
        //creating the map by segmenting each pixel as foreground or 
        // background
         if(match)
         {
          map[k*cols + l] = COLOR_BACKGROUND;
          //cout <<  (int)map[120*cols + 160] << endl;        
         }
         else
         {
           map[k*cols + l] = COLOR_FOREGROUND;
         // cout << "not background" << endl;          
         }  
              
       }
     }
  
}                      


void update_model(unsigned char* segmentmap,
                  unsigned char* c_image,pixel* init_model,
                  int row,int col,int time_sample,
                  int numsamples,int frames)
{
  int map_index;
  int neigh_index;
  int x,y;
  int time_stamp;
 /// int sizeRGB = numsamples;
 
// check if each pixel is background and udpdate
// model of pixel and its neighbour if pixel gets 
// selected to get updated
  for(int a = 0;a < row;a++)
  {
   for(int b = 0;b < col; b++)
    { 
      map_index = a*col + b;
      
      if(segmentmap[map_index] == 0)
      { 
       // if pixel is chosen for update
        time_stamp = rand()%16;
       
        if(time_stamp==0)
        {
          
         /* if(a ==120 && b==160)
         {
          
          for(int n =0; n < numsamples; n++)
           {
            cout << (int)init_model[a*col*numsamples+b*numsamples + n].b << (int)init_model[a*col*numsamples+b*numsamples + n].g
            << (int)init_model[a*col*numsamples+b*numsamples + n].r << endl;
           }
         }*/

          //update the pixel model    
          Updatepixelmodel(c_image,init_model,map_index,numsamples);
           
          /*if(a == 120 && b ==160)
          {
           //iter = iter +1 ;
          for(int n =0; n < numsamples; n++)
           {          
            cout << (int)init_model[a*col*numsamples+b*numsamples + n].b << (int)init_model[a*col*numsamples+b*numsamples + n].g
            << (int)init_model[a*col*numsamples+b*numsamples + n].r << endl;
           }
          }*/

          // choose neighbouring pixel to get updated
          x =  rand()%3-1;                
          y = rand()%3-1;
          neigh_index = (x+a)*col + (y+b);

               
          //update the pixel of neoghbouring pixel
          Updatepixelmodel(c_image,init_model,neigh_index,numsamples);
         }
       }
     }
   }
}

int main(int argc,char* argv[])
{   

  
  string filename;
  int N;
  int R;
  int time_sample;
  int match_count;


  clock_t t1_start,t1_end,t2_start,t2_end,t3_start,t3_end;
  clock_t start_time,end_time;
  float initialize_time,segment_time,update_time,memory_transfer_time;
 
//Constructor for parsing command line arguments
  
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

  pixel* image = NULL;
  pixel* start_image = NULL;
  unsigned char* segmap = NULL;
  pixel* model = NULL;

// Parameters of the image
 
  int height;
  int width;
  int numofFrames;

// Initializing variables

  int frameCount = 0;
  
// Getting frame info
  string& fp = filename;
  VideoCapture capture(fp);

//get frame info
 #ifdef VIDEO
  numofFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
  width  = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
  height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
 #endif
 
//Allocating memory to pixels
 #ifdef VIDEO
  model = new pixel[width*height*N];
  image = new pixel[width*height];
  start_image = image;
 #endif
// to create different random numbers every time
  srand(time(NULL));
   
// Reading all the frames and initializing the model 
  Mat frame;
  capture >> frame;

  start_time = clock();
  while(!frame.empty())
  {

    #ifdef FRAME
     height = frame.rows;
     width = frame.cols;
    #endif

 
// allocate memory for segmented map for each frame
    #ifdef FRAME
      image = new pixel[width*height];
    #endif

    segmap = new unsigned char[width*height];
  

// point image to start of space allocated    
   #ifdef VIDEO
    image = start_image;
   #endif

// initialize pixel model and time 
    
    if(frameCount ==0)
   {
      #ifdef FRAME
       model = new pixel[width*height*N];
      #endif 
      t1_start = clock();
      Initialize_model(frame.data,model,height,width,N); 
      t1_end = clock();
      float diff(float(t1_end) - float(t1_start));
      diff = diff/CLOCKS_PER_SEC;
      initialize_time += diff; 
      
    }
  
// segment the frame and time

    t2_start = clock();
    segment_frames(segmap,frame.data,model,height,width,N,R,match_count);
    t2_end = clock();
    float diff(float(t2_end) - float(t2_start));
    diff = diff/CLOCKS_PER_SEC;
    segment_time += diff;
    

// update the model and its neighbour and time
    if(frameCount > 0)
    {
      t3_start = clock();
      update_model(segmap,frame.data,model,height,width,
                 time_sample,N,frameCount);
      t3_end = clock();
      float diff(float(t3_end) -float(t3_start));
      diff = diff/CLOCKS_PER_SEC;
      update_time += diff;
      
    }
    
    frameCount++;
    
    delete[] segmap;
    #ifdef FRAME
    delete[] image;
    #endif
    capture >> frame;
   
  }
  end_time = clock();
  
//total time for all frames 
  
  float total_time(float(end_time) - float(start_time));
  total_time = total_time/CLOCKS_PER_SEC;

//delete the allocated arrays
  #ifdef VIDEO
   delete[] start_image;
  #endif

  delete[] model;

//release the cpture instance
  capture.release();

//print time measurements
  Measurements measure;
  measure.setField("Initialize_model",initialize_time);
  measure.setField("Segment_frames",segment_time);
  measure.setField("Update_frames",update_time);
  measure.setField("Total_time",total_time);
  measure.setField("Memory_transfer_nocopy",memory_transfer_time);
  
  measure.getFieldFloat("Initialize_model");
  measure.getFieldFloat("Segment_frames");
  measure.getFieldFloat("Update_frames");
  measure.getFieldFloat("Total_time");
  measure.getFieldFloat("Measure_transfer_nocopy");
  string result = measure.toLDAPString();
  cout << result << endl;

//print the string configuration
  string lda = config.toLDAPString();
  cout << lda << endl;

  return 0;
}
