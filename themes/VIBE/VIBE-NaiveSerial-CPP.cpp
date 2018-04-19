#include<iostream>
#include<stdio.h>
#include<string>
#include<string.h> // for memset
#include<cmath>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include "opencv2/opencv.hpp"   
#include "verification.h"
#include "../common/Configuration.h"
#include "../common/Measurements.h"
#include "util.h"
#include<stack>
//#include "cuda.h"
//#include "cuda_runtime_api.h"
#include<assert.h>

using namespace cv;
using namespace std;

#define VIDEO
int iter = 0 ;
#define COLOR_BACKGROUND 255
#define COLOR_FOREGROUND 0
//#define MAIN // uncomment the MAIN if you want to write out only frame 350.
//#define dimensions 

const int WHITE = 255;
const int BLACK = 0;
const int RADIUS = 0; 

inline int max(int a, int b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }

struct Data {
    int maxx;
    int maxy;
    int minx;
    int miny;
};

typedef struct Data Data;

Data extremes(Data p1, Data p2, Data p3, Data p4) {
    // pairwise comparison
    int max_x_12 = max(p1.maxx, p2.maxx);
    int max_y_12 = max(p1.maxy, p2.maxy);
    int max_x_34 = max(p3.maxx, p4.maxx);
    int max_y_34 = max(p3.maxy, p4.maxy);
    // find the largest between two pairs
    int max_x = max(max_x_12, max_x_34);
    int max_y = max(max_y_12, max_y_34);
   
    // pairwise comparison
    int min_x_12 = min(p1.minx, p2.minx);
    int min_y_12 = min(p1.miny, p2.miny);
    int min_x_34 = min(p3.minx, p4.minx);
    int min_y_34 = min(p3.miny, p4.miny);
    // find the largest between two pairs
    int min_x = min(min_x_12, min_x_34);
    int min_y = min(min_y_12, min_y_34);

    return (Data){max_x, max_y, min_x, min_y};
}

Data dfs(unsigned char* image, bool* visited, int x, int y, int width, int height) {
  stack<Point> stack;
  int xs[8] = {0, 0, 1, -1, 1, 1, -1, -1};
  int ys[8] = {1, -1, 0, 0, 1, -1, -1, 1};
  int maxx = 0, minx = 1234567, miny = 1234567, maxy = 0;
  stack.push(Point(x, y));
  while (!stack.empty()) {
    Point top = stack.top();
    stack.pop();
    if (top.x >= width || top.y >= height || top.x < 0 || top.y < 0) continue;
    if (*((visited + top.y*width) + top.x)) continue; 
    if (image[top.y*width + top.x] == BLACK) {
      bool done = false;
      int rxs[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      int rys[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
      for (int cnt = 1; cnt <= RADIUS; ++cnt) {
        rxs[2]++; rxs[3]--; rxs[4]++; rxs[5]++; rxs[6]--; rxs[7]--;
        rys[0]++; rys[1]--; rys[4]++; rys[5]--; rys[6]--; rys[7]++;
        for (int i = 0; i < 8; ++i) {  
          done |= image[(top.y + rys[i])*width + (top.x+rxs[i])] == WHITE;
          if (done) break;
        }
      }
      if (!done) continue;  
    }
    *((visited + top.y*width) + top.x) = true;
    // get the extremes
    maxx = max(maxx, top.x);
    maxy = max(maxy, top.y);
    minx = min(minx, top.x);
    miny = min(miny, top.y);
    // visit adjacents
    for (int i = 0; i < 8; ++i) {
      stack.push(Point(top.x + xs[i], top.y + ys[i]));
    }
  }
  return (Data) { maxx, maxy, minx, miny };
}


int main(int argc,char* argv[])
{   

  string filename ;
  int N;
  int R;
  int time_sample;
  int match_count;
  int globalseed;

  struct timeval t1_start,t1_end,t2_start,t2_end,t3_start,t3_end,t4_start,t4_end;
  struct timeval start_time,end_time,t5_start,t5_end;
  float capture_time = 0.0;
  float initialize_time = 0.0;
  float segment_time =0.0;
  float update_time = 0.0;
  float memory_transfer_time=0.0;
  float total_time = 0.0;
  float median_blur =0.0;

  // Constructor for parsing command line arguments
  Configuration config;
  
  /**************************************************************************
   ** Parameter options. Help and verify are constructor defaults.          *
   **************************************************************************/ 
  config.addParamString("filename", 'f', "NULL", "--The filename of the video");
  config.addParamInt("numberofSamples", 's' , 20 , "--The number of samples in model for each pixel");
  config.addParamInt("radiusofSphere", 'r' , 20 , "--The radius of euclidean sphere");
  config.addParamInt("timeSampling",'i', 16 , "--Sampling in time factor");
  config.addParamInt("matchingSamples", 'c' , 2 , "--Number of samples within the radius");
  config.addParamString("output", 'o',"NULL", " --The filename of the result");
  config.addParamInt("global_seed", 'g', 1524, "--global_seed " 
                                         "<global_seed>, seed for rng"); 
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
  globalseed = config.getInt("global_seed");
  assert(globalseed != 1524);

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
  int count = 0;
// Initializing variables
  int frameCount = 0;
  long long segcount = 0;
// Getting frame info
  string& fp = filename;
  VideoCapture capture(fp);

//get frame info

  #ifdef VIDEO
  numofFrames = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
  width  = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
  height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    #ifdef dimensions
      cout << "Frame height = " << height << endl;
      cout << "Frame width = " << width << endl;
    #endif
  #endif
  
//Allocating memory to pixel

 #ifdef VIDEO
  model = new pixel[width*height*N];
  image = new pixel[width*height];
  start_image = image;
 #endif
// to create different random numbers every time
  assert(globalseed != 1524);
  srand(globalseed);
// Reading all the frames and initializing the model 
  Mat frame;
  capture >> frame;
  
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
 
// reading in data into image from capture instance frame
   
    pixel* frame_pixel = (pixel*)frame.data;
    memcpy(image,frame_pixel,height*width*sizeof(pixel));
   
// initialize pixel model and time 
    gettimeofday(&start_time,NULL); 
    if(frameCount ==0){ 
     #ifdef FRAME
      model = new pixel[width*height*N];
     #endif
     // initialize_timer();
      gettimeofday(&t1_start,NULL);
      // doen't modify image: OK
      initialize_model(image,model,height,width,N); 
      gettimeofday(&t1_end,NULL);  
      initialize_time += ((t1_end.tv_sec  - t1_start.tv_sec) * 1000000u + 
         t1_end.tv_usec - t1_start.tv_usec) / 1.e6;
    }

   // segment the frame and time
   
    gettimeofday(&t2_start,NULL);

    // modifies segmap, doesn't modify image: OK
    segment_frame(segmap,image,model,height,width,N,R,match_count,frameCount);
    gettimeofday(&t2_end,NULL);
    segment_time += ((t2_end.tv_sec  - t2_start.tv_sec) * 1000000u + 
         t2_end.tv_usec - t2_start.tv_usec) / 1.e6;


    /* Do DFS to find all foreground objects */
    bool visited[height][width];
    memset(visited, false, sizeof(visited));

    vector<Rect> forest; 
     
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (segmap[y*width + x] == WHITE) {
                if (!visited[y][x]) {
                    Data bounds = dfs(segmap, (bool*) visited, x, y, width, height);
                    Rect rect(Point(bounds.minx, bounds.miny), Point(bounds.maxx, bounds.maxy));
                    forest.push_back(rect);                    
                }
            }
        }
    }

   // update the model and its neighbour and time

   // initialize_timer();
    gettimeofday(&t3_start,NULL);
    
    // update_model doesn't modify segmap: OK
    segcount = update_model(segmap,image,model,height,width,
                 time_sample,N,frameCount,segcount);
    gettimeofday(&t3_end,NULL);
    update_time += ((t3_end.tv_sec  - t3_start.tv_sec) * 1000000u +
         t3_end.tv_usec - t3_start.tv_usec) / 1.e6;

   //  update_time += elapsed_time();
     
   // Mat output_frame = Mat(frame.rows,frame.cols,CV_8UC1);
    Mat output_frame = frame; // old fram
    // modify draw bounding box on the frame
    for (int i = 0; i < forest.size(); ++i) {
        //if (i == 5) {
            Rect rect = forest[i];
            Data bounds = { rect.x + rect.width, rect.y + rect.height, rect.x, rect.y }; 
            rectangle(output_frame, rect, Scalar(0, 0, 255), 2);
        //}
    }
    // cout << "Draw the ractangle. See it." << endl;
    //output_frame.data = segmap;

    gettimeofday(&t5_start,NULL);
    medianBlur(output_frame,output_frame,3);
    gettimeofday(&t5_end,NULL);
    median_blur += ((t5_end.tv_sec  - t5_start.tv_sec) * 1000000u +
    t5_end.tv_usec - t5_start.tv_usec) / 1.e6;
   
    #ifdef MAIN
    if (frameCount == 350) {
    #endif
      string output = config.getString("output");
      output += "_" + to_string(frameCount) + ".png";
      imwrite(output,output_frame);
    #ifdef MAIN
    }
    #endif

    frameCount++; 
    delete[] segmap;
    #ifdef FRAME
    delete[] image;
    #endif

    gettimeofday(&t4_start,NULL);
    capture >> frame;
    gettimeofday(&t4_end,NULL);
    capture_time += ((t4_end.tv_sec  - t4_start.tv_sec) * 1000000u +
    t4_end.tv_usec - t4_start.tv_usec) / 1.e6;
 
    gettimeofday(&end_time,NULL);
    total_time += ((end_time.tv_sec  - start_time.tv_sec) * 1000000u +
    end_time.tv_usec - start_time.tv_usec) / 1.e6;
  }
  
//delete the allocated arrays
 #ifdef VIDEO
  delete[] start_image;
 #endif
  delete[] model;

//release the cpture instance
  capture.release();

//print the time information
  Measurements measure;
  measure.setField("Initialize_model",initialize_time);
  measure.setField("Segment_frames",segment_time);
  measure.setField("Update_frames",update_time);
  measure.setField("Total_time",total_time);
  measure.setField("capture_time",capture_time);
  measure.setField("median_blur",median_blur);
  measure.getFieldFloat("Initialize_model");
  measure.getFieldFloat("Segment_frames");
  measure.getFieldFloat("Update_frames");
  measure.getFieldFloat("Total_time");
 // measure.getFieldFloat("capture_time");
  string result = measure.toLDAPString();
  cout << result << endl;
  cout << segcount << endl;
//print the string configuration
  string lda = config.toLDAPString();
  cout << lda << endl;

  return 0;
}

