/**
 * @file main-opencv.cpp
 * @date July 2014 
 * @brief An exemplative main file for the use of ViBe and OpenCV
 */
#include <iostream>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "vibe-background-sequential.h"
#include "../../common/Configuration.h"
#include "../../common/Measurements.h"
#include <time.h>
#include<assert.h>

#define dimensions

using namespace cv;
using namespace std;

/** Function Headers */
void processVideo(string videoFilename,string output,int globalseed);

/**
 * Displays instructions on how to use this program.
 */

void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use ViBe with OpenCV                            " << endl
    << "Usage:"                                                                     << endl
    << "./main-opencv -f <video filename> -g <seed> -o <output name>"               << endl
    << "for example: ./main-opencv video.avi"                                       << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}

/**
 * Main program. It shows how to use the grayscale version (C1R) and the RGB version (C3R). 
 */
int main(int argc, char* argv[])
{
  /* Print help information. */
  help();
  
  string filename;
  string output_name;
  int globalseed;

  Configuration config;

  config.addParamString("filename", 'f', "NULL", "--The filename of the video");
  config.addParamString("output", 'o',"NULL", " --The filename of the result");
  config.addParamInt("global_seed", 'g', 1524, "--global_seed "
                                         "<global_seed>, seed for rng");
  config.parse(argc,argv);
  filename = config.getString("filename");
  output_name = config.getString("output");
  globalseed = config.getInt("global_seed");
  /* Create GUI windows. */
  //namedWindow("Frame");
  //namedWindow("Segmentation by ViBe");
  assert(globalseed != 1524);
  processVideo(filename,output_name,globalseed);

  /* Destroy GUI windows. */
  destroyAllWindows();
  return EXIT_SUCCESS;
}

/**
 * Processes the video. The code of ViBe is included here. 
 *
 * @param videoFilename  The name of the input video file. 
 */
void processVideo(string videoFilename,string output,int globalseed)
{
  /* Create the capture object. */
  VideoCapture capture(videoFilename);

  if (!capture.isOpened()) {
    /* Error in opening the video input. */
    cerr << "Unable to open video file: " << videoFilename << endl;
    exit(EXIT_FAILURE);
  }

  /* Variables. */
  static int frameNumber = 1; /* The current frame number */
  Mat frame;                  /* Current frame. */
  Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
  int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */

  clock_t t1_start,t1_end,t2_start,t2_end,t3_start,t3_end;
  clock_t start_time,end_time;
  float initialize_time = 0.0;
  float segment_time = 0.0;
  float update_time = 0.0;
  float total = 0.0;
  assert(globalseed != 1524);
  srand(globalseed);
  /* Model for ViBe. */
  vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

  /* Read input data. ESC or 'q' for quitting. */
  while ((char)keyboard != 'q' && (char)keyboard != 27) {
    /* Read the current frame. */
    if (!capture.read(frame)) {
      cerr << "Unable to read next frame." << endl;
      cerr << "Exiting..." << endl;
           Measurements measure;
      initialize_time = 0;
      measure.setField("Initialize_model",initialize_time);
      measure.setField("Segment_frames",segment_time);
      measure.setField("Update_frames",update_time);
      measure.setField("Total_time",total);
      measure.getFieldFloat("Initialize_model");
      measure.getFieldFloat("Segment_frames");
      measure.getFieldFloat("Update_frames");
      measure.getFieldFloat("Total_time");
      string result = measure.toLDAPString();
      cout << result << endl;

      exit(EXIT_FAILURE);
    }

    /* Applying ViBe.
     * If you want to use the grayscale version of ViBe (which is much faster!):
     * (1) remplace C3R by C1R in this file.
     * (2) uncomment the next line (cvtColor).
     */
    /* cvtColor(frame, frame, CV_BGR2GRAY); */

    start_time = clock();
    if (frameNumber == 1) {
      segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
      model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
      t1_start = clock();
      libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows,frameNumber);
      t1_end = clock();
      float diff(float(t1_end) - float(t1_start));
      diff = diff/CLOCKS_PER_SEC;
      initialize_time = diff;
      #ifdef dimensions 
        cerr << "Height = " << frame.rows << endl;
        cerr << "Width = " << frame.cols << endl;
      #endif
    }

    /* ViBe: Segmentation and updating. */
    t2_start = clock();
    libvibeModel_Sequential_Segmentation_8u_C3R(model, frame.data, segmentationMap.data, frameNumber);
    t2_end = clock();
    float segment_diff(float(t2_end) - float(t2_start));
    segment_diff = segment_diff/CLOCKS_PER_SEC;
    segment_time += segment_diff;

    t3_start = clock();
    libvibeModel_Sequential_Update_8u_C3R(model, frame.data, segmentationMap.data);
    t3_end = clock();
    float update_diff(float(t3_end) - float(t3_start));
    update_diff = update_diff/CLOCKS_PER_SEC;
    update_time += update_diff;

    /* Post-processes the segmentation map. This step is not compulsory. 
       Note that we strongly recommend to use post-processing filters, as they 
       always smooth the segmentation map. For example, the post-processing filter 
       used for the Change Detection dataset (see http://www.changedetection.net/ ) 
       is a 5x5 median filter. */
   
    medianBlur(segmentationMap, segmentationMap, 3); /* 3x3 median filtering */
    
    // Aza's contribution: select only specific frame
    if(frameNumber == 350) {
        imwrite(output + "_" + to_string(frameNumber) + ".png", segmentationMap);
    }
    
    /* Shows the current frame and the segmentation map. */
    //imshow("Frame", frame);
    //imshow("Segmentation by ViBe", segmentationMap);

    ++frameNumber;
    
    end_time = clock();
    float total_time(float(end_time) - float(start_time));
    total_time = total_time/CLOCKS_PER_SEC;
    total += total_time;

    /* Gets the input from the keyboard. */
    keyboard = waitKey(1);
  }

  /* Delete capture object. */
  capture.release();

  /* Frees the model. */
  libvibeModel_Sequential_Free(model);
}
