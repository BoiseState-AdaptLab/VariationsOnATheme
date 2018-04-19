#include"opencv2/opencv.hpp"
#include<iostream>
#include "../common/Configuration.h"
#include<stdio.h>
#include<stdlib.h>
#include<string>
using namespace std;
using namespace cv;

int main(int argc,char*argv[])
{
  string filename;
  Mat frame;
  int frameCount=0;

    // Constructor for parsing command line arguments
   Configuration interface;
  
   /**************************************************************************
    ** Parameter options. Help and verify are constructor defaults.          *
    **************************************************************************/ 

    interface.addParamString("filename", 'f', argv[1], "--The filename of the video");
    // changed the 'o' to 'n' (conflict with "output")
    interface.addParamInt("numberofSamples", 's' , 20 , "--The number of samples in model for each pixel");
    interface.addParamInt("radiusofSphere", 'r' , 16 , "--The radius of euclidean sphere");
    interface.addParamInt("timeSampling",'i', 16 , "--Sampling in time factor");
    interface.addParamInt("matchingSamples", 'c' , 2 , "--Number of samples within the radius");
    interface.addParamString("output", 'o',"NULL", " --The filename of the result");
    // added the following addParam's from vibe_naive: common/Configuration.cpp 
    interface.addParamInt("Nx",'k',100,"--Nx <problem-size> for x-axis in elements (N)");
    interface.addParamInt("Ny",'l',100,"--Ny <problem-size> for y-axis in elements (N)");
    interface.addParamInt("Nz",'m',100,"--Nz <problem-size> for z-axis in elements (N)");
    interface.addParamInt("bx",'x',1,"--bx <block-size> for x-axis in elements");
    interface.addParamInt("by",'y',1,"--by <block-size> for y-axis in elements");
    interface.addParamInt("bz",'z',1,"--bz <block-size> for z-axis in elements");
    interface.addParamInt("tilex",'a',128,"--tilex <tile-size> for x-axis in elements");
    interface.addParamInt("tiley",'b',128,"--tiley <tile-size> for y-axis in elements");
    // changed c to q (conflict with "matchingSamples" 
    interface.addParamInt("tilez",'q',128,"--tilez <tile-size> for z-axis in elements");
    interface.addParamInt("T",'T',100,"-T <time-steps>, the number of time steps");
    // changed h to q (conflict with "help")
    interface.addParamInt("height",'j',-1,"height <#>, number of time steps in tile");
    interface.addParamInt("tau",'t',30,"--tau <tau>, distance between tiling"
                                "hyperplanes (all diamond(slab))");
    interface.addParamInt("num_threads",'p',1,"-p <num_threads>, number of cores");
    interface.addParamInt("global_seed", 'g', 1524, "--global_seed " 
                                           "<global_seed>, seed for rng"); 
    interface.addParamBool("n",'n', false, "-n do not print time");

    interface.parse(argc, argv);

  filename = interface.getString("filename");
  
  int width;
  int height;
  Mat fg_pixel_count = imread("pixel_count.png",-1);
 
  int count = 0;
  frame = imread(filename,-1);
  width = frame.cols;
  height =frame.rows;
  if(!frame.empty())  
  {
    
    for(int i = 0;i < height;i++)
    {
      for(int j = 0;j < width;j++)
      {   
         
         if(frame.at<uchar>(i,j) == 255) // white = 255
         {
           //cout << i <<" " <<  j << " " << (int)frame.at<uchar>(i,j) << endl;
           uchar pixel = fg_pixel_count.at<uchar>(i,j) ;
           pixel += 1; // black=0, so pixel+=1 will start from 0 base.
           fg_pixel_count.at<uchar>(i,j) = pixel;
         }
        
        }
     
      }
     imwrite("pixel_count.png",fg_pixel_count);
  }
  else
  {
    
     cout << "unable to load frame" << endl; 
  
  }
  
  return 0;
}

