#include"opencv2/opencv.hpp"
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<vector>
#include<assert.h>
#include<fstream>

//#define _DEBUG

using namespace std;
using namespace cv;

int main(int argc,char*argv[])
{
 
  Mat hist_diff;
  string filename_1 = argv[1];
  string filename_2 = argv[2];
  Mat pixel_map_1 = imread(filename_1,-1);
  Mat pixel_map_2 = imread(filename_2,-1);
  Mat diff_map = Mat::zeros(pixel_map_1.rows,pixel_map_1.cols,CV_8UC1);
  //cout << pixel_map_1.rows << pixel_map_1.cols << endl;
  
  #ifdef _DEBUG
  cout << "Diff Map before the absdiff" << endl;
  namedWindow("Before", WINDOW_AUTOSIZE);
  imshow("Before", diff_map);
  waitKey(0);
  #endif

  absdiff(pixel_map_1,pixel_map_2,diff_map);
  
  #ifdef _DEBUG
  cout << "Diff Map after the absdiff" << endl;
  namedWindow("After", WINDOW_AUTOSIZE);
  imshow("After", diff_map);
  waitKey(0);
  #endif

  //absdiff(frame_1,frame_2,diff_map);
  vector<int> hist;
  double scale_factor  = ceil(255.0/32.00);
  for(int l = 0; l< 33;l++)
  {
    int tmp = 0;
    int count = 0;
   for(int h=0;h< 480;h++)
   {
    for(int k = 0;k < 640;k++)
    {
       if(diff_map.at<uchar>(h,k) == l)
        count += 1;
       //else cout << "debug=" << diff_map.at<uchar>(h,k) << endl;
       tmp ++;
    }
   }
    assert(tmp == 480*640);
    hist.push_back(count);
  }

  ofstream fout("histogram.out", fstream::app);   

  for(int k = 0;k < 33;k++)
  {
    fout << hist[k] << endl;
  }
   for(int i = 0;i < pixel_map_1.rows;i++)
  {
   for(int j = 0;j < pixel_map_2.cols;j++)
   {
     //if((int)diff_map.at<uchar>(i,j) == 32){
      //cout << i << j << endl;}
   }  
  }


  for(int i = 0;i < pixel_map_1.rows;i++)
  {
    for(int j = 0; j < pixel_map_2.cols;j++)
    {
    
      uchar pixel;
      pixel = diff_map.at<uchar>(i,j);
      // if the pixels from 2 counter maps had identical values (i.e. diff=0), then 
      // the pixel at the diff map will be white. Else it will be some shade of gray / black 
      if(pixel == 0)
      {
        pixel = 255;
      }
      else{
      pixel = (32-(uchar)pixel)*7;
      }
      diff_map.at<uchar>(i,j) = pixel;
      //pixel_map_1.at<uchar>(i,j) = pixel;
    }
  }

  /*string output; //= config.getString("output");
    if(h< 10)
    output = "diff_map/diff_00" + to_string(h) + ".png";
    else if(h >= 10 && h< 100)
    output = "diff_map/diff_0" + to_string(h) + ".png";
    else
    output = "diff_map/diff_" + to_string(h) + ".png";*/

  imwrite("diff_map_" + filename_2,diff_map);
  //cout << count << endl;

  return 0;

}      
    

