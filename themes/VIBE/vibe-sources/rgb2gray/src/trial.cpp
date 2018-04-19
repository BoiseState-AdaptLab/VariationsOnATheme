#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<ctime>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
//#include "opencv2/gpu/device/vec_traits.hpp"
#include "utilities.h"
//#include "timer.cu"
//#include "cuda.h"
//#include "cuda_runtime.h"
//#include "cuda_runtime_api.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
  
  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);
 
  VideoCapture capture(cmdLineArgs.filename);
  
  if(capture.isOpened())
  {
   int start = time(NULL);
   while(true)
   {
    cout << "inside" << endl;
    Mat frame,ImageGRAY;
    capture >> frame;
    if(!frame.empty())
    { 
    cvtColor(frame,ImageGRAY,CV_BGR2GRAY);
    }
    else
    {
    cout << "FRame not found" << endl;
    int stop = time(NULL);
    int elapsed = stop-start;
    cout << "the time elapsed is: %d" << elapsed << endl;
    break;
    }
   // cvtColor(ImageGRAY,ImageGRAY,CV_RGB2GRAY);
   }
  }
  else
  {
   cout << " Cannot open Video" << endl;
  }
  return 0;
}
    
