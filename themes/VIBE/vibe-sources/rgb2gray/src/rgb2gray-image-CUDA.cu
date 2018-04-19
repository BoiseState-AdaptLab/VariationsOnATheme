#include "opencv2/opencv.hpp"
#include "utilities.h"
#include "timer.cu"
#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<bitset>

using namespace std;
using namespace cv;

int frameCount = 1;
float time1 = 0.0;
float time2 = 0.0;
float time3 = 0.0;
float time4 = 0.0;
float time5 = 0.0;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }

}
/******************************************************************************
* void RGBtoGRAY(const uchar* const InputRGB,
                                unsigned char* const OutputGRAY, 
                                int rows,int cols)

* Rutuja
* It is a kernel for each thread to convert RGB
                                pixel to grayscale.
*
******************************************************************************/
__global__ void RGBtoGRAY(const uchar* const InputRGB,
                          unsigned char* const OutputGRAY, 
                          int rows,int cols){

int i = blockIdx.y*blockDim.y + threadIdx.y;
int j = blockIdx.x*blockDim.x + threadIdx.x;
//printf("%d\t%d\t%d\t%d\n",blockIdx.x,blockIdx.y,i,j);
//fflush(0);
unsigned char temp_red,temp_green,temp_blue;

  if((i<rows) && (j<cols))
   {
    temp_red = InputRGB[((i*cols + j)*3)];
    temp_green = InputRGB[((i*cols + j)*3)+1];
    temp_blue = InputRGB[((i*cols + j)*3)+2];
    float result = 0.299f*(float)temp_red+0.587f*(float)temp_green+0.114f*(float)temp_blue;
   // printf("%d\t%d\t%f\n",i,j,result);
    float k = result -(long)result;
     if(k > (float)0.500){
      result = ceilf(result);}  
    OutputGRAY[i*cols + j] = (int)result;
    //printf("%d\n",(int)result);
   }
}

/******************************************************************************
*void my_RGBtoGRAY(const uchar* const h_ImageRGB,
                           uchar* const d_ImageRGB,
                           unsigned char* const d_ImageGray,
                           int rows,int cols)
* Rutuja
* It sets the dimensions for launching the kernel
                                and launches the kernel
*
******************************************************************************/
__host__ void my_RGBtoGRAY(const uchar* const h_ImageRGB,
                           uchar* const d_ImageRGB,
                           unsigned char* const d_ImageGray,
                           int rows,int cols,int x_block,int y_block)
{
const dim3 blockSize(x_block,y_block,1);
const dim3 gridSize(ceil((cols*1.0)/x_block),ceil((rows*1.0)/y_block),1);
RGBtoGRAY<<<gridSize, blockSize>>>(d_ImageRGB,d_ImageGray,rows,cols);
if ( cudaSuccess != cudaGetLastError())
    printf( "Error!\n" );
}


/******************************************************************************
*int verify_result(Mat& RGB,unsigned char* gray,
                   int vidlen,int Framecount)
*Rutuja
*Verification Function to perform pixel by pixel verification of float conversion
 of RGB to GRAY values as compared to OpenCV CvtColor() values.
********************************************************************************/

void verify_result(Mat& RGB,unsigned char* gray,
                   int vidlen,int Framecount)
{
  Mat GRAY;
  GRAY.create(RGB.rows,RGB.cols,CV_8UC1);
 // cvtColor(RGB,GRAY,CV_RGB2GRAY);
/*  int h= 0;
 // cout << RGB.rows << endl;
 // cout << RGB.cols << endl;
  for(int i = 0;i < (RGB.rows);i++)
  {
   for(int j = 0; j< RGB.cols ;j++)
   {  
      if(GRAY.data[i*RGB.cols + j] == gray[i*RGB.cols + j])
      {
     // cout << i << "\t" <<(int)GRAY.data[i] << "\t" << (int)gray[i] << "\n" << endl;
       h++;
      }
      else 
      {
       cout << i << "\t" << j << "\t" << (int)GRAY.data[i*RGB.cols+j] << "\t" << (int)gray[i*RGB.cols + j] << "\n" << endl; 
     }
   }
 }
 
  cout << h << endl;
  if(h == RGB.rows*RGB.cols)
  {
   Framecount++;
  }
  else
  {
   Framecount++;
   cout << "Failed Comparison for framecount :" << Framecount - 1 << endl;
  }
  //cout << "The framecount is:" << Framecount << endl;*/

  //Testing RGB2GRAY in C

  unsigned char r,g,b;
  float result;
  unsigned char* Image_gray;
  Image_gray=(unsigned char*)malloc(sizeof(unsigned char)*RGB.rows*RGB.cols);
  GpuTimer timer5;
  timer5.Start();
  for(int i = 0;i < RGB.rows ;i++)
  {
   for(int j = 0; j< RGB.cols ; j++)
    {
     r = RGB.data[(i*RGB.cols + j)*3];
     g = RGB.data[((i*RGB.cols + j)*3)+1];
     b = RGB.data[((i*RGB.cols + j)*3)+2];
     result  = 0.299f*(float)r + 0.587f*(float)g + 0.114f*(float)b;
     float p = result -long(result);
     if(p >(float)0.500){
     result = ceilf(result);}
     Image_gray[(i*RGB.cols) + j] = result;
    }
  }
  timer5.Stop();
  time5 = time5+ timer5.Elapsed();
  printf("The time taken for serial computation is:%f\n",time5);
// Verification of image matching of c code and GpU

   int h = 0;
   for (int a = 0; a < (RGB.rows); a++)
   {
    for(int b = 0 ; b < RGB.cols ; b++)
    {
    if(Image_gray[a*RGB.cols + b]==gray[a*RGB.cols + b])
       {
        h++;
       }
    else
       {
        printf("Failed comparison");
       }
    }
   } 

    cout << h << endl;
   if(h==(RGB.rows*RGB.cols))
   {
   
    cout << "Successful Comparison" << endl;
   }
   else
   {
    
    cout << "Failed Comparison" << endl;
    
   }

  //cout << Framecount << "\n" << endl; 
  if(Framecount == vidlen)
  {
   cout << "Successful" << endl;
  }
 

}

int allocate(Mat& ImageRGB,
              unsigned char* h_imageRGB,
              unsigned char* h_imageGRAY,
              unsigned char* d_imageRGB,
              unsigned char* d_imageGRAY, 
              size_t numPixels,int len,int framecount,
              bool verify,int x_block,int y_block)
{
GpuTimer timer1;
GpuTimer timer4;
timer4.Start();
timer1.Start();

cout << "Copying Memory" << endl;
gpuErrchk(cudaMemcpy(d_imageRGB,h_imageRGB,sizeof(unsigned char)*numPixels*3,cudaMemcpyHostToDevice));
timer1.Stop();
time1 = time1 + timer1.Elapsed();
cout << time1 << endl;
cout << "Finished Copying" << endl;
cout << "Launching Kernel" << endl;

GpuTimer timer2;
timer2.Start();
my_RGBtoGRAY(h_imageRGB,d_imageRGB,d_imageGRAY,ImageRGB.rows,ImageRGB.cols,
             x_block,y_block);
timer2.Stop();
time2 = time2 + timer2.Elapsed();
cout << time2 << endl;
cout << " Finished Launching kernel" << endl;
 cout << " Copying Memory back from GPU to CPU"<< endl;

GpuTimer timer3;
timer3.Start(); 
gpuErrchk(cudaMemcpy(h_imageGRAY,d_imageGRAY,sizeof(unsigned char)*numPixels,cudaMemcpyDeviceToHost));
timer3.Stop();
timer4.Stop();

time3 = time3 + timer3.Elapsed();
time4 = time4 + timer4.Elapsed();
cout << time3 << endl;
cout << time4 << endl;

cout << "Finshed copying back memory from GPU to CPU" << endl;
cout << "Freeing Memory" << endl;
cudaFree(d_imageRGB);
cudaFree(d_imageGRAY);
 return 0;
}


/******************************************************************************
* int main(int argc, char* argv[])
* Rutuja
* The main loads the image,converts it from BGR to
                                RGB,allocates pointer to the host image,allocates
                                memory to the host and device pointers,copy 
                                memory from CPU to GPU,call to launch the kernel,
                                copy memory back from GPU to CPU,call the 
                                function to match the gray images generated by
                                cuda code and the serial code,free device 
                                memory.  
*******************************************************************************/

int main(int argc, char* argv[]){

// 1 - Command line parsing
  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);

// Converting from BGR to RGB
 
  int x;
  int y;
  double frameLength;
  
//Initializing pointers
  unsigned char *h_imageRGB;
  unsigned char *h_imageGRAY;
  unsigned char* d_imageRGB, *d_imageGRAY;
  h_imageRGB = NULL;
  h_imageGRAY = NULL;
 
// Reading the frame
  VideoCapture cap(cmdLineArgs.filename);
 
  bool v = false;
  v = cmdLineArgs.verify;
  x = cmdLineArgs.blocksize_x;
  y = cmdLineArgs.blocksize_y;

  frameLength = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if(cap.isOpened()){
    uchar *test_gray,*src;
    while(cap.isOpened() && frameCount == 1)
     {
       //cout << "inside" << endl;
       Mat frame,ImageRGB,Gray;
       cap >> frame;
       //printf("inside1");
       if(!frame.empty())
        { 
        int row =  frame.rows;
        //printf("%d\n",row);
        int columns=frame.cols;
        //printf("%d\n",columns); 
        cvtColor(frame,ImageRGB,CV_BGR2RGB);
        const size_t numPixels = ImageRGB.rows*ImageRGB.cols;
        //allocating memory on the host
        h_imageRGB =(unsigned char*)malloc(sizeof(unsigned char)*numPixels*3);
        h_imageGRAY =(unsigned char*)malloc(sizeof(unsigned char)*numPixels);
        h_imageRGB = ImageRGB.data;
        //alloacting memory on the device
        gpuErrchk(cudaMalloc((void**)&d_imageRGB,sizeof(unsigned char)*numPixels*3));
        gpuErrchk(cudaMalloc((void**)&d_imageGRAY,sizeof(unsigned char)*numPixels));

        allocate(ImageRGB,h_imageRGB,h_imageGRAY,
                 d_imageRGB,d_imageGRAY,numPixels,
                 frameLength,frameCount,v,x,y);
        
        if(v == true)
        {
          verify_result(ImageRGB,h_imageGRAY,frameLength,frameCount);
        }


        }
       else
        {
        cout << "cannot load image" << endl;
        break;
        }
        frameCount++;

      }
    }
 cap.release();
 //cout << frameCount << endl;
 
 return 0;
 }


