#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<bitset>
#include "timer.cu"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/videoio/videoio.hpp"
#include "opencv/cv.h"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "utilities.h"

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
                          const int* const look_up_table,
                          int rows,int cols){
 int yuv_shift=14;
// int size = rows*cols;
 int i = blockIdx.y*blockDim.y + threadIdx.y;
 int j = blockIdx.x*blockDim.x + threadIdx.x;
//printf("%d\t%d\t%d\t%d\n",blockIdx.x,blockIdx.y,i,j);
//fflush(0);
 if(i < rows && j < cols)
{
   OutputGRAY[i*cols + j] = (uchar)((look_up_table[InputRGB[((i*cols)+j)*3]]
                + look_up_table[InputRGB[((i*cols)+j)*3+1]+256] + look_up_table[InputRGB[((i*cols)+j)*3+2]+512]) >> yuv_shift);
}
//printf("%d",OutputGRAY[1000]);
}

/******************************************************************************
* uchar* test_OpenCV_int(Mat& Image1, Mat& Image,int rows,int cols)
* Rutuja
* Implementation of OpenCV cvtColor in C
*
******************************************************************************/
void  test_OpenCV_int(const unsigned char *Image1,const uchar *gray,const int* const look_tab,int rows,int cols,int frame,int len)
{
  //printf("inside4");
 // uchar* src= Image1.data;
  unsigned char *dst;
  int k,l,h,a,b;
  h =0;
  int size  = rows*cols;
 // printf("%d",size);
  int yuv_shift = 14;
  dst = (uchar*)malloc(size*sizeof(uchar));
  //int tab[3*256];
  int srccn = 3;
  GpuTimer timer5;
  timer5.Start();
  for(a = 0;a < rows;a++)
  {
   for( b = 0;b < cols ; b++)
   {
  // for(l = 0 ;l < size ;l++,Image1 += srccn){ 
    dst[a*cols+b] = (uchar)((look_tab[Image1[(a*cols+b)*3]] + look_tab[Image1[(a    *cols+b)*3+1]+256] +look_tab[Image1[((a*cols+b)*3)+2]+512]) >> yuv_shift);
  // dst[l] = (uchar)((look_tab[Image1[0]] + look_tab[Image1[1]+256] + look_tab[Image1[2]+512]) >> yuv_shift);
   
   }
  }
  timer5.Stop();
  time5 += timer5.Elapsed();
  printf("%f\n",time5);
  //printf("%d",dst[921610]);
  for(k = 0; k< size ;k++)
  {
    if(dst[k]==gray[k])
    {
     h++;
    }
    else
    {
      printf("Failed Comparison");
      printf("%d",k);
    }
  }
  cout << h << endl;
  if(h==size)
  {
    printf("frame matches");
  }
  
  if(frame == len)
  {
    printf("Succesful Comparison");
  }
  else
  {
    printf("All the frames do not match");
  }
 
  free(dst);
  
  
}

/******************************************************************************
*int* look_up()
* Rutuja
* It builds the table required for floating point approximation of integer 
  calculations from RGB TO GRAY values.

******************************************************************************/
int* look_up()
{

  int *tab;
  int k;
  int yuv_shift = 14;
  int R2Y = 4899;
  int G2Y = 9617;
  int B2Y = 1868;
  tab = (int*)malloc((3*256)*sizeof(int));
  //int tab[3*256];
 // printf("inside2");
  const int coeffs0[] = {R2Y,G2Y,B2Y};
  int b = 0,g = 0, r = (1 << (yuv_shift -1));
  int db = coeffs0[2],dg = coeffs0[1],dr = coeffs0[0];
  for(k = 0; k < 256;k++,b +=db,g+=dg,r+=dr)
   {
     tab[k]=b;
     tab[k+256] = g;
     tab[k+512] = r;
   }
 
 return tab;
}
     
int allocate(Mat& ImageRGB,
              unsigned char* h_imageRGB,
              unsigned char* h_imageGRAY,
              unsigned char *d_imageRGB,
              unsigned char *d_imageGRAY,
              int* tab,int *d_tab,size_t numPixels,
              int len,int framecount,
              bool verify,int x_block,int y_block)
{

GpuTimer timer1;
GpuTimer timer4;
timer4.Start();
timer1.Start();
cout << "Copying Memory" << endl;
gpuErrchk(cudaMemcpy(d_imageRGB,h_imageRGB,sizeof(unsigned char)*numPixels*3,cudaMemcpyHostToDevice));
gpuErrchk(cudaMemcpy(d_tab,tab,sizeof(int)*256*3,cudaMemcpyHostToDevice));
timer1.Stop();

time1 = time1 + timer1.Elapsed();
cout << time1 << endl;
cout << "Finished Copying" << endl;
cout << "Launching Kernel" << endl;
GpuTimer timer2;
timer2.Start();
const dim3 blockSize(x_block,y_block,1);
const dim3 gridSize(ceil((ImageRGB.cols*1.0)/x_block),ceil((ImageRGB.rows*1.0)/y_block),1);
//printf("\n%d\t%d",(int)ceil((ImageRGB.cols*1.0)/x_block),(int)ceil((ImageRGB.rows*1.0)/y_block));
RGBtoGRAY<<<gridSize, blockSize>>>(d_imageRGB,d_imageGRAY,d_tab,ImageRGB.rows,ImageRGB.cols);
if ( cudaSuccess != cudaGetLastError())
    printf( "Error!\n" );

timer2.Stop();
time2 = time2 + timer2.Elapsed();
cout << time2 << endl;
cout << " Finished Launching kernel" << endl;
cout << " Copying Memory back from GPU to CPU" << endl;
GpuTimer timer3;
timer3.Start(); 
gpuErrchk(cudaMemcpy(h_imageGRAY,d_imageGRAY,sizeof(unsigned char)*numPixels,cudaMemcpyDeviceToHost));
timer3.Stop();
timer4.Stop();

time3 = time3 + timer3.Elapsed();
time4 = time4 + timer4.Elapsed();
cout << time3 << endl;
cout << time4 << endl;

//cout << "Finshed copying back memory from GPU to CPU" << endl;
// cout << "Freeing Memory" << endl;
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
 // Mat ImageRGB,ImageBGR;
   double frameLength;
  
//Initializing pointers
  int *table;
  unsigned char *h_imageRGB;
  unsigned char *h_imageGRAY;
  unsigned char *d_imageRGB, *d_imageGRAY;
  int* d_tab;
  h_imageRGB = NULL;
  h_imageGRAY = NULL;
 
// Calculating the lookup table
  table = look_up();
 // printf("%d \t%d",table[1],table[767]);
// Reading the frame
  VideoCapture cap(cmdLineArgs.filename);
  frameLength = cap.get(CV_CAP_PROP_FRAME_COUNT);
 // ImageBGR = imread(cmdLineArgs.filename,CV_LOAD_IMAGE_COLOR);
 
  bool v = false;
  int x , y = 0;
  v = cmdLineArgs.verify;
  x = cmdLineArgs.blocksize_x;
  y = cmdLineArgs.blocksize_y;
  if(cap.isOpened()){
    while(cap.isOpened() && frameCount == 1)
     {
       //cout << "inside" << endl;
       Mat frame,ImageRGB;
       cap >> frame;
       //printf("inside1");
       if(!frame.empty())
        { 
        int row =  frame.rows;
       // printf("%d\n",row);
        int columns=frame.cols;
       // printf("%d\n",columns); 
        cvtColor(frame,ImageRGB,CV_BGR2RGB);
        const size_t numPixels = ImageRGB.rows*ImageRGB.cols;
        h_imageRGB =(unsigned char*)malloc(sizeof(unsigned char)*numPixels*3);
        h_imageGRAY =(unsigned char*)malloc(sizeof(unsigned char)*numPixels);
        h_imageRGB = ImageRGB.data;
        gpuErrchk(cudaMalloc((void**)&d_imageRGB,sizeof(unsigned char)*numPixels*3));
        gpuErrchk(cudaMalloc((void**)&d_imageGRAY,sizeof(unsigned char)*numPixels));
        gpuErrchk(cudaMalloc((void**)&d_tab,sizeof(int)*256*3));

        allocate(ImageRGB,h_imageRGB,h_imageGRAY,d_imageRGB,d_imageGRAY,table,d_tab,numPixels,frameLength,frameCount,v,x,y);
        if(v == true)
          {
            test_OpenCV_int(h_imageRGB,h_imageGRAY,table,ImageRGB.rows,ImageRGB.cols,frameCount,frameLength);
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
 
 
 return 0;
 }


