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
                          int rows,int cols,int xblock_size,
                          int yblock_size,int xtile_size,
                          int ytile_size,int xfactor,int yfactor)
{

  // Memory allocated in shared space for just this block
  extern __shared__ int array[];
  // Used for a shift when approximating GRAY value
  int yuv_shift=14;

  int g_idx,s_idx,k,l,m,n;
  // i,j starting co-ordinates of this block
  int i = blockIdx.x*xtile_size + threadIdx.x;
  int j = blockIdx.y*ytile_size + yfactor*threadIdx.y;

  // locate the first global memory index for this block/tile
  m = 3*j*cols + i*3;
  n = j*cols + i;
  
  // k,l tile numbers to be executed by each block 
  for(k=0;k < (yfactor);k++){

    for(l=0;l < (xfactor);l++){

      // shared memory data index
      s_idx = 3*threadIdx.y*blockDim.x+3*threadIdx.x;
      // read the blue,green,red values into shared memory
      array[s_idx]   = InputRGB[m];
      array[s_idx+1] = InputRGB[m+1];
      array[s_idx+2] = InputRGB[m+2];

      // This is to sync the reads
      __syncthreads();

      // write the data to global memory 
      OutputGRAY[n] = (unsigned char)((look_up_table[array[(s_idx)]]
             + look_up_table[array[(s_idx)+1]+256] 
             + look_up_table[array[(s_idx)+2]+512]) >> yuv_shift);



     // foreach step in we need to jump the size of the block
     m += xblock_size*3;
     n += xblock_size;
   } // this is the end of the l loop

   // This jump needs to be down one row
   m = 3*(j+k+1)*cols + i*3;
   n = (j+k+1)*cols + i;

  } // this is the end of the k loop
 

}


/******************************************************************************
* uchar* test_OpenCV_int(Mat& Image1, Mat& Image,int rows,int cols)
* Rutuja
* Implementation of OpenCV cvtColor in C
*
******************************************************************************/
void test_OpenCV_int(const unsigned char *Image1,const uchar *gray,const int* const look_tab,int rows,int cols,int frame,int len)
{

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
    dst[a*cols+b] = (uchar)((look_tab[Image1[(a*cols+b)*3]] + look_tab[Image1[(a*cols+b)*3+1]+256] +look_tab[Image1[((a*cols+b)*3)+2]+512]) >> yuv_shift);
  // dst[l] = (uchar)((look_tab[Image1[0]] + look_tab[Image1[1]+256] + look_tab[Image1[2]+512]) >> yuv_shift);
   
   }
  }
  timer5.Stop();
  time5 += timer5.Elapsed();
  //printf("%d",dst[921610]);
  printf("\nThe serial timing of the code is:%f\n",time5);


  // Verification
  int failed = 0;
  for(k = 0; k< size && !failed ;k++)
  {
    if(dst[k]==gray[k])
    {
     h++;
    }
    else
    {
      failed = 1;
    }
  }
  if(failed){
    printf("Failed Comparison %d\n",k);
  }else{
    printf("SUCCESS\n");
  }
  
 /* if(frame == len)
  {
    printf("Succesful Comparison");
  }
  else
  {
    printf("All the frames do not match");
  }*/
 
  free(dst);
  
  
} 

/******************************************************************************
*int* look_up()
* Rutuja
* It builds the table required for floating point approximation of integer 
  calculations from RGB TO GRAY values.
*
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


/*******************************************************************************
int main()- The main loads the image,converts it from BGR to
            RGB,allocates pointer to the host image,allocates
            memory to the host and device pointers,copy 
            memory from CPU to GPU,call to launch the kernel,
            copy memory back from GPU to CPU,call the 
            function to match the gray images generated by
            cuda code and the serial code,free device 
            memory.  
*******************************************************************************/

int main(int argc, char* argv[])
{

// 1 - Command line parsing
  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);

// Mat ImageRGB,ImageBGR;
  double frameLength;
  
//Initializing pointers
  int *table;
  unsigned char *h_imageRGB;
  unsigned char *h_imageGRAY;
  unsigned char *d_imageRGB, *d_imageGRAY;
  int *d_tab;
  h_imageRGB = NULL;
  h_imageGRAY = NULL;

//Initializing timers

// timer 1 is the memcpy from cpu to gpu time
  GpuTimer timer1;
// timer 2 is the GPU kernel processing time
  GpuTimer timer2;
// timer 3 is the memcpy gpu to cpu time
  GpuTimer timer3;
// timer 4 is the whole time
  GpuTimer timer4;

 
// Calculating the lookup table
  table = look_up();

// Reading the frame
  VideoCapture cap(cmdLineArgs.filename);
  frameLength = cap.get(CV_CAP_PROP_FRAME_COUNT);
 
  bool v = false;
  int x_block,y_block,x_tile,y_tile,x_factor,y_factor,avg_factor,iteration;
  x_block =0;
  y_block =0;
  x_tile =0;
  y_tile =0;
  x_factor=0;
  y_factor=0;
  avg_factor=1;
  iteration = 1;
  v = cmdLineArgs.verify;
  x_block = cmdLineArgs.blocksize_x;
  y_block = cmdLineArgs.blocksize_y;
  x_tile = cmdLineArgs.tilesize_x;
  y_tile = cmdLineArgs.tilesize_y;
  avg_factor = cmdLineArgs.average_factor;
// Checking for tile and block size
  
  if(x_tile%x_block==0 && y_tile%y_block==0)
  {
    x_factor = x_tile/x_block;
    y_factor = y_tile/y_block;
  }
  else
  {
    
    printf("\nThe block and tile sizes are not multiples.Enter tile size to be multiple of block size");
    return 0;   
     
  }

  
  if(cap.isOpened()){
    while(cap.isOpened() && frameCount == 1)
     {
       Mat frame,ImageRGB;
       cap >> frame;
       while(iteration<=avg_factor)
       {
        if(!frame.empty())
         { 
         int row =  frame.rows;
         int columns=frame.cols; 
         cvtColor(frame,ImageRGB,CV_BGR2RGB);
         const size_t numPixels = ImageRGB.rows*ImageRGB.cols;
        
        // Checking for gridDims and tile size
         if(columns%x_tile!=0 || row%y_tile!=0)
         { 
          printf("\nThe grid dimensions is not a multiple of frame size");
          break;
         }
     
         if(x_factor==0 || y_factor==0)
         {
          break;
         }

        //allocate memory on the host
         size_t  sizeRGB = sizeof(unsigned char)*numPixels*3;
         size_t  sizeGRAY = sizeof(unsigned char)*numPixels;
         size_t sizeTAB = sizeof(int)*256*3;
        
        // allocate data to the host
         h_imageRGB =(unsigned char*)malloc(sizeRGB);
         h_imageGRAY =(unsigned char*)malloc(sizeGRAY);
         h_imageRGB = ImageRGB.data;


        // allocate memory on the device
        gpuErrchk(cudaMalloc((void**)&d_imageRGB,sizeRGB));
        gpuErrchk(cudaMalloc((void**)&d_imageGRAY,sizeGRAY));
        gpuErrchk(cudaMalloc((void**)&d_tab,sizeTAB));
      
        //timer started for total calculation
        //timer staretd for memory transfer from CPU to GPU    
        timer4.Start();
        timer1.Start();
        gpuErrchk(cudaMemcpy(d_imageRGB,h_imageRGB,sizeRGB,
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_tab,table,sizeTAB,cudaMemcpyHostToDevice));
        timer1.Stop();
        time1 = time1 + timer1.Elapsed();
        
        //timer started for processing on the GPU
        timer2.Start();
        const dim3 blockSize(x_block,y_block,1);
        const dim3 gridSize(columns/x_tile,row/y_tile,1);
        size_t dyn_mem_size = 3*x_block*y_block*sizeof(int);
       /* printf("Configuration: block(%d,%d)\n\tgrid(%d,%d)",
                x_block,y_block,columns/x_tile,row/y_tile);
            
        printf("\n\tfactors(%d,%d)\n",x_factor,y_factor);
        printf("\ttile(%d,%d)\n",x_tile,y_tile);*/
     
    
        RGBtoGRAY<<<gridSize, blockSize, dyn_mem_size>>>(d_imageRGB,
                                                         d_imageGRAY,
                                                         d_tab,
                                                         ImageRGB.rows,
                                                         ImageRGB.cols,
                                                         x_block,
                                                         y_block,x_tile,
                                                         y_tile,x_factor,
                                                         y_factor);

        gpuErrchk(cudaThreadSynchronize());

        timer2.Stop();
        time2 = time2 + timer2.Elapsed();
        
        // timer started for memory transfer from GPU to CPU
        timer3.Start();
        gpuErrchk(cudaMemcpy(h_imageGRAY,d_imageGRAY,sizeGRAY,
                  cudaMemcpyDeviceToHost));
        timer3.Stop();
        timer4.Stop();

        time3 = time3 + timer3.Elapsed();
        time4 = time4 + timer4.Elapsed();
        // all the timers stopped
         
        //Freeing GPU memory
        cudaFree(d_imageRGB);
        cudaFree(d_imageGRAY);
                
        // Verification for each frame pixel by pixel
        if(v == true)
         {
           test_OpenCV_int(h_imageRGB,h_imageGRAY,table,
                                 ImageRGB.rows,ImageRGB.cols,
                                 frameCount,frameLength);
         }
       }
       else
       {
        cout << "cannot load image" << endl;
        break;
       }
      iteration++;
     }
     frameCount++;
     time1 = time1/avg_factor;
     time2 = time2/avg_factor;
     time3 = time3/avg_factor;
     time4 = time4/avg_factor;
     printf("\nx_block:%d,y_block:%d,x_factor:%d,y_factor:%d,x_tile:%d,y_tile:%d,MemCpyCPUtoGPU:%f,KernelProcessing:%f,MemCpyGPUtoCPU:%f,Totaltime:%f",x_block,y_block,x_factor,y_factor,x_tile,y_tile,time1,time2,time3,time4);

    }
   }
 cap.release();
 return 0;
 }


