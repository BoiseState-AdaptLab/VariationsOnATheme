#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<curand.h>
#include<curand_kernel.h>


#include "VIBE-FusedParallelShared-CUDA-kernel.h"

#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255


/* Function cuda_init_rand
 * Input: seed for the number generator
 *        state - an array of them already allocated on device
 *        cols - the width of our 2D data space
 */
__global__ void cuda_init_rand(unsigned long long seed,
                               curandState_t *state,
                               int cols){

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;

  curand_init(seed, // the seed 
              (unsigned long long)i+j,  // the sequence 
              0,    //the offset 
              &state[i*cols + j]);

}
/* Function cuda_init_rand_wrapper
 * input: A seed to get the randoma number generator started.
 *        A pointer to an array of states that have been already
 *          allocated on the device
 *        The number of columns in our 2D data layout 
 *        The grid and block configuration
 */
extern "C" void cuda_init_rand_wrapper(unsigned long long seed,
                                       curandState_t *state,
                                       int cols, dim3 grid, dim3 block){
    (cuda_init_rand<<<grid,block>>>(seed,state,cols));
    if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in curand init!\n" );
    }
}


/* Function median_blur*/

__global__ void median_blur(unsigned char* map,int rows,
                            int cols,int kernel_size){

 // allocating shared memory
   extern __shared__ unsigned char local_map[];
  
   int g_idx,s_idx;
   
 // global thread index

   int j = blockIdx.x*blockDim.x + threadIdx.x;
   int i = blockIdx.y*blockDim.y + threadIdx.y;

   //int i =0;
   //int  j =0;
// Copying global memory to shared memory except for the boundary 
// pixels
   if( i > 0 && i < rows-1 && j > 0  && j < cols-1){  
    //printf("%d,%d",i,j);
    if(threadIdx.x == 0 && threadIdx.y >0 &&  threadIdx.y < blockDim.y-1  )
    {
      g_idx = i*cols + j-1;
      s_idx = (threadIdx.y+1)*(blockDim.x+2) + threadIdx.x; 
      local_map[s_idx] = map[g_idx];

    }

    else if(threadIdx.y == 0 && threadIdx.x > 0 && threadIdx.x < blockDim.x-1)
    {
      g_idx = (i-1)*cols + j;
      s_idx = (threadIdx.y)*(blockDim.x+2) + threadIdx.x+1;
      local_map[s_idx] = map[g_idx]; 
    }
   
    else if(threadIdx.x == blockDim.x -1 && threadIdx.y > 0 && threadIdx.y < blockDim.y-1)
    {

      g_idx = i*cols + j + 1;
      s_idx = (threadIdx.y+1)*(blockDim.x+2) + threadIdx.x +2;
      local_map[s_idx] = map[g_idx];
    }

    else if(threadIdx.y == blockDim.y -1 && threadIdx.x > 0 &&  threadIdx.x < blockDim.x-1)
    {

      g_idx = (i+1)*cols + j;
      s_idx = (threadIdx.y+2)*(blockDim.x+2) + threadIdx.x +1;
      local_map[s_idx] = map[g_idx];
    }

    else if(threadIdx.x == 0 && threadIdx.y == 0)
    {
      g_idx = (i-1)*cols + j-1;
      s_idx = threadIdx.y*(blockDim.x+2) + threadIdx.x;
      local_map[s_idx] = map[g_idx]; 

    }

    else if(threadIdx.x == blockDim.x -1  && threadIdx.y == 0)
    {
      g_idx = (i-1)*cols + j+1;
      s_idx = (threadIdx.y)*(blockDim.x+2) + threadIdx.x+2;
      local_map[s_idx] = map[g_idx];

    }

    else if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
    {
      g_idx = (i+1)*cols + j-1;
      s_idx = (threadIdx.y+2)*(blockDim.x+2) + threadIdx.x;
      local_map[s_idx] = map[g_idx];

    }
    else if(threadIdx.x == blockDim.x -1  && threadIdx.y == blockDim.y - 1)
    {
      g_idx = (i+1)*cols + j+1;
      s_idx = (threadIdx.y+2)*(blockDim.x+2) + threadIdx.x+2;
      local_map[s_idx] = map[g_idx];

    }
    
    g_idx = i*cols + j;
    s_idx = (threadIdx.y+1)*(blockDim.x+2) + threadIdx.x+1;
    local_map[s_idx] = map[g_idx];
    
   }
    
   __syncthreads();
   
// Counting the zeros in the neighbourhood 
   if(i > 0 &&  i < rows-1 && j > 0 && j < cols-1){
    int count = 0;

    for(int a = -1;a < 2; a++)
    {
     for(int b = -1; b < 2;b++)
     {
      if(local_map[s_idx+ (a*(blockDim.x+2))+b ] == 0 && count < 5)
      { 
        count++;
      }
     }
    }
   
// assigning median value of neighbourhood as 0 or 255   
   if(count >= 4)
  {
    map[i*cols + j] = 0;
  }
   else
  {
    map[i*cols + j] = 255;
  }    
 }
}

/* Function median_blur wrapper*/

extern "C" void cuda_median_blur(unsigned char*map,int rows,int cols,
                                 int kernel_size,size_t mem_size,
                                 dim3 grid,dim3 block){
 
  median_blur<<<grid,block,mem_size>>>(map,rows,cols,kernel_size);
  if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in median!\n" );
    }

}
__global__ void segment_update_model(unsigned char* map,pixel* frame,
                                     unsigned char* init_model,int time_sample,
                                     int rows,int cols,int samples,
                                     int radius,int match_threshold,
                                     curandState_t *state)
{

   int frame_index=0;
   int global_model_index=0;
   int local_model_index = 0;
   int dist = 0;
   int count = 0;
   int index = 0;
   int dist_threshold = 4.5*radius;
   int neigh_index;
   int x,y,n;
   int time_stamp;

   extern __shared__ unsigned char localmodel[];

   unsigned char current_pixel[3];
  

   int j = blockIdx.x*blockDim.x + threadIdx.x;
   int i = blockIdx.y*blockDim.y + threadIdx.y;

   //get the pixel to be segmented and updated
   frame_index = i*cols + j;
   current_pixel[0] = frame[frame_index].b;
   current_pixel[1] = frame[frame_index].g;
   current_pixel[2] = frame[frame_index].r;

   //copying model into shared memory
   if(i < rows  && j < cols)
   {
   
    //index of model
     global_model_index = (i*cols + j)*samples*3;
     local_model_index = ((threadIdx.y)*(blockDim.x) + threadIdx.x)*samples*3;

     for(int h =0;h < 3*samples;h+=3)
     {
       localmodel[local_model_index + h] = init_model[global_model_index+h];
       localmodel[local_model_index + h +1] = init_model[global_model_index+h+1];
       localmodel[local_model_index + h +2] = init_model[global_model_index+h+2];
     }

   } 
       
   __syncthreads();
  
   if(i < rows  && j < cols)
   {
     while(index < samples && count < match_threshold)
     {

       dist =  abs((current_pixel[0] - localmodel[local_model_index])) +
               abs((current_pixel[1] - localmodel[local_model_index+1])) +
               abs((current_pixel[2] - localmodel[local_model_index+2]));
        
           
       if(dist < dist_threshold)
       {
           count++;
       }

       index++;
       local_model_index += 3;
          
     }


      if(count >= match_threshold)
      {
         map[frame_index] = COLOR_BACKGROUND;
      }
      else
      {
         map[frame_index] = COLOR_FOREGROUND;
      }

      time_stamp = (u_int)curand(&state[i*cols + j]) %time_sample;
      if(time_stamp == 0){

      if(map[frame_index] == 0){

        int modelsample_index;
        int global_model_index;

        // which value in the model to update (0-19)
        modelsample_index = (u_int)curand(&(state[frame_index]))%samples;

        // memory index of the pixel + the model offset 
        global_model_index = frame_index*samples*3 + modelsample_index*3;

        // set the model value to the current pixel value  
        init_model[global_model_index] = frame[frame_index].b;
        init_model[global_model_index+1] = frame[frame_index].g;
        init_model[global_model_index+2] = frame[frame_index].r;

        if(i > 0 && i < rows - 1  && j > 0 && j < cols - 1)
        {
          n = (u_int)curand(&(state[frame_index]))%9;
          x = n/3 - 1;
          y = n - 1 - 3*x;
          neigh_index = (x+i)*cols + (y+j);

          global_model_index = neigh_index*samples*3 + modelsample_index*3;
          init_model[global_model_index] = frame[frame_index].b;
          init_model[global_model_index+1] = frame[frame_index].g;
          init_model[global_model_index+2] = frame[frame_index].r;
        }
       }
     }
   }
 }


void cuda_segment_update_model(unsigned char* map,pixel* frame,
                               unsigned char* init_model,int time_sample,
                               int rows,int cols,int samples,
                               int radius,int match_threshold,
                               curandState_t *state,size_t model_size,
                               dim3 grid,dim3 block)
{
  
   segment_update_model<<<grid,block,model_size>>>(map,frame,init_model,
                                      time_sample,rows,cols,
                                      samples,radius,match_threshold,
                                      state);

   if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in update kernel!\n" );
     }

}
