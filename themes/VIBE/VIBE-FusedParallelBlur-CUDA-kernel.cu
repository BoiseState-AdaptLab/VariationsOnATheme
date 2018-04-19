#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<curand.h>
#include<curand_kernel.h>


#include "VIBE-FusedParallelBlur-CUDA-kernel.h"

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

__global__ void median_blur(unsigned char* map,
                            unsigned char *output_map,
                            int rows,int cols,
                            int kernel_size){

 // allocating shared memory
   extern __shared__ unsigned char local_map[];
   
   int g_idx,s_idx;
   
 // global thread index

   int j = blockIdx.x*blockDim.x + threadIdx.x;
   int i = blockIdx.y*blockDim.y + threadIdx.y;

   //int i =0;
   //int  j =0;
// Copying global memory to shared memory 
    if(threadIdx.x == 0 && j > 0)
    {
      g_idx = i*cols + j-1;
      s_idx = (threadIdx.y+1)*(blockDim.x+2) + threadIdx.x; 
      local_map[s_idx] = map[g_idx];

    }

    else if(threadIdx.y == 0 && i > 0)
    {
      g_idx = (i-1)*cols + j;
      s_idx = (threadIdx.y)*(blockDim.x+2) + threadIdx.x+1;
      local_map[s_idx] = map[g_idx]; 
    }
   
    else if(threadIdx.x == blockDim.x -1 && j < cols-1)
    {

      g_idx = i*cols + j + 1;
      s_idx = (threadIdx.y+1)*(blockDim.x+2) + threadIdx.x +2;
      local_map[s_idx] = map[g_idx];
    }

    else if(threadIdx.y == blockDim.y -1 && i  < rows - 1)
    {

      g_idx = (i+1)*cols + j;
      s_idx = (threadIdx.y+2)*(blockDim.x+2) + threadIdx.x +1;
      local_map[s_idx] = map[g_idx];
    }

    else if(threadIdx.x == 0 && threadIdx.y == 0 && i != 0 && j !=0)
    {
      g_idx = (i-1)*cols + j-1;
      s_idx = threadIdx.y*(blockDim.x+2) + threadIdx.x;
      local_map[s_idx] = map[g_idx]; 

    }

    else if(threadIdx.x == blockDim.x -1  && threadIdx.y == 0 && j != cols-1 && i != 0)
    {
      g_idx = (i-1)*cols + j+1;
      s_idx = (threadIdx.y)*(blockDim.x+2) + threadIdx.x+2;
      local_map[s_idx] = map[g_idx];

    }

    else if(threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && i != rows-1 && j != 0)
    {
      g_idx = (i+1)*cols + j-1;
      s_idx = (threadIdx.y+2)*(blockDim.x+2) + threadIdx.x;
      local_map[s_idx] = map[g_idx];

    }
    else if(threadIdx.x == blockDim.x -1  && threadIdx.y == blockDim.y - 1 && i != rows -1 && j != cols -1)
    {
      g_idx = (i+1)*cols + j+1;
      s_idx = (threadIdx.y+2)*(blockDim.x+2) + threadIdx.x+2;
      local_map[s_idx] = map[g_idx];

    }
    
    g_idx = i*cols + j;
    s_idx = (threadIdx.y+1)*(blockDim.x+2) + threadIdx.x+1;
    local_map[s_idx] = map[g_idx];
    
   
    
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
    output_map[i*cols + j] = 0;
  }
   else
  {
    output_map[i*cols + j] = 255;
  }    
 }
}

/* Function median_blur wrapper*/

extern "C" void cuda_median_blur(unsigned char*map,unsigned char *output_map,
                                 int rows,int cols,
                                 int kernel_size,size_t mem_size,
                                 dim3 grid,dim3 block){
 
  median_blur<<<grid,block,mem_size>>>(map,output_map,
                                       rows,cols,kernel_size);
  if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in median!\n" );
    }

}
__global__ void segment_update_model(unsigned char* map,pixel* frame,
                                     pixel* init_model,int time_sample,
                                     int rows,int cols,int samples,
                                     int radius,int match_threshold,
                                     curandState_t *state)
{
   int frame_index=0;
   int model_index=0;
   int dist = 0;
   int count = 0;
   int index = 0;
   int dist_threshold = 4.5*radius;
   int neigh_index;
   int x,y,n;
   int time_stamp;
   //int modelsample_index = 0;
   //int global_model_index = 0;
   
   // conversion factor for gnerating random number
   //float factor_model = samples/time_sample;
   //float factor_neighbour = 9/time_sample;

   pixel* pix_pointer;

   int j = blockIdx.x*blockDim.x + threadIdx.x;
   int i = blockIdx.y*blockDim.y + threadIdx.y;

   //index of each pixel
   if(i < rows && j < cols)
   {
     frame_index = i*cols + j;
     pix_pointer = &frame[frame_index];

     //index of model
     model_index = i*samples*cols + j*samples;

     while(index < samples && count<match_threshold)
     {

       model_index += 1;

       dist =  abs((pix_pointer->b - init_model[model_index].b)) +
               abs((pix_pointer->g -  init_model[model_index].g)) +
               abs((pix_pointer->r -  init_model[model_index].r));
           
       if(dist < dist_threshold)
       {
           count++;
       }

          index++;
          
      }

      time_stamp = (u_int)curand(&state[i*cols + j]) %time_sample;
      if(count >= match_threshold)
      {
         map[frame_index] = COLOR_BACKGROUND;
        //check if pixel gets chosen to be updated
       /*  if(time_stamp == 0)
         {
          //get index of model to update
          modelsample_index = time_stamp*factor_model;

          // memory index of the pixel + the model offset 
          global_model_index = frame_index*samples + modelsample_index;

          // set the model value to the current pixel value  
          init_model[global_model_index] = frame[frame_index];
          
          // update the neighbour
          if(i > 0 && i < rows - 1  && j > 0 && j < cols - 1)
          {
             n = time_stamp*factor_neighbour;
             x = n/3 - 1;
             y = n - 1 - 3*x;
             neigh_index = (x+i)*cols + (y+j);
             global_model_index = neigh_index*samples + modelsample_index;
             init_model[global_model_index] = frame[frame_index];
          }
         }*/
      }
      else
      {
         map[frame_index] = COLOR_FOREGROUND;
      }
      
      if(time_stamp == 0){

       if(map[frame_index] == 0){

        int modelsample_index;
        int global_model_index;

        // which value in the model to update (0-19)
        modelsample_index = (u_int)curand(&(state[frame_index]))%samples;

        // memory index of the pixel + the model offset 
        global_model_index = frame_index*samples + modelsample_index;

        // set the model value to the current pixel value  
        init_model[global_model_index] = frame[frame_index];

        if(i > 0 && i < rows - 1  && j > 0 && j < cols - 1)
        {
          n = (u_int)curand(&(state[frame_index]))%9;
          x = n/3 - 1;
          y = n - 1 - 3*x;
          neigh_index = (x+i)*cols + (y+j);

          //modelsample_index = (u_int)curand( &(state[frame_index]))%samples;
          global_model_index = neigh_index*samples + modelsample_index;
          init_model[global_model_index] = frame[frame_index];
        }
       }
     }     
 
   }
 }


void cuda_segment_update_model(unsigned char* map,pixel* frame,
                               pixel* init_model,int time_sample,
                               int rows,int cols,int samples,
                               int radius,int match_threshold,
                               curandState_t *state,dim3 grid,
                               dim3 block)
{
  
   segment_update_model<<<grid,block>>>(map,frame,init_model,
                                      time_sample,rows,cols,
                                      samples,radius,match_threshold,
                                      state);

   if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in update kernel!\n" );
     }

}
