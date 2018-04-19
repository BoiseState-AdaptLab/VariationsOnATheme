#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>
#include<curand.h>
#include<curand_kernel.h>
#include"util.h"

#include "VIBE-NaiveParallel_Overlap-CUDA-kernel.h"

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


__global__ void segment_frames(unsigned char* map,pixel* frame,
                               pixel* init_model,int rows,int cols,
                               int samples,int radius,int match_threshold)
{
   bool match = false;
   int frame_index=0;
   int model_index=0;
   int dist = 0;
   int count = 0;
   int index = 0;
   int dist_threshold = 4.5*radius;

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
   
       while(count < match_threshold &&  index < samples)
        {

          

          dist =  abs((pix_pointer->b - init_model[model_index].b)) +
                  abs((pix_pointer->g -  init_model[model_index].g)) +
                  abs((pix_pointer->r -  init_model[model_index].r));

           if(dist < dist_threshold)
          {
             count++;
          }

          model_index += 1;
          index++;
       }

        if(count >= match_threshold)
       {
         match = true;
       }
        else
       {
         match =  false;
       }
       
        // creating the map by segmenting each pixel as foreground or 
        // background
       if(match)
         {
          map[i*cols + j] = COLOR_BACKGROUND;
          //cout <<  (int)map[120*cols + 160] << endl;        
         }
       else
         {
           map[i*cols + j] = COLOR_FOREGROUND;
         // cout << "not background" << endl;          
         }
     }

}

__global__ void update_model(unsigned char* segmentmap,
                             pixel* c_image,pixel* init_model,
                             curandState_t *state,
                             int rows,int cols,int time_sample,
                             int numsamples,int frames){

    int map_index;
    int neigh_index;
    int x,y,n;
    int time_stamp;

    
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
     
    
    map_index = i*cols + j;
    
    time_stamp = (u_int)curand(&state[i*cols + j])%time_sample;
    if(time_stamp == 0){

     if(segmentmap[map_index] == 0){
           
      int modelsample_index;
      int global_model_index;

      // which value in the model to update (0-19)
      modelsample_index = (u_int)curand(&(state[i*cols+j]))%numsamples;
         
      // memory index of the pixel + the model offset 
      global_model_index = map_index*numsamples + modelsample_index;

      // set the model value to the current pixel value  
      init_model[global_model_index] = c_image[map_index];

      n = (u_int)curand(&state[i*cols+j])%9; 
      x = n/3 - 1;
      y = n%3 - 1;

      if( i > 0 && i < rows -1 && j > 0 && j < cols - 1)
      {
        neigh_index = (x+i)*cols + (y+j);
        global_model_index = neigh_index*numsamples + modelsample_index;
        init_model[global_model_index] = c_image[map_index];
      }
    }
  }
}

extern "C" void cuda_segment_frames(unsigned char* map,pixel* frame,
                                   pixel* init_model,int rows,int cols,
                                   int samples,int radius,int match_threshold,
                                   dim3 grid,dim3 block)
{
      segment_frames<<<grid, block>>>(map,frame,
                                      init_model,rows,
                                      cols,samples,radius,
                                      match_threshold);
      if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in segment kernel!\n" );
      }
  
}
extern "C" void cuda_update_model(unsigned char* map,pixel* frame,
                                  pixel* init_model, curandState_t *state,
                                  int rows,int cols,int time_samples,
                                  int modelsamples,int framecount,
                                  dim3 grid,dim3 block) {

     update_model<<<grid,block>>>(map,frame,init_model,state,
                                rows,cols,time_samples,
                                modelsamples,framecount);
     if ( cudaSuccess != cudaGetLastError()){
          printf( "Error in update kernel!\n" );
     }


}

                                      
