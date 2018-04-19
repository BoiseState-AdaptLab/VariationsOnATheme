
#include<iostream>
#include<stdio.h>
#include<string>
#include<cmath>


#include"util.h"

#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255


__global__ void segment_frames(unsigned char* map,pixels* frame,
                               pixels* init_model,int rows,int cols,
                               int samples,int radius,int match_threshold){

  bool match = false;
  int frame_index=0;
  int model_index=0;
  int dist = 0;
  int count = 0;
  int index = 0;

  pixels* pix_pointer;

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int i = blockIdx.y*blockDim.y + threadIdx.y;

  //index of each pixel
  if(i < rows && j < cols){

    frame_index = i*cols + j;
    pix_pointer = &frame[frame_index];

    //index of model
    model_index = i*samples*cols + j*samples;
   
    while(count <= match_threshold &&  index < samples){

       model_index =+ 1;

       dist =  abs((pix_pointer->b - init_model[model_index].b)*
                   (pix_pointer->b-init_model[model_index].b)) +
               abs((pix_pointer->g-init_model[model_index].g)*
                   (pix_pointer->g-init_model[model_index].g)) +
               abs((pix_pointer->r-init_model[model_index].r)*
                   (pix_pointer->r-init_model[model_index].r));

       if(dist < radius*radius) {
         count++;
       }

      index++;
    }

    if(count >= match_threshold){
      match = true;
    }else{
      match =  false;
    }

    // creating the map by segmenting each pixel as foreground or 
    // background
    if(match){
      map[i*cols + j] = COLOR_BACKGROUND;
      //cout <<  (int)map[120*cols + 160] << endl;        
    }else{
      map[i*cols + j] = COLOR_FOREGROUND;
      // cout << "not background" << endl;          
    }
  }
}
                                          
