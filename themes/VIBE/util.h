#include<iostream>
#include<stdlib.h>


#ifndef UTIL_H
#define UTIL_H 1

struct pixel 
{

  unsigned char b;
  unsigned char g;
  unsigned char r;

};


void initialize_model(pixel* first_frame,
                      pixel* init_model,
                      int rows, int cols,
                      int samples);

void Initialize_model(unsigned char*first_frame,
                      pixel* init_model,
                      int rows, int cols,
                      int samples);

int find_euclidean_distance(unsigned char b1,unsigned char g1,
                            unsigned char r1,unsigned char b2,
                            unsigned char g2,unsigned char r2);

pixel* updatepixelmodel(pixel* c_frame,pixel* c_mod,int index,int size);

pixel* Updatepixelmodel(unsigned char* c_frame,pixel* c_mod,int index,int size);

int count_samples(pixel* pointer,pixel* curr_model,
                  int threshold,int sample,int rad,int mod_index,int frame);

int Count_samples(unsigned char* pointer,pixel* curr_model,
                  int threshold,int sample,int rad,int mod_index);

void segment_frame(unsigned char* map,pixel* frame,pixel* init_model,
                    int rows,int cols,int samples,int radius,
                    int match_threshold,int frameNumber);


long long update_model(unsigned char* segmentmap,
                  pixel* c_image,pixel* init_model,
                  int row,int col,int time_sample,
                  int numsamples,int frames,long long seg_count);


#endif
