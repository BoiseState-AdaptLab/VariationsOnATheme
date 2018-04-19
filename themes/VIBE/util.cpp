#include<iostream>
#include<stdlib.h>
//#include<random>
#include<time.h>
#include "util.h"
#include "../common/Configuration.h"

#ifndef init_model
#define init_model(i,j,n) init_model[(i*20*480) + (j*20) + n]

#define COLOR_BACKGROUND 0
#define COLOR_FOREGROUND 255

using namespace std;

/*************************************************************************
 ** Function: segment_frame(unsigned char* map,pixel* frame,
                    pixel* init_model,
                    int rows,int cols,int samples,
                    int radius,int match_threshold,int frameNumber)
   -function to segment frame pixel by pixel into background or foreground
***************************************************************************/

void segment_frame(unsigned char* map,pixel* frame,
                    pixel* init_model,
                    int rows,int cols,int samples,
                    int radius,int match_threshold,int frameNumber)
{
   bool match = false;
   int frame_index=0;
   int model_index=0;
   pixel* pix_pointer;
   //cout << rows << cols << endl;

   for(int k = 0;k < rows;k++)
   {
     for(int l = 0; l < cols; l++)
     {
        //index of each pixel
        frame_index = k*cols + l;
        pix_pointer = &frame[frame_index];

        //index of model
        model_index = k*samples*cols + l*samples;

        // calculate the samples that lie within the radius threshold 
        match = count_samples(pix_pointer,init_model,match_threshold,
                              samples,radius,model_index,frameNumber);

        //creating the map by segmenting each pixel as foreground or
        /*if(k == 257 && l == 225)
        {
          cout << match << endl;
        }*/
        // background
         if(match)
         {
          map[k*cols + l] = COLOR_BACKGROUND;
         }
         else
         {

           map[k*cols + l] = COLOR_FOREGROUND;
         // cout << "not background" << endl;          
         }

      }
     }

}

/******************************************************************************************
 * Function update_frame();
 
  - this functions updates the pixel model acording to a random policy

*******************************************************************************************/
long long update_model(unsigned char* segmentmap,
                  pixel* c_image,pixel* init_model,
                  int row,int col,int time_sample,
                  int numsamples,int frames,long long seg_count)
{
  int map_index;
  int neigh_index;
  int x,y;
  int time_stamp;

  int n;
  // check if each pixel is background and udpdate
  // model of pixel and its neighbour if pixel gets 
  // selected to get updated
  for(int a = 0;a < row;a++)
  {
   for(int b = 0;b < col; b++)
    {
      map_index = a*col + b;

      time_stamp = rand()%time_sample;

      if(time_stamp  == 0)
      {
       // if pixel is chosen for update

        if(segmentmap[map_index] == 0)
        {
         seg_count += 1;
         // updatepixelmodel(c_image,init_model,map_index,numsamples);
         int global_model_index;
         int modelsample_index;
         modelsample_index = rand()%numsamples;
         global_model_index = map_index*numsamples + modelsample_index;
         init_model[global_model_index] = c_image[map_index];

          // choose neighbouring pixel to get updated
          if(a > 0 && a < row-1 && b > 0 && b < col-1)
          { 
            n =  rand()%9;
            x = n/3 - 1;
            y = n%3 - 1;
          
            neigh_index = (x+a)*col + (y+b);
            //update neighbouring pixel
            int neigh_model_index = neigh_index*numsamples + modelsample_index;
            init_model[neigh_model_index] = c_image[map_index];
          }
         }
       }
     }
   
   }
  return(seg_count);
}



/*********************************************************************************************
  
  Function to find distance between pixel and and its corresponding model

******************************************************************************************/
int find_euclidean_distance(unsigned char b1,unsigned char g1,
                            unsigned char r1,unsigned char b2,
                            unsigned char g2,unsigned char r2)
{
  //cout << (abs((b1-b2)*(b1-b2)) + abs((g1-g2)*(g1-g2)) + abs((r1-r2)*(r1-r2))) << endl;
  return(abs((b1-b2)) + abs((g1-g2)) + abs((r1-r2)));
}

pixel* updatepixelmodel(pixel* c_frame,pixel* c_mod,int index,int size_model)
{
     int modelsample_index;
     int global_model_index;

     modelsample_index = rand()%size_model;
     global_model_index = index*size_model + modelsample_index;
     c_mod[global_model_index] = c_frame[index];
     
  
return c_mod;
}

/***********************************************************************************************
*
* Function to update model without pixel struct
*
************************************************************************************************/
pixel* Updatepixelmodel(unsigned char* c_frame,pixel* c_mod,int index,int size_model)
{
     int modelsample_index;
     int global_model_index;

     modelsample_index = rand()%size_model;
     global_model_index = index*size_model + modelsample_index;
     c_mod[global_model_index].b = c_frame[index*3];
     c_mod[global_model_index].g = c_frame[(index*3)+1];
     c_mod[global_model_index].r = c_frame[(index*3)+2];

  return c_mod;
}
 
/********************************************************************************************
 
  Function to count number of samples of pixel model that lie within radius threshold


**********************************************************************************************/ 

int count_samples(pixel* pointer,pixel* curr_model,
                  int threshold,int sample,int rad,int mod_index,int frame)
{
 int dist = 0;
 int count = 0;
 int c = 4.5*rad;
 //cout << c << endl;
 int index = 0;
 //cout << mod_index << endl;
 while(count <= threshold &&  index < sample)
   {
     
     dist = find_euclidean_distance(pointer->b,
                                    pointer->g,
                                    pointer->r,
                                    curr_model[mod_index].b,
                                    curr_model[mod_index].g,
                                    curr_model[mod_index].r);
    
       
     if(dist <= c)
     {
       count++;
     }
     

     mod_index += 1;
     index++;
     
     //cout << count << endl;
   }

if(count >= threshold)
 
  return true;

else
 
  return false;
}

/**********************************************************************************************
 *
 * count samples without pixel struct
 *
 * ********************************************************************************************/
 int Count_samples(unsigned char* pointer,pixel* curr_model,
                  int threshold,int sample,int rad,int mod_index)
{
 
 int dist = 0;
 int count = 0;
 int index = 0;

 while(count <= threshold &&  index < sample)
   {

     mod_index =+ 1;
     dist = find_euclidean_distance(pointer[0],
                                    pointer[1],
                                    pointer[2],
                                    curr_model[mod_index].r,
                                    curr_model[mod_index].g,
                                    curr_model[mod_index].b);

     if(dist < rad*rad)
     {
       count++;
     }

     index++;
   }

if(count >= threshold)

  return true;

else

  return false;
}

/************************************************************************************************
 *
 * initialize the model
 *
 * *********************************************************************************************/

void initialize_model(pixel* first_frame,pixel* init_model,
                      int rows, int cols,int samples)
{

  int x;
  int y;
  int count_0 = 0;
  int count_1 = 0;
  int count_neg_1 = 0;
  pixel value;
  int r = 0;
  int g = 0;
  int b = 0;
  pixel value_plus_noise;

  for(int i = 0; i < rows;i++)
  {
   for(int j = 0;j < cols;j++)
   {

      init_model[(i*samples*cols) + (j*samples)+0] = first_frame[i*cols + j];
      init_model[(i*samples*cols) + (j*samples)+1] = first_frame[i*cols + j];

      for(int index = 2;index < samples;index++)
       {
        // x = rand()%3 - 1;
         value = first_frame[i*cols + j];
       //  y = rand()%3 - 1;       
         /*if(x_offset == 0 && y_offset ==0)
         {
           count_0 +=2;
         }
         else if(x_offset == 0 || y_offset == 0)
         {
          count_0 +=1;
         }

         if(x_offset == 1 && y_offset ==1)
         {
           count_1 +=2;
         }
         else if(x_offset == 1 || y_offset == 1)
         {
          count_1 +=1;
         }

         if(x_offset == -1 && y_offset == -1)
         {
           count_neg_1 +=2;
         }
         else if(x_offset == -1 || y_offset == -1)
         {
          count_neg_1 +=1;
         }*/


         /*while(0 > (x+j) || (x+j) > cols || 0 > y+i || (y+i) > rows)
         {
             x = rand()%3-1;
             y = rand()%3-1;
         }

         
         value = first_frame[(i+y)*cols + (j+x)];

         init_model[(i*samples*cols) + (j*samples) + index] = value;*/
 
         
         b = value.b + rand() % 20 - 10; 
         g = value.g + rand() % 20 - 10;
         r = value.r + rand() % 20 - 10;
         if(r < 0) 
         { 
            r = 0;
            value_plus_noise.r = r;
         }
         else
         {
            value_plus_noise.r = r;
         }
         if(g < 0) 
         {
           g = 0;
           value_plus_noise.g = g; }
         else
         {
           value_plus_noise.g = g;}
         if (b < 0) 
         { 
           b = 0;
           value_plus_noise.b = b;
         }
         else
         {
           value_plus_noise.b = b;
         }
         if (r > 255) { value_plus_noise.r = 255; }
         else{value_plus_noise.r= r;}
         if (g > 255) { value_plus_noise.g = 255; }
         else{value_plus_noise.g = g;}
         if (b > 255) { value_plus_noise.b = 255; }
         else{value_plus_noise.b = b;}
    
         init_model[(i*samples*cols) + (j*samples)+index] = value_plus_noise;
               
      }
 
    } 

  }

 // cout << count_neg_1 << endl;
 // cout << count_0 << endl;
 // cout << count_1 << endl;
  
}

/********************************************************************************************
 *
 * initialize model without use of pixel struct
 *
 * ******************************************************************************************/        
void Initialize_model(unsigned char* first_frame,pixel* init_model,
                      int rows, int cols,int samples)

{
  int x;
  int y;

  unsigned char value,value1,value2;
  cout << rows << endl;
  cout << cols << endl;
  for(int i = 1; i < rows;i++)
  {

   for(int j = 1;j < cols;j++)
   {


       for(int index = 0;index < samples;index++)
       {


         x = rand()%3-1;
         y  = rand()%3-1;



          while(0 > (x+j) || (x+j) > cols || 0 > y+i || (y+i) > rows)
           {
             x = rand()%3-1;
             y = rand()%3-1;
           }


         value = first_frame[((i+y)*cols + (j+x))*3];
         value1 = first_frame[((i+y)*cols + (j+x))*3+1];
         value2 = first_frame[((i+y)*cols + (j+x))*3+2];

         init_model(i,j,index).b = value;
         init_model(i,j,index).g = value1;
         init_model(i,j,index).r = value2;
      }
    }
  }
}
                        
#endif
