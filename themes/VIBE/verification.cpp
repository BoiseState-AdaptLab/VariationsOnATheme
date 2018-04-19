#include "verification.h"


using namespace std;

int collect_verification_statistics(unsigned char* map_gpu,
                                     unsigned char* map_serial,
                                     int rows,int cols)
{
    
    int count = 0;
    for(int i =0;i < rows*cols;i++)
    {
       if(map_gpu[i] - map_serial[i] != 0)
       {
          count += 1;
       }
    
    }

  return count;   

}       
