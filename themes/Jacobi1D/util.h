/*****************************************************************************
 * util.h
 *
 * This file contains headers for verification, memory allocation,
 * memory initalization, and checking (for command line options) functions.
 *
 * It also has macros for stencil computation, type definition, and 
 * instrumentation.
 *
 * The function are:
 *
 *  bool verifyResultJacobi1D(real*,Configuration&);
 *  bool verifyResultJacobi1DCuda(real*,Configuration&);
 *  bool allocateSpace(real**,Configuration&);
 *  void initializeSpace(real*,Configuration&);
 *  bool checkCommandLineOptions(Configuration&);
 *
 * Example usage of functions:
 *
 *  verifyResultJacobi1D(pointerToData,configuration);
 *   if verification is successful
 *     return true
 *   else
 *     return false
 *
 *  verifyResultJacobi1DCuda(pointerToData,configuration);
 *   if	verification is successful
 *     return true
 *   else
 *     return false
 *
 *  allocateSpace(&pointerTodata,configuration);
 *   if allocation is successful
 *     return true
 *   else
 *     return false
 *
 *  initializeSpace(pointerTodata,configuration);
 *
 *  checkCommandLineOptions(configuration);
 *   if checking is successful
 *    return true
 *   else
 *    return false
 *
*****************************************************************************/

#include <stdio.h>
#include <omp.h>
#include "../common/Configuration.h"

/*****************************************************************************
 * This macro is used for selecting the data type in the benchmarks
 *
 *
*****************************************************************************/
#ifdef DATATYPEDOUBLE
   typedef double real;
#else
   typedef float real;
#endif

/*****************************************************************************
 * Function headers
 * See the util.cpp for the functions
 *
*****************************************************************************/
bool verifyResultJacobi1D(real* result, Configuration& configuration);
bool verifyResultJacobi1DCuda(real* result, Configuration& configuration);
bool allocateSpace(real** data, Configuration& configuration);
void initializeSpace(real* data, Configuration& configuration);
bool checkCommandLineOptions(Configuration &configuration);

/*****************************************************************************
 *
 * Macros are used for the   calculations
 *
 *
*****************************************************************************/
#define space(ptr,t,i)  (ptr)[ ((t) & 1) * (Nx + 2) +                 \
                                (i) ]
#define stencil(ptr,t,i) space(ptr,t,i) =  ( space(ptr, t-1, i-1)  +  \
                                             space(ptr, t-1, i  )  +  \
                                             space(ptr, t-1, i+1) ) / 3


/*****************************************************************************
 *   The instrumentation macros are for CUDA benchmarks shown
 *   below:
 *
 * COUNT_MACRO_INIT()
 * COUNT_MACRO_KERNEL_CALL()
 * COUNT_MACRO_RESOURCE_ITERS(x)
 * COUNT_MACRO_BYTES_FROM_GLOBAL(x)
 * COUNT_MACRO_BYTES_TO_GLOBAL(x) 
 * COUNT_MACRO_NUMTILES(x)
 * COUNT_MACRO_UNIQUE_WAVEFRONT_SIZE(x) 
 * COUNT_MACRO_PRINT() 
*****************************************************************************/
#ifdef COUNT
#define COUNT_MACRO_INIT() unsigned long long int resourceIters = 0,     \
bytesFromGlobal = 0, bytesToGlobal = 0, countTiles = 0;                  \
int kernelCalls = 0, uniqueWavefronts[10]={0};
#define COUNT_MACRO_KERNEL_CALL() kernelCalls +=1
#define COUNT_MACRO_RESOURCE_ITERS(x) resourceIters +=                   \
((unsigned long long int)x)
#define COUNT_MACRO_BYTES_FROM_GLOBAL(x) bytesFromGlobal +=              \
((unsigned long long int)x)
#define COUNT_MACRO_BYTES_TO_GLOBAL(x) bytesToGlobal +=                  \
((unsigned long long int)x)
#define COUNT_MACRO_NUMTILES(x) countTiles += ((unsigned long long int)x)
#define COUNT_MACRO_UNIQUE_WAVEFRONT_SIZE(x)                             \
do {char flag = false;                                                   \
 for (int i = 0; i < 10;  i++ ){                                         \
   if(x == uniqueWavefronts[i]){                                         \
     flag = true;                                                        \
   }                                                                     \
 }                                                                       \
  if(flag == false){                                                     \
    for (int i = 0; i < 10;  i++ ){                                      \
      if(uniqueWavefronts[i] == 0){                                      \
        uniqueWavefronts[i] = x;                                         \
        break;                                                           \
      }                                                                  \
    }                                                                    \
  }                                                                      \
}while(0);
#define COUNT_MACRO_PRINT()                                              \
printf("KernelCalls:%d,ResourceIters:%llu,bytesFromGlobal:%llu,\
bytesToGlobal:%llu,countTiles:%llu,",kernelCalls,resourceIters,          \
bytesFromGlobal,bytesToGlobal,countTiles);                               \
do {                                                                     \
int counter = 0;                                                         \
printf("uniqueWavefronts:[");                                            \
while(uniqueWavefronts[counter]!=0){                                     \
  printf("%d",uniqueWavefronts[counter]);                                \
  if(uniqueWavefronts[counter+1]!=0){                                    \
    printf(",");                                                         \
  }                                                                      \
  counter++;                                                             \
}                                                                        \
printf("]");                                                             \
}while(0)

#else
#define COUNT_MACRO_INIT()
#define COUNT_MACRO_KERNEL_CALL()
#define COUNT_MACRO_RESOURCE_ITERS(x)
#define COUNT_MACRO_NUMTILES(x)
#define COUNT_MACRO_BYTES_FROM_GLOBAL(x)
#define COUNT_MACRO_BYTES_TO_GLOBAL(x)
#define COUNT_MACRO_NUMTILES(x)
#define COUNT_MACRO_UNIQUE_WAVEFRONT_SIZE(x)
#define COUNT_MACRO_PRINT()
#endif

