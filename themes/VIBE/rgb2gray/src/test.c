/************************************************
 * lets test some stuff.
************************************************/
#include "utilities.h"
#include <string.h>
#include <stdlib.h>


int testVerifyParam(int argc, char* argv[]);
int testFilepath(int argc,char* argv[]);
int testblocksize(int argc,char* argv[]);

int testVerify(){

  int failed = 0;
  //  - Command line parsing
  Params cmdLineArgs;
  int argc = 1;
  char** argv;

  optind = 0;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);
  if(cmdLineArgs.verify){
    failed = 1;
    fprintf(stderr,"TEST FAILED testVerify(1)\n");
  }

  argc = 4;
  argv = (char**)malloc(sizeof(char*)*5);
  argv[0] = (char*)"tester";
  argv[1] = (char*)"-v";
  argv[2] = (char*)"-f";
  argv[3] = (char*)"testfile";
  optind = 0;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);
  if(!cmdLineArgs.verify){
    failed = 1; 
    fprintf(stderr,"TEST FAILED testVerify(2)\n");
  }
  return !failed;
}

int testFilepath(){
 
  int failed = 0; 
  Params cmdLineArgs;
  int argc = 0;
  char** argv;  

  argv = (char**)malloc(sizeof(char*)*5);

  optind = 0;
  if(parseCmdLineArgs(&cmdLineArgs,argc,argv)){
    fprintf(stderr,"TEST FAILED testFilepath(1)\n");
    failed = 1;
  }

  argc = 3;
  argv[0] = (char*)"tester";
  argv[1] = (char*)"-f";
  argv[2] = (char*)"testfile";
  

  optind = 0;
  if(parseCmdLineArgs(&cmdLineArgs,argc,argv)){
    printf("TEST:%s\n",cmdLineArgs.filename);
  }else{
    fprintf(stderr,"TEST FAILED testFilepath(2)\n");
    failed = 1;
  }
  return !failed;
}   

int testblocksize(){
  int failed = 0;
  //  - Command line parsing
  Params cmdLineArgs;
  int argc = 1;
  char** argv;
  argv = (char**)malloc(sizeof(char*)*7);
  optind = 0;
  if(parseCmdLineArgs(&cmdLineArgs,argc,argv)){
    fprintf(stderr,"Test Failed testblocksize(1)\n");
    failed = 1;
  }  
  argc = 5; 
  //argv = (char**)malloc(sizeof(char*)*5);
  argv[0] = (char*)"tester";
  argv[1] = (char*)"-f";
  argv[2] = (char*)"sdf";
  argv[3] = (char*)"-y";
  argv[4] = (char*)"16";
  optind = 0;
  if(parseCmdLineArgs(&cmdLineArgs,argc,argv)){
    printf("blocksize_y %d\n",cmdLineArgs.blocksize_y);
   }else{
    fprintf(stderr,"Test Failed testblocksiz(2)\n");
   failed = 1;
  }
 return !failed;
}

int testtilesize(){
  int failed = 0;
  Params cmdLineArgs;
  int argc = 1;
  char** argv;
  argv = (char**)malloc(sizeof(char*)*7);
  optind = 0;
  if(parseCmdLineArgs(&cmdLineArgs,argc,argv)){
    fprintf(stderr,"Test Failed testtilesize(1)\n");
    failed = 1;
  }
  argc = 5;
  argv[0] = (char*)"tester";
  argv[1] = (char*)"-f";
  argv[2] = (char*)"sdf";
  argv[3] = (char*)"-q";
  argv[4] = (char*)"32";
  optind = 0;
  if(parseCmdLineArgs(&cmdLineArgs,argc,argv)){
    printf("tilesize_y %d\n",cmdLineArgs.tilesize_y);
   }else{
    fprintf(stderr,"Test Failed testtilesiz(2)\n");
   failed = 1;
  }
 return !failed;
}

int main(int argc, char* argv[]){
  
  int total_test_passed = 0;
  int total_test_count = 0;

  // check for -v not set
  total_test_count++;
  if(testVerify()){
   total_test_passed++;
  }
 

  //check for -f not set
  total_test_count++;
  if(testFilepath()){
    total_test_passed++;
  }
//check for -bx not set 
  total_test_count++;
  if(testblocksize()){
    total_test_passed++;
  }

//test for -tx not set
  total_test_count++;
  if(testtilesize()){
    total_test_passed++;
  }
 
 // End -- print result
  if(total_test_passed == total_test_count){
    printf("Success:");
  }else{
    printf("Failure:");
  }
  printf("\n%d out of %d tests passed\n",total_test_passed,total_test_count);


  }


