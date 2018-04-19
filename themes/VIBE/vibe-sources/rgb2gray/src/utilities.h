#include<iostream>
#include<stdio.h>
#include<string.h>
#include<unistd.h>
#include<stdlib.h>

int parseInt( char* string ){
   return (int) strtol( string, NULL, 10 );
  }

typedef struct {
  // common parameters
  int verify;
  char filename[512];
  int blocksize_x;
  int blocksize_y;
  int tilesize_x;
  int tilesize_y;
  int average_factor;
} Params;

int parseCmdLineArgs(Params *cmdLineArgs, int argc, char* argv[]){

 // int s;
 // set default values
 cmdLineArgs->verify = false;
 cmdLineArgs->blocksize_x = 16;
 cmdLineArgs->blocksize_y = 16;
 cmdLineArgs->filename[0] = '\0';
 cmdLineArgs->tilesize_x = 16;
 cmdLineArgs->tilesize_y = 16;
 cmdLineArgs->average_factor = 1;
  // process incoming
  char c;
  while ((c = getopt (argc, argv, "hvf:x:y:t:q:a:")) != -1){
    switch( c ) {

      case 'h': // help
        printf("usage: %s\n"
                  "-h\tusage help, this dialogue\n"
                  "-v\tverify output\n", argv[0]);
           return 1;

      case 'v': // verify;
         cmdLineArgs->verify = true;
         break;

      case 'f'://filepath name
         strncpy(cmdLineArgs->filename,optarg,512);
         break;
   
      case 'x'://blocksize_x
         cmdLineArgs->blocksize_x = parseInt(optarg);
         break;
    
      case 'y'://blocksize_y
         cmdLineArgs->blocksize_y = parseInt(optarg);
         break;
      
      case 't'://tilesixe in x
         cmdLineArgs->tilesize_x =parseInt(optarg);
         break;

      case 'q'://tilesize in y
         cmdLineArgs->tilesize_y = parseInt(optarg);
         break;
      
      case 'a'://averaging factor
         cmdLineArgs->average_factor = parseInt(optarg);
         break;

      case ':':
        return 0;

      default:
        fprintf(stderr, "default\n");
        return(0);
		 }
		}
    if(cmdLineArgs->filename[0] == '\0'){
      fprintf(stderr,"-f is a required argument\n");
      return 0;
    }
    return 1;
}
