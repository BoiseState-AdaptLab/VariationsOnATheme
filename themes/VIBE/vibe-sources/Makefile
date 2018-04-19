LIBS_OPENCV = `$(PREFIX)pkg-config --libs opencv`
INCLUDE_OPENCV = `$(PREFIX)pkg-config --cflags opencv`

default: 
	gcc -std=c99 -O3 -Wall -pedantic -Wno-unused-function -Wno-unused-parameter -Wno-deprecated -Wno-deprecated-declarations -Wno-sign-compare -Wno-unused-but-set-parameter -c vibe-background-sequential.c
	gcc -o main -std=c99 -O3 -Wall -Werror -pedantic your-main-file-sequential.c vibe-background-sequential.o
	g++ -c ../../common/Configuration.cpp -o Configuration.o
	g++ -c ../../common/Measurements.cpp -o Measurements.o
	g++ -o main-opencv -O3 -Wall -pedantic $(INCLUDE_OPENCV) main-opencv.cpp vibe-background-sequential.o Configuration.o Measurements.o $(LIBS_OPENCV)

clean:
	rm -f *.o
