CC = g++
EXEC = blurSEQ
OBJS = timer.o
H_FILE = timer.h
MATHFLAG = -lm
FLAGS = -O3 -fopenmp
SEQFLAGS = -O3

all: $(EXEC)

blurSEQ: BlurRoberts-NaiveSerial-CPP.cpp $(OBJS) $(H_FILE)
	$(CC) $(FLAGS) $(MATHFLAG) -o $@ BlurRoberts-NaiveSerial-CPP.cpp $(OBJS)

# blurOMP: BlurRoberts-NaiveOpenMP-C.c $(OBJS) $(H_FILE)
# 	$(CC) $(FLAGS) $(MATHFLAG) -o $@ BlurRoberts-NaiveOpenMP-C.c $(OBJS)
#
# blurSmart: BlurRoberts-NaiveSmartFuse-C.c $(OBJS) $(H_FILE)
# 	$(CC) $(FLAGS) $(MATHFLAG) -o $@ BlurRoberts-NaiveSmartFuse-C.c $(OBJS)

timer.o: timer.c
	$(CC) $(SEQFLAGS) -o $@ -c timer.c

clean:
	rm -f *.o $(OBJS) $(EXEC)
