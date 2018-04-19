CPPFLAGS = -std=c++0x -Wno-write-strings

test: config_test measure_test valgrind

config_test: Configuration_test.cpp Configuration.cpp Configuration.h
	g++ -g -O0 -c ${CPPFLAGS} Configuration.cpp -o Configuration.o
	g++ -g -O0 -c ${CPPFLAGS} Configuration_test.cpp -o Configuration_test.o
	g++ -g -O0 ${CPPFLAGS} Configuration.o Configuration_test.o -o Configuration_test
	./Configuration_test

measure_test: Measurements_test.cpp Measurements.cpp Measurements.h
	g++ -g -O0 -c ${CPPFLAGS} Measurements.cpp -o Measurements.o
	g++ -g -O0 -c ${CPPFLAGS} Measurements_test.cpp -o Measurements_test.o
	g++ -g -O0 ${CPPFLAGS} Measurements.o Measurements_test.o -o Measurements_test
	./Measurements_test

valgrind:./Configuration_test ./Measurements_test
	valgrind -q --leak-check=yes ./Configuration_test ./Measurements_test

clean:
	rm *.o Configuration_test Measurements_test

