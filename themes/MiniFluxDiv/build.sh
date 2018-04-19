g++ -std=c++11 -g -I../common -I../common/Benchmark -I../common/ISLInt -I/usr/include -I/usr/local/include -fopenmp -O4 ../common/Measurements.cpp ../common/Configuration.cpp ../common/Benchmark/Benchmark.cpp MiniFluxBenchmark.cpp main.cpp -o $1

