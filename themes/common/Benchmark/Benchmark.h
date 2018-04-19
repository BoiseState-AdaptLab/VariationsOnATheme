/**
    This file is part of VariationsDev.

    VariationsDev is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    VariationsDev is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with VariationsDev. If not, see <http://www.gnu.org/licenses/>.
 **/

#ifndef BENCHMARK_BENCHMARK_H
#define BENCHMARK_BENCHMARK_H

//#include <memory>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <omp.h>

#include <Configuration.h>
#include <Measurements.h>

using namespace std;

typedef double DType;       // TODO: Maybe make this a template parameter later...

class Benchmark {
public:
    Benchmark() {}
    virtual ~Benchmark() {}

    bool valid();
    double runTime();
    double speedup();
    string& name();

    function<void(DType**, DType**, DType*, Configuration&)>& execFunction();
    function<void(DType**, DType**, DType*, Configuration&)>& evalFunction();
    function<bool(DType**, DType**, Configuration&, vector<int>&)> compFunction();

    void start();
    void stop();

    bool setNumThreads(int nThreads);

    virtual void init() = 0;
    virtual void finish() = 0;
    virtual string error() = 0;

    void run();
    void report();

protected:
    Configuration _config;
    Measurements _meas;

    bool _status = false;
    bool _verify = false;

    double _startTime = 0.0;
    double _stopTime = 0.0;
    double _runTime = 0.0;
    double _evalTime = 0.0;

    DType **_inputData = nullptr;
    DType **_outputData = nullptr;
    DType **_verifyData = nullptr;

    int _nRows = 0;
    int _nCols = 1;

    string _name = "";

    vector<int> _errorLoc;

    function<void(DType**, DType**, DType*, Configuration&)> _execFxn;
    function<void(DType**, DType**, DType*, Configuration&)> _evalFxn;
    function<bool(DType**, DType**, Configuration&, vector<int>&)> _compFxn;
};


#endif //BENCHMARK_BENCHMARK_H
