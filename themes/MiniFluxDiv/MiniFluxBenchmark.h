/**
    This file is part of MiniFluxDiv.

    MiniFluxDiv is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    MiniFluxDiv is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MiniFluxDiv. If not, see <http://www.gnu.org/licenses/>.
 **/

#ifndef BENCHMARK_MINIFLUXBENCHMARK_H
#define BENCHMARK_MINIFLUXBENCHMARK_H

#include "Benchmark.h"
//#include "ISLFunctions.h"

//<Include>
#include "miniFluxDiv-seriesHalide-gen.h"

#ifndef dx
#define dx 0.5
#define factor1 (1.0/12.0)
#define factor2 2.0
#endif

typedef DType Real;

MiniFluxDivData *mini_flux_div_init(Configuration& config);
void mini_flux_div_truth(Real ** old_boxes, Real** new_boxes, Real *timer, Configuration& config);
void mini_flux_div_lc(Real** old_boxes, Real** new_boxes, Real *timer, Configuration& config);
bool mini_flux_div_comp(Real** new_boxes, Real** ref_boxes, Configuration& config, vector<int>& loc);

class MiniFluxBenchmark : public Benchmark {
public:
    MiniFluxBenchmark();
    MiniFluxBenchmark(int argc, char *argv[]);
    virtual ~MiniFluxBenchmark();

    void init();
    void finish();
    string error();
    void toCSV();
};

#endif //BENCHMARK_MINIFLUXBENCHMARK_H
