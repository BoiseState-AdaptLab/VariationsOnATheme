#!/usr/bin/env python3
#
#   This file is part of MiniFluxDiv.

#   MiniFluxDiv is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.

#   MiniFluxDiv is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with MiniFluxDiv. If not, see <http://www.gnu.org/licenses/>.
#

import os
import sys
import math
import subprocess as sub
import traceback as tb
import argparse as ap

DEBUG = False #True

def ncpus():
    import multiprocessing
    return multiprocessing.cpu_count()

def run(args, vars={}, stream=None):
    output = ''

    for key in vars:
        os.environ[key] = vars[key]

    try:
        data = sub.check_output(args, env=os.environ, stderr=sub.STDOUT)
        output = data.decode()
    except Exception as e:
        eprint(str(e))

    if stream is not None:
        stream.write(output)

    return output

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parseArgs():
    parser = ap.ArgumentParser('runMiniFluxDiv', description='Execute miniFluxDiv test cases.')
    parser.add_argument('-e', '--exec', default='', help='Executable to run.')
    parser.add_argument('-l', '--legend', default='', help='Chart legend for output table.')
    parser.add_argument('-b', '--boxes', default='32', help='List of box sizes.')
    parser.add_argument('-c', '--cells', default='128', help='List of cell sizes.')
    parser.add_argument('-p', '--threads', default='1', help='List of thread counts.')
    parser.add_argument('-n', '--runs', default='3', help='Number of runs.')
    parser.add_argument('-d', '--delim', default=',', help='List delimiter.')
    parser.add_argument('-v', '--verify', default='0', help='Perform verification.')

    return parser.parse_known_args()

def main():
    if len(sys.argv) < 2:
        sys.argv.append('-h')

    (args, unknowns) = parseArgs()

    executable = args.exec
    delim = args.delim

    boxStr = args.boxes.split(delim)
    boxList = [int(val) for val in boxStr]

    cellStr = args.cells.split(delim)
    cellList = [int(val) for val in cellStr]

    maxThreads = ncpus()
    thrStr = args.threads.split(delim)

    thrList = []
    for val in thrStr:
        val = int(val)
        if val > 0 and val <= maxThreads:
            thrList.append(val)

    nRuns = int(args.runs)
    doVerify = int(args.verify) > 0

    program = executable.split('/')[-1]
    vars = {'OMP_SCHEDULE': 'static'}

    legend = args.legend
    if len(legend) < 1:
        legend = program.replace('miniFluxdiv-', '')

    header = 'Program,Legend,nBoxes,nCells,nThreads,'
    for i in range(nRuns):
        header += 'Time' + str(i) + ','
    header += 'MeanTime,MedianTime,StdevTime'

    print(header)

    nPairs = len(boxList)
    for threads in thrList:
        for i in range(nPairs):
            nBoxes = boxList[i]
            nCells = cellList[i]

            args = [executable, '-B', '%d' % nBoxes, '-C', '%d' % nCells, '-p', '%d' % threads]
            if doVerify and i == 0:     # Only verify first run...
                args.append('-v')

            cmd = ' '.join(args)

            times = []
            tsum = 0.0
            tsumsq = 0.0

            for i in range(nRuns):
                output = run(args, vars)
                isValid = len(output) > 0

                if isValid:
                    lines = output.rstrip().split("\n")
                    line = lines[-1].rstrip(',')
                    items = line.split(',')

                    dct = {}
                    for item in items:
                        (key, val) = item.split(':')
                        dct[key] = val

                    if doVerify and 'verification' in dct and 'FAIL' in dct['verification']:
                        print("VERIFICATION FAILED: on '%s' with '%s'" % (cmd, lines[0]))
                        isValid = False
                        break
                    else:
                        rtime = float(dct['RunTime'])
                        times.append(rtime)
                        tsum += rtime
                        tsumsq += (rtime * rtime)

            if isValid:
                tList = ['%lf' % time for time in times]

                # Calc mean & stdev
                fRuns = float(nRuns)
                mean = tsum / fRuns
                stdev = math.sqrt((tsumsq - (tsum * tsum) / fRuns) / float(nRuns - 1))

                # Calc median
                times = sorted(times)
                mid = int(nRuns / 2)
                if nRuns % 2 == 1:
                    median = times[mid]
                else:
                    median = (times[mid] + times[mid + 1]) / 2.0

                line = '%s,%s,%d,%d,%d,%s,%lf,%lf,%lf' % (program, legend, nBoxes, nCells, threads, ','.join(tList), mean, median, stdev)
                print(line)
                sys.stdout.flush()

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")

    except Exception as e:
        print('Unexpected Exception: ' + str(e))
        tb.print_exc()
        os._exit(1)

