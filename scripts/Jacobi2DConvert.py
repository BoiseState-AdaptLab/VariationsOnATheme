#!/usr/bin/env python3

# Author:  Eddie C. Davis
# Program: Jacobi2DConvert
# Project: VariationsDev
# File:    Jacobi2DConvert.py
# Purpose: Convert CSV output from Jacobi2DRun.py into tab separated file used by graph generator.

import os
import sys
import csv
import traceback as tb

def main():
    args = sys.argv[1:]
    nArgs = len(args)

    if nArgs > 0:
        fname = args[0]
        if '-h' in fname:
            print('usage: Jacobi2DConvert <filename>.csv')
            return
        else:
            file = open(fname)
    else:
        file = sys.stdin

    reader = csv.reader(file, delimiter=',')
    header = next(reader)
    schedules = {}

    cols = {}
    for col in header:
        cols[col] = len(cols)

    outs = {}

    for row in reader:
        # Skip error rows for now...
        if len(row) > 0:
            if 'FAILED' not in row[0]:
                schedule = row[cols['Schedule']]
                if schedule not in schedules:
                    schedules[schedule] = len(schedules) + 1

                threads = int(row[cols['nThreads']])
                if threads not in outs:
                    outs[threads] = {}

                time = row[cols['Time']]
                if float(time) > 0:
                    schedNum = schedules[schedule]
                    times = outs[threads]
                    if schedNum in times:
                        prevTime = outs[threads][schedNum]
                        if time < prevTime:
                            outs[threads][schedNum] = time
                    else:
                        outs[threads][schedNum] = time

    if nArgs > 0:
        file.close()

    # Print output...
    schedMax = len(schedules)
    threadCounts = sorted(outs.keys())

    # position\tthreads\tnaiveTime\tNUMTime(9ofthese)
    header = "position\tthreads\tnaiveTime\t"
    schedNum = 2
    while schedNum <= schedMax:
        header += str(schedNum) + "Time\t"
        schedNum += 1

    print(header)

    for threadCount in threadCounts:
        item = outs[threadCount]
        schedNum = 1
        line = "%d\t%d\t%s\t" % (threadCount, threadCount, item[schedNum])

        schedNum += 1
        while schedNum <= schedMax:
            if schedNum in item:
                line += item[schedNum]

            line += "\t"
            schedNum += 1

        print(line)

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")

    except Exception as e:
        print('Unexpected Exception: ' + str(e))
        tb.print_exc()
        os._exit(1)

