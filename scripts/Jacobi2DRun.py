#!/usr/bin/env python3

# Author:  Eddie C. Davis
# Program: Jacobi2DRun
# Project: VariationsDev
# File:    Jacobi2DRun.py

import os
import sys
import subprocess as sub
import traceback as tb

DEBUG = False #True

def ncpus():
    import multiprocessing
    return multiprocessing.cpu_count()


def run(args, vars={}, stream=None):
    for key in vars:
        os.environ[key] = vars[key]

    try:
        data = sub.check_output(args, env=os.environ, stderr=sub.STDOUT)
        output = data.decode()
    except: # CalledProcessError:
        output = ''

    if stream is not None:
        stream.write(output)

    return output


def main():
    executable = ''
    nX = 4096
    nY = nX
    nTimes = 100
    thrBegin = 1
    thrStep = 1
    nThreads = ncpus()
    nRuns = 5
    tileSizes = [64, 1024]

    schedules = ['baseline', 'unrolled', 'fuse_0_0_1_1', 'tile_inner_inner_serial_serial_fuse_0_0_1_1',
                 'fuse_0_0_1_1_tile_inner_inner_serial_serial', 'tile_inner_inner_serial_tile_outer_outer_fuse_0_0_1_1',
                 'tile_inner_inner_tile_outer_outer_serial_fuse_0_0_1_1', 'tile_inner_inner_tile_10_10_parallel_serial_serial',
                 'tile_outer_outer_tile_parallel_serial']

    parallelSchedules = ['baseline', 'tile_inner_inner_tile_10_10_parallel_serial_serial', 'tile_outer_outer_tile_parallel_serial']

    innerTileSchedules = ['tile_inner_inner_serial_serial_fuse_0_0_1_1', 'fuse_0_0_1_1_tile_inner_inner_serial_serial']
    innerOuterTileSchedules = ['tile_inner_inner_serial_tile_outer_outer_fuse_0_0_1_1', 'tile_inner_inner_tile_outer_outer_serial_fuse_0_0_1_1',
                               'tile_inner_inner_tile_10_10_parallel_serial_serial']
    outerTileSchedules = ['tile_outer_outer_tile_parallel_serial']

    args = sys.argv[1:]
    nArgs = len(args)
    if nArgs > 0:
        executable = args[0]
    if nArgs > 1:
        nX = nY = int(args[1])
    if nArgs > 2:
        nY = int(args[2])
    if nArgs > 3:
        nTimes = int(args[3])
    if nArgs > 4:
        thrBegin = int(args[4])
    if nArgs > 5:
        nThreads = int(args[5])
    if nArgs > 6:
        thrStep = int(args[6])
    if nArgs > 7:
        nRuns = int(args[7])
    if nArgs > 8:
        tileSizes = args[8].split(',')
        for i in range(0, len(tileSizes)):
            tileSizes[i] = int(tileSizes[i])
    if nArgs > 9:
        schedules = args[9].split(',')

    if '-h' in executable:
        print("usage: Jacobi2DRun.py <executable=%s> <Nx=%d> <Ny=%d> <T=%d> <p_begin=%d> <p_end=%d> <p_step=%d> <runs=%d> <tile_size1,tile_size2,...>[%s] <schedule1,schedule2,...>" %
              (executable, nX, nY, nTimes, thrBegin, nThreads, thrStep, nRuns, '64,1024'))
        return

    #path = os.path.dirname(executable)
    program = executable.split('/')[-1]
    vars = {'OMP_SCHEDULE': 'static'}

    print('Program,Schedule,Nx,Ny,T,nThreads,ITS,OTS,Time')

    for schedule in schedules:
        thrEnd = thrBegin
        if schedule in parallelSchedules:
            thrEnd = nThreads

        # for thrNum in range(thrBegin, thrEnd + 1):
        thrNum = thrBegin
        while thrNum <= thrEnd:
            for innerSize in tileSizes:
                # ./ Jacobi2D-NaiveParallel-OMP --Nx 4096 --Ny 4096 -T 100 -p 4 -v
                args = [executable, '--Nx', '%d' % nX, '--Ny', '%d' % nY, '-T', '%d' % nTimes, '-p', '%d' % thrNum,
                        '-s', schedule, '-v']
                mtime = 0.0

                isTiled = False
                if 'tile_inner' in schedule:
                    args.append('-i')
                    args.append(str(innerSize))
                    isTiled = True

                outerSizes = []
                if 'tile_outer' in schedule:
                    isTiled = True
                    for tileSize in tileSizes:
                        if tileSize > innerSize:
                            outerSizes.append(tileSize)
                else:
                    outerSizes.append(innerSize)

                args.append('-o')
                args.append(str(innerSize))

                for outerSize in outerSizes:
                    args[-1] = str(outerSize)
                    cmd = ' '.join(args)

                    for i in range(nRuns):
                        if DEBUG:
                            print(cmd)
                            isValid = False
                        else:
                            output = run(args, vars)
                            isValid = len(output) > 0

                            if isValid:
                                output = output.replace('schedule:0,', '')
                                lines = output.rstrip().split("\n")
                                line = lines[-1].rstrip(',')
                                items = line.split(',')

                                dct = {}
                                for item in items:
                                    (key, val) = item.split(':')
                                    dct[key] = val

                                if 'SUCCESS' in dct['verification']:
                                    rtime = float(dct['elapsedTime'])
                                    mtime += rtime
                                else:
                                    print("VERIFICATION FAILED: on '%s' with '%s'" % (cmd, lines[0]))

                    if isValid:
                        mtime /= float(nRuns)
                        print('%s,%s,%d,%d,%d,%d,%d,%d,%lf' % (program, schedule,nX, nY, nTimes, thrNum, innerSize, outerSize, mtime))
                        sys.stdout.flush()

                    if not isTiled:
                        break

                if not isTiled:
                    break

            if thrNum == 1:
                thrNum = 2
            else:
                thrNum += thrStep


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")

    except Exception as e:
        print('Unexpected Exception: ' + str(e))
        tb.print_exc()
        os._exit(1)

