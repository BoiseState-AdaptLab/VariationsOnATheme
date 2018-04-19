#!/bin/python3

# Will erase the output files from the experiments.
# Author: Aza Tulepbergenov, Spring 2017

from subprocess import call
import os

cur_dir = os.getcwd()
os.chdir(cur_dir + '/..')

call('make clean', shell=True)

files = ['histogram.out', 'diff_map_*', 'my1', 'my2', 'pixel_count*', 'v1', 'v2', 'vibe-sources/main', 'vibe-sources/main-opencv']

for f in files:
    print('Deleting {}'.format(f))
    call('rm -rf {}'.format(f), shell=True)

print('Done...')

