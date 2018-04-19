#!/bin/python3

from subprocess import call
import os

files = ['histogram.out', 'diff_map_*', 'experiment_log', 'gmon.out', 'my1', 'my2', 'pixel_count_1.png', 'pixel_count_2.png', 'pixel_count_3.png', 'pixel_count_4.png', 'v1', 'v2']
assert len(files) == 12

num = 3

print('Starting...')
for i in range(num):
    call('./experiment.py', shell=True)
    print('Before: {}'.format(os.getcwd()))
    os.chdir('..')
    print('After: {}'.format(os.getcwd()))
    call('rm -f pixel_count.png', shell=True) 
    new_dir = 'experiment_02_26_{}'.format(i)
    call('mkdir ' + new_dir, shell=True)
    for f in files:
        call('mv {} {}'.format(f, new_dir), shell=True)
    os.chdir('scripts')
    call('./clean.py', shell=True)
    print('Finished {} experiment'.format(i))
print('Done...')
