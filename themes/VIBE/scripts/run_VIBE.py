#!/bin/python3
# Author: Aza Tulepbergenov

# Will run the VIBE algorithm on the given video. Results in 600+ frames in each of the 3 newly created folders. 

import sys
from subprocess import call
import os

cur_dir = os.getcwd()
print('Debug: {}'.format(cur_dir))

idx = 0

if len(sys.argv) != 6:
    print('Usage: dir1 dir2 dir3 video')
    sys.exit(1)

dir_name_1 = sys.argv[1]
dir_name_2 = sys.argv[2]
dir_name_3 = sys.argv[3]
dir_name_4 = sys.argv[4]

video = sys.argv[5]

call('mkdir {}'.format(dir_name_1), shell=True)
call('mkdir {}'.format(dir_name_2), shell=True)
call('mkdir {}'.format(dir_name_3), shell=True)
call('mkdir {}'.format(dir_name_4), shell=True)

seed_offset = 32 # offset to ensure that the 2 runs of my1 vs my2 and v1 vs v2 actually have differences...

# radius of sphere, matching samples etc. are being set inside the code via Configuration object.
for i in range(1, 33):
    call('./VIBE-NaiveSerial-CPP -f {} -o {}/output_{} -g {}'.format(video, dir_name_1, i, i), shell=True)

for i in range(1, 33):
    call('./vibe-sources/main-opencv -f {} -o {}/output_{} -g {}'.format(video, dir_name_2, i, i), shell=True)

for i in range(1, 33):
    call('./vibe-sources/main-opencv -f {} -o {}/output_{} -g {}'.format(video, dir_name_3, i, i + seed_offset), shell=True)

for i in range(1, 33):
    call('./VIBE-NaiveSerial-CPP -f {} -o {}/output_{} -g {}'.format(video, dir_name_4, i, i + seed_offset), shell=True)


