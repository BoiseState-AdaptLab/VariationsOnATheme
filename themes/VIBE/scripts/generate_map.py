#!/bin/python3

# Will generate the foreground counter map for given images. 
# Results in:
#   pixel_count_3.png -> foreground counter map for our VIBE implementation
#   pixel_count_1.png -> first foreground counter map for original VIBE
#   pixel_count_2.png -> second foregounr counter map for original VIBE

import sys
from subprocess import call
import os

if len(sys.argv) != 6:
    print('Usage: option dir1 dir2 dir3 dir3')
    sys.exit(1)

cur_dir = os.getcwd()
print('Debug: {}'.format(cur_dir))

option = int(sys.argv[1])

dir_1 = sys.argv[2]
dir_2 = sys.argv[3]
dir_3 = sys.argv[4]
dir_4 = sys.argv[5]

# this will break, since the name of my images is something like this:
# output_idx_frameNum..
if option == 1:
    for idy in range(1, 33):
        call('./vibe_generate_map -f {}/output_{}_350.png'.format(dir_1, idy), shell=True)
    # my1
    call('cp pixel_count.png pixel_count_3.png', shell=True)
elif option == 2:
    for idy in range(1, 33):
        call('./vibe_generate_map -f {}/output_{}_350.png'.format(dir_2, idy), shell=True)
    # v1
    call('cp pixel_count.png pixel_count_1.png', shell=True)
elif option == 3:
    for idy in range(1, 33):
        call('./vibe_generate_map -f {}/output_{}_350.png'.format(dir_3, idy), shell=True)
    # v2
    call('cp pixel_count.png pixel_count_2.png', shell=True)
elif option == 4:
    for idy in range(1, 33):
        call('./vibe_generate_map -f {}/output_{}_350.png'.format(dir_4, idy), shell=True)
    # my2
    call('cp pixel_count.png pixel_count_4.png', shell=True)

