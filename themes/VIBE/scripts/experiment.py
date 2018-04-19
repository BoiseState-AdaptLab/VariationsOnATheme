#!/bin/python3

# The following description is accurate as of Spring 2017. 
# Will run the experiment according to the steps outlined in the README to verify the accuracy of two VIBE versions.

import subprocess
import os
import time

print('Running the experiment...')


def call(cmd, f):
    """
    :cmd: command to run
    :f: reference to an open file
    """
    subprocess.call(cmd, shell=True, stdout=f, stderr=f)

cur_dir = os.getcwd()
os.chdir(cur_dir+'/../')
cur_dir = os.getcwd()
print('Debug: {}'.format(cur_dir))

path_to_scripts = 'scripts/'

print('Opening file...')
start = time.time()
with open('{}/experiment_log'.format(cur_dir), 'w') as f:
    print('Generating results...', file=f)
    print('Generating results...')
    call('make VIBE-NaiveSerial-CPP', f)
    call('cd vibe-sources; make;', f)
    call('./{}run_VIBE.py my1 v1 v2 my2 videos/sequence.avi'.format(path_to_scripts), f)
    
    # generate an empty image for the map generation step
    # for our VIBE
    print('Generating an empty 480x640 image...', file=f)
    print('Generating an empty 480x640 image...')
    # the script below has a hard-coded "el_count.png" name
    call('./{}generate_empty_image.py 640 480'.format(path_to_scripts), f) 
    print('Generating map for our implementation of VIBE...', file=f)
    print('Generating map for our implementation of VIBE...')
    call('make vibe_generate_map', f)
    call('./{}generate_map.py 1 my1 v1 v2 my2'.format(path_to_scripts), f)
    
    # generate an empty image for the map generation step
    # for v1 of their VIBE
    print('Generating an empty 480x640 image...', file=f)
    print('Generating an empty 480x640 image...')
    call('./{}generate_empty_image.py 640 480'.format(path_to_scripts), f) 
    print('Generating first map for original VIBE...', file=f)
    print('Generating first map for original VIBE...')
    call('./{}generate_map.py 2 my1 v1 v2 my2'.format(path_to_scripts), f)

    # generate an empty image for the map generation step
    # for v2 of their VIBE
    print('Generating an empty 480x640 image...', file=f)
    print('Generating an empty 480x640 image...')
    call('./{}generate_empty_image.py 640 480'.format(path_to_scripts), f) 
    print('Generating second map for original VIBE...', file=f)
    print('Generating second map for original VIBE...')
    call('./{}generate_map.py 3 my1 v1 v2 my2'.format(path_to_scripts), f)

    # generate an empty image for the map generation step
    # for v2 of our VIBE
    print('Generating an empty 480x640 image...', file=f)
    print('Generating an empty 480x640 image...')
    call('./{}generate_empty_image.py 640 480'.format(path_to_scripts), f) 
    print('Generating second map for my VIBE...', file=f)
    print('Generating second map for my VIBE...')
    call('./{}generate_map.py 4 my1 v1 v2 my2'.format(path_to_scripts), f)

    call('make vibe_generate_histogram', f)
    
    print('Generating the histogram "Their v1 vs. Their v2"...')
    call('./vibe_generate_histogram pixel_count_1.png pixel_count_2.png'.format(path_to_scripts), f)
    
    print('Generating the histogram for "My vs. Their"...')
    call('./vibe_generate_histogram pixel_count_1.png pixel_count_3.png'.format(path_to_scripts), f)
    
    print('Generating the histogram "My 1 vs. My 2"...')
    call('./vibe_generate_histogram pixel_count_3.png pixel_count_4.png'.format(path_to_scripts), f)

#os.chdir('scripts')
print('Total time = {} seconds'.format(time.time() - start))
print('Done! Check the log file')
