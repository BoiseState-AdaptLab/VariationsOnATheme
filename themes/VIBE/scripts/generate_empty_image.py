#!/bin/python3

# This script will generate an empty image of size H x W, where W and H are command line arguments
# Assumes you have a Linux program called "convert". You can install it via "sudo dnf install ImageMagick"

from subprocess import call
import sys
import os

cur_dir = os.getcwd()
print('Debug: {}'.format(cur_dir))

if len(sys.argv) != 3:
    print('Usage: width height')
    sys.exit(1)

h = sys.argv[1]
w = sys.argv[2]

#if int(h) > int(w):
#    print('Incorrect args. Width must be greater than the height')
#    sys.exit(1)

call('convert -size {}x{} xc:black pixel_count.png'.format(h, w), shell=True)


