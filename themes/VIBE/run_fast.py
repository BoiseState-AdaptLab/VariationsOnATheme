#!/bin/python3

# Must be run from the ~/.../VIBE 
import sys
from subprocess import call
import os

#video = 'rgb2gray/etc/7-0001.mp4'

call('make clean', shell=True)
call('make VIBE-NaiveSerial-CPP', shell=True)
video = 'videos/sequence.avi'
for i in range(1):
    call('./VIBE-NaiveSerial-CPP -f {} -o output_{} -g {}'.format(video, i, i), shell=True)


