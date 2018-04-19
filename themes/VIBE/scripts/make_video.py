#!/usr/bin/python3

import subprocess
import sys
import argparse

# Should be called from the VIBE root directory. 
# Make sure you modified VIBE_NaiveSerial-CPP file to spit out all the frames instead of just frame 350.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('v', help='video file')
    parser.add_argument('src', help='folder containing the frames of a video')
    parser.add_argument('dst', help='name of the destination video')
    return parser.parse_args()

def main():
    args = parse_args()
    subprocess.call('make clean', shell=True)
    subprocess.call('make VIBE-NaiveSerial-CPP', shell=True)
    subprocess.call('mkdir {}'.format(args.src), shell=True)
    subprocess.call('./VIBE-NaiveSerial-CPP -f {} -o {}/output -g {}'.format(args.v, args.src, 777), shell=True)
    subprocess.call('ffmpeg -start_number 1 -i {}/output_%d.png -vcodec mpeg4 {}/{}'.format(args.src, args.src, args.dst), shell=True)

main()
