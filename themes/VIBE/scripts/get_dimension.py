#!/usr/bin/python2

from __future__ import print_function
import sys
import argparse

if sys.version_info.major > 2: raise Exception('Need to use python2')

import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('v', help='video file')
    return parser.parse_args()

args = parse_args()    
vcap = cv2.VideoCapture(args.v)
if vcap.isOpened():
    width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    print('-'*10)
    print('Video {}'.format(args.v))
    print('height = {}, width = {}'.format(height, width))
    print('-'*10)

