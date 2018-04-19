#!/bin/python3

from scipy.spatial.distance import correlation as cor

FILENAME = 'histogram.out'
OUT = 'score.out'

num_lines = 66

histograms = []

with open(FILENAME, 'r') as f:
    for line in f:
        if not line.strip(): continue
        histograms.append(int(line.strip()))
         
assert len(histograms) == num_lines
assert num_lines % 2 == 0

hist1 = histograms[:num_lines//2]
hist2 = histograms[num_lines//2:]

assert len(hist1) == num_lines//2
assert len(hist1) == len(hist2)

score = cor(hist1, hist2)

with open(OUT, 'a') as f:
    f.write('actual={}, rounded={}'.format(score, round(score, 3)))

print('Check {}'.format(OUT))
