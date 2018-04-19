import os, subprocess
import csv
from collections import OrderedDict

Configuration={}
Configuration['numCells']= { '16':'12288',
                           '32':'1536',
                           '64':'192',
                           '128':'24'}
Configuration['numThreads']=['1','2','4','8','16']
Configuration['numTime']= 10
Configuration['file']= {'Within': "./miniFluxdiv-baseline-OMP_within",
                         'Over' : "./miniFluxdiv-baseline-OMP_over"}
print Configuration

for key in Configuration.get('numCells'):
    print key
    print Configuration.get('numCells').get(key)

#for k in Configuration.get('file').keys(): 
from subprocess import call
with open("miniFluxdiv-baseline_OmpReadings.txt","w") as f:
    for k in Configuration.get('file').keys():
        for key in Configuration.get('numCells'):
            for thread in Configuration.get('numThreads'):
                #for run in range(0,Configuration.get('numTime')):
                for run in range(1,3):
                   # pass
                    print k,key,thread,run
                    print Configuration.get('numCells').get(key)
                    print Configuration.get('file').get(k)
                    call([str(Configuration.get('file').get(k)),"-b",str(Configuration.get('numCells').get(key)),"-c",str(key),"-p",str(thread)],stdout =f)




