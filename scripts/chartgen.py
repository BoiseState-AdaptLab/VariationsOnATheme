#!/usr/bin/env python3
#
#   This file is part of MiniFluxDiv.

#   MiniFluxDiv is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.

#   MiniFluxDiv is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with MiniFluxDiv. If not, see <http://www.gnu.org/licenses/>.
#

import os
import sys
import time
import argparse as ap
import traceback as tb

import plotly.offline as py
import plotly.graph_objs as go
#import numpy as np
import pandas as pd
#import cufflinks as cf
import shutil as shu
import subprocess as sp

from codegen import settings

def barCharts(chartdict, datadir, df):
    legends = chartdict['legends']
    figures = chartdict['figures']
    colors = chartdict['colors']
    symbols = chartdict['symbols']
    dashes = chartdict['dashes']

    outfiles = []

    sizes = [16, 32, 64, 128]


def scatterPlots(chartdict, datadir, df):
    legends = chartdict['legends']
    figures = chartdict['figures']
    colors = chartdict['colors']
    symbols = chartdict['symbols']
    dashes = chartdict['dashes']

    if 'order' in chartdict and len(chartdict['order']) > 0:
        order = chartdict['order']
    else:
        order = []
        for j in range(len(legends)):
            order.append(j)


    outfiles = []

    #sizes = [16, 32, 64, 128]
    sizes = [16, 128]
    for i in range(len(sizes)):
        size = sizes[i]
        sub = df.ix[df['nCells'] == size]

        data = []
        for j in order:
            legend = legends[j]
            figure = figures[j]
            color = colors[j]
            symbol = symbols[j]
            xcol = chartdict['xcol']
            ycol = chartdict['ycol']

            ser = sub.ix[sub['Legend'] == legend]

            # Create a trace
            trace = go.Scatter(
                x=ser[xcol],
                y=ser[ycol],
                name=figure,
                marker=dict(
                    symbol=symbol,
                    size=chartdict['markerSize']
                ),
                mode='lines+markers',
                line=dict(
                    color=color,
                    width=chartdict['lineSize'],
                    dash=dashes[j]
                )
            )

            data.append(trace)

        legendpos = chartdict['legendpos']

        # Edit the layout
        layout = dict(
            font=dict(family=chartdict['fontName'], size=chartdict['fontSize'], color=chartdict['fontColor']),
            title='MiniFluxDiv Performance on 28-Core Intel Xeon E5-2680 (Box Size %d<sup>3</sup>)' % size,
            legend=dict(x=legendpos[0], y=legendpos[1]),
            xaxis=dict(
                title='Thread Count',
                # type='log',
                autorange=True,
                dtick=2
            ),
            yaxis=dict(
                title='Execution Time (s)',
                type='log',
                #autorange=True
                range=[-0.8,1.0],
                tickfont=dict(
                    family=chartdict['fontName'],
                    size=36),
                titlefont=dict(
                    family=chartdict['fontName'],
                    size=36)
            )
        )

        # Plot and embed in ipython notebook!
        fig = dict(data=data, layout=layout)
        outfile = 'mfd-c%d-line.html' % size
        print("Generating file '%s/%s'..." % (datadir, outfile))
        py.plot(fig, filename=outfile)
        outfile = "%s/%s" % (datadir, outfile)
        outfiles.append(outfile)

    return outfiles


def main():
    bindir = sys.argv[0]
    pos = bindir.rfind('/')
    if pos >= 0:
        bindir = bindir[0:pos]
    else:
        bindir = '.'

    phantom = '%s/js/phantomjs %s/js/render.js' % (bindir, bindir)

    datafile = sys.argv[1]
    pos = datafile.rfind('/')
    if pos >= 0:
        datadir = datafile[0:pos]
    else:
        datadir = '.'

    os.chdir(datadir)

    # Is this line needed to run offline?
    #py.init_notebook_mode(connected=True)

    chartname = sys.argv[2]
    if chartname not in settings.charts:
        print("ERROR: Unknown chart name '%s'" % chartname)
        sys.exit(-1)

    chartdict = settings.charts[chartname]

    print("Reading data file '%s'..." % datafile)
    df = pd.read_csv(datafile)

    outfiles = scatterPlots(chartdict, datadir, df)

    for outfile in outfiles:
        # Insert polyfill include statement into the HTML output to satisfy phantomjs...
        print("Updating file '%s'..." % outfile)
        polyfill = '<script src="../js/polyfill.min.js"></script>'
        tmpfile = '%s~' % outfile
        fin = open(outfile, 'r')
        fout = open(tmpfile, 'w')

        for line in fin:
            if '<head>' in line:
                line = line.replace('</head>', '%s</head>' % polyfill)
            fout.write(line)

        fin.close()
        fout.close()
        shu.move(tmpfile, outfile)

        # Convert html to png using phantomjs...
        imgfile = outfile.replace('.html', '.png')
        print("Creating file '%s'..." % imgfile)
        imgcmd = '%s %s %s' % (phantom, outfile, imgfile)
        data = sp.check_output(imgcmd.split(' '), env=os.environ, stderr=sp.STDOUT)

        # Now convert png to pdf with convert...
        docfile = imgfile.replace('.png', '.pdf')
        print("Converting file '%s'..." % docfile)
        convcmd = '/usr/bin/convert %s %s' % (imgfile, docfile)
        data = sp.check_output(convcmd.split(' '), env=os.environ, stderr=sp.STDOUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e: # Ctrl-C
        print("Closing gracefully on keyboard interrupt...")
    except Exception as e:
        print('ERROR: ' + str(e))
        tb.print_exc()
