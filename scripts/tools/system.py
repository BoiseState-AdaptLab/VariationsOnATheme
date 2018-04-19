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

import subprocess as sub

def run(cmd='', input='', sh=True):
    output = ''
    error = ''

    proc = sub.Popen(cmd, shell=sh, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    if len(input) > 0:
        data = proc.communicate(input=input.encode('ascii'))
        output = [item.decode() for item in data]
    else:
        output = proc.stdout.read().decode()
        error = proc.stderr.read().decode()

    return (output, error)

def ncpus():
    n = 0
    try:
        import multiprocessing as mp
        n = mp.cpu_count()
    except (ImportError, NotImplementedError):
        n = 1

    return n

def threaded():
    try:
        import concurrent.futures
        return True
    except (ImportError, NotImplementedError):
        return False
