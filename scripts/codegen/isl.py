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

from abc import abstractmethod

from tools import strings
from tools import system

import copy as cp
#import islpy as isl
import re
import shutil

class ISLError(Exception):
    def __init__(self, message):    # Call the base class constructor with the parameters it needs
        super(ISLError, self).__init__(message)


class ISLVar(object):
    def __init__(self, name='', lower='', upper=''):
        self._name = name
        self._lower = lower
        self._upper = upper

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, lower):
        self._lower = lower

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, upper):
        self._upper = upper

    def __str__(self):
        if len(self._lower) < 1:
            out = self._name + " <= " + self._upper
        elif len(self._upper) < 1:
            out = self._name + " >= " + self._lower
        else:
            out = self._lower + " <= " + self._name + " <= " + self._upper

        return out

    def __repr__(self):
        return self.__str__()


class ISLSpace(object):
    def __init__(self, name='', vars=[]):
        self._name = name
        self._vars = vars

    def add(self, var):
        self._vars.append(var)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def vars(self):
        return self._vars

    @vars.setter
    def vars(self, vars):
        self._vars = vars

    def __str__(self):
        # S[t, i]: 1 <= t <= T and 1 <= i <= N
        out = self._name + "["

        nVars = len(self._vars)
        for i in range(nVars):
            out += self._vars[i]._name
            if i < nVars - 1:
                out += ","

        out += "] : ";
        for i in range(nVars):
            out += str(self._vars[i])
            if i < nVars - 1:
                out += " and "

        return out

    def __repr__(self):
        return self.__str__()


class ISLDomain(object):
    def __init__(self, name='', spaces=[]):
        self._name = name
        self._spaces = spaces

    def add(self, space):
        self._spaces.append(space)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __str__(self):
        # Domain := [T,N] -> { S[t,i] : 1<=t<=T and 1<=i<=N };
        out = self._name + " := ["

        nSpaces = len(self._spaces)
        for i in range(nSpaces):
            vars = self._spaces[i]._vars
            nVars = len(vars)
            for j in range(nVars):
                out += vars[j].upper
                if j < nVars - 1:
                    out += ","

        out += "] -> { "
        for i in range(nSpaces):
            out += str(self._spaces[i])
            if i < nSpaces - 1:
                out += ";\n"

        out += " };"

        return out


class ISLSchedule(object):
    def __init__(self, name='', spaces=[]):
        self._name = name
        self._spaces = spaces
        self._mapping = []

    def add(self, space):
        self._spaces.append(space)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __str__(self):
        # Schedule := [T,N] -> { S[t,i] -> [t,i] };
        out = self.name + " := ["

        nSpaces = len(self._spaces)
        for i in range(nSpaces):
            vars = self._spaces[i].vars
            nVars = len(vars)

            for j in range(nVars):
                out += vars[j].upper
                if j < nVars - 1:
                    out += ","

        out += "] -> { "
        for i in range(nSpaces):
            space = self._spaces[i]
            out += space.name + "["

            vars = self._spaces[i].vars
            nVars = len(vars)
            for j in range(nVars):
                out += vars[j].name
                if j < nVars - 1:
                    out += ","

            out += "] -> ["

            mapping = self._mapping
            for j in range(len(mapping)):
                out += mapping[j]
                if j < nVars - 1:
                    out += ","

            out += "]"

            if i < nSpaces - 1:
                out += ";\n"

        out += " };"

        return out


class ISLTransform(object):
    def __init__(self, name=''):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @abstractmethod
    def transform(self, schedule):
        pass


class ISLSkew(ISLTransform):
    def __init__(self, name='skew', offsets=[]):
        super().__init__(name)
        self._offsets = offsets

    def transform(self, schedule):
        newSched = cp.copy(schedule)

        mapping = newSched._mapping
        vars = newSched._spaces[0]._vars
        for i in range(len(self._offsets)):
            offset = self._offsets[i]
            mapStr = vars[i].name
            if len(offset) > 0:
                if offset[0] != '-' and offset[0] != '+':
                    mapStr += '+'
                mapStr = mapStr + str(offset)

            if i < len(mapping):
                mapping[i] = mapStr
            else:
                mapping.append(mapStr)

        newSched._mapping = mapping
        newSched._name = self._name + newSched._name

        return newSched


class ISLGenerator(object):
    def __init__(self, benchmark={}, out=None):
        self._benchmark = benchmark
        self._out = out
        self._isl = ''
        self._gen = ''

    def benchmark(self):
        return self._benchmark

    def code(self):
        return self._gen

    def isl(self):
        return self._isl

    def readISL(self, file=''):
        if len(file) > 0:
            code = ''
            f = open(file, 'r')
            for line in f:
                code += line
            f.close()
            self._isl = code

    def transform(self, schedule, domain):
        code = ''
        code += str(domain) + "\n"
        code += str(schedule) + "\n"
        code += "codegen(" + schedule.name + " * " + domain.name + ");\n"
        self._isl = code

        return code

    def getISL(self):
        code = self._isl
        benchmark = self._benchmark
        vars = benchmark['vars']

        for key in vars:
            val = str(vars[key])
            patt = re.compile("(\$%s)" % key)  # parentheses for capture groups
            code = re.sub(patt, val, code)

        return code

    def generate(self):
        benchmark = self._benchmark
        code = ''

        ompRegion = False
        ompLevel = -1
        ompPragma = ''
        ompLines = []

        # Start reading the template...
        template = '%s/%s' % (benchmark['path'], benchmark['template'])
        file = open(template, 'r')

        # Peak at first line to look for <Name> tag.
        line = file.readline().rstrip()

        if '<Name' in line:
            if '=' in line:
                benchmark['output'] = line.split('=')[1].replace('"', '').rstrip('>')
            else:
                code += line.replace('<Name>', benchmark['name'])

            file.close()
            file = open(template, 'r')
        else:
            code += line + "\n"

        if 'output' not in benchmark or len(benchmark['output']) < 1:
            benchmark['output'] = benchmark['name']

        if '.h' not in benchmark['output']:
            benchmark['output'] = '%s.h' % benchmark['output']

        outFile = '%s/%s' % (benchmark['path'], benchmark['output'])
        print("Creating output file '%s'..." % outFile)
        self._out = open(outFile, 'w')

        # Read the rest of the template...
        for line in file:
            line = line.rstrip()
            if '<Variables' in line and 'vars' in benchmark:
                varnames = sorted(benchmark['vars'].keys())
                for var in varnames:
                    code += '#define ' + var + ' ' + str(benchmark['vars'][var]) + "\n"
                    if var == 'TILE_SIZE':
                        code += '#define TILE_MASK ' + str(int(benchmark['vars'][var]) - 1) + "\n"
                        code += '#define TILE_SIZE2 ' + str(int(benchmark['vars'][var]) * 2) + "\n"
                        code += '#define TILE_MASK2 ' + str(int(benchmark['vars'][var]) * 2 - 1) + "\n"
            elif '<OMP' in line:
                ompRegion = True
                if 'Level' in line:
                    begin = line.find('=', line.find('Level') + 5) + 1
                    end = line.find(' ', begin)
                    if end < begin:
                        end = line.find('>', begin)
                    ompLevel = int(line[begin:end].replace('"', ''))
                if 'Pragma' in line:
                    begin = line.find('"', line.find('Pragma') + 6) + 1
                    end = line.find('"', begin)
                    if end < begin:
                        end = line.find('>', begin)
                    ompPragma = line[begin:end].replace('"', '')
            elif '</OMP' in line:
                ompRegion = False
            elif ompRegion:
                ompLines.append(line)
            elif '<ISL' in line:
                # Get ISL code...
                islFile = benchmark['isl']
                if len(islFile) < 1 and 'File' in line:
                    begin = line.find('=', line.find('File') + 4) + 1
                    end = line.find(' ', begin)
                    if end < begin:
                        end = line.find('>', begin)
                    islFile = line[begin:end].replace('"', '')

                islFile = '%s/%s' % (benchmark['path'], islFile)
                self.readISL(islFile)
                islCode = self.getISL()

                (islLines, error) = system.run([benchmark['iscc']], islCode)

                islLines = islLines[0].split("\n")

                if len(islLines) < 1:
                    raise ISLError("Unable to generate code from ISL file '%s'." % islFile)
                elif len(strings.grep('error', islLines, True)) > 0:
                    raise ISLError("ISL error in file '%s': %s" % islFile, "\n".join(islLines))

                if benchmark['skipguard'] and 'if (' in islLines[0]:
                    guard = islLines[0]
                    islLines = islLines[1:]
                    while len(islLines[-1]) < 1:
                        del islLines[-1]
                    if '{' in guard and '}' in islLines[-1]:
                        del islLines[-1]

                # Merge ISL code and OMP code...
                index = 0
                if ompLevel > 0 and len(ompPragma) > 0:
                    ompLevel -= 1    # This was implemented when box loop was not part of the kernel so back it up one...
                    nesting = 0
                    while index < len(islLines):
                        if 'for (' in islLines[index]:
                            if nesting == ompLevel:
                                break
                            nesting += 1
                        index += 1

                    # Insert OMP pragma before index...
                    islLines.insert(index, ompPragma)
                    index += 1

                # And any code after index...
                if len(ompLines) > 0:
                    if index > 0:
                        newLines = islLines[0:index + 1]
                        wrapped = False
                        if '{' not in newLines[-1]:
                            newLines[-1] += ' {'  # Add opening brace...
                            wrapped = True
                        newLines.extend(ompLines)
                        newLines.extend(islLines[index + 1:])
                        if wrapped:
                            #newLines[-1] += "    }\n"  # Add closing brace...
                            newLines.append('}')
                        islLines = newLines
                    else:
                        ompLines.append('')
                        ompLines.extend(islLines)
                        islLines = ompLines

                    code += "\n".join(islLines).rstrip() + "\n"

            else:
                code += line + "\n"
                if '#pragma omp' in line:
                    ompPragma = line.lstrip()

        file.close()

        self._gen = code

        if self._out is not None:
            self._out.write(code)
            self._out.close()

        # Now we have to modify the header file for the benchmark class...
        classFile = '%s/%s' % (benchmark['path'], benchmark['header'])
        tempFile = '%s~' % classFile
        print("Updating class file '%s'..." % classFile)
        fin = open(classFile, 'r')
        fout = open(tempFile, 'w')

        for line in fin:
            fout.write(line)
            if '<Include' in line:
                fout.write("#include \"%s\"\n" % benchmark['output'])
                line = fin.readline()   # Discard the next line

        fin.close()
        fout.close()

        shutil.move(tempFile, classFile)

        return code
