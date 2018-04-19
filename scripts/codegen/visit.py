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

import sys

from abc import abstractmethod

class Visitor(object):
    @abstractmethod
    def visit(self, node):
        pass

    def enter(self, node):
        # Default implementaiton: do nothing...
        pass

    def exit(self, node):
        # Default implementaiton: do nothing...
        pass

class DFLRVisitor(Visitor):
    def __init__(self):
        self._nVisited = 0

    def visit(self, node):
        self.enter(node)

        for child in node.getChildren():
            self.nVisited += 1
            child.accept(self)

        self.exit(node)

    @property
    def nVisited(self):
        return self._nVisited

    @nVisited.setter
    def nVisited(self, nVisited):
        self._nVisited = nVisited

# TODO: Modify this class to use graphviz package...
class DOTVisitor(DFLRVisitor):
    def __init__(self, out=sys.stdout):
        super().__init__()
        self.out = out

    def enter(self, node):
        if node.isRoot():
            self.out.write("digraph ASTGraph {\n")

    def visit(self, node):
        self.enter(node)
        self.out.write("%d [ label=\"%s\" ];\n" % (self.nVisited, node.getLabel()))

        nodeID = self.nVisited
        for child in node.getChildren():
            self.nVisited += 1
            self.out.write("%d -> %d\n" % (nodeID, self.nVisited))
            child.accept(self)

        self.exit(node)

    def exit(self, node):
        if node.isRoot():
            self.out.write("}\n")

class LoopVisitor(DFLRVisitor):
    def __init__(self):
        super().__init__()
        self.nests = []

def DFGVisitor(Visitor):
    def __init__(self, dfg=None):
        self._dfg = dfg

    def visit(self, node):
        self.enter(node)

        for child in node.getChildren():
            self.nVisited += 1
            child.accept(self)

        self.exit(node)