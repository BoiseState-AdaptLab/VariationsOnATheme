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

import networkx as nx
import pydotplus as dot
import sys

class FlowGraph(object):
    def __init__(self, name='', nodes=[], edges=[], comps=[]):
        self._name = name
        self._path = ''
        self._nodelist = []
        self._edgelist = []
        self._nodedict = {}
        self._comps = comps
        self._factory = NodeFactory()
        self._g = None #nx.DiGraph()
        self._nrows = 0
        self._ncols = 0
        self.nodes = nodes
        self.edges = edges

    def __contains__(self, key):
        # Assume it's a tuple right now...
        (x, y) = key
        key = '%d,%d' % (x, y)
        return key in self._nodedict

    def __getitem__(self, key):
        (x, y) = key
        key = '%d,%d' % (x, y)
        return self._nodedict[key]

    def __setitem__(self, key, val):
        (x, y) = key
        key = '%d,%d' % (x, y)
        self._nodedict[key] = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def graph(self):
        return self._g

    @property
    def nodes(self):
        #return self._g.nodes(data=True)
        return self._nodelist

    @nodes.setter
    def nodes(self, rows=[]):
        self._nrows = len(rows)

        i = 0
        for row in rows:
            self._ncols = max(self._ncols, len(row))

            j = 0
            for tuple in row:
                (type, label) = tuple
                node = self._factory.get(type, label)

                node['row'] = i
                node['col'] = j
                if j < len(self._comps):
                    node['comp'] = self._comps[j]

                nodekey = '%d,%d' % (i, j)
                self._nodedict[nodekey] = node
                self._nodelist.append(node)
                j += 1
            i += 1

    @property
    def edges(self):
        #return self._g.edges(data=True)
        return self._edgelist

    @edges.setter
    def edges(self, edges=[]):
        for edge in edges:
            (src, dest) = edge
            if src in self and dest in self:
                self._edgelist.append(Edge(self[src], self[dest]))
        #self._g.add_edges_from(e)

    def node(self, x, y):
        nodekey = '%d,%d' % (x, y)
        return self._nodedict[nodekey]

    def nxgraph(self):
        g = nx.DiGraph()

        for node in self._nodelist:
            g.add_node(node.label, node.attrs)

        for edge in self._edgelist:
            g.add_edge(edge.src.label, edge.dest.label, edge.attrs)

        self._g = g

        return g

    def draw(self):
        if self._g is None:
            self.nxgraph()
        nx.draw_networkx(self._g)

    def read(self, path=''):
        if len(path) < 1:
            path = '%s.dot' % self.name
        self._g = nx.read_dot(path)

    def write(self, path=''):
        if len(path) < 1:
            path = '%s.dot' % self.name

        fout = open(path, 'w')      # sys.stdout
        fout.write("strict digraph \"%s\" {\n" % self.name)

        for i in range(self._nrows):
            for j in range(self._ncols):
                id = i * self._ncols + j
                node = self[(i, j)]
                node['id'] = id

                # 0 [ label="Program" color="gray" shape="rect"];
                label = node.label
                if 'comp' in node and len(node['comp']) > 0:
                    label = '%s\\n%s' % (label, node['comp'])
                line = "%d [label = \"%s\"" % (id, label)

                if len(node['shape']) > 0:
                    line = "%s shape=\"%s\"" % (line, node['shape'])
                if len(node['color']) > 0:
                    line = "%s fillcolor=\"%s\" style=\"filled\"" % (line, node['color'])

                line = "%s];\n" % line
                fout.write(line)

        for edge in self.edges:
            fout.write("%d -> %d\n" % (edge.src['id'], edge.dest['id']))

        fout.write("}\n")
        fout.close()

        #if self._g is None:
            #self.nxgraph()
        #nx.drawing.nx_pydot.write_dot(self._g, path)
        #nx.drawing.nx_agraphwrite_dot(g, '%s.dot' % graph)


class Edge(object):
    def __init__(self, u=None, v=None, label='', attrs={}, directed=True):
        self._u = u
        self._v = v
        self._label = label
        self._attrs = attrs
        self._directed = directed

    def __contains__(self, item):
        return item in self._attrs

    def __getitem__(self, key):
        return self._attrs[key]

    def __setitem__(self, key, val):
        self._attrs[key] = val

    @property
    def key(self):
        vals = [str(val) for val in sorted(self._attrs.values())]
        return '-'.join(vals)

    @property
    def attrkeys(self):
        return self._attrs.keys()

    @property
    def src(self):
        return self._u

    @src.setter
    def src(self, u=None):
        self._u = u

    @property
    def dest(self):
        return self._v

    @dest.setter
    def dest(self, v=None):
        self._v = v

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label=''):
        self._label = label

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, attrs={}):
        self._attrs = attrs

    @property
    def directed(self):
        return self._directed

    @directed.setter
    def directed(self, directed=True):
        self._directed = directed


class Node(object):
    def __init__(self, label='', attrs={}):
        self._label = label
        self._attrs = attrs
        self._attrs['color'] = ''
        self._attrs['shape'] = ''

    def __contains__(self, item):
        return item in self._attrs

    def __getitem__(self, key):
        return self._attrs[key]

    def __setitem__(self, key, val):
        self._attrs[key] = val

    @property
    def key(self):
        vals = [str(val) for val in sorted(self._attrs.values())]
        return '-'.join(vals)

    @property
    def attrkeys(self):
        return self._attrs.keys()

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label=''):
        self._label = label

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, attrs={}):
        self._attrs = attrs

    def accept(self, visitor):
        visitor.visit(self)


class StmtNode(Node):
    def __init__(self, label='', attrs={}):
        super().__init__(label, attrs.copy())
        self._attrs['shape'] = 'invtriangle'


class DataNode(Node):
    def __init__(self, label='', attrs={}):
        super().__init__(label, attrs.copy())
        self._attrs['shape'] = 'rect'


class TempNode(DataNode):
    def __init__(self, label='', attrs={}):
        super().__init__(label, attrs.copy())
        self._attrs['color'] = 'gray'

class NodeFactory(object):
    class __NodeFactory:
        def __init__(self):
            pass

    _instance = None

    def __init__(self):
        if not NodeFactory._instance:
            NodeFactory._instance = NodeFactory.__NodeFactory()

    def __getattr__(self, name):
        return getattr(self._instance, name)

    def get(self, type='', label='', attrs={}):
        if type.startswith('S'):
            node = StmtNode(label, attrs)
        elif type.startswith('D'):
            node = DataNode(label, attrs)
        elif type.startswith('T'):
            node = TempNode(label, attrs)
        else:
            node = Node(label, attrs)

        return node
