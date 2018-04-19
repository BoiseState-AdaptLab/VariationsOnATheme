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

CC=gcc
CXX=g++
COMMON_DIR=../common

CFLAGS = -I/usr/include -I/usr/local/include 
COPTS = -fopenmp -O4

CVARIANTS = miniFluxdiv-explain-serial\
            miniFluxdiv-explain-baseline\
            miniFluxdiv-explainTripleCache-lc\
            miniFluxdiv-explainTripleCache-fuse\
            miniFluxdiv-explainTripleCache-tile888\
            miniFluxdiv-explainTripleCache-tile161616\
            miniFluxdiv-explainTripleCache-tile323232

.PHONY: clean all test

all: $(CVARIANTS)


$(CVARIANTS): % : %.cpp util.cpp $(COMMON_DIR)/Measurements.cpp $(COMMON_DIR)/Configuration.cpp
	  $(CXX) -g  util.cpp $(COMMON_DIR)/Measurements.cpp $(COMMON_DIR)/Configuration.cpp $@.cpp -o $@ $(COPTS)

%.gen: %.in
		cat $^ | iscc > $@

test: $(CVARIANTS)
	for v in  $(CVARIANTS);\
	do\
	  $$v -C 16 -B 100 -v;\
	done;\
	echo "Testing Complete"

clean:
	rm -f *o
	rm -f $(CVARIANTS)
	rm -f *_over *_within

realclean: clean
	rm -f *gen
