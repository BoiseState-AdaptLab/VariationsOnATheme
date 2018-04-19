/**
    This file is part of VariationsDev.

    VariationsDev is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    VariationsDev is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with VariationsDev. If not, see <http://www.gnu.org/licenses/>.
 **/

#ifndef ISLINT_INTSETLIB_H
#define ISLINT_INTSETLIB_H

#if ! defined ASSUME_POSITIVE_INTMOD
#define ASSUME_POSITIVE_INTMOD 0
#endif

#if ASSUME_POSITIVE_INTMOD
#define intDiv(x,y) (eassert(((x)%(y)) >= 0), ((x)/(y)))
#define intMod(x,y) (eassert(((x)%(y)) >= 0), ((x)%(y)))
#else
#define intDiv_(x,y)  ((((x)%(y))>=0) ? ((x)/(y)) : (((x)/(y)) -1))
#define intMod_(x,y)  ((((x)%(y))>=0) ? ((x)%(y)) : (((x)%(y)) +y))
#define checkIntDiv(x,y) (eassert((y) > 0 && intMod_((x),(y)) >= 0 && intMod_((x),(y)) <= (y) && x==((y)*intDiv_((x),(y)) + intMod_((x),(y)))))
#define intDiv(x,y) (checkIntDiv((x),(y)), intDiv_((x),(y)))
#define intMod(x,y) (checkIntDiv((x),(y)), intMod_((x),(y)))
#endif

#define ceild(n, d) intDiv_((n), (d)) + ((intMod_((n),(d))>0)?1:0)
#define floord(n, d)  intDiv_((n), (d))

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef dx
#define dx 0.5
#define factor1 (1.0/12.0)
#define factor2 2.0
#endif

typedef double Real;

#endif //ISLINT_INTSETLIB_H
