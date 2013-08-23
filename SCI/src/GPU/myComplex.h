//# myComplex.h: Single and double precision complex numbers
//# Copyright (C) 2000,2001,2002,2004
//# Associated Universities, Inc. Washington DC, USA.
//#
//# This library is free software; you can redistribute it and/or modify it
//# under the terms of the GNU Library General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or (at your
//# option) any later version.
//#
//# This library is distributed in the hope that it will be useful, but WITHOUT
//# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
//# License for more details.
//#
//# You should have received a copy of the GNU Library General Public License
//# along with this library; if not, write to the Free Software Foundation,
//# Inc., 675 Massachusetts Ave, Cambridge, MA 02139, USA.
//#
//# Correspondence concerning AIPS++ should be addressed as follows:
//#        Internet email: aips2-request@nrao.edu.
//#        Postal address: AIPS++ Project Office
//#                        National Radio Astronomy Observatory
//#                        520 Edgemont Road
//#                        Charlottesville, VA 22903-2475 USA
//#
//# $Id: myComplex.h 21130 2011-10-18 07:39:05Z gervandiepen $


#ifndef CASA_COMPLEX1_H
#define CASA_COMPLEX1_H


//# Includes
#include <casa/aips.h>
#include <myComplexfwd.h>
#include <mycomplex.h>

namespace casa { //# NAMESPACE CASA - BEGIN

  inline cuComplex operator*(const cuComplex& val1, const cuComplex &val2) { return cuCmulf(val1, val2);};
  inline cuComplex operator/(const cuComplex& val1, const cuComplex &val2) { return cuCdivf(val1, val2);};
  inline cuComplex operator+(const cuComplex& val1, const cuComplex &val2) { return cuCaddf(val1, val2);};
  inline cuComplex operator-(const cuComplex& val1, const cuComplex &val2) { return cuCsubf(val1, val2);};

  inline cuComplex operator/(const cuComplex& val1, const Float &val2) { return cuCdivf(val1, make_cuFloatComplex(val2,0.0));};
  inline cuComplex operator*(const cuComplex& val1, const Float &val2) { return cuCmulf(val1, make_cuFloatComplex(val2,0.0));};

  inline cuComplex set(cuComplex& val1, const cuComplex &val2) { val1.x=val2.x; val1.y=val2.y;return val1;}
  inline cuComplex negate(cuComplex& val1) { val1.x=-val1.x; val1.y=-val1.y;return val1;}

  //  inline Complex operator*(const Complex& val, Double f) { val.x*=Float(f); val.y*=Float(f); return val; }
  // inline cuComplex operator*(Double f, const cuComplex& val) { val.x*=Float(f); val.y*=Float(f); return val; }
  // inline cuComplex operator/(const cuComplex& val, Double f) { val.x/=Float(f); val.y/=Float(f); return val; }
  // inline cuComplex operator/(Double f, const cuComplex& val) { val.x/=Float(f); val.y/=Float(f); return val; }
// </group>
// These operators are useful, otherwise both Float and Double are applicable
// for Ints.
// <group>
// inline cuComplex operator*(const cuComplex& val, Int f) { return val*Float(f); }
// inline cuComplex operator*(Int f, const cuComplex& val) { return val*Float(f); }
// inline cuComplex operator/(const cuComplex& val, Int f) { return val/Float(f); }
// inline cuComplex operator/(Int f, const cuComplex& val) { return Float(f)/val; }

} //# NAMESPACE CASA - END

// Define real & complex conjugation for non-complex types
// and put comparisons into std namespace.

#endif
