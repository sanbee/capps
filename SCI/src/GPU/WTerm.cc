//# WTerm.cc: implementation of WTerm
//# Copyright (C) 2007
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
//# Correspondence concerning AIPS++ should be adressed as follows:
//#        Internet email: aips2-request@nrao.edu.
//#        Postal address: AIPS++ Project Office
//#                        National Radio Astronomy Observatory
//#                        520 Edgemont Road
//#                        Charlottesville, VA 22903-2475 USA
//#
//#
//# $Id$
#include <synthesis/TransformMachines/WTerm.h>
#ifdef HAS_OMP
#include <omp.h>
#endif


namespace casa { //# NAMESPACE CASA - BEGIN
  
  void WTerm::applySky(Matrix<Complex>& screen, 
		       const Vector<Double>& sampling,
		       const Double wValue,
		       const Int inner)
  {
    Int convSize = screen.shape()(0);
    Double twoPiW=2.0*C::pi*Double(wValue);
#ifdef HAS_OMP
    Int Nth=max(omp_get_max_threads()-2,1);
#endif

    if (!isNoOp())
      {
        #pragma acc kernels
	for (Int iy=-inner/2;iy<inner/2;iy++) 
	  {
	    Double m=sampling(1)*Double(iy);
	    Double msq=m*m;
#pragma omp parallel default(none) firstprivate(msq,iy) shared(sampling,twoPiW,convSize,screen) num_threads(Nth)
    {
#pragma acc kernel
#pragma omp for
	    for (Int ix=-inner/2;ix<inner/2;ix++) 
	      {
		Double l=sampling(0)*Double(ix);
		Double rsq=l*l+msq;
		if(rsq<1.0) 
		  {
		    Double phase=twoPiW*(sqrt(1.0-rsq)-1.0);
		    screen(ix+convSize/2,iy+convSize/2)*=Complex(cos(phase),sin(phase));
		  }
	      }
    }
	  }
      }
  }

  void WTerm::applySky(Matrix<Complex>& screen, 
		       const Int wPixel, 
		       const Vector<Double>& sampling,
		       const Double wScale,
		       const Int inner)
  {
    //UNUSED: Int convSize = screen.shape()(0);
    if(wPixel>0) 
      {
	Double wValue=(wPixel*wPixel)/wScale;
	applySky(screen, sampling, wValue, inner);
      }
  }
};
