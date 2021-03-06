#include <images/Images/ImageInterface.h>
#include <images/Images/TempImage.h>
#include <casa/Logging/LogOrigin.h>
#include <casa/Logging/LogSink.h>
#include <casa/Logging/LogIO.h>
#include <casa/Arrays/Vector.h>
#include <synthesis/MSVis/VisBuffer.h>
#include "BeamCalc.h"
#include <casa/OS/Timer.h>
#include <cuLatticeFFT.h>

#ifndef SYNTHESIS_ANTENNAATERM_H
#define SYNTHESIS_ANTENNAATERM_H

namespace casa{
  class AntennaATerm
  {
  public:
    AntennaATerm(cuLatticeFFT* cuLFFT): timer_p(), cufft_p(cuLFFT), Ad_buf_p(NULL), skyCoords_p(), skyShape_p() {initAP(ap_p);};
    ~AntennaATerm () {delete ap_p.aperture;};
    
    void initAP(ApertureCalcParams& ap);
    void setDeviceBuffer(Complex* d_buf) {Ad_buf_p = d_buf;};
    void applyPB(ImageInterface<Complex>& pbImage, 
		 Float pa, Float Freq,
		 //const VisBuffer& vb, 
		 Int bandID,
		 Bool doSquint);
    
    void setSkyCoords(const CoordinateSystem& skyCoords, const IPosition& shape) {skyCoords_p = skyCoords; skyShape_p=shape;};
    const CoordinateSystem& skyCoords() {return skyCoords_p;};
    const IPosition& skyShape() {return skyShape_p;};

    void setApertureParams(ApertureCalcParams& ap,
			   const Float& pa, const Float& Freq, 
			   const Int& bandID,
			   IPosition& skyShape,
			   Vector<Double>& uvIncr);
    inline void setApertureParams(const Float& pa, const Float& Freq, 
				  const Int& bandID,
				  IPosition& skyShape,
				  Vector<Double>& uvIncr)
    {setApertureParams(ap_p, pa, Freq, bandID, skyShape, uvIncr);};
    
    void removeSquint(ApertureCalcParams& ap);
    void regridApertureEngine(ApertureCalcParams& ap,Int inStokes);
    inline void regridApertureEngine(Int inStokes) {regridApertureEngine(ap_p,inStokes);}
    void regridAperture(CoordinateSystem& skyCS,
			IPosition& skyShape,
			Float pa, Float Freq,
			//const VisBuffer &vb,
			Bool doSquint, Int& bandID);
    
    void ftAperture(ApertureCalcParams& ap, const Int& inStokes);
    void skyMuller(Array<Complex>& skyJones,
		   const IPosition& shape,
		   const Int& inStokes);
    
    void fillPB(ImageInterface<Complex>& inImg,
		ImageInterface<Complex>& outImg,
		Bool Square=False);
    CoordinateSystem makeUVCoords(CoordinateSystem& imageCoordSys,
				  IPosition& shape);

    inline void cfft2d(Lattice<Complex>& cLattice) {cufft_p->cfft2d(cLattice);};
    cuLatticeFFT *cufft_p;
  private:
    ApertureCalcParams ap_p;
    Timer timer_p;
    Double fftTime_p, beamCalcTime_p;
    Complex *Ad_buf_p;
    CoordinateSystem skyCoords_p;
    IPosition skyShape_p;
  };
  
};
#endif
