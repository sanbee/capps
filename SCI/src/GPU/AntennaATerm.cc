#include "AntennaATerm.h"
#include <ms/MeasurementSets/MSColumns.h>
#include <lattices/Lattices/LatticeFFT.h>
#include <coordinates/Coordinates/SpectralCoordinate.h>
#include <coordinates/Coordinates/LinearCoordinate.h>
#include <coordinates/Coordinates/DirectionCoordinate.h>
#include <synthesis/TransformMachines/Utils.h>
#include <casa/OS/Timer.h>

#include "cuda_calls.h"
#include "/usr/local/cuda-5.5/include/cufft.h"


namespace casa{
  
  void AntennaATerm::initAP(ApertureCalcParams& ap)
  {
    fftTime_p=beamCalcTime_p=0.0;

    ap.oversamp = 1;
    ap.x0=-13.0; ap.y0=-13.0;
    ap.dx=0.5; ap.dy=0.5;
    
    ap.nx=ap.ny=104;
    ap.pa=18000000;
    ap.freq=1.4;
    ap.band = BeamCalc_VLA_L;
    IPosition shape(4,ap.nx,ap.ny,1,1);
    ap.aperture = new TempImage<Complex>();
    //   if (maximumCacheSize() > 0) ap.aperture->setMaximumCacheSize(maximumCacheSize());
    ap.aperture->resize(shape);
  };
  
  void AntennaATerm::applyPB(ImageInterface<Complex>& pbImage, 
			     //			     const VisBuffer& vb, 
			     Float pa, Float Freq,
			     Int bandID,
			     Bool doSquint)
  {
    CoordinateSystem skyCS(pbImage.coordinates());
    IPosition skyShape(pbImage.shape());
    
    regridAperture(skyCS, skyShape, pa, Freq, doSquint, bandID);
    pbImage.setCoordinateInfo(skyCS);
    fillPB(*(ap_p.aperture),pbImage);
    {
      string name("ftaperture.im");
      storeImg(name,*(ap_p.aperture));
      storeImg("pbimage.im",pbImage);
    }
    cerr << "ATerm timing breakup: CUFFT = " << fftTime_p << " BeamCalc::calculateAperture = " << beamCalcTime_p << endl;
  }
  
  void AntennaATerm::setApertureParams(ApertureCalcParams& ap,
				       const Float& pa, const Float& Freq, 
				       const Int& bandID,
				       IPosition& skyShape,
				       Vector<Double>& uvIncr)
  {
    Double Lambda = C::c/Freq;
    
    ap.pa=pa;
    ap.band = bandID;
    ap.freq = Freq/1E9;
    ap.nx = skyShape(0);           ap.ny = skyShape(1);
    ap.dx = abs(uvIncr(0)*Lambda); ap.dy = abs(uvIncr(1)*Lambda);
    ap.x0 = -(ap.nx/2)*ap.dx;      ap.y0 = -(ap.ny/2)*ap.dy;
  }
  
  void AntennaATerm::removeSquint(ApertureCalcParams& ap)
  {
    {
      IPosition PolnRIndex(4,0,0,0,0), PolnLIndex(4,0,0,3,0);
      IPosition tndx(4,0,0,0,0);
      IPosition apertureShape = ap.aperture->shape();
      for(tndx(3)=0;tndx(3)<apertureShape(3);tndx(3)++)   // The freq. axis
	for(tndx(2)=0;tndx(2)<apertureShape(2);tndx(2)++) // The Poln. axis
	  for(tndx(1)=0;tndx(1)<apertureShape(1);tndx(1)++)   // The spatial
	    for(tndx(0)=0;tndx(0)<apertureShape(0);tndx(0)++) // axis.
	      {
		PolnRIndex(0)=PolnLIndex(0)=tndx(0);
		PolnRIndex(1)=PolnLIndex(1)=tndx(1);
		Complex val, Rval, Lval;
		Float phase;
		val = ap.aperture->getAt(tndx);
		Rval = ap.aperture->getAt(PolnRIndex);
		Lval = ap.aperture->getAt(PolnLIndex);
		phase = arg(Rval); Rval=Complex(cos(phase),sin(phase));
		phase = arg(Lval); Lval=Complex(cos(phase),sin(phase));
		
		if      (tndx(2)==0) ap.aperture->putAt(val*conj(Rval),tndx);
		else if (tndx(2)==1) ap.aperture->putAt(val*conj(Lval),tndx);
		else if (tndx(2)==2) ap.aperture->putAt(val*conj(Rval),tndx);
		else if (tndx(2)==3) ap.aperture->putAt(val*conj(Lval),tndx);
	      }
    }
  }
  
  void AntennaATerm::regridApertureEngine(ApertureCalcParams& ap, Int inStokes)
  {
    IPosition apertureShape(ap.aperture->shape());
    apertureShape(0) = ap.nx;  apertureShape(1) = ap.ny;
    ap.aperture->resize(apertureShape);
    ap.aperture->set(0.0);
    timer_p.mark();
    BeamCalc::Instance()->calculateAperture(&ap,inStokes);
    beamCalcTime_p += timer_p.all();
    // {
    //   string name("aperture.im");
    //   storeImg(name,*(ap_p.aperture));
    // }
  }
  
  void AntennaATerm::regridAperture(CoordinateSystem& skyCS,
				    IPosition& skyShape,
				    //				    const VisBuffer &vb,
				    Float pa, Float Freq,
				    Bool doSquint, Int& bandID)
  {
    CoordinateSystem skyCoords(skyCS);
    // Float pa, Freq;
    // AlwaysAssert(ap_p.band>=-1, AipsError);
    // const ROMSSpWindowColumns& spwCol = vb.msColumns().spectralWindow();
    // ROArrayColumn<Double> chanfreq = spwCol.chanFreq();
    // ROScalarColumn<Double> reffreq = spwCol.refFrequency();
    // Freq = max(chanfreq.getColumn());

    //--------------------------------------------------------------------
    IPosition imsize(skyShape);
    CoordinateSystem uvCoords = makeUVCoords(skyCoords,imsize);

    Int index= uvCoords.findCoordinate(Coordinate::STOKES);
    Int inStokes = uvCoords.stokesCoordinate(index).stokes()(0);
    
    index = uvCoords.findCoordinate(Coordinate::LINEAR);
    LinearCoordinate lc=uvCoords.linearCoordinate(index);
    Vector<Double> uvIncr = lc.increment();
    Double Lambda = C::c/Freq;
    
    Vector<Int> poln(1);
    poln(0) = inStokes;
    StokesCoordinate polnCoord(poln);
    SpectralCoordinate spectralCoord(MFrequency::TOPO,Freq,1.0,0.0);
    //    uvCoords.addCoordinate(dirCoord);
    index = uvCoords.findCoordinate(Coordinate::STOKES);
    uvCoords.replaceCoordinate(polnCoord,index);
    index = uvCoords.findCoordinate(Coordinate::SPECTRAL);
    uvCoords.replaceCoordinate(spectralCoord,index);

    
    //--------------------------------------------------------------------
    ap_p.aperture->setCoordinateInfo(uvCoords);
    
    setApertureParams(ap_p, pa, Freq, bandID, skyShape, uvIncr);

    cerr << "Computing aperture @ " << ap_p.freq << " " << ap_p.pa << " " << ap_p.band << " " << skyShape << " " << uvIncr << endl;
    
    regridApertureEngine(ap_p,inStokes);
    {
      string name("aperture.im");
      storeImg(name,*(ap_p.aperture));
    }

    if (!doSquint) removeSquint(ap_p);
    
    ftAperture(ap_p,inStokes);
    //--------------------------------------------------------------------
  }
  
  
  void AntennaATerm::ftAperture(ApertureCalcParams& ap,const Int& inStokes)
  {
    Array<Complex> skyJonesBuf;
    (ap.aperture)->get(skyJonesBuf);
    IPosition shape = skyJonesBuf.shape().asVector();

    // Pradeep:
    //
    // skyJonesBuf is a CASA Array of type Complex.  I have extracted
    // its shape in the shape variable.  The following code will tell
    // you: (1) No. of dimensions, and (2) the size of the Array along
    // each dimension.
    //   Int nDim = shape.nelements();
    //   for(Int i=0;i<nDim;i++) cerr << "Size " << i << " = " << shape(i) <<  endl;

    // If you need to get a C-pointer to the storage inside the Array,
    // you can do that as follows:
      Bool dummy;
      Complex *pointer = skyJonesBuf.getStorage(dummy);
      int NX = shape(0);
      int NY = shape(1);
      int ret;
    //#ifdef CUDA
    cerr << "CUFFT call start, NX = " << NX << " NY = " << NY << " pointer = " << pointer << endl; 
    timer_p.mark();
    ret = call_cufft((cufftComplex*)pointer, NX, NY);
    fftTime_p += timer_p.all();
    cerr << "CUFFT call ends.  CYFFT Time = \n" << fftTime_p << endl;
   // #else
   // LatticeFFT::cfft2d(*(ap.aperture));
    //#endif
    
    skyMuller(skyJonesBuf, shape, inStokes);
  }
  
  void AntennaATerm::skyMuller(Array<Complex>& buf,
			       const IPosition& shape,
			       const Int& inStokes)
  {
    Array<Complex> tmp;
    IPosition t(4,0,0,0,0),n0(4,0,0,0,0),n1(4,0,0,0,0);
    
    Float peak;
    peak=0;
    for(t(2)=0;t(2)<shape(2);t(2)++)
      for(t(1)=0;t(1)<shape(1);t(1)++)
	for(t(0)=0;t(0)<shape(0);t(0)++)
	  if (abs(buf(t)) > peak) peak = abs(buf(t));
    if (peak > 1E-8)
      for(t(3)=0;t(3)<shape(3);t(3)++)       // Freq axis
	for(t(2)=0;t(2)<shape(2);t(2)++)     // Poln axis
	  for(t(1)=0;t(1)<shape(1);t(1)++)   // y axis
	    for(t(0)=0;t(0)<shape(0);t(0)++) // X axis
	      buf(t) = buf(t)/peak;
    // {
    //   skyJones.put(buf);
    //   String name("skyjones.im");
    //   storeImg(name,skyJones);
    // }
    
    tmp = buf;
    
    t(0)=t(1)=t(2)=t(3)=0;
    
    if ((inStokes == Stokes::RR) || (inStokes == Stokes::LL))
      {
	t(2)=0;n0(2)=0;n1(2)=0; //RR
	for(  n0(0)=n1(0)=t(0)=0;n0(0)<shape(0);n0(0)++,n1(0)++,t(0)++)
	  for(n0(1)=n1(1)=t(1)=0;n0(1)<shape(1);n0(1)++,n1(1)++,t(1)++)
	    buf(t) = (tmp(n0)*conj(tmp(n1)));
      }
    
    if (inStokes == Stokes::LR)
      {
	t(2)=0;n0(2)=1;n1(2)=0; //LR
	for(  n0(0)=n1(0)=t(0)=0;n0(0)<shape(0);n0(0)++,n1(0)++,t(0)++)
	  for(n0(1)=n1(1)=t(1)=0;n0(1)<shape(1);n0(1)++,n1(1)++,t(1)++)
	    buf(t) = (tmp(n0)*conj(tmp(n1)));
      }
    
    if (inStokes == Stokes::RL)
      {
	t(2)=0;n0(2)=0;n1(2)=1; //LR
	for(  n0(0)=n1(0)=t(0)=0;n0(0)<shape(0);n0(0)++,n1(0)++,t(0)++)
	  for(n0(1)=n1(1)=t(1)=0;n0(1)<shape(1);n0(1)++,n1(1)++,t(1)++)
	    buf(t) = (tmp(n0)*conj(tmp(n1)));
      }
  }
  
  void AntennaATerm::fillPB(ImageInterface<Complex>& inImg,
			    ImageInterface<Complex>& outImg,
			    Bool Square)
  {
    IPosition imsize(outImg.shape());
    IPosition ndx(outImg.shape());
    IPosition inShape(inImg.shape()),inNdx;
    Vector<Int> inStokes,outStokes;
    Int index,s,index1;
    index = inImg.coordinates().findCoordinate(Coordinate::STOKES);
    inStokes = inImg.coordinates().stokesCoordinate(index).stokes();
    index = outImg.coordinates().findCoordinate(Coordinate::STOKES);
    outStokes = outImg.coordinates().stokesCoordinate(index).stokes();
    index = outImg.coordinates().findCoordinate(Coordinate::SPECTRAL);
    index1 = inImg.coordinates().findCoordinate(Coordinate::SPECTRAL);
    SpectralCoordinate inSpectralCoords = inImg.coordinates().spectralCoordinate(index1);
    CoordinateSystem outCS = outImg.coordinates();
    outCS.replaceCoordinate(inSpectralCoords,index);
    outImg.setCoordinateInfo(outCS);
    
    outImg.put(inImg.get());

    // ndx(3)=0;
    // for(ndx(2)=0;ndx(2)<imsize(2);ndx(2)++) // The poln axes
    //   {
    // 	for(s=0;s<inShape(2);s++) if (inStokes(s) == outStokes(ndx(2))) break;
	
    // 	for(ndx(0)=0;ndx(0)<imsize(0);ndx(0)++)
    // 	  for(ndx(1)=0;ndx(1)<imsize(1);ndx(1)++)
    // 	    {
    // 	      Complex cval;
    // 	      inNdx = ndx; inNdx(2)=s;
    // 	      cval = inImg.getAt(inNdx);
    // 	      if (Square) cval = cval*conj(cval);
    // 	      outImg.putAt(cval*outImg.getAt(ndx),ndx);
    // 	    }
    //   }
  }

  CoordinateSystem AntennaATerm::makeUVCoords(CoordinateSystem& imageCoordSys,
					      IPosition& shape)
  {
    CoordinateSystem FTCoords = imageCoordSys;

    Int dirIndex=FTCoords.findCoordinate(Coordinate::DIRECTION);
    DirectionCoordinate dc=imageCoordSys.directionCoordinate(dirIndex);
    Vector<Bool> axes(2); axes=True;
    Vector<Int> dirShape(2); dirShape(0)=shape(0);dirShape(1)=shape(1);
    Coordinate* FTdc=dc.makeFourierCoordinate(axes,dirShape);

    FTCoords.replaceCoordinate(*FTdc,dirIndex);
    delete FTdc;

    return FTCoords;
  }

};
