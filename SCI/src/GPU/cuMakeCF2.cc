#include <casa/aips.h>
#include <AntennaATerm.h>
#include <cuWTerm.h>
#include <images/Images/PagedImage.h>
#include <lattices/Lattices/LatticeFFT.h>
#include <synthesis/TransformMachines/Utils.h>
#include <casa/OS/Timer.h>
#include <cuConvolver.h>

#ifdef cuda
#include <cuda_calls.h>
extern "C" {
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
};
#endif

#include <scimath/Mathematics/FFTServer.h>

#define CONVSIZE (2*1024)
#define OVERSAMPLING 20
#define TILE_WIDTHx (32)
#define TILE_WIDTHy (32)

#define DO_PROFILE

//#undef HAS_OMP

using namespace std;
using namespace casa;

Bool resizeCF(Array<Complex>& func, Int& xSupport, Int& ySupport,
	      const Float& sampling, const Complex& peak);

//
// Debugging code.
//
void storeDeviceBuf(TempImage<Complex>& theCF, cuComplex* d_buf, String name)
{ 
  Bool dummy0; 
  Array<Complex> cfBuf=theCF.get();
  IPosition skyShape=cfBuf.shape();
  cufftComplex *tt=(cufftComplex *)cfBuf.getStorage(dummy0);
  getBufferFromDevice(tt, d_buf,skyShape(0)*skyShape(1)*sizeof(cufftComplex)); 
  Complex *ttc =(Complex *)tt; cfBuf.putStorage(ttc,dummy0); theCF.put(cfBuf);
  storeImg(name,theCF); 
}

int main(int argc, char *argv[])
{
  //
  // Set up the parameters required to compute the A- and the W-terms.
  //
  Timer timer;
  Double timeATerm=0.0, timeWTerm=0.0, timeFFT=0.0, timeResize=0.0;

  Float Freq = 1.4e9, pa=-1.2;
  Float sampling=OVERSAMPLING;
  Int bandID= BeamCalc::Instance()->getBandID(Freq,"EVLA");
  IPosition skyShape(4,CONVSIZE,CONVSIZE,1,1);
  Vector<Double> uvIncr(2,0.01);
  Int xSupport, ySupport;
  Int wConvSize = 1024,iw=7000,nW=1024;
  Vector<Double> cellSize(2);cellSize=OVERSAMPLING*8*(M_PI/180.0)/3600;
  Double maxUVW = 0.25/cellSize(0);
  Double wScale=Float((wConvSize-1)*(wConvSize-1))/maxUVW;

  if (argc > 1)
    if (sscanf(argv[1],"%d",&nW) == EOF) nW=1024;

  //
  // Allocate the buffers and make the co-ordinate systems.  Do the
  // latter by loading a image from the disk.
  //
  PagedImage<Complex> thisAVGPB("./TemplateATerm_2_0.im");
  //TempImage<Complex> theCF(thisAVGPB.shape(), thisAVGPB.coordinates());
  TempImage<Complex> theCF(skyShape,thisAVGPB.coordinates());
  TempImage<Complex> thePB(skyShape,thisAVGPB.coordinates());
  Array<Complex> cfBuf=theCF.get();

  //
  // Setup the WTerm and ATerm objects.
  //
  // A number of w-terms are computed for one computation of the
  // ATerm.  Hence, compute and ATerm once re-use it with all the
  // w-terms we need.
  //
  cuLatticeFFT cuLatFFT;
  AntennaATerm aTerm(&cuLatFFT);
  cuConvolver cuConv(&cuLatFFT);
  cuWTerm cuWTerm;

  cuLatFFT.init(skyShape(0), skyShape(1));
  cuWTerm.setParams(skyShape(0), skyShape(1), TILE_WIDTHx, TILE_WIDTHy,
		  cellSize(0), wScale, cfBuf.shape()(0));
  //
  // Make storage on the device to hold the PB and a buffer for the CF
  //
  Int nBytes_p=skyShape(0)*skyShape(1)*sizeof(cufftComplex);
  cufftComplex *Ad_buf_p = (cufftComplex *) allocateDeviceBuffer(nBytes_p);
  cufftComplex *CFd_buf_p = (cufftComplex *) allocateDeviceBuffer(nBytes_p);
  aTerm.setDeviceBuffer((Complex *)Ad_buf_p);
  //
  // Apply the A-term to a buffer and re-use this buffer with multiple
  // w-terms.
  //
  Timer atimer;
  atimer.mark();
  //  cudaProfilerStart();
  aTerm.setApertureParams(pa, Freq, bandID, skyShape, uvIncr);
  aTerm.applyPB(thePB, pa,Freq, bandID, True);

  {// The un-necessary flip....
    Array<Complex>  tmp=thePB.get();
    FFTServer<Float, Complex> fftServer;
    fftServer.flip(tmp, True, False);
    thePB.put(tmp);

    // NON-OPTIMAL
    // Send the FT(ATerm -- the PB) back to the device.  This, will be
    // eliminated once ATerm is computed on the device too.
    //
    cufftComplex *Ah_buf_p = (cufftComplex *)tmp.data();
    sendBufferToDevice(Ad_buf_p, Ah_buf_p, nBytes_p);
  }

  //  storeDeviceBuf(theCF, Ad_buf_p, String("ATerm.im"));

  //  cudaProfilerStop();

  timeATerm+=atimer.all();
  //
  // Apply the w-term to a buffer and then multiply that buffer with
  // the A-term. FFT of the resulting buffer is the CF.  Re-size the
  // resulting buffer where the CF values fall by 1e-4 of the peak.
  //
  // For now, there is only one W-term.  Later we can put the
  // following code in a loop over nWterms.
  //
  Int iw0=700;
  Double ffttime=0,wtime=0,fliptime=0;

  for (iw=iw0;iw<nW+iw0;iw++)
  {
    cuWTerm.setWPixel(iw);
    cuConv.convolve(Ad_buf_p, CFd_buf_p, cuWTerm,
		  skyShape(0), skyShape(1), TILE_WIDTHx, TILE_WIDTHy);
  }

  //
  // Write out the CF.  Transform the image co-oridinates to uv-coordinates before saving.
  //
  storeDeviceBuf(theCF, CFd_buf_p, String("MyATerm.im"));
  
  return 0;
}


#ifdef HAS_OMP
#include <omp.h>
#endif

//
//----------------------------------------------------------------------
// A global method for use in OMP'ed findSupport() below
//
// void archPeak(const Float& threshold, const Int& origin, const Block<Int>& cfShape, const Complex* funcPtr, 
// 		const Int& nCFS, const Int& PixInc,const Int& th, const Int& R, 
// 		Block<Int>& maxR)
void archPeak(const Float& threshold, const Int& origin, const Int* cfShape, const Complex* funcPtr, 
	      const Int& nCFS, const Int& PixInc,const Int& th, const Int& R, 
	      Int* maxR)
{
  Block<Complex> vals;
  Block<Int> ndx(nCFS);	ndx=0;
  Int NSteps;
  //Check every PixInc pixel along a circle of radius R
  NSteps = 90*R/PixInc; 
  vals.resize((Int)(NSteps+0.5));
  uInt valsNelements=vals.nelements();
  vals=0;
  
  for(Int pix=0;pix<NSteps;pix++)
    {
      ndx[0]=(int)(origin + R*sin(2.0*M_PI*pix*PixInc/R));
      ndx[1]=(int)(origin + R*cos(2.0*M_PI*pix*PixInc/R));
      
      if ((ndx[0] < cfShape[0]) && (ndx[1] < cfShape[1]))
	//vals[pix]=func(ndx);
	vals[pix]=funcPtr[ndx[0]+ndx[1]*cfShape[1]+ndx[2]*cfShape[2]+ndx[3]*cfShape[3]];
    }
  
  maxR[th]=-R;
  for (uInt i=0;i<valsNelements;i++)
    if (fabs(vals[i]) > threshold)
      {
	maxR[th]=R;
	break;
      }
  //		th++;
}
Bool findSupport(Array<Complex>& func, Float& threshold, 
		 Int& origin, Int& radius)
{
  LogIO log_l(LogOrigin("AWConvFunc", "findSupport[R&D]"));
  
  Int nCFS=func.shape().nelements(),
    PixInc=1, R0, R1, R, convSize;
  Block<Int> cfShape(nCFS);
  Bool found=False;
  Complex *funcPtr;
  Bool dummy;
  uInt Nth=1, threadID=0;
  
  for (Int i=0;i<nCFS;i++)
    cfShape[i]=func.shape()[i];
  convSize = cfShape[0];
  
#ifdef HAS_OMP
  Nth = max(omp_get_max_threads()-2,1);
  cerr << "Firing " << Nth << " threads." << endl;
#endif
  
  Block<Int> maxR(Nth);
  Int *maxR_p, *cfShape_p;
  
  maxR_p=maxR.storage();
  cfShape_p = cfShape.storage();
  
  funcPtr = func.getStorage(dummy);
  
  R1 = convSize/2-2;
  radius=R1;
  while (R1 > 1)
    {
      R0 = R1; R1 -= Nth;
      
#pragma omp parallel default(none) firstprivate(R0,R1)  private(R,threadID) shared(origin, threshold, PixInc,maxR_p ,cfShape_p,nCFS,funcPtr) num_threads(Nth)
      { 
#pragma omp for
	for(R=R0;R>R1;R--)
	  {
#ifdef HAS_OMP
	    threadID=omp_get_thread_num();
#endif
	    archPeak(threshold, origin, cfShape_p, funcPtr, nCFS, PixInc, threadID, R, maxR_p);
	  }
	///#pragma omp barrier
      }///omp 	    
      
      for (uInt th=0;th<Nth;th++)
	{
	  if (maxR[th] > 0)
	    {
	      found=True; 
	      if (maxR[th] < radius) radius=maxR[th]; 
	    }
	}
      if (found) 
	return found;
    }
  return found;
}

Bool setUpCFSupport(Array<Complex>& func, Int& xSupport, Int& ySupport,
		    const Float& sampling, const Complex& peak)
{
  LogIO log_l(LogOrigin("AWConvFunc", "setUpCFSupport[R&D]"));
  //
  // Find the convolution function support size.  No assumption
  // about the symmetry of the conv. func. can be made (except that
  // they are same for all poln. planes).
  //
  xSupport = ySupport = -1;
  Int convFuncOrigin=func.shape()[0]/2, R; 
  Bool found=False;
  Float threshold;
  // Threshold as a fraction of the peak (presumed to be the center pixel).
  if (abs(peak) != 0) threshold = abs(peak);
  else 
    threshold   = (abs(func(IPosition(4,convFuncOrigin,convFuncOrigin,0,0))));
  cerr << "Threshold = " << threshold << endl;
  
  //    threshold *= aTerm_p->getSupportThreshold();
  threshold *= 1e-3;
  //    threshold *=  0.1;
  // if (aTerm_p->isNoOp()) 
  //   threshold *= 1e-3; // This is the threshold used in "standard" FTMchines
  // else
  
  //
  // Find the support size of the conv. function in pixels
  //
  // Timer tim;
  // tim.mark();
  Timer timer;
  timer.mark();
  if (found = findSupport(func,threshold,convFuncOrigin,R))
    xSupport=ySupport=Int(0.5+Float(R)/sampling)+1;
  
  // tim.show("findSupport:");
  cerr << "Time for findSupport = " << timer.all() << endl;
  
  if (xSupport*sampling > convFuncOrigin)
    {
      log_l << "Convolution function support size > N/2.  Limiting it to N/2, but this should be considered a bug"
	    << "(threshold = " << threshold << ")"
	    << LogIO::WARN;
      xSupport = ySupport = (Int)(convFuncOrigin/sampling);
    }
  
  if(xSupport<1) 
    log_l << "Convolution function is misbehaved - support seems to be zero"
	  << LogIO::EXCEPTION;
  return found;
}

Bool resizeCF(Array<Complex>& func, Int& xSupport, Int& ySupport,
	      const Float& sampling, const Complex& peak)
{
  Int ConvFuncOrigin=func.shape()[0]/2;  // Conv. Func. is half that size of convSize
  
  Bool found = setUpCFSupport(func, xSupport, ySupport, sampling,peak);
  
  return True;
  
  //Int supportBuffer = aTerm_p->getOversampling()*2;
  Int supportBuffer = OVERSAMPLING*2;
  Int bot=(Int)(ConvFuncOrigin-sampling*xSupport-supportBuffer),//-convSampling/2, 
    top=(Int)(ConvFuncOrigin+sampling*xSupport+supportBuffer);//+convSampling/2;
  //    bot *= 2; top *= 2;
  bot = casa::max(0,bot);
  top = casa::min(top, func.shape()(0)-1);
  
  Array<Complex> tmp;
  IPosition blc(4,bot,bot,0,0), trc(4,top,top,0,0);
  //
  // Cut out the conv. func., copy in a temp. array, resize the
  // CFStore.data, and copy the cutout version to CFStore.data.
  //
  tmp = func(blc,trc);
  func.resize(tmp.shape());
  func = tmp; 
  return found;
}
