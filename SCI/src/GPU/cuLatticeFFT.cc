#include <cuLatticeFFT.h>
#include <lattices/Lattices/Lattice.h>

namespace casa{

  void cuLatticeFFT::init(const Int& nx, const Int& ny, cufftType type)
  {
    if ((h_buf_Nx != nx) || (h_buf_Ny != ny))
      {
	timer_p.mark();
	{
	  h_buf_Nx = nx; h_buf_Ny=ny;
	  makeCUFFTPlan2D(nx,ny,type);
	  setCUCompatibilityMode();
	  nBytes_p=nx*ny*sizeof(cufftComplex);
	  d_buf_p = (cufftComplex *) allocateDeviceBuffer(nBytes_p);
	}
	cerr << "cuLatticeFFT::init+device mem. allocation took " << timer_p.all() << "sec." << endl;
      }
  }

  void cuLatticeFFT::cfft2d(Lattice<Complex> & cLattice, 
			    const Bool toFrequency)
  {
    timer_p.mark();
    // Bool dummy;
    // Array<Complex> tmp;
    // cLattice.get(tmp);
    // h_buf_p = (cufftComplex *)tmp.getStorage(dummy);
    // sendBufferToDevice(d_buf_p, h_buf_p, nBytes_p);
    // inPlaceCUFFTC2C(cufftPlan_p, (cufftComplex *)d_buf_p, CUFFT_FORWARD);

    cfft2d(cLattice, d_buf_p, toFrequency);
    getBufferFromDevice(h_buf_p,d_buf_p,nBytes_p);

    cerr << "cuLatticeFFT::cfft2d took " << timer_p.all() << "sec." << endl;
  }
  //
  // Use an externally supplied device buffer.  Leave the result in
  // the supplied device buffer.
  //
  void cuLatticeFFT::cfft2d(Lattice<Complex> & cLattice, 
			    cufftComplex *d_buf,
			    const Bool toFrequency)
  {
    Bool dummy;
    Array<Complex> tmp;
    cLattice.get(tmp);
    h_buf_p = (cufftComplex *)tmp.getStorage(dummy);
    sendBufferToDevice(d_buf, h_buf_p, nBytes_p);
    inPlaceCUFFTC2C(cufftPlan_p, (cufftComplex *)d_buf, CUFFT_FORWARD);
    //    getBufferFromDevice(h_buf_p,d_buf_p,nBytes_p);
  }
  //
  // Use the supplied device buffer to do in-place FFT
  //
  void cuLatticeFFT::cfft2d(cufftComplex *d_buf,
			    const Bool toFrequency)
  {
    inPlaceCUFFTC2C(cufftPlan_p, (cufftComplex *)d_buf, CUFFT_FORWARD);
  }
};
