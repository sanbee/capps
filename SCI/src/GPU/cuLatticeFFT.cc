#include <cuLatticeFFT.h>
#include <lattices/Lattices/Lattice.h>

namespace casa{

  void cuLatticeFFT::init(const Int& nx, const Int& ny, cufftType type)
  {
    timer_p.mark();
    if ((h_buf_Nx != nx) || (h_buf_Ny != ny))
      {
	h_buf_Nx = nx; h_buf_Ny=ny;
	makeCUFFTPlan2D(nx,ny,type);
	setCUCompatibilityMode();
	nBytes_p=nx*ny*sizeof(cufftComplex);
	d_buf = (cufftComplex *) allocateDeviceBuffer(nBytes_p);
      }
    cerr << "cuLatticeFFT::init took " << timer_p.all() << "sec." << endl;
  }

  void cuLatticeFFT::cfft2d(Lattice<Complex> & cLattice, const Bool toFrequency)
  {
    timer_p.mark();
    Bool dummy;
    Array<Complex> tmp;
    cLattice.get(tmp);
    h_buf = (cufftComplex *)tmp.getStorage(dummy);
    sendBufferToDevice(d_buf, h_buf, nBytes_p);
    inPlaceCUFFTC2C(cufftPlan_p, (cufftComplex *)d_buf, CUFFT_FORWARD);
    getBufferFromDevice(h_buf,d_buf,nBytes_p);

    cerr << "cuLatticeFFT::cfft2d took " << timer_p.all() << "sec." << endl;
  }
};
