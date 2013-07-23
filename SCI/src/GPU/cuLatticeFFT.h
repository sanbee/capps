#include <lattices/Lattices/LatticeFFT.h>
#include <casa/OS/Timer.h>

#include <cuUtils.h>
#ifndef CULATTICEFFT_H
#define CULATTICEFFT_H

namespace casa{

  class cuLatticeFFT
  {
  public:
    cuLatticeFFT(): timer_p() {cufftPlan_p=h_buf_Nx=h_buf_Ny=0; h_buf_p = d_buf_p = NULL;};
    cuLatticeFFT(const cuLatticeFFT& other) {operator=(other);}

    void init(const Int& nx, const Int& ny, cufftType type=CUFFT_C2C);

    int makeCUFFTPlan2D(const Int nx, const Int ny, cufftType type=CUFFT_C2C)
    {return makeCUFFTPlan(&cufftPlan_p, nx, ny, type);};

    int setCUCompatibilityMode(cufftCompatibility mode=CUFFT_COMPATIBILITY_NATIVE)
    {return setCompatibilityMode(cufftPlan_p, mode);}

    void cfft2d(Lattice<Complex> & cLattice, const Bool toFrequency=True);
    void cfft2d(Lattice<Complex> & cLattice, cufftComplex *d_buf,
		const Bool toFrequency);
    void cfft2d(cufftComplex *d_buf, const Bool toFrequency=True);

    cuLatticeFFT& operator=(const cuLatticeFFT& other);


  private:
    cufftHandle cufftPlan_p;
    cufftComplex *h_buf_p, *d_buf_p;
    Int h_buf_Nx, h_buf_Ny;
    Int nBytes_p;
    Timer timer_p;
  };

};
#endif
