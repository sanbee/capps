#include <cuLatticeFFT.h>
#include <cuWTerm.h>

#ifndef CUCONVOLVER_H
#define CUCONVOLVER_H

namespace casa{
  class cuConvolver
  {
  public:
    cuConvolver(cuLatticeFFT* cuLatFFT) {cufft_p = cuLatFFT;}
    ~cuConvolver() {};
    
    void convolve(const cuComplex *buf1_d, cuComplex *result_d, cuWTerm& wTerm);
    void convolve(const cuComplex *buf1_d, cuComplex *result_d, cuWTerm& wTerm,
		  const int nx, const int ny, const int tileWidthX, const int tileWidthY);

  private:
    cuLatticeFFT *cufft_p;
    
  };
};
#endif
