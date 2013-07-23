#include <cuConvolver.h>
#include <cuUtils.h>

namespace casa{
  void cuConvolver::convolve(const cuComplex *buf1_d, cuComplex *result_d, cuWTerm& wTerm)
  {
    wTerm.apply(result_d, buf1_d);
    cufft_p->cfft2d(result_d);
  }

  void cuConvolver::convolve(const cuComplex *buf1_d, cuComplex *result_d, cuWTerm& wTerm,
			     const int nx, const int ny, const int tileWidthX, const int tileWidthY)
  {
    wTerm.apply(result_d, buf1_d);
    cufft_p->cfft2d(result_d);
    flip(result_d, nx,ny,tileWidthX,tileWidthY);
  }
};
