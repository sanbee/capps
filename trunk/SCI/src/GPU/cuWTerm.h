#include <casa/Logging/LogIO.h>
#include <cuUtils.h>

#ifndef CUWTERM_H
#define CUWTERM_H

namespace casa{
  class cuWTerm
  {
  public:
    cuWTerm(const bool isNoOp=False):isNoOp_p(isNoOp) {};
    ~cuWTerm() {};
    
    void setWPixel(const double& iw) {wPixel_p = iw;}
    void setParams(const int nx,         const int ny,
		   const int tileWidthX, const int tileWidthY, 
		   const float sampling, const double wScale, 
		   const int inner,      const bool isNoOp=False)
    {
      nx_p=nx; ny_p=ny; 
      tileWidthX_p = tileWidthX;
      tileWidthY_p = tileWidthY;
      sampling_p = sampling;
      wScale_p = wScale;
      inner_p = inner;
      isNoOp_p=isNoOp;
    };
    
    void apply(cuComplex* buf_d, const cuComplex* aTerm_d, const int nx=0, const int ny=0)
    {
      if (nx > 0) nx_p=nx;
      if (ny > 0) ny_p=ny;
      
      wTermApplySky(buf_d, aTerm_d,  
		    nx_p,  ny_p, tileWidthX_p, tileWidthY_p, 
		    wPixel_p, sampling_p,  wScale_p, 
		    inner_p,  isNoOp_p);
    };

  private:
    bool isNoOp_p;
    int nx_p, ny_p, tileWidthX_p, tileWidthY_p, inner_p;
    float sampling_p;
    double wScale_p, wPixel_p;
  };
}
#endif
