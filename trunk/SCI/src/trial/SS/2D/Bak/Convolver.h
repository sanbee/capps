//-*- C++ -*-
#ifndef CONVOLVER_H
#define CONVOLVER_H
#include <iostream.h>

#include <fftw.h>
#include <rfftw.h>
#include "Common.h"
#include <complex.h>
#include <iostream.h>
#include <vector.h>

template <class T> class Convolver
{
public:
  Convolver<T>() {};
  Convolver<T>(IMAGETYPE<T> &PSF);

  ~Convolver<T>() 
  {
    //delete XFR;
  }
  
  void makeXFR(IMAGETYPE<T> &PSF);
  void Flip(IMAGETYPE<T> &Img);
  void Flip(IMAGETYPE<T> &Img,int dir);
  void convolve(IMAGETYPE<T> &Inp);

private:
  //
  // This is a real image, resized later to [Nx][2*(Ny/2+1)].
  // Should be really complex image of size [Nx][Ny/2] - but
  // can't make that work.  So till then...
  //
  IMAGETYPE<T> XFR;
  rfftwnd_plan r2cplan,c2rplan;
};

template class Convolver<float>;
template class Convolver<double>;

#endif
