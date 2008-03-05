//-*- C++ -*-
#ifndef CONVOLVER_H
#define CONVOLVER_H
#include <iostream>

#include <fftw.h>
//#include <rfftw.h>
#include "Common.h"
#include <complex.h>
#include <vector.h>

using namespace std;

template <class T> class Convolver
{
public:
  Convolver<T>() {};
  Convolver<T>(IMAGETYPE<T> &PSF);

  ~Convolver<T>() 
  {
    //delete XFR;
  }
  
  void makeXFR(IMAGETYPE<T> &PSF,int doFlip=1);
  void Flip(IMAGETYPE<T> &Img);
  void convolve(IMAGETYPE<T> &Inp);
  void toFTDomain(IMAGETYPE<T> &Inp, IMAGETYPE<T> &tmp,int doNorm=1);
  void ifft(IMAGETYPE<T> &inImg, IMAGETYPE<T> &outImg);

  IMAGETYPE<T> & getXFR() {return XFR;}

private:
  //
  // This is a real image, resized later to [Nx][2*(Ny/2+1)].
  // Should be really complex image of size [Nx][Ny/2] - but
  // can't make that work.  So till then...
  //
  IMAGETYPE<T> XFR;
  //  rfftwnd_plan r2cplan,c2rplan;
  fftw_plan r2cplan,c2rplan;
};

template class Convolver<float>;
template class Convolver<double>;

#endif
