#include "Common.h"
#include "Convolver.h"
#include <vector.h>
#include <iostream.h>
//#include <rfftw.h>
#include <fftw.h>
//
//------------------------------------------------------------------------
//
template <class T> Convolver<T>::Convolver<T>(IMAGETYPE<T> &PSF)
{
  makeXFR(PSF);
}
//
//------------------------------------------------------------------------
//
template <class T> void Convolver<T>::makeXFR(IMAGETYPE<T> &PSF,int doFlip)
{
  int Nx,Ny,i,j;

  PSF.size(Nx,Ny);
  //
  // Resize the transfer function to hold one half of the
  // Hermitian function
  //
  XFR.resize((Nx),2*(Ny/2+1));

  if (doFlip) Flip(PSF);

  for (i=0;i<Nx;i++) for (j=0;j<Ny;j++) XFR.setVal(i,j,PSF(i,j));
  //
  // Make plans for the future!
  //
//   r2cplan = rfftw2d_create_plan(Nx,Ny,FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_IN_PLACE);
//   c2rplan = rfftw2d_create_plan(Nx,Ny,FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);
  r2cplan = fftw_plan_dft_r2c_2d(Nx,Ny,, FFTW_ESTIMATE|FFTW_IN_PLACE);
  c2rplan = fft_plan_dft_c2r_2d(Nx,Ny,FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);

  //
  // Make the transfer function and remember it
  //
  rfftwnd_one_real_to_complex(r2cplan,(fftw_real *)XFR.getStorage(),NULL);
  /*
  for (i=0;i<Nx;i++)
    {
      for (j=0;j<Ny;j+=2)
	cout << i << " " << j << " " << XFR(i,j) << " " << XFR(i,j+1) << endl;
      cout << endl;
    }
  */
}
//
//------------------------------------------------------------------------
//
template <class T> void Convolver<T>::ifft(IMAGETYPE<T> &inImg, IMAGETYPE<T> &outImg)
{
  int Nx,Ny;
  inImg.size(Nx,Ny);
  Ny = (Ny/2 - 1)*2;
  outImg.resize(Nx,Ny);
  rfftwnd_one_complex_to_real(c2rplan,
			      (fftw_complex *)inImg.getStorage(),
			      (fftw_real *)outImg.getStorage());
}
//
//------------------------------------------------------------------------
//
template <class T> void Convolver<T>::convolve(IMAGETYPE<T> &Img)
{
  static IMAGETYPE<T> tmp;
  toFTDomain(Img,tmp);
  ifft(tmp,Img);
}
//
//------------------------------------------------------------------------
//
template <class T> void Convolver<T>::toFTDomain(IMAGETYPE<T> &Img, IMAGETYPE<T> &tmp,int doNorm)
{
  T Norm;
  int Nx, Ny, tNx, tNy;
  
  Img.size(Nx,Ny);
  tNx = (Nx);  tNy = 2*(Ny/2+1);  tmp.resize(tNx,tNy);
  
  Norm=Nx*Ny;

  //
  // Copy the image to a tmp image.
  //
  for (int i=0;i<Nx;i++) for (int j=0;j<Ny;j++) tmp.setVal(i,j,Img(i,j));
  //
  // Take the R2C FFT of the input image
  //
  rfftwnd_one_real_to_complex(r2cplan, (fftw_real *)tmp.getStorage(),NULL);
  //
  // Point-by-point complex multiplication with the XFR...and normalize 
  // on the fly...
  //
  complex<T> *XFRbuf, *Imgbuf;

  XFRbuf = (complex<T> *)XFR.getStorage();
  Imgbuf = (complex<T> *)tmp.getStorage();

  tNy /= 2;

  //
  // Use an extra variable, but reduce multiplications by a factor of
  // O(N)!  Aaaahh...I am "over optimizing"!  But what the heck...
  //
  int ndx;
  if (doNorm)
    for (int i=0;i<tNx;i++)
      {
	ndx = i*tNy;
	for(int j=0;j<tNy;j++)
	  {
	    int ndy = ndx + j;
	    Imgbuf[ndy] *= XFRbuf[ndy]/Norm;
	  }
      }
  else
    for (int i=0;i<tNx;i++)
      {
	ndx = i*tNy;
	for(int j=0;j<tNy;j++)
	  {
	    int ndy = ndx + j;
	    Imgbuf[ndy] = XFRbuf[ndy];
	  }
      }
  /*
  //
  // C2R FFT of the result
  //
  rfftwnd_one_complex_to_real(c2rplan,
			      (fftw_complex *)tmp.getStorage(),
			      (fftw_real *)Img.getStorage());
  */
}
//
//------------------------------------------------------------------------
//
template <class T> void Convolver<T>::Flip(IMAGETYPE<T> &Img)
{
  int Nx,Ny,i,j;
  vector<T> tmpBuf;

  Img.size(Nx,Ny);

  /*
  for (i=0;i<Nx;i++)
    {
      for (j=0;j<Ny;j++) fprintf(stderr,"%3.f ", Img(i,j));
      fprintf(stderr,"\n");
    }
  fprintf(stderr,"----------------------------\n");
  */
  
  int N;

  N=Ny/2;   tmpBuf.resize(N);

  for (i=0;i<Nx;i++)
    {
      for (j=0;j<N;j++)  tmpBuf[j] = Img(i,j);
      for (j=0;j<N;j++)  Img.setVal(i,j, Img(i,j+N));
      for (j=N;j<Ny;j++) Img.setVal(i,j,tmpBuf[j-N]);
    }

  N=Nx/2;   tmpBuf.resize(N);

  for (j=0;j<Ny;j++)
    {
      for (i=0;i<N;i++)  tmpBuf[i] = Img(i,j);
      for (i=0;i<N;i++)  Img.setVal(i,j, Img(i+N,j));
      for (i=N;i<Nx;i++) Img.setVal(i,j,tmpBuf[i-N]);
    }

  /*
  for (i=0;i<Nx;i++)
    {
      for (j=0;j<Ny;j++) fprintf(stderr,"%3.f ", Img(i,j));
      fprintf(stderr,"\n");
    }
  fprintf(stderr,"----------------------------\n");
  */
}

