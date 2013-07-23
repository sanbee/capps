// -*- C -*-
#include <math.h>
#include <cufft.h>

void wTermApplySky(cufftComplex* screen, const int& nx, const int& ny, const int TILE_WIDTH,
		   const int* sampling, const double& wValue, const int& inner,
		   const bool& isNoOp)
{
      /* int WIDTH=ny; */
      /* dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ; */

      /* dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ; */

      /* kernel_setBuf<<<dimGrid,dimBlock>>> ( d_buf,nx,ny,TILE_WIDTH,val); */

  kernel_wTermApplySky(screen, nx, ny, TILE_WIDTH,sampling, wValue, inner,isNoOp);
}
void kernel_wTermApplySky(cufftComplex* screen, const int& nx, const int& ny, const int TILE_WIDTH,
		   const int* sampling, const double& wValue, const int& inner,
		   const bool& isNoOp)
{
    int convSize = nx;
    double twoPiW=2.0*M_PI*double(wValue);
    if (!isNoOp)
      {
	for (int iy=-inner/2;iy<inner/2;iy++) 
	  {
	    double m=sampling[1]*double(iy);
	    double msq=m*m;
	    for (int ix=-inner/2;ix<inner/2;ix++) 
	      {
		double l=sampling[0]*double(ix);
		double rsq=l*l+msq;
		if(rsq<1.0) 
		  {
		    double phase=twoPiW*(sqrt(1.0-rsq)-1.0);
		    screen[ix+convSize/2 + (iy+convSize/2)*ny].x*=cos(phase);
		    screen[ix+convSize/2 + (iy+convSize/2)*ny].y*=sin(phase);
		  }
	      }
	  }
      }
}
