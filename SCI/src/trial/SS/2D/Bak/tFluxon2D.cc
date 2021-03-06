#include <stdio.h>
#include <fstream>
#include <iostream>
#include <regFluxon2D.h>
#include <Array2D.h>
#include <Exp.h>
#include <ErrorObj.h>

#define EPS 1E-3
#define FTYPE double

Array2D<FTYPE> A2;

int main(int argc, char **argv)
{
  if (argc < 2 )
    {
      cerr << "Usage: " << argv[0] << " Scale" << endl;
      exit(-1);
    }

  int N=1000000, ImSize=256;
  double dScale;
  Exp<FTYPE> ExpTab(N,(float)(ImSize*ImSize)/N);
  GlobalParams::NSigma=5;
  GlobalParams::TableOfExponential = ExpTab;
  GlobalParams::XCenter = ImSize/2.0;
  GlobalParams::YCenter = ImSize/2.0;

  sscanf(argv[1],"%lf",&dScale);
  Array2D<FTYPE> PSFImg;
  Fluxon2D<FTYPE> f0(ExpTab,1.0,ImSize/2,ImSize/2,1);
  Fluxon2D<FTYPE> f1(ExpTab,1.0,ImSize/4,ImSize/2,1/dScale),f2;
  FTYPE A,S,P,Step;

  PSFImg.resize(ImSize,ImSize);
  f0.setScale(1/dScale);
  f2=f1.convolve(f0);
  PSFImg += f2;

  cerr << f0 << endl << f1 << endl << f2 << endl;

  int nx0,nx1,ny0,ny1;

  f2.getRange(ImSize, nx0,nx1,ny0,ny1);

  for (int i=0;i<ImSize;i++)
    {
      for (int j=0;j<ImSize;j++)
	{
	  cout << i << " " << j << " " << f1(i,j) << " ";
	  if (f1.inRange(i,j,nx0,nx1,ny0,ny1))
	    {
	      f1.evalDerivatives((FTYPE)i,(FTYPE)j);
	      cout << " " << f1.dF(f2.DA) 
		   << " " << f1.dF(f2.DXP) 
		   << " " << f1.dF(f2.DYP) 
		   << " " << f1.dF(f2.DS) 
		   << endl;
	    }
	  else  cout << "0 0 0 0" << endl;
	}
      cout << endl;
    }
}
