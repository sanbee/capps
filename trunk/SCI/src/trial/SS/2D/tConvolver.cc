#include "Common.h"
#include <Convolver.n.h>
#define N 64
#define TYPE double

main()
{
  IMAGETYPE<TYPE> tst,PSF,img;
  
  tst.resize(N,N); tst.assign(0);

  PSF.resize(N,N); PSF.assign(0);

  //  for (int i=0;i<N;i++)
  //    for (int j=0;j<N;j++)
  //      PSF.setVal(i,j,i*j);
  
  //  tst.setVal(N/2,N/2,10.0);
  tst.setVal(32,32,10.0);
  //  for (int i=N/2-5;i<N/2+5;i++)    for (int j=N/2-5;j<N/2+5;j++)      tst.setVal(i,j,1.0);

  for (int i=N/2-5;i<N/2+5;i++)    for (int j=N/2-5;j<N/2+5;j++)      PSF.setVal(i,j,10.0);

  //  PSF.setVal(N/2,N/2,2.0);
 
  Convolver<TYPE> Convo;
  Convo.makeXFR(PSF);
  img=tst;
  Convo.convolve(img);
  cout << img << endl;
}
