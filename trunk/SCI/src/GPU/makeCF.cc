#include <casa/aips.h>
#include <AntennaATerm.h>
#include <images/Images/PagedImage.h>

using namespace std;
using namespace casa;

int main(int argc, char **argv[])
{
  Float Freq = 1.4e9, pa=-1.2;
  Int bandID= BeamCalc::Instance()->getBandID(Freq,"EVLA");
  IPosition skyShape(4,1024,1024,1,1);
  Vector<Double> uvIncr(2,0.01);

  PagedImage<Complex> thisAVGPB("./TemplateATerm_2_0.im");
  TempImage<Complex> theCF(thisAVGPB.shape(), thisAVGPB.coordinates());

  AntennaATerm aat;
  aat.setApertureParams(pa, Freq, bandID, skyShape, uvIncr);
  aat.applyPB(theCF, pa,Freq, bandID, True);
  {
    PagedImage<Complex> tmp(theCF.shape(), theCF.coordinates(), String("MyATerm.im"));
    LatticeExpr<Complex> le(theCF);
    tmp.copyData(le);
  }
  //  aat.regridApertureEngine();

  return 0;
}
