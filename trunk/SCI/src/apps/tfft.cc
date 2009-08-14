#include <stdlib.h>
#include <casa/aips.h>
#include <synthesis/MeasurementEquations/ImagerMultiMS.h>
#include <synthesis/MeasurementComponents/Utils.h>
#include <images/Images/ImageOpener.h>
#include <images/Images/ImageInterface.h>
#include <images/Images/ImageRegrid.h>
#include <images/Images/PagedImage.h>
#include <lattices/Lattices/ArrayLattice.h>
#include <lattices/Lattices/SubLattice.h>
#include <lattices/Lattices/LCBox.h>
#include <lattices/Lattices/LatticeExpr.h>
#include <lattices/Lattices/LatticeCache.h>
#include <lattices/Lattices/LatticeFFT.h>
#include <lattices/Lattices/LatticeIterator.h>
#include <lattices/Lattices/LatticeStepper.h>
#include <cl.h>
#include <clinteract.h>
#include <xmlcasa/Quantity.h>
#include <casa/OS/Directory.h>

using namespace std;
using namespace casa;
//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}


void UI(Bool restart, int argc, char **argv, string& InImgName, string& OutImgName)
{
  if (!restart)
    {
      BeginCL(argc,argv);
      char TBuf[FILENAME_MAX];
      clgetConfigFile(TBuf,argv[0]);strcat(TBuf,".config");
      clloadConfig(TBuf);
      clInteractive(0);
    }
  else
   clRetry();
  try
    {
      int i;
      {
	i=1;clgetSValp("inimg", InImgName,i);  
	i=1;clgetSValp("outimg", OutImgName,i);  
      }
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}
//
//-------------------------------------------------------------------------
//
int main(int argc, char **argv)
{
  //
  //---------------------------------------------------
  //
  string InImgName, OutImgName;
  Bool restartUI=False;
 RENTER:// UI re-entry point.
  //
  // Factory defaults
  //
  InImgName=OutImgName="";
  //
  // The user interface
  //
  UI(restartUI,argc, argv, InImgName, OutImgName);
  restartUI = False;
  //---------------------------------------------------
  try
    {
      if (!(getenv("AIPSPATH") || getenv("CASAPATH")))
	throw(AipsError("Neither AIPSPATH nor CASAPATH environment variable found.  "
			"Perhaps you forgot to source casainit.sh/csh?"));
      LatticeBase* lattPtr = ImageOpener::openImage (String(InImgName));
      ImageInterface<Complex> *cImage;
      cImage = dynamic_cast<ImageInterface<Complex>* >(lattPtr);
      ImageInterface<Complex> outImg(*cImage);
      LatticeFFT::cfft(outImg,False);
      String Name(OutImgName);
      storeImg(Name, outImg);

    }
  catch (clError& x)
    {
      x << x.what() << endl;
      restartUI=True;
    }
  //
  // Catch any exception thrown by AIPS++ libs.  Do your cleanup here
  // before returning to the UI (if you choose to).  Without this, all
  // exceptions (AIPS++ or otherwise) are caught in the default
  // exception handler (which is installed by the CLLIB as the
  // clDefaultErrorHandler).
  //
  catch (AipsError& x)
    {
      cerr << "###AipsError: " << x.getMesg() << endl;
      restartUI=True;
    }
  if (restartUI) RestartUI(RENTER);
}
