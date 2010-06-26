#include <images/Images/PagedImage.h>
#include <casa/Logging.h>
#include <casa/Logging/LogIO.h>
#include <casa/Logging/LogMessage.h>
#include <casa/BasicSL/String.h>
#include <synthesis/MeasurementEquations/ImagerMultiMS.h>
#include <display/Display/StandAloneDisplayApp.h>
#include <tables/Tables/Table.h>
#include <iostream>
#include "./casaChecks.h"
#include <cl.h>
#include <clinteract.h>
#include <iostream>
using namespace std;
using namespace casa;

//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}
void UI(Bool restart, int argc, char **argv, string& image, string& mask)
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
	i=1;clgetSValp("image", image,i);
	i=1;clgetSValp("mask",mask,i);
      }
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}

Bool clone(const String& imageName, const String& newImageName)
{
  //if(!valid()) return False;
  // This is not needed if(!assertDefinedImageParameters()) return False;
  LogIO os(LogOrigin("imager", "clone()", WHERE));
  try {
    PagedImage<Float> oldImage(imageName);
    PagedImage<Float> newImage(oldImage.shape(), oldImage.coordinates(),
                               newImageName);
  } catch (AipsError x) {
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  }
  return True;
}


int main(int argc, char **argv)
{
  String image(""), mask("");
  Bool restartUI=False;
 RENTER:// UI re-entry point.
  // if (argc < 3)
  //   {
  //     cerr << argv[0] << " usage: " << "<ImageFileName> <MaskImageFileName>" << endl;
  //     return -1;
  //   }
  //  String image(argv[1]), mask(argv[2]);
  image = mask = "";
  UI(restartUI,argc, argv, image, mask);
  restartUI=False;

   if(Table::isReadable(mask)) 
     {
       if (! Table::isWritable(mask)) 
	 {
	   cerr << "Mask image is not modifiable " << endl;//LogIO::WARN << LogIO::POST;
	   return -1;
	 }
    //we should regrid here is image and mask do not match
     }
   Imager imager;
   String threshold("0mJy");
   Int niter=0, ncycle=0;
   try
     {
       imager.interactivemask(image,mask,niter,ncycle,threshold);
     }
    catch (clError& x)
      {
	x << x.what() << endl;
	restartUI=True;
	//      RestartUI(RENTER);
      }
    catch (AipsError& x)
      {
	cerr << "###AipsError: " << x.getMesg() << endl;
	restartUI=True;
	//      RestartUI(RENTER);
      }
   if (restartUI) RestartUI(RENTER);
}
