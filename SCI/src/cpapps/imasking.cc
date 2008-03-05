#include <images/Images/PagedImage.h>
#include <casa/Logging.h>
#include <casa/Logging/LogIO.h>
#include <casa/Logging/LogMessage.h>
#include <casa/BasicSL/String.h>
#include <display/QtViewer/QtClean.qo.h>
#include <display/Display/StandAloneDisplayApp.h>
#include <tables/Tables/Table.h>
#include <iostream>
using namespace std;
using namespace casa;

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
  //interactivemask(const String& image, const String& mask){

  //  LogIO os(LogOrigin(argv[0], argv[0], WHERE));
  if (argc < 3)
    {
      cerr << argv[0] << " usage: " << "<ImageFileName> <MaskImageFileName>" << endl;//LogIO::POST;
      return -1;
    }
  String image(argv[1]), mask(argv[2]);

   if(Table::isReadable(mask)) 
     {
       if (! Table::isWritable(mask)) 
	 {
	   cerr << "Mask image is not modifiable " << endl;//LogIO::WARN << LogIO::POST;
	   return -1;
	 }
    //we should regrid here is image and mask do not match
     }
   else
     {
       clone(image, mask);
     }
   QtApp::init();
   QtClean vwrCln(image, mask); 
   //  if(!vwrCln.loadImage(image, mask)){
   if(!vwrCln.imageLoaded())
     {
       cerr << "Failed to load image and mask in viewer" << endl;
	 //	  << LogIO::SEVERE << LogIO::POST;
       return -1;
     }
   return vwrCln.go();
}
