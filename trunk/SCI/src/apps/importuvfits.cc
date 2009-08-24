#include <casa/Logging.h>
#include <casa/Logging/LogIO.h>
#include <casa/Logging/LogMessage.h>
#include <casa/BasicSL/String.h>
#include <msfits/MSFits/MSFitsInput.h>
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
void UI(Bool restart, int argc, char **argv, string& fitsFileNameBuf, string& msFileNameBuf, 
	bool& newNameStyle)
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
	i=1;clgetSValp("fitsfile", fitsFileNameBuf,i);
	i=1;clgetSValp("msfile",msFileNameBuf,i);
	i=1;clgetBValp("newnames",newNameStyle,i);
      }
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}

int main(int argc, char **argv)
{
  string msFileNameBuf, fitsFileNameBuf;
  Bool newNameStyle=1;
  Bool restartUI=False;

 RENTER:// UI re-entry point.
  msFileNameBuf=fitsFileNameBuf="";
  UI(restartUI,argc, argv, fitsFileNameBuf, msFileNameBuf, newNameStyle);
  restartUI = False;

  //Bool newNameStyleBool=(newNameStyle >= 1);
  try
    {
      if (fitsFileNameBuf=="") throw(AipsError("Input file name is blank"));
      if (msFileNameBuf=="") throw(AipsError("Output file name is blank"));
      MSFitsInput uvfits2MS(msFileNameBuf, fitsFileNameBuf, newNameStyle);
      uvfits2MS.readFitsFile();
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
