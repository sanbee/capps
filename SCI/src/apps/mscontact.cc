#include <casa/aips.h>
#include <casa/Containers/Block.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelectionTools.h>
#include <ms/MeasurementSets/MSSelectableTable.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/PlainTable.h>
#include <cl.h> // C++ized version
#include <clinteract.h>

using namespace std;
using namespace casa;

//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}
//#define RestartUI(Label)  {if(clIsInteractive()) {clRetry();goto Label;}}
//
void UI(Bool restart, int argc, char **argv, Block<String>& MSNBuf, string& OutMSBuf, Block<String>& SubTablesBuf)
{
  int i,nMS=0, nST=0;
  vector<string> MSBufs,STBufs;
  if (!restart)
    {
      BeginCL(argc,argv);
      clInteractive(0);
    }
  else
   clRetry();
  try
    {
      clgetNSValp("ms", MSBufs,nMS);  
      i=1;clgetSValp("outms",OutMSBuf,i);  
      clgetNSValp("subtables",STBufs,nST);
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
  nMS=MSBufs.size();
  nST=STBufs.size();
  MSNBuf.resize(nMS);
  for (i=0;i<nMS;i++) 
    {
      MSNBuf[i]=String(MSBufs[i]);
      cerr << MSNBuf[i] << endl;
    }
  SubTablesBuf.resize(nST);
  for (i=0;i<nST;i++) 
    {
      SubTablesBuf[i]=String(STBufs[i]);
      cerr << SubTablesBuf[i] << endl;
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
  //  MSSelection msSelection;
  string OutMSBuf;
  Block<String> MSNBuf,SubTablesBuf;
  Bool deepCopy=0;
  Bool restartUI=False;;

 RENTER:// UI re-entry point.
  MSNBuf.resize(0);OutMSBuf.resize(0); SubTablesBuf.resize(0);
  UI(restartUI,argc, argv, MSNBuf,OutMSBuf, SubTablesBuf);
  restartUI = False;
  //
  // Make a new scope, outside which there should be no tables lefts open.
  //
  {
    try
      {
	Table mms(MSNBuf,SubTablesBuf);
	String outname=OutMSBuf;
	mms.rename(outname,Table::New);
      }
    catch (clError& x)
      {
	x << x.what() << endl;
	restartUI=True;
      }
    catch (MSSelectionError& x)
      {
	cerr << "###MSSelectionError: " << x.getMesg() << endl;
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
  }

  //if (restartUI) 
  restartUI=True;RestartUI(RENTER);
}
