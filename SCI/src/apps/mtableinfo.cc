#include <casa/aips.h>
#include <ms/MeasurementSets/MSSummary.h>
#include <ms/MeasurementSets/MeasurementSet.h>
#include <casa/Logging/LogIO.h>
#include <casa/Logging/StreamLogSink.h>
#include <images/Images/TempImage.h>
#include <images/Images/ImageInterface.h>
#include <images/Images/ImageUtilities.h>
#include <images/Images/ImageOpener.h>
#include <images/Images/ImageSummary.h>
//#include <images/Images/ImageSummary.cc>
#include <lattices/Lattices/PagedArray.h>
#include <fstream>
//
#include <cl.h>
#include <clinteract.h>

using namespace std;
using namespace casa;

//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}
//#define RestartUI(Label)  {if(clIsInteractive()) {clRetry();goto Label;}}
//
void UI(Bool restart, int argc, char **argv, 
	vector<string>& MSNBuf, string& OutBuf,
	string& RefMSName,bool& verbose)
{
  if (!restart)
    {
      BeginCL(argc,argv);
      clInteractive(0);
    }
  else
   clRetry();
  try
    {
      int i;
      OutBuf="";
      i=0;clgetNSValp("table", MSNBuf,i);  
      i=1;clgetSValp("outfile",OutBuf,i);  
      i=1;clgetSValp("refmsname",RefMSName,i);
      i=1;clgetBValp("verbose",verbose,i);  
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
  //  MSSelection msSelection;
  string MSNBuf,OutBuf, RefMSName;
  vector<string> MSNBufv;
  Bool verbose=0, restartUI=False;;

 RENTER:// UI re-entry point.
  MSNBuf=OutBuf=RefMSName="";
  UI(restartUI,argc, argv, MSNBufv,OutBuf,RefMSName,verbose);
  restartUI = False;
  //
  //---------------------------------------------------
  //
  try
    {
      if (MSNBufv.size() == 0)	throw(AipsError("No table name given"));
      //      Table table(MSNBuf,Table::Update);
      String type;
      {
	Table table(MSNBufv[0],TableLock(TableLock::AutoNoReadLocking));
	TableInfo& info = table.tableInfo();
	type=info.type();
      }
      ofstream ofs(OutBuf.c_str());
      
      ostringstream os;
      //      StreamLogSink sink(LogMessage::NORMAL, ofs);
      LogSink sink(LogMessage::NORMAL,&os);
      LogIO logio(sink);
      //      ostream os(logio.output());

      //
      // We need polymorphic TableSummary class.
      //
      if (type=="Measurement Set")
	{
	  //	  MS ms(MSNBuf,Table::Update);
	  Block<String> strBlock(MSNBufv.size());
	  Block<String> subTables;
	  //	  Block<String> subTables(1);subTables[0]="ANTENNA";
	  for (uInt i=0;i<strBlock.nelements();i++) strBlock[i] = MSNBufv[i];
	  Table mmsTable(strBlock,subTables,TableLock(TableLock::AutoNoReadLocking));
	  if (RefMSName != "")
	    {
	      mmsTable.rename(RefMSName,Table::New);
	      //	      MS ms(MSNBuf,TableLock(TableLock::AutoNoReadLocking));
	    }
	  MS ms(mmsTable);
	  
	  MSSummary mss(ms);
	  
	  mss.list(logio, verbose!=0);
	  ofs << (os.str().c_str()) << endl;
//	  exit(0);
	}
      else if (type=="Image")
	{
	  LatticeBase* lattPtr = ImageOpener::openImage (MSNBuf);
	  ImageInterface<Float> *fImage;
	  ImageInterface<Complex> *cImage;

	  fImage = dynamic_cast<ImageInterface<Float>*>(lattPtr);
	  cImage = dynamic_cast<ImageInterface<Complex>*>(lattPtr);

	  if (fImage != 0)
	    {
	      logio << "Image data type  : Float" << LogIO::NORMAL;
	      ImageSummary<Float> ims(*fImage);
	      Vector<String> list = ims.list(logio);
	      ofs << (os.str().c_str()) << endl;
	    }
	  else if (cImage != 0)
	    {
	      logio << "Image data type  : Complex" << LogIO::NORMAL;
	      ImageSummary<Complex> ims(*cImage);
	      Vector<String> list = ims.list(logio);
	      ofs << (os.str().c_str()) << endl;
	    }
	  else
	    logio << "Unrecognized image data type." << LogIO::EXCEPTION;
//	  exit(0);
	}
      else 
	{
	  logio << "Unrecognized table type " << type << LogIO::EXCEPTION;
	}
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
  // exception handler clDefaultErrorHandler (installed by the CLLIB).
  //
  catch (AipsError& x)
    {
      cerr << "###AipsError: " << x.getMesg() << endl;
      restartUI=True;
      exit(0);
    }
  if (restartUI) RestartUI(RENTER);
}
