#include <casa/aips.h>
#include <synthesis/CalTables/NewCalTable.h>
#include <synthesis/CalTables/CTInterface.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelectionTools.h>
#include <tables/Tables/PlainTable.h>
// Stuff from http://code.google.com/p/parafeed
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
void UI(Bool restart, int argc, char **argv, 
	string& CalNBuf, string& OutCTBuf,
	bool& deepCopy, 
	string& fieldStr, 
	string& antennaStr, 
	string& spwStr,
	string& timeStr,
	string& scanStr,
	string& obsidStr)
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
      CalNBuf=OutCTBuf="";
      i=1;clgetSValp("ct", CalNBuf,i);  
      i=1;clgetSValp("outct", OutCTBuf,i);  
      i=1;clgetBValp("deepcopy",deepCopy,i);
      clgetFullValp("time",timeStr);
      clgetFullValp("field",fieldStr);
      clgetFullValp("baseline",antennaStr);
      clgetFullValp("spw",spwStr);
      clgetFullValp("scan",scanStr);
      clgetFullValp("obsid",obsidStr);

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
void showTableCache()
{
  const TableCache& cache = PlainTable::tableCache();
  if(cache.ntable()!=0)
    cerr << endl << "####WARNING!!!!: The Table Cache has the following " << cache.ntable() << " entries:"  << endl;
  
  for (uInt i=0; i<cache.ntable(); ++i) 
    cerr << "    " << i << ": \"" <<  cache(i)->tableName() << "\"" << endl;
}
//
//-------------------------------------------------------------------------
//
void printInfo(MSSelection& msSelection)
{
  cout << "FieldId        = " << msSelection.getFieldList()    << endl;
  cout << "Ant1Id         = " << msSelection.getAntenna1List() << endl;
  cout << "Ant2Id         = " << msSelection.getAntenna2List() << endl;
  cout << "SpwId          = " << msSelection.getSpwList()      << endl;
  cout << "ScanList       = " << msSelection.getScanList() << endl;
  cout << "ObsIDList      = " << msSelection.getObservationList() << endl;
  cout << "ChanList       = " << msSelection.getChanList()      << endl;
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
  string CalNBuf,OutCTBuf,fieldStr, antennaStr, spwStr, timeStr, scanStr, obsidStr;
  Bool deepCopy=False;
  Bool restartUI=False;;

 RENTER:// UI re-entry point.
  CalNBuf=OutCTBuf=fieldStr=antennaStr="";
  fieldStr=antennaStr=spwStr=obsidStr="";
  deepCopy=False;
  UI(restartUI,argc, argv, CalNBuf, OutCTBuf, deepCopy, fieldStr, antennaStr, 
     spwStr,timeStr,scanStr,obsidStr);
  restartUI = False;
  //
  //---------------------------------------------------
  //
  try
    {
      NewCalTable calTab(CalNBuf), selectedCalTable(calTab);
      CTInterface msLike(calTab);

      MSSelection mss;
      MSSelectionLogError mssLE;
      mss.setErrorHandler(MSSelection::ANTENNA_EXPR,&mssLE);

      mss.setFieldExpr(fieldStr);
      mss.setAntennaExpr(antennaStr);
      mss.setSpwExpr(spwStr);
      mss.setTimeExpr(timeStr);
      mss.setScanExpr(scanStr);
      mss.setObservationExpr(obsidStr);
      //mss.setStateExpr(fieldStr);
      //      mss.reset(msLike, MSSelection::PARSE_LATE,"",antennaStr,fieldStr);

      TableExprNode ten=mss.toTableExprNode(&msLike);
      printInfo(mss);
      getSelectedTable(selectedCalTable, calTab, ten, "");

      if (OutCTBuf != "")
	{
	  if (deepCopy) selectedCalTable.deepCopy(OutCTBuf,Table::New);
	  else          selectedCalTable.rename(OutCTBuf,Table::New);
	  selectedCalTable.flush();
	}

      cerr << "Selected " << selectedCalTable.nrow() 
	   << " rows out of " << calTab.nrow() 
	   << " rows." << endl;
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
  showTableCache();
  //if (restartUI) 
  restartUI=True;RestartUI(RENTER);
}
