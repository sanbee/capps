#include <casa/aips.h>
#include <synthesis/CalTables/NewCalTable.h>
#include <synthesis/CalTables/CTInterface.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelectionTools.h>
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
void UI(Bool restart, int argc, char **argv, 
	string& CalNBuf, string& OutCTBuf,
	bool& deepCopy, string& fieldStr)
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
      clgetFullValp("field",fieldStr);

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
    cerr << endl << "The Table Cache has the following " << cache.ntable() << " entries:"  << endl;
  
  for (uInt i=0; i<cache.ntable(); ++i) 
    cerr << "    " << i << ": \"" <<  cache(i)->tableName() << "\"" << endl;
}
//
//-------------------------------------------------------------------------
//
void printInfo(MSSelection& msSelection)
{
  cout << "Field        = " << msSelection.getFieldList()    << endl;
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
  string CalNBuf,OutCTBuf,fieldStr;
  Bool deepCopy=False;
  Bool restartUI=False;;

 RENTER:// UI re-entry point.
  CalNBuf=OutCTBuf=fieldStr="";
  fieldStr="*";
  deepCopy=False;

  UI(restartUI,argc, argv, CalNBuf, OutCTBuf, deepCopy, fieldStr);
  restartUI = False;
  //
  //---------------------------------------------------
  //
  try
    {
      NewCalTable calTab(CalNBuf), selectedCalTable(calTab);
      CTInterface msLike(calTab);
      MSSelection mss;
      // MSSelectionLogError mssLE;
      // mss.setErrorHandler(MSSelection::ANTENNA_EXPR,&mssLE);

      mss.reset(msLike, MSSelection::PARSE_LATE,
		"","",fieldStr);
      TableExprNode ten=mss.toTableExprNode(&msLike);
      printInfo(mss);
      getSelectedTable(selectedCalTable, calTab, ten, "");

      if (OutCTBuf != "")
	if (deepCopy) selectedCalTable.deepCopy(OutCTBuf,Table::New);
	else          selectedCalTable.rename(OutCTBuf,Table::New);
      selectedCalTable.flush();

      cerr << "Selected " << selectedCalTable.nrow() << " rows out of " << calTab.nrow() << " rows." << endl;
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
