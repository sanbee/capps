#include <casa/aips.h>
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
void UI(Bool restart, int argc, char **argv, string& MSNBuf, string& OutMSBuf, bool& deepCopy,
	string& fieldStr, string& timeStr, string& spwStr, string& baselineStr,
	string& scanStr, string& arrayStr, string& uvdistStr,string& taqlStr, string& polnStr,
	string& stateObsModeStr, string& observationStr)
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
      MSNBuf=OutMSBuf=timeStr=baselineStr=uvdistStr=scanStr=arrayStr=polnStr=stateObsModeStr="";
      i=1;clgetSValp("ms", MSNBuf,i);  
      i=1;clgetSValp("outms",OutMSBuf,i);  
      i=1;clgetBValp("deepcopy",deepCopy,i);
      clgetFullValp("field",fieldStr);
      clgetFullValp("time",timeStr);  
      clgetFullValp("spw",spwStr);  
      clgetFullValp("poln",polnStr);  
      clgetFullValp("baseline",baselineStr);  
      clgetFullValp("scan",scanStr);  
      clgetFullValp("array",arrayStr);  
      clgetFullValp("uvdist",uvdistStr);  
      clgetFullValp("stateobsmode",stateObsModeStr);  
      clgetFullValp("observation",observationStr);  
      dbgclgetFullValp("taql",taqlStr);  
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
void printBaselineList(Matrix<Int> list,ostream& os)
{
  os << "Baselines = ";
  IPosition shp=list.shape();
  for(Int j=0;j<shp(1);j++)
    {
      for(Int i=0;i<shp(0);i++)
	os << list(i,j) << " ";
      os << endl << "            " ;
    }
  os << endl;
}
//
//-------------------------------------------------------------------------
//
void printInfo(MSSelection& msSelection)
{
  cout << "Ant1         = " << msSelection.getAntenna1List() << endl;
  cout << "Ant2         = " << msSelection.getAntenna2List() << endl;
  cout << "Baselines    = " << msSelection.getBaselineList() << endl;
  cout << "Field        = " << msSelection.getFieldList()    << endl;
  cout << "SPW          = " << msSelection.getSpwList()      << endl;
  cout << "Chan         = " << msSelection.getChanList(NULL,1,True)     << endl;
  cout << "Freq         = " << msSelection.getChanFreqList(NULL,True)     << endl;
  cout << "Scan         = " << msSelection.getScanList()     << endl;
  cout << "StateObsMode = " << msSelection.getStateObsModeList()     << endl;
  cout << "Array        = " << msSelection.getSubArrayList() << endl;
  cout << "Time         = " << msSelection.getTimeList()     << endl;
  cout << "UVRange      = " << msSelection.getUVList()       << endl;
  cout << "UV in meters = " << msSelection.getUVUnitsList()  << endl;
  cout << "DDIDs        = " << msSelection.getDDIDList()     << endl;
  cout << "PolMap       = " << msSelection.getPolMap()       << endl;
  cout << "CorrMap      = " << msSelection.getCorrMap( )     << endl;
  cout << "StateList    = " << msSelection.getStateObsModeList() << endl;
  cout << "ObservationIDList    = " << msSelection.getObservationList() << endl;
  //  printBaselineList(msSelection.getBaselineList(),cout);
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
  string MSNBuf,OutMSBuf,fieldStr,timeStr,spwStr,baselineStr,
    uvdistStr,taqlStr,scanStr,arrayStr, polnStr,stateObsModeStr,
    observationStr;
  Bool deepCopy=0;
  Bool restartUI=False;;

 RENTER:// UI re-entry point.
  MSNBuf=OutMSBuf=fieldStr=timeStr=spwStr=baselineStr=
    uvdistStr=taqlStr=scanStr=arrayStr=polnStr=stateObsModeStr=observationStr="";
  deepCopy=0;
  fieldStr=spwStr="*";
  fieldStr=spwStr="";
  UI(restartUI,argc, argv, MSNBuf,OutMSBuf, deepCopy,
     fieldStr,timeStr,spwStr,baselineStr,scanStr,arrayStr,
     uvdistStr,taqlStr,polnStr,stateObsModeStr,observationStr);
  restartUI = False;
  //
  //---------------------------------------------------
  //
  //      MS ms(MSNBuf,Table::Update),selectedMS(ms);
  
  //
  // Make a new scope, outside which there should be no tables lefts open.
  //
  {
    MS ms(MSNBuf,TableLock(TableLock::AutoNoReadLocking)),selectedMS(ms);
    //
    // Setup the MSSelection thingi
    //
    
    MSInterface msInterface(ms);
    MSSelection msSelection;
    MSSelectionLogError mssLE;
    msSelection.setErrorHandler(MSSelection::ANTENNA_EXPR, &mssLE);
    try
      {
	// msSelection.reset(ms,MSSelection::PARSE_NOW,
	// 			timeStr,baselineStr,fieldStr,spwStr,
	// 			uvdistStr,taqlStr,polnStr,scanStr,arrayStr,
	// 			stateObsModeStr,observationStr);
	msSelection.reset(msInterface,MSSelection::PARSE_NOW,
			  timeStr,baselineStr,fieldStr,spwStr,
			  uvdistStr,taqlStr,polnStr,scanStr,arrayStr,
			  stateObsModeStr,observationStr);
	// TableExprNode ten=msSelection.toTableExprNode(&msInterface);
	// cerr << "TEN rows = " << ten.nrow() << endl;
	printInfo(msSelection);
	
	if (!msSelection.getSelectedMS(selectedMS))
	  {
	    cerr << "###Informational:  Nothing selected.  ";
	    if (OutMSBuf != "")
	      cout << "New MS not written." << endl;
	    else
	      cout << endl;
	  }
	else
	  if (OutMSBuf != "")
	    if (deepCopy) selectedMS.deepCopy(OutMSBuf,Table::New);
	    else          selectedMS.rename(OutMSBuf,Table::New);
	cerr << "Number of selected rows: " << selectedMS.nrow() << endl;
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

  showTableCache();
  //if (restartUI) 
  restartUI=True;RestartUI(RENTER);
}
