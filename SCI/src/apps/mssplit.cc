#include <casa/aips.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelectionTools.h>
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
  cout << "Ant1         = " << msSelection.getAntenna1List() << endl
       << "Ant2         = " << msSelection.getAntenna2List() << endl
       << "Baselines = "    << msSelection.getBaselineList() << endl
       << "Field        = " << msSelection.getFieldList()    << endl
       << "SPW          = " << msSelection.getSpwList()      << endl
       << "Chan         = " << msSelection.getChanList(NULL,1,True)     << endl
       << "Scan         = " << msSelection.getScanList()     << endl
       << "StateObsMode  = " << msSelection.getStateObsModeList()     << endl
       << "Array        = " << msSelection.getSubArrayList() << endl
       << "Time         = " << msSelection.getTimeList()     << endl
       << "UVRange      = " << msSelection.getUVList()       << endl
       << "UV in meters = " << msSelection.getUVUnitsList()  << endl
       << "DDIDs        = " << msSelection.getDDIDList()     << endl
       << "PolMap       = " << msSelection.getPolMap()       << endl
       << "CorrMap      = " << msSelection.getCorrMap( )     << endl
       << "StateList    = " << msSelection.getStateObsModeList() << endl;
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
  UI(restartUI,argc, argv, MSNBuf,OutMSBuf, deepCopy,
     fieldStr,timeStr,spwStr,baselineStr,scanStr,arrayStr,
     uvdistStr,taqlStr,polnStr,stateObsModeStr,observationStr);
  restartUI = False;
  //
  //---------------------------------------------------
  //
  try
    {
      //      MS ms(MSNBuf,Table::Update),selectedMS(ms);
      MS ms(MSNBuf,TableLock(TableLock::AutoNoReadLocking)),selectedMS(ms);
      //
      // Setup the MSSelection thingi
      //
      MSSelection msSelection(ms,MSSelection::PARSE_NOW,
			      timeStr,baselineStr,fieldStr,spwStr,
			      uvdistStr,taqlStr,polnStr,scanStr,arrayStr,
			      stateObsModeStr,observationStr);

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
  //if (restartUI) 
  restartUI=True;RestartUI(RENTER);
}
