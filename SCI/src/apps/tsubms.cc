#include <casa/aips.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelectionTools.h>
#include <msvis/MSVis/SubMS.h>
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
void UI(Bool restart, int argc, char **argv, string& MSNBuf, string& OutMSBuf, 
	string& WhichColStr,int& deepCopy,string& fieldStr, string& timeStr, 
	string& spwStr, string& baselineStr,string& scanStr, string& arrayStr, 
	string& uvdistStr,string& taqlStr,float& integ)
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
      MSNBuf=OutMSBuf=timeStr=baselineStr=uvdistStr=scanStr=arrayStr="";
      i=1;clgetSValp("ms", MSNBuf,i);  
      i=1;clgetSValp("outms",OutMSBuf,i);  
      clgetFullValp("whichcol",WhichColStr);
      //      i=1;clgetIValp("deepcopy",deepCopy,i);
      clgetFullValp("field",fieldStr);
      clgetFullValp("time",timeStr);  
      i=1;clgetFValp("integ",integ,i);
      clgetFullValp("spw",spwStr);  
      clgetFullValp("baseline",baselineStr);  
      clgetFullValp("scan",scanStr);  
      clgetFullValp("array",arrayStr);  
      clgetFullValp("uvdist",uvdistStr);  
      dbgclgetFullValp("taql",taqlStr);  
      VString options;
      options.resize(5);
      options[0]="data"; options[1]="model"; options[2]="corrected";
      options[3]="all"; options[4]="(a list of comma-separated column names)";
      clSetOptions("whichcol", options);

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
  string MSNBuf,OutMSBuf,WhichColStr="data",fieldStr="*",timeStr,spwStr="*",
    baselineStr,uvdistStr, taqlStr,scanStr,arrayStr,corrStr;
  Int deepCopy=0;
  Bool restartUI=False;;
  Float integ=-1;
 RENTER:// UI re-entry point.
  MSNBuf=OutMSBuf=WhichColStr=fieldStr=timeStr=spwStr=baselineStr=uvdistStr=
    taqlStr=scanStr=corrStr=arrayStr="";
  WhichColStr="data"; fieldStr="*"; spwStr="*";
  deepCopy=0;
  UI(restartUI,argc, argv, MSNBuf,OutMSBuf, WhichColStr, deepCopy,
     fieldStr,timeStr,spwStr,baselineStr,scanStr,arrayStr,uvdistStr,taqlStr,integ);
  restartUI = False;
  corrStr.resize(0);
  //
  //---------------------------------------------------
  //
  try
    {
      if (OutMSBuf == "")
	throw(clError("Need an output MS name", "###UserError"));

      //MS ms(MSNBuf,Table::Update),selectedMS(ms);
      MS ms(MSNBuf,TableLock(TableLock::AutoNoReadLocking)),selectedMS(ms);

	if (OutMSBuf != "")
	  {
	    //
	    // Damn CASA::Strings!
	    //
	    String OutMSName(OutMSBuf), WhichCol(WhichColStr);
	    //	    SubMS splitter(selectedMS);
	    //
	    // SubMS class is not msselection compliant (it's a strange
	    // mix of msselection and selection-by-hand)!
	    //
	    SubMS splitter(ms);
	    Vector<Int> nchan(1,10), start(1,0), step(1,1);
	    String CspwStr(spwStr), CfieldStr(fieldStr), CbaselineStr(baselineStr),
	      CscanStr(scanStr), CuvdistStr(uvdistStr), CtaqlStr(taqlStr), CtimeStr(timeStr);
	    splitter.setmsselect(CspwStr, CfieldStr, CbaselineStr, CscanStr, CuvdistStr,
				 CtaqlStr);//, nchan,start, step);

	    splitter.selectTime(integ,CtimeStr);
	    splitter.makeSubMS(OutMSName, WhichCol);
	  }
	//      cerr << "Number of selected rows: " << selectedMS.nrow() << endl;
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
  if (restartUI) RestartUI(RENTER);
  //  restartUI=True;RestartUI(RENTER);
}
