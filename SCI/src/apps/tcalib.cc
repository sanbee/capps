#include <casa/aips.h>
#include <casa/System/Aipsrc.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <synthesis/MeasurementComponents/Calibrater.h>
#include "./casaChecks.h"
#include <cl.h>
#include <clinteract.h>
#include <xmlcasa/Quantity.h>

using namespace std;
using namespace casa;
//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}
//#define RestartUI(Label)  {if(clIsInteractive()) {clRetry();goto Label;}}
//
void UI(Bool restart, int argc, char **argv, string& MSNBuf, string& CTNBuf, string& MINBuf,
	string& OutCTNBuf, string& OutDCBuf, string& fieldStr, string& timeStr, string& spwStr, string& antStr,
	string& uvrangeStr, string& jonesType, Float &Gain, Int &niter, Float &tol, string& integStr, /*Float &integ, */ Float &paInc,
	Int &wplanes, Int& nchan, Int& start, Int& step)
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
 	SMap watchPoints; vector<string> watchedKeys;

	i=1;clgetSValp("ms", MSNBuf,i);  
	i=1;clgetSValp("model",MINBuf,i);  
	i=1;clgetSValp("incal",CTNBuf,i);  
	i=1;clgetSValp("outcal",OutCTNBuf,i);  
	//	i=1;clgetFValp("integ",integ,i);  
	i=1;clgetSValp("integ",integStr,i);  

	i=1;clgetIValp("nchan",nchan,i);
	i=1;clgetIValp("start",start,i);
	i=1;clgetIValp("step",step,i);

        ClearMap(watchPoints);
	watchedKeys.resize(2);
	watchedKeys[0]="cfcache";  watchedKeys[1]="painc";
	watchPoints["EP"]=watchedKeys;

	i=1;clgetSValp("type",jonesType,i,watchPoints);
	i=1;clgetSValp("cfcache",OutDCBuf,i);  
	i=1;clgetFValp("painc",paInc,i);  

	clgetFullValp("field",fieldStr);
	clgetFullValp("spw",spwStr);  
	clgetFullValp("uvrange",uvrangeStr);
	clgetFullValp("antenna",antStr);
	clgetFullValp("time",timeStr);
	//
	// For the brave
	//
	i=1;dbgclgetFValp("gain",Gain,i);  
	i=1;dbgclgetIValp("niter",niter,i);  
	i=1;dbgclgetFValp("tolerance",tol,i);  
	i=1;dbgclgetIValp("wplanes",wplanes,i);  
      }
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
  string MSNBuf, MINBuf, CTNBuf, OutCTNBuf,
    OutDC, fieldStr, timeStr, spwStr, antStr, uvrangeStr, jonesType, integStr;
  Float Gain=0.1, Tolerance=1E-7, Integ=0,paInc=360.0;

  Int Niter=100, wPlanes=1, nchan=1, start=0, step=1;
  Bool restartUI=False;;
  MSSelection msSelection;
  static uInt cairc = Aipsrc::registerRC("calibrater.activity.interval", "3600.0");
  String cai_str = Aipsrc::get(cairc);
  Float cai; std::sscanf(std::string(cai_str).c_str(), "%f", &cai);

 RENTER:// UI re-entry point.
  MSNBuf = MINBuf = CTNBuf = OutCTNBuf = OutDC = timeStr = antStr ="";
  //
  // Factory defaults
  //
  jonesType="EP"; spwStr="*"; fieldStr="*";

  UI(restartUI,argc, argv, MSNBuf,CTNBuf,MINBuf, OutCTNBuf, OutDC, 
     fieldStr, timeStr, spwStr, antStr, uvrangeStr, jonesType, Gain, Niter, 
     Tolerance, integStr, paInc, wPlanes, nchan,start,step);
  restartUI = False;
  //
  //---------------------------------------------------
  //
  try
    {
//       if (!getenv("AIPSPATH"))
// 	throw(AipsError("Environment variable AIPSPATH not found.  "
// 			"Perhaps you forgot to source casainit.sh/csh?"));

      checkCASAEnv();

      Calibrater calib;
      String MSName(MSNBuf),CalTableName(CTNBuf),ModImgName(MINBuf),
	OutCalTableName(OutCTNBuf), diskCacheDir(OutDC);
      //
      // Make the MS 
      //
      //
      // Setup the MSSelection thingi
      //
      msSelection.setFieldExpr(fieldStr);
      msSelection.setTimeExpr(timeStr);
      msSelection.setSpwExpr(spwStr);
      msSelection.setAntennaExpr(antStr);
      msSelection.setUvDistExpr(uvrangeStr);
      MS ms(MSName,TableLock(TableLock::AutoLocking),Table::Update),selectedMS(ms);
      TableExprNode exprNode=msSelection.toTableExprNode(&ms);
      if (!exprNode.isNull()) selectedMS = MS(ms(exprNode));
	
      cout << "Opened MS: " << ms.tableName() << endl;

//     Block<Int> nosort(0);
//     Matrix<Int> noselection;
//     Double timeInterval=0;
//     VisSet* vs_p=new VisSet(selectedMS,nosort,noselection,timeInterval);
      //
      // Set up the Calibrater
      //
      calib.initialize(selectedMS);
//       calib.setdata("none",1,0,1,
// 		    Quantity(std::vector<double> (1, 0.0),"km/s"), 
// 		    Quantity(std::vector<double> (1, 0.0),"km/s")
// 		    Vector<Double>(1,0),
// 		    "");
      Vector<Double> uvrange(1,0);
//       calib.setdata("channel",  // Mode
// 		    nchan,       // NChan
// 		    start,       // Start
// 		    step       // Step
// 		    );  
      calib.selectvis("","","","","","","channel",nchan,start,step);
      calib.reset(True,True);
      Vector<Int> spwmap(1,-1);
      if (CTNBuf != "")
	calib.setapply("",           //Type
		       0.0,          //Integ
		       CTNBuf,       //Input Table
		       "nearest",    //Interp-type
		       "",           //select
		       False,        //calwt
		       spwmap,
		       0.0);         //opacity
//       calib.setsolve(String(jonesType), String(integStr), String(OutCalTableName), 
// 		     False, (Double)Integ, String(""), String(""), 
// 		     False, (Double)1.0, String(""), 0,
// 		     String(diskCacheDir), (Double)paInc);
      
      ostringstream IntegStr;
      //      IntegStr << Integ << "s";

      calib.setsolve(jonesType,       // Type
		     integStr,           // Solution interval
		     OutCalTableName,
		     False,           // Append to the cal. table?
		     Integ,           // Data integratin (before solver)
		     "AP",4,"",False,0.0f,"",0,
		     diskCacheDir,    // Name of the CF disk cache
		     paInc);          // PA increment (in deg).
      if (ModImgName != "") calib.setmodel(ModImgName);
      calib.solve();
      return 0;
    }
  catch (clError& x)
    {
      x << x.what() << endl;
      restartUI=True;
      //      RestartUI(RENTER);
    }
  catch (MSSelectionTimeError& x)
    {
      cerr << "###MSSelectionError: " << x.getMesg() << endl;
      restartUI=True;
      //      RestartUI(RENTER);
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
      //      RestartUI(RENTER);
    }
  if (restartUI) RestartUI(RENTER);
}
