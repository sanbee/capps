#include <casa/aips.h>
#include <stdlib.h>
#include <casa/System/Aipsrc.h>
#include <ms/MSSel/MSSelection.h>
#include <ms/MSSel/MSSelectionError.h>
//#include <msvis/MSVis/VisSet.h>
//#include <msvis/MSVis/VisSetUtil.h>
#include <synthesis/ImagerObjects/SynthesisImagerVi2.h>
#include <synthesis/ImagerObjects/SynthesisUtilMethods.h>
#include <synthesis/TransformMachines/Utils.h>
#include <cl.h>
#include <clinteract.h>
//#include <xmlcasa/Quantity.h>
#include <casa/OS/Directory.h>

using namespace std;
using namespace casacore;
using namespace casa;
//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}

void toCASAVector(std::vector<int>& stdv, Vector<Int>& tmp)
{
  Int n=stdv.size();
//   if ((n > 0) && (stdv[0] < 0)) tmp.resize(0);
//   else
    {
      tmp.resize(n);
      for(Int i=0;i<n;i++) tmp(i) = stdv[i];
    }
}

void toCASASVector(std::vector<string>& stdv, Vector<String>& tmp)
{
  Int n=stdv.size();
  tmp.resize(n);
  for(Int i=0;i<n;i++) tmp(i) = stdv[i];
}

void UI(Bool restart, int argc, char **argv, string& MSName, string& timeStr, string& spwStr, 
	string& antStr, string& fieldStr, string& uvDistStr, float& paInc, string& cfcache, 
	string& pointingTable, float& cellx, float& celly, string& phasecenter, string& stokes,string& mode, 
	string& ftmac,string& wtType, string& rmode,double &robust,Int &niter, Int &wplanes, 
	Int& nx, Int& ny, Vector<Int>& datanchan, Vector<Int>& datastart, Vector<Int>& datastep,
	Int &imnchan, Int &imstart, Int &imstep,Int& facets,Float& gain, Float& threshold,
	Vector<String>& models,Vector<String>& restoredImgs,Vector<String>& residuals, 
	Vector<String>& masks,string& complist,string&algo,string& taql,string& operation,
	float& pblimit,float& cycleFactor,int& applyOffsets,int& dopbcorr,
	Bool& interactive,Long& cache,	
	float& computepainc, bool& psterm, bool& aterm, bool& mterm, bool& wbawp, bool& conjbeams)
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
	vector<int> imsize(2,0);
	vector<float> cell(2,0);
	vector<string> tmods, trest, tresid,tmasks, tms;

	i=1;clgetSValp("ms", MSName,i);  
	//	toCASAVector(tms, MSName);

	i=2;clgetNIValp("imsize",imsize,i);
	if (i==1) ny=nx=imsize[0];
	else {nx = imsize[0]; ny=imsize[1];}

	i=2;clgetNFValp("cellsize",cell,i);
	if (i==1) celly=cellx=cell[0];
	else {cellx = cell[0]; celly=cell[1];}

	i=1;clgetSValp("phasecenter",phasecenter,i);

	i=0;clgetNSValp("model",tmods,i);
	i=0;clgetNSValp("restored",trest,i);
	i=0;clgetNSValp("residual",tresid,i);
	i=0;clgetNSValp("mask",tmasks,i);
	toCASASVector(tmods,models);
	toCASASVector(trest,restoredImgs);
	toCASASVector(tresid,residuals);
	toCASASVector(tmasks,masks);

	i=1;clgetSValp("complist",complist,i);
	
	ClearMap(watchPoints);
	watchedKeys.resize(2);
	watchedKeys[0]="facets";  watchedKeys[1]="wplanes";
	watchPoints["wprojectft"]=watchedKeys;

	watchedKeys.resize(13);
	watchedKeys[0]="facets";        watchedKeys[1]="wplanes";
	watchedKeys[2]="cfcache";       watchedKeys[3]="painc";
	watchedKeys[4]="computepainc";      watchedKeys[5]="psterm";
	watchedKeys[6]="aterm";         watchedKeys[7]="wbawp";
	watchedKeys[8]="mterm";         watchedKeys[9]="conjbeams";     
	watchedKeys[10]="pointingtable";watchedKeys[11]="applyoffsets";	
	watchedKeys[12]="dopbcorr";
	watchPoints["awprojectft"]=watchedKeys;
	watchPoints["awp"]=watchedKeys;

	i=1;clgetSValp("ftmachine",ftmac,i,watchPoints);
	i=1;clgetIValp("facets",facets,i);
	i=1;clgetIValp("wplanes",wplanes,i);  
	i=1;clgetBValp("psterm",psterm,i);  
	i=1;clgetBValp("aterm",aterm,i);  
	i=1;clgetBValp("mterm",mterm,i);  
	i=1;clgetBValp("wbawp",wbawp,i);  
	i=1;clgetBValp("conjbeams",conjbeams,i);  
	i=1;clgetFValp("computepainc",computepainc,i);  
	
	i=1;clgetFValp("painc",paInc,i);  
	i=1;clgetIValp("applyoffsets",applyOffsets,i);
	i=1;clgetIValp("dopbcorr",dopbcorr,i);  
	i=1;clgetSValp("pointingtable",pointingTable,i);  
	i=1;clgetSValp("cfcache",cfcache,i);  

	i=1;clgetSValp("algorithm",algo,i);
	
	i=1;clgetSValp("stokes",stokes,i);

	ClearMap(watchPoints);
 	watchedKeys.resize(2);
	watchedKeys[0]="rmode";watchedKeys[1]="robust";
 	watchPoints["briggs"]=watchedKeys;
	i=1;clgetSValp("weighting",wtType,i,watchPoints);

	i=1;clgetSValp("rmode",rmode,i);
	float frobust=0;
	i=1;clgetFValp("robust",frobust,i);
	robust=frobust;

	i=1;clgetFullValp("field",fieldStr);  
	i=1;clgetFullValp("spw",spwStr);  
	i=1;clgetFullValp("time",timeStr);  
	i=1;clgetFullValp("baseline",antStr);  
	i=1;clgetFullValp("uvrange",uvDistStr);  

	ClearMap(watchPoints);
	watchedKeys.resize(3);
	watchedKeys[0]="imnchan";  watchedKeys[1]="imstart";
	watchedKeys[2]="imstep";   
	watchPoints["pseudo"]=watchedKeys;

	i=1;clgetSValp("mode",mode,i, watchPoints);
	vector<int> dnc(1,1),dstrt(1,0),dstp(1,1);
	i=0;clgetNIValp("datanchan",dnc,i);
	i=0;clgetNIValp("datastart",dstrt,i);
	i=0;clgetNIValp("datastep",dstp,i);
	toCASAVector(dnc,datanchan);
	toCASAVector(dstrt,datastart);
	toCASAVector(dstp,datastep);

	i=1;clgetIValp("imnchan",imnchan,i);
	i=1;clgetIValp("imstart",imstart,i);
	i=1;clgetIValp("imstep",imstep,i);

	ClearMap(watchPoints);
	watchedKeys.resize(4);
	watchedKeys[0]="gain";      watchedKeys[1]="niter";
	watchedKeys[2]="threshold"; watchedKeys[3]="interactive";
	watchPoints["clean"]=watchedKeys;
	i=1;clgetSValp("operation",operation,i,watchPoints);

	i=1;clgetFValp("gain",gain,i);
	i=1;clgetIValp("niter",niter,i);
	i=1;clgetFValp("threshold",threshold,i);
	i=1;clgetBValp("interactive",interactive,i);
	//
	// Hidden stuff for the brave
	//
	i=1;dbgclgetFValp("cyclefactor",cycleFactor,i);  
	i=1;dbgclgetFValp("pblimit",pblimit,i);  
	i=1;dbgclgetFullValp("taql",taql);
	Float fcache=1024*1024*1024*2.0; 
	i=1;dbgclgetFValp("cache",fcache,i); cache=(Long)fcache;
	//
	// Do some user support!;-) Set the possible options for various keywords.
	//
	VString options;

	options.resize(5);
	options[0]="clean"; options[1]="predict"; options[2]="psf";
	options[3]="dirty"; options[4]="residual";
	clSetOptions("operation",options);

	options.resize(3);
	options[0]="mfs";options[1]="spectral";options[2]="pseudo";
	clSetOptions("mode",options);

	options.resize(3);
	options[0]="uniform";options[1]="natural";options[2]="briggs";
	clSetOptions("weighting",options);

	options.resize(4);
	options[0]="gridft";options[1]="wprojectft";options[2]="awprojectft"; options[3]="awp";
	clSetOptions("ftmachine",options);

	options.resize(2);
	//	options[0]="cs";options[1]="clark";options[2]="hogbom";options[3]="mfclark";
	options[0]="mtmfs";options[1]="hogbom";
	clSetOptions("algorithm",options);

	options.resize(3);
	options[0]="I";options[1]="IV";options[2]="IQUV";
	clSetOptions("stokes",options);
      }
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}
bool mdFromString(casacore::MDirection &theDir, const casacore::String &in)
{
  bool rstat(false);
  if (in.length()==0)
    {
      theDir=MDirection();
      rstat=false;
    }
  else
    {
   String tmpA, tmpB, tmpC;
   std::istringstream iss(in);
   iss >> tmpA >> tmpB >> tmpC;
   casacore::Quantity tmpQA;
   casacore::Quantity tmpQB;
   casacore::Quantity::read(tmpQA, tmpA);
   casacore::Quantity::read(tmpQB, tmpB);
   if(tmpC.length() > 0){
      MDirection::Types theRF;
      MDirection::getType(theRF, tmpC);
      theDir = MDirection (tmpQA, tmpQB, theRF);
      rstat = true;
   } else {
      theDir = MDirection (tmpQA, tmpQB);
      rstat = true;
   }
    }
  return rstat;
}
//
//-------------------------------------------------------------------------
//
// void copyMData2Data(MeasurementSet& theMS, Bool incremental=False)
// {
//   Block<int> sort(0);
//   sort.resize(5);
//   sort[0] = MS::FIELD_ID;
//   sort[1] = MS::FEED1;
//   sort[2] = MS::ARRAY_ID;
//   sort[3] = MS::DATA_DESC_ID;
//   sort[4] = MS::TIME;
//   Matrix<Int> noselection;

//   VisSet vs_p(theMS, sort, noselection);
//   VisIter& vi = vs_p.iter();
//   VisBuffer vb(vi);
//   vi.origin();
//   vi.originChunks();

//   for (vi.originChunks();vi.moreChunks();vi.nextChunk())
//     for (vi.origin(); vi.more(); vi++) 
//       if (incremental) 
// 	{
// 	  vi.setVis( (vb.modelVisCube() + vb.visCube()),
// 		     VisibilityIterator::Corrected);
// 	  vi.setVis(vb.correctedVisCube(),VisibilityIterator::Observed);
// 	  vi.setVis(vb.correctedVisCube(),VisibilityIterator::Model);
// 	} 
//       else 
// 	{
// 	  vi.setVis(vb.modelVisCube(),VisibilityIterator::Observed);
// 	  vi.setVis(vb.modelVisCube(),VisibilityIterator::Corrected);
// 	}
// };
//
//-------------------------------------------------------------------------
//
int main(int argc, char **argv)
{
  //
  //---------------------------------------------------
  //
  string MSName, timeStr, spwStr, antStr, fieldStr, uvDistStr, cfcache,pointingTable;
  string stokes,mode, casaMode, ftmac,wtType, rmode, algo, taql;
  Float padding=1.0, pblimit, paInc,cellx,celly;
  Long cache=2*1024*1024*1024L;
  Double robust=0.0;
  Int Niter=0, wPlanes=1, nx,ny, facets=1, imnchan=1, imstart=0, imstep=1, 
    applyOffsets=0,dopbcorr=1;
  Vector<int> datanchan(1,1),datastart(1,0),datastep(1,1);
  Bool restartUI=False;;
  Bool applyPointingOffsets=False, applyPointingCorrections=True, usemodelcol=True;
  Bool psterm_b=True, aterm_b=True, mterm_b=True, wbawp_b=True, conjbeams_b=True;
  Float gain,threshold;
  Vector<String> models, restoredImgs, residuals,masks,startModels;
  String complist,operation;
  //MSSelection msSelection;
  
  Float cycleFactor=1.0, cycleSpeedup=-1, constPB=0.4, minPB=0.1;
  Float computepainc=360.0;
  Int stopLargeNegatives=2, stopPointMode = -1;
  Bool interactive=false;
  String scaleType = "NONE";
  Vector<String> fluxScale; fluxScale.resize(0);
  String phasecenter;
 RENTER:// UI re-entry point.
  MSName=timeStr=antStr=uvDistStr=cfcache=complist=pointingTable="";
  //
  // Factory defaults
  //
  pblimit=0.05;
  stokes="I"; ftmac="gridft"; algo="hogbom"; operation="clean";
  wtType="uniform"; rmode="none"; mode="mfs";
  casaMode="FREQ";
  gain=0.1; paInc = 360.0;
  spwStr=""; fieldStr=""; threshold=0;
  phasecenter="";
  //
  // The user interface
  //
  UI(restartUI,argc, argv, MSName, timeStr, spwStr, antStr, fieldStr, uvDistStr, paInc, 
     cfcache, pointingTable, cellx, celly, phasecenter, stokes,mode,ftmac,wtType,rmode,robust,
     Niter, wPlanes,nx,ny, datanchan,datastart,datastep,imnchan,imstart,imstep,
     facets,gain,threshold,models,restoredImgs,residuals,masks,complist,algo,taql,
     operation,pblimit,cycleFactor,applyOffsets,dopbcorr,interactive,cache,computepainc, psterm_b, aterm_b, mterm_b, wbawp_b, conjbeams_b);
  
  if (applyOffsets==1) applyPointingOffsets=True;else applyPointingOffsets=False;
  if (dopbcorr==1) applyPointingCorrections=True;else applyPointingCorrections=False;
  restartUI = False;
  //---------------------------------------------------
  try
    {
      if (!(getenv("AIPSPATH") || getenv("CASAPATH")))
	throw(AipsError("Neither AIPSPATH nor CASAPATH environment variable found.  "
			"Perhaps you forgot to source casainit.sh/csh?"));
      
      SynthesisImagerVi2 imager;
      String AMSName(MSName),diskCacheDir(cfcache);
      vector<double> pa(1);pa[0]=paInc;
      
      SynthesisParamsSelect selectParams;
      selectParams.msname=MSName;
      selectParams.spw=spwStr;
      selectParams.field=fieldStr;
      selectParams.antenna=antStr;
      selectParams.timestr=timeStr;
      selectParams.uvdist=uvDistStr;
      selectParams.taql=taql;
      
      imager.selectData(selectParams);
      
      Bool doshift=False;
      MDirection mphaseCenter;
      //String phasecenter("18h00m00.00 -23d00m00.000 J2000");
      mdFromString(mphaseCenter, phasecenter);
      
      //      Int field0=getPhaseCenter(selectedMS,mphaseCenter,0);
      
      // Int field0=getPhaseCenter(selectedMS,mphaseCenter);
      // cerr << "####Putting phase center on field no. " << field0 << endl;
      //      mdFromString(mphaseCenter, phasecenter);
      doshift=True;
      
      if (mode=="mfs") {imnchan=1;imstart=datastart[0];imstep=datanchan[0];}
      else if (mode=="pseudo") {}
      else if (mode=="spectral") {imnchan=datanchan[0];imstart=datastart[0];imstep=datastep[0];}
      else throw(AipsError("Incorrect setting for keyword \"mode\".  Possible values are \"mfs\", \"pseudo\", or \"spectral\""));
      Int centerFieldId=-1;
      String casaStokes(stokes), casaModeStr(casaMode);
      casacore::MRadialVelocity mvel;
      casacore::MFrequency mfreq;
      casacore::Quantity qstart;
      
      Vector<Quantity> restFreqs(1);
      restFreqs[0]=Quantity(0,"km/s");
      startModels.resize(1);startModels[0]="";
      if (operation == "predict") startModels[0]=models[0];
      
      SynthesisParamsImage imageParams;
      imageParams.imageName=models[0];
      imageParams.startModel=startModels[0];
      imageParams.imsize[0]=nx; imageParams.imsize[1]=ny;
      imageParams.cellsize[0]=Quantity((double)cellx,"arcsec"); 
      imageParams.cellsize[1]=Quantity((double)celly,"arcsec"); 
      imageParams.stokes=casaStokes;
      imageParams.phaseCenter=mphaseCenter;
      imageParams.mode=mode;
      imageParams.chanStart=imstart;
      imageParams.chanStep=imstep;
      imageParams.deconvolver=algo;
      
      SynthesisParamsGrid gridParams;
      gridParams.ftmachine=ftmac;
      gridParams.wprojplanes=wPlanes;
      gridParams.aTermOn=aterm_b;
      gridParams.psTermOn=psterm_b;
      gridParams.mTermOn=mterm_b;
      gridParams.wbAWP=wbawp_b;
      gridParams.doPointing=true;
      gridParams.conjBeams=conjbeams_b;
      gridParams.rotatePAStep=paInc;
      gridParams.computePAStep=computepainc;
      gridParams.cfCache=cfcache;
      gridParams.facets=facets;
      gridParams.useDoublePrec=true; 
      
      imager.defineImage(imageParams, gridParams);
      
      
      imager.weight(wtType,                        // Def="natural"
		    rmode,                         // Def="none"
		    casacore::Quantity(0.0,"Jy"),  //noise, // Def="0.0Jy"
		    robust,                        // Def=0
		    casacore::Quantity(0.0,"arcsec"),//fieldOfView,// Def="0.0.arcsec"
		    0);        
      //      if (operation != "predict")
      //imager.makePSF();
      Record majorCycleControls;
      majorCycleControls.define("lastcycle", False);
      imager.executeMajorCycle(majorCycleControls);
      imager.makePSF();
      return 0;
    }
  catch (clError& x)
    {
      x << x.what() << endl;
      restartUI=True;
    }
  catch (MSSelectionTimeError& x)
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
}
