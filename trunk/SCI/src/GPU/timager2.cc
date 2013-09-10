#include <cuda_runtime.h>

#include <stdlib.h>
#include <casa/aips.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSSelectionError.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <synthesis/MSVis/VisSet.h>
#include <synthesis/MSVis/VisSetUtil.h>
//#include <synthesis/MeasurementEquations/Imager.h>
#include <synthesis/MeasurementEquations/ImagerMultiMS.h>
#include <synthesis/TransformMachines/Utils.h>
#include <cl.h>
#include <clinteract.h>
//#include <xmlcasa/Quantity.h>
#include <casa/OS/Directory.h>

using namespace std;
using namespace casa;
//
//-------------------------------------------------------------------------
//
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}

void toCASAVector(std::vector<float>& stdv, Vector<Float>& tmp)
{
  Int n=stdv.size();
//   if ((n > 0) && (stdv[0] < 0)) tmp.resize(0);
//   else
    {
      tmp.resize(n);
      for(Int i=0;i<n;i++) tmp(i) = stdv[i];
    }
}

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
	string& antStr, string& fieldStr, string& scanStr, string& uvDistStr, float& paInc, string& cfcache, 
	string& pointingTable, float& cellx, float& celly, string& stokes,string& mode, 
	string& ftmac,string& wtType, string& rmode,double &robust,Int &niter, Int &wplanes, 
	Int& nx, Int& ny, Vector<Int>& datanchan, Vector<Int>& datastart, Vector<Int>& datastep,
	Int &imnchan, Int &imstart, Int &imstep,Int& facets,Float& gain, Float& threshold,
	Vector<String>& models,Vector<String>& restoredImgs,Vector<String>& residuals, 
	Vector<String>& psfs, Vector<String>& masks,string& complist,string&algo,string& taql,
	string& operation,float& pblimit,float& cycleFactor,bool& applyOffsets,bool& dopbcorr,
	bool& interactive,Long& cache, bool& copydata, bool& copyboth, Vector<Float>& MSScales,
	bool& useScratch,
	float& rotpainc, bool& psterm, bool& aterm, bool& mterm, bool& wbawp, bool& conjbeams)
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
 	SMap watchPoints; vector<string> exposedKeys;
	vector<int> imsize(2,0);
	vector<float> cell(2,0);
	vector<string> tmods, trest, tresid,tmasks, tms, tpsfs;

	i=1;clgetSValp("ms", MSName,i);  
	//	toCASAVector(tms, MSName);

	i=2;clgetNIValp("imsize",imsize,i);
	if (i==1) ny=nx=imsize[0];
	else {nx = imsize[0]; ny=imsize[1];}

	i=2;clgetNFValp("cellsize",cell,i);
	if (i==1) celly=cellx=cell[0];
	else {cellx = cell[0]; celly=cell[1];}

	i=0;clgetNSValp("model",tmods,i);
	i=0;clgetNSValp("restored",trest,i);
	i=0;clgetNSValp("residual",tresid,i);
	i=0;clgetNSValp("psf",tpsfs,i);
	i=0;clgetNSValp("mask",tmasks,i);
	toCASASVector(tmods,models);
	toCASASVector(trest,restoredImgs);
	toCASASVector(tresid,residuals);
	toCASASVector(tpsfs,psfs);
	toCASASVector(tmasks,masks);

	i=1;clgetSValp("complist",complist,i);
	
	InitMap(watchPoints, exposedKeys);
	exposedKeys.push_back("facets");  exposedKeys.push_back("wplanes");
	watchPoints["wproject"]=exposedKeys;

	ClearKeys(exposedKeys);
	exposedKeys.push_back("facets");   exposedKeys.push_back("wplanes");
	exposedKeys.push_back("cfcache");  exposedKeys.push_back("painc");
	exposedKeys.push_back("dopbcorr"); exposedKeys.push_back("applyoffsets");  
	exposedKeys.push_back("pointingtable");
	watchPoints["pbwproject"]=exposedKeys;	
	watchPoints["pbmosaic"]=exposedKeys;	

	ClearKeys(exposedKeys);
	exposedKeys.push_back("facets");        exposedKeys.push_back("wplanes");
	exposedKeys.push_back("cfcache");       exposedKeys.push_back("painc");
	exposedKeys.push_back("rotpainc");      exposedKeys.push_back("psterm");
	exposedKeys.push_back("aterm");         exposedKeys.push_back("wbawp");
	exposedKeys.push_back("mterm");         exposedKeys.push_back("conjbeams");     
	exposedKeys.push_back("pointingtable");exposedKeys.push_back("applyoffsets");	
	exposedKeys.push_back("dopbcorr");
	watchPoints["awproject"]=exposedKeys;
	watchPoints["protoft"]=exposedKeys;

	i=1;clgetSValp("ftmachine",ftmac,i,watchPoints);
	i=1;clgetIValp("facets",facets,i);
	i=1;clgetIValp("wplanes",wplanes,i);  
	i=1;clgetBValp("psterm",psterm,i);  
	i=1;clgetBValp("aterm",aterm,i);  
	i=1;clgetBValp("mterm",mterm,i);  
	i=1;clgetBValp("wbawp",wbawp,i);  
	i=1;clgetBValp("conjbeams",conjbeams,i);  
	i=1;clgetFValp("rotpainc",rotpainc,i);  
	
	//
	// Key of "1" and "0" implies logical True and False for watchPoints.
	//
	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("pointingtable");
	watchPoints["1"]=exposedKeys;
	i=1;clgetBValp("applyoffsets",applyOffsets,i,watchPoints);
	i=1;clgetSValp("pointingtable",pointingTable,i);  
	i=1;clgetBValp("dopbcorr",dopbcorr,i);  
	i=1;clgetSValp("cfcache",cfcache,i);  
	i=1;clgetFValp("painc",paInc,i);  

	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("scales");
	watchPoints["multiscale"]=exposedKeys;
	i=1;clgetSValp("algorithm",algo,i,watchPoints);

	vector<float> dscales(1,0.0);
	i=0;clgetNFValp("scales",dscales,i);
	toCASAVector(dscales,MSScales);
	MSScales = dscales;

	i=1;clgetSValp("stokes",stokes,i);

	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("rmode");exposedKeys.push_back("robust");
 	watchPoints["briggs"]=exposedKeys;
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
	i=1;clgetFullValp("scan",scanStr);  

	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("imnchan");  
	exposedKeys.push_back("imstart");
	exposedKeys.push_back("imstep");   
	watchPoints["pseudo"]=exposedKeys;

	i=1;clgetSValp("mode",mode,i, watchPoints);
	// vector<int> dnc(1,1),dstrt(1,0),dstp(1,1);
	// i=0;clgetNIValp("datanchan",dnc,i);
	// i=0;clgetNIValp("datastart",dstrt,i);
	// i=0;clgetNIValp("datastep",dstp,i);
	// toCASAVector(dnc,datanchan);
	// toCASAVector(dstrt,datastart);
	// toCASAVector(dstp,datastep);

	i=1;clgetIValp("imnchan",imnchan,i);
	i=1;clgetIValp("imstart",imstart,i);
	i=1;clgetIValp("imstep",imstep,i);

	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("gain");  exposedKeys.push_back("niter");
	exposedKeys.push_back("threshold"); exposedKeys.push_back("interactive");
	watchPoints["clean"]=exposedKeys;

	ClearKeys(exposedKeys);
	exposedKeys.push_back("copydata");
	watchPoints["predict"]=exposedKeys;
	i=1;clgetSValp("operation",operation,i,watchPoints);

	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("copyboth");
	watchPoints["1"]=exposedKeys;
	i=1;clgetBValp("copydata",copydata,i,watchPoints);
	i=1;clgetBValp("copyboth",copyboth,i);
	

	i=1;clgetFValp("gain",gain,i);
	i=1;clgetIValp("niter",niter,i);
	i=1;clgetFValp("threshold",threshold,i);
	i=1;clgetBValp("interactive",interactive,i);
	//
	// Hidder stuff for the brave
	//
	i=1;dbgclgetFValp("cyclefactor",cycleFactor,i);  
	i=1;dbgclgetFValp("pblimit",pblimit,i);  
	i=1;dbgclgetFullValp("taql",taql);
	Float fcache=1024*1024*1024*2.0; 
	i=1;dbgclgetFValp("cache",fcache,i); cache=(Long)fcache;
	i=1;dbgclgetBValp("usescratch",useScratch,i);
	//
	// Do some user support!;-) Set the possible options for various keywords.
	//
	VString options;

	options.resize(0);
	options.push_back("clean"); options.push_back("predict"); options.push_back("psf");
	options.push_back("dirty"); options.push_back("residual");
	clSetOptions("operation",options);

	options.resize(0);
	options.push_back("continuum");
	options.push_back("spectral");
	options.push_back("pseudo");
	clSetOptions("mode",options);

	options.resize(0);
	options.push_back("uniform");options.push_back("natural");options.push_back("briggs");
	clSetOptions("weighting",options);

	options.resize(0);
	options.push_back("ft");options.push_back("wproject");options.push_back("pbwproject");
	options.push_back("pbmosaic");options.push_back("awproject");options.push_back("protoft");
	clSetOptions("ftmachine",options);

	options.resize(0);
	options.push_back("cs");options.push_back("clark");options.push_back("hogbom");
	options.push_back("mfclark");
	options.push_back("multiscale");
	clSetOptions("algorithm",options);

	options.resize(0);
	options.push_back("I");options.push_back("IV");options.push_back("IQUV");
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
bool mdFromString(casa::MDirection &theDir, const casa::String &in)
{
   bool rstat(false);
   String tmpA, tmpB, tmpC;
   std::istringstream iss(in);
   iss >> tmpA >> tmpB >> tmpC;
   casa::Quantity tmpQA;
   casa::Quantity tmpQB;
   casa::Quantity::read(tmpQA, tmpA);
   casa::Quantity::read(tmpQB, tmpB);
   if(tmpC.length() > 0){
      MDirection::Types theRF;
      MDirection::getType(theRF, tmpC);
      theDir = MDirection (tmpQA, tmpQB, theRF);
      rstat = true;
   } else {
      theDir = MDirection (tmpQA, tmpQB);
      rstat = true;
   }
   return rstat;
}
//
//-------------------------------------------------------------------------
//
void copyMData2Data(MeasurementSet& theMS, Bool both=False, Bool incremental=False)
{
  Block<int> sort(0);
  sort.resize(5);
  sort[0] = MS::FIELD_ID;
  sort[1] = MS::FEED1;
  sort[2] = MS::ARRAY_ID;
  sort[3] = MS::DATA_DESC_ID;
  sort[4] = MS::TIME;
  Matrix<Int> noselection;

  VisSet vs_p(theMS, sort, noselection);
  VisIter& vi = vs_p.iter();
  VisBuffer vb(vi);
  vi.origin();
  vi.originChunks();

  for (vi.originChunks();vi.moreChunks();vi.nextChunk())
    for (vi.origin(); vi.more(); vi++) 
      if (incremental) 
	{
	  vi.setVis( (vb.modelVisCube() + vb.visCube()),
		     VisibilityIterator::Corrected);
	  if (both)
	    {
	      vi.setVis(vb.correctedVisCube(),VisibilityIterator::Observed);
	      //	      vi.setVis(vb.correctedVisCube(),VisibilityIterator::Model);
	    }
	} 
      else 
	{
	  vi.setVis(vb.modelVisCube(),VisibilityIterator::Corrected);
	  if (both) vi.setVis(vb.modelVisCube(),VisibilityIterator::Observed);
	}
};
//
//-------------------------------------------------------------------------
//
int main(int argc, char **argv)
{
  //
  //---------------------------------------------------
  //
  string MSName, timeStr, spwStr, antStr, fieldStr, scanStr, uvDistStr, cfcache,pointingTable;
  string stokes,mode, casaMode, ftmac,wtType, rmode, algo, taql;
  Float padding=1.0, pblimit, paInc,cellx,celly;
  Long cache=2*1024*1024*1024L;
  Double robust=0.0;
  Int Niter=0, wPlanes=1, nx,ny, facets=1, imnchan=1, imstart=0, imstep=1;
  bool applyOffsets=false,dopbcorr=true, copydata=false, copyboth=false, interactive=false;
  Vector<int> datanchan(1,1),datastart(1,0),datastep(1,1);
  Vector<Float> MSScales(1,0.0);
  Bool restartUI=False;;
  Bool applyPointingOffsets=False, applyPointingCorrections=True, usemodelcol=True;
  Bool psterm_b=True, aterm_b=True, mterm_b=True, wbawp_b=True, conjbeams_b=True;
  Float gain,threshold;
  Vector<String> models, restoredImgs, residuals,masks,psfs;
  String complist,operation;
  MSSelection msSelection;
  Bool useScratchColumns=True;
  Float cycleFactor=1.0, cycleSpeedup=-1, constPB=0.4, minPB=0.1, cycleMaxPSFFraction=0.8;
  Float rotpainc=5.0;
  Int stopLargeNegatives=2, stopPointMode = -1;
  String scaleType = "NONE";
  Vector<String> fluxScale; fluxScale.resize(0);

  //  cudaSetDevice(0);
  cerr << "###Info: Initializing device...";
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess)
      throw(AipsError("Cuda error:  Failed to initialize"));

  cerr << "done." << endl;
 RENTER:// UI re-entry point.
  MSName=timeStr=antStr=uvDistStr=cfcache=complist=pointingTable="";
  //
  // Factory defaults
  //
  pblimit=0.05;
  stokes="I"; ftmac="ft"; algo="cs"; operation="clean";
  wtType="natural"; rmode="none"; mode="continuum";
  casaMode="channel";
  gain=0.1; paInc = 360.0;
  spwStr=""; fieldStr=""; scanStr=""; threshold=0;
  useScratchColumns=True;
  //
  // The user interface
  //
  UI(restartUI,argc, argv, MSName, timeStr, spwStr, antStr, fieldStr, scanStr, uvDistStr, paInc, 
     cfcache, pointingTable, cellx, celly, stokes,mode,ftmac,wtType,rmode,robust,
     Niter, wPlanes,nx,ny, datanchan,datastart,datastep,imnchan,imstart,imstep,
     facets,gain,threshold,models,restoredImgs,residuals,psfs,masks,complist,algo,taql,
     operation,pblimit,cycleFactor,applyOffsets,dopbcorr,interactive,cache,copydata,copyboth,
     MSScales,useScratchColumns,rotpainc, psterm_b, aterm_b, mterm_b, wbawp_b, conjbeams_b);

  // if (applyOffsets==1) applyPointingOffsets=True;else applyPointingOffsets=False;
  // if (dopbcorr==1) applyPointingCorrections=True;else applyPointingCorrections=False;
  applyPointingOffsets=applyOffsets;
  applyPointingCorrections=dopbcorr;
  restartUI = False;
  //---------------------------------------------------
  try
    {
      if (!(getenv("AIPSPATH") || getenv("CASAPATH")))
	throw(AipsError("Neither AIPSPATH nor CASAPATH environment variable found.  "
			"Perhaps you forgot to source casainit.sh/csh?"));
      // if (cfcache=="") 
      // 	throw(AipsError("CF cache directory name is blank!"));

      // cDataToGridImpl_p is an overloaded function.  The two names
      // below are the same, but the complier, via the ComplexGridder
      // and DComplexGridder typedefs from cDataToGridImpl.h has all
      // the information needed to pick up the correct instantiation
      // of cDataToGridImpl_p.

      // ComplexGridder fC = cDataToGridImpl_p;
      // DComplexGridder fD = cDataToGridImpl_p;
      // ComplexGridder fC = cuDataToGridImpl_p;
      // DComplexGridder fD = cuDataToGridImpl_p;

      Imager imager;
      //      imager.setGridder(fC, fD);
      
      String AMSName(MSName),diskCacheDir(cfcache);
      //
      // Make the MS 
      //
      //
      // Setup the MSSelection thingi
      //
      msSelection.setTimeExpr(timeStr);
      msSelection.setSpwExpr(spwStr);
      msSelection.setAntennaExpr(antStr);
      msSelection.setFieldExpr(fieldStr);
      msSelection.setScanExpr(scanStr);
      msSelection.setUvDistExpr(uvDistStr);
      MS ms(AMSName,Table::Update),selectedMS(ms);
      Vector<int> spwid, fieldid;
      TableExprNode exprNode=msSelection.toTableExprNode(&ms);
      Matrix<Int> chanList = msSelection.getChanList(NULL,1,True);

      datanchan.resize(chanList.shape()(0));
      datastart.resize(chanList.shape()(0));
      datastep.resize(chanList.shape()(0));
      for (int i=0;i<chanList.shape()(0);i++)
	{
	  datastart[i]=chanList(i,1);
	  datastep[i]=chanList(i,3);
	  datanchan[i]=chanList(i,2)-chanList(i,3)+1;
	}
      //      cerr << datanchan << " " << datastart << " " << datastep << endl; 

      if (mode=="continuum") {casaMode="mfs";imnchan=1;imstart=datastart[0];imstep=datanchan[0];}
      else if (mode=="spectral") {imnchan=datanchan[0];imstart=datastart[0];imstep=datastep[0];}
      else if (mode=="pseudo") {}
      else throw(AipsError("Incorrect setting for keyword \"mode\".  "));

      if (!exprNode.isNull())
	{
	  selectedMS = MS(ms(exprNode));
	  spwid=msSelection.getSpwList();
	  fieldid=msSelection.getFieldList();
	}
      //
      // Imager requires the list of spectral window IDs and field IDs
      // present in the MS that is supplied to it.  This sort of
      // defeats the advantage of pre-selected MS.
      //
      if (spwid.nelements()   == 0) {spwid.resize(1);   spwid(0)=0;}
      if (fieldid.nelements() == 0) {fieldid.resize(1); fieldid(0)=0;}
      //      fieldid.resize();
      //      indgen(spwid);indgen(fieldid);
      //
      // Set up the imager
      //
      Bool compress=False;
      if (operation == "predict") useScratchColumns=True;
      imager.open(selectedMS,compress,useScratchColumns);
      vector<double> pa(1);pa[0]=paInc;

      imager.setvp(False,       //dovp
		   True,        //userdefaultvp
		   "",          // vptable,
		   True,        //dosquint
		   casa::Quantity(paInc,"deg"),
		   casa::Quantity(180,"deg"), //skyposthreshold
		   ""           //telescope
		   );
      Vector<Int> antIndex;
      imager.setdata(casaMode,
		     datanchan,    //vector<int>
		     datastart,    //vector<int>
		     datastep,     //vector<int>
		     casa::Quantity(0,"km/s"),//mstart
		     casa::Quantity(1,"km/s"),//mstep
		     spwid,    //vector<int>
		     fieldid,  //<vector<int>
		     String(taql)//msselect
		     //		     "","",antIndex,"","","","",useScratchColumns
		     );
      Bool doshift=False;
      MDirection mphaseCenter;
      String phasecenter("18h00m02.978 -23d00m01.411 J2000");
      mdFromString(mphaseCenter, phasecenter);
      
      //      Int field0=getPhaseCenter(selectedMS,mphaseCenter,6);

      Int field0=getPhaseCenter(selectedMS,mphaseCenter);
      //      cerr << "####Putting phase center on field no. " << field0 << endl;
      //      mdFromString(mphaseCenter, phasecenter);
      doshift=True;


      Int centerFieldId=-1;
      String casaStokes(stokes), casaModeStr(casaMode);
      imager.defineImage(nx,ny,
			 casa::Quantity((Double)cellx,"arcsec"),
			 casa::Quantity((Double)celly,"arcsec"),
			 stokes,
			 mphaseCenter,
			 centerFieldId,
			 casaMode,
			 imnchan,imstart,imstep,
			 casa::Quantity(0,"Hz"),    //mstart, // Def=0 km/s
			 casa::Quantity(1,"km/s"),  //mstep, // Def=1 km/s
			 casa::Quantity(0,"km/s"),
			 spwid,
			 facets, 
			 casa::Quantity(0,"Hz"),   // Rest frequency (we don't care yet)
			 casa::MFrequency::LSRK,   // Rest freq. frame
			 casa::Quantity(0,"m"));   // Distance (==> not in the near field)
      if (operation != "predict")
	imager.weight(wtType,                        // Def="natural"
		      rmode,                         // Def="none"
		      casa::Quantity(0.0,"Jy"),      //noise, // Def="0.0Jy"
		      robust,    // Def=0
		      casa::Quantity(0.0,"arcsec"),//fieldOfView,// Def="0.0.arcsec"
		      0);        

      imager.setmfcontrol(cycleFactor,
			  cycleSpeedup,
			  cycleMaxPSFFraction,
			  stopLargeNegatives, 
			  stopPointMode,
			  scaleType,
			  minPB,
			  constPB,
			  fluxScale);
      MPosition mlocation;
      //mpFromString(mlocation, location);
      if (cache <= 0) cache=nx*ny*2;
      const String& freqinterpmethod="linear";
      const Int imageTileSizeInPix=0;
      const Bool singleprecisiononly=False;
      const Int numthreads=-1;

      imager.setoptions(ftmac,            //Def="ft"
			cache,            // Def=4194304
			16,               // tile Def=16
			"sf",             // gridfunction Def="sf"
			mlocation,        // Def=""
			padding,          // Def=1.0
			//			usemodelcol,
			wPlanes,
			pointingTable,    //epjTableName
			applyPointingOffsets,//Def=True
			applyPointingCorrections,//Def=true
			cfcache,          //Def=""
			rotpainc,
			paInc,            // Def=4.0
			pblimit,           // Def=0.05
			freqinterpmethod,
			imageTileSizeInPix,
			singleprecisiononly,
			numthreads,
			psterm_b, aterm_b, mterm_b, wbawp_b,conjbeams_b);
      if (algo == "multiscale")
	imager.setscales(String("uservector"), (Int)MSScales.nelements(), MSScales);

      Vector<Bool> fixed(1,False); // If True, this will make the rest
				   // of the code not go through
				   // deconv.
      if (operation=="clean")
	{
	  String thresholdStr("0.0mJy");
	  Int nPerCycle=100,status;
	  Int interactiveNiter=(Int)(Niter/nPerCycle + 0.5);
	  if (restoredImgs.nelements() == 0) restoredImgs.resize(1);
	  if (restoredImgs[0] == "") restoredImgs[0] = models[0] + ".clean";
	  if (residuals.nelements() == 0) residuals.resize(1);
	  if (residuals[0] == "") residuals[0] = models[0] + ".res";

	  imager.iClean(algo,
			Niter,
			gain,
			casa::Quantity(threshold,"mJy"),
			//			  "0.0mJy",
			False,                 //displayProgress
			models,                //Vector<String>
			fixed,                 //Vector<Bool>
			complist,              //String
			masks,                 //Vector<String>
			restoredImgs,          //Vector<String>
			residuals,             //Vector<String>
			psfs,
			interactive, 
			nPerCycle,String(""));
			// False
			// );
	}
      else if (operation=="predict")
	{
	  imager.ft(models,complist,False);
	  if (copydata)
	    {
	      if (copyboth)
		cerr << "###Info: Copying MODEL_DATA to DATA and CORRECTED_DATA columns." << endl;
	      else
		cerr << "###Info: Copying MODEL_DATA to CORRECTED_DATA columns." << endl;
		
	      copyMData2Data(selectedMS,copyboth);
	    }
	}
      else if (operation=="dirty")
	imager.makeimage("model",models[0]);
      else
	imager.makeimage(operation,models[0]);

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
