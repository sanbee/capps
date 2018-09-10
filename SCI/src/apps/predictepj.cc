#include <stdlib.h>
#include <casa/aips.h>
#include <casa/IO/AipsIO.h>
#include <ms/MSSel/MSSelection.h>
#include <ms/MSSel/MSSelectionError.h>
#include <ms/MSSel/MSSelectionTools.h>
#include <ms/MSSel/MSSelectableTable.h>
#include <msvis/MSVis/VisBuffer.h>
#include <msvis/MSVis/VisSet.h>
#include <msvis/MSVis/VisSetUtil.h>
//#include <synthesis/MeasurementEquations/Imager.h>
//#include <synthesis/MeasurementEquations/ImagerMultiMS.h>
#include <synthesis/MeasurementComponents/nPBWProjectFT.h>
#include <synthesis/MeasurementComponents/EPJones.h>
#include <synthesis/CalTables/CalTable2.h>
#include <synthesis/CalTables/CalSet.h>
#include <synthesis/MeasurementEquations/VisEquation.h>
#include <casa/System/ProgressMeter.h>
#include <tables/Tables/TableProxy.h>

//#include <synthesis/TransformMachines/Utils.h>
#include <cl.h>
#include <clinteract.h>
//#include <xmlcasa/Quantity.h>
#include <casa/OS/Directory.h>

using namespace std;
using namespace casacore;
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
	float& rotpainc, bool& psterm, bool& aterm, bool& mterm, bool& wbawp, bool& conjbeams,
	bool& singlePrecision)
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
	vector<string> tmods;

	i=1;clgetSValp("ms", MSName,i);  
	i=0;clgetNSValp("model",tmods,i);
	toCASASVector(tmods,models);
	
	i=1;clgetFullValp("field",fieldStr);  
	i=1;clgetFullValp("spw",spwStr);  
	i=1;clgetFullValp("time",timeStr);  
	i=1;clgetFullValp("baseline",antStr);  
	i=1;clgetFullValp("uvrange",uvDistStr);  
	i=1;clgetFullValp("scan",scanStr);  

	InitMap(watchPoints, exposedKeys);
	ClearKeys(exposedKeys);
	exposedKeys.push_back("cfcache");  exposedKeys.push_back("painc");
	watchPoints["pbwproject"]=exposedKeys;	

	ClearKeys(exposedKeys);
	exposedKeys.push_back("cfcache");       exposedKeys.push_back("painc");
	exposedKeys.push_back("rotpainc");
	watchPoints["awproject"]=exposedKeys;

	i=1;clgetSValp("ftmachine",ftmac,i,watchPoints);
	i=1;clgetFValp("rotpainc",rotpainc,i);  
	
	i=1;clgetSValp("pointingtable",pointingTable,i);  
	i=1;clgetBValp("dopbcorr",dopbcorr,i);  
	i=1;clgetSValp("cfcache",cfcache,i);  
	i=1;clgetFValp("painc",paInc,i);  

	InitMap(watchPoints,exposedKeys);

	ClearKeys(exposedKeys);
	exposedKeys.push_back("copydata");
	watchPoints["predict"]=exposedKeys;
	i=1;clgetSValp("operation",operation,i,watchPoints);

	InitMap(watchPoints,exposedKeys);
	exposedKeys.push_back("copyboth");
	watchPoints["1"]=exposedKeys;
	i=1;clgetBValp("copydata",copydata,i,watchPoints);
	i=1;clgetBValp("copyboth",copyboth,i);
	//
	// Do some user support!;-) Set the possible options for various keywords.
	//
	VString options;

	options.resize(0);
	options.push_back("predict");
	options.push_back("residual");
	clSetOptions("operation",options);

	options.resize(0);
	options.push_back("pbwproject");
	options.push_back("awproject");

	clSetOptions("ftmachine",options);
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

  casa::VisSet vs_p(theMS, sort, noselection);
  casa::VisIter& vi = vs_p.iter();
  casa::VisBuffer vb(vi);
  vi.origin();
  vi.originChunks();

  for (vi.originChunks();vi.moreChunks();vi.nextChunk())
    for (vi.origin(); vi.more(); vi++) 
      if (incremental) 
  	{
  	  vi.setVis( (vb.modelVisCube() + vb.visCube()),
  		     casa::VisibilityIterator::Corrected);
  	  if (both)
  	    {
  	      vi.setVis(vb.correctedVisCube(),casa::VisibilityIterator::Observed);
  	      //	      vi.setVis(vb.correctedVisCube(),VisibilityIterator::Model);
  	    }
  	} 
      else 
  	{
  	  vi.setVis(vb.modelVisCube(),casa::VisibilityIterator::Corrected);
  	  if (both) vi.setVis(vb.modelVisCube(),casa::VisibilityIterator::Observed);
  	}
};
//
//-------------------------------------------------------------------------
//
  void makeComplexGrid(TempImage<Complex>& Grid, 
		       PagedImage<Float>& ModelImage,
		       casa::VisBuffer& vb)
  {
    Vector<Int> whichStokes(0);
    CoordinateSystem cimageCoord =
      casa::StokesImageUtil::CStokesCoord(//cimageShape,
				    ModelImage.coordinates(),
				    whichStokes,
				    casa::StokesImageUtil::CIRCULAR);
    
    Grid.resize(IPosition(ModelImage.ndim(),
			  ModelImage.shape()(0),
			  ModelImage.shape()(1),
			  ModelImage.shape()(2),
			  ModelImage.shape()(3)));
    
    Grid.setCoordinateInfo(cimageCoord);
    
    Grid.setMiscInfo(ModelImage.miscInfo());
    casa::StokesImageUtil::From(Grid,ModelImage);
    
    if(vb.polFrame()==MSIter::Linear) 
      casa::StokesImageUtil::changeCStokesRep(Grid,casa::StokesImageUtil::LINEAR);
    else casa::StokesImageUtil::changeCStokesRep(Grid,casa::StokesImageUtil::CIRCULAR);
  }
  //
  //-----------------------------------------------------------------------
  //  
void setModel(casa::VisSet& vs, casa::nPBWProjectFT *pbwp_p, casacore::TempImage<casacore::Complex>& uvGrid,
	      const String& modelImageName)
  {
    Block<int> sort(0);
    sort.resize(5);
    sort[0] = MS::FIELD_ID;
    sort[1] = MS::FEED1;
    sort[2] = MS::ARRAY_ID;
    sort[3] = MS::DATA_DESC_ID;
    sort[4] = MS::TIME;
    Matrix<Int> noselection;
    casa::ROVisIter& vi(vs.iter());
    casa::VisBuffer vb(vi);
    
    PagedImage<Float> modelImage(modelImageName);
    makeComplexGrid(uvGrid,modelImage,vb);
    vi.originChunks();
    vi.origin();
    pbwp_p->initializeToVis(uvGrid,vb);

    casacore::Vector<casacore::Int> polMap_p;
    polMap_p = pbwp_p->getPolMap();
    //    cout << "Pol Map = " << polMap_p << endl;
  }

void fillPointingOffsets(casacore::String name, casacore::Cube<Float>& offsets)
{
  casacore::Record tlock; tlock.define("option","user");
  casacore::TableProxy *pointing = new casacore::TableProxy(name,tlock,casacore::Table::Old);
  casacore::ValueHolder vh=pointing->getColumn(casacore::String("GAIN"),0,-1,1);
  offsets=vh.asArrayFloat();
}
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
  Bool restartUI=False;
  Bool applyPointingOffsets=False, applyPointingCorrections=True, usemodelcol=True;
  Bool psterm_b=True, aterm_b=True, mterm_b=True, wbawp_b=True, conjbeams_b=True;
  Float gain,threshold;
  Vector<String> models, restoredImgs, residuals,masks,psfs;
  String complist,operation;
  MSSelection msSelection;
  Bool useScratchColumns=True, singlePrecision=True;
  Float cycleFactor=1.0, cycleSpeedup=-1, constPB=0.4, minPB=0.1, cycleMaxPSFFraction=0.8;
  Float rotpainc=360.0;
  Int stopLargeNegatives=2, stopPointMode = -1;
  String scaleType = "NONE";
  Vector<String> fluxScale; fluxScale.resize(0);

  //  cudaSetDevice(0);
  //cerr << "###Info: Initializing device...";
  //cudaDeviceSynchronize();
  //if (cudaGetLastError() != cudaSuccess)
      //throw(AipsError("Cuda error:  Failed to initialize"));

  //cerr << "done." << endl;
 RENTER:// UI re-entry point.
  MSName=timeStr=antStr=uvDistStr=cfcache=complist=pointingTable="";
  //
  // Factory defaults
  //
  pblimit=0.05;
  stokes="I"; ftmac="pbwproject"; algo="cs"; operation="predict";
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
     MSScales,useScratchColumns,rotpainc, psterm_b, aterm_b, mterm_b, wbawp_b, conjbeams_b,
     singlePrecision);

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
	  datastep[i]=chanList(i,2)-chanList(i,3)+1;//chanList(i,3);
	  datanchan[i]=chanList(i,2)-chanList(i,3)+1;
	}
      //      cerr << datanchan << " " << datastart << " " << datastep << endl; 

      if (mode=="continuum") 
	{
	  casaMode="MFS";
	  imnchan=-1;
	  imstart=0;//datastart[0];
	  imstep=0;//datanchan[0];
	  datanchan.resize(1);datanchan[0]=-1;
	  datastart.resize(1);datastart[0]=0;
	  datastep.resize(1);datastep[0]=1;
	}
      else if (mode=="spectral") {imnchan=datanchan[0];imstart=datastart[0];imstep=datastep[0];}
      else if (mode=="pseudo") {}
      else throw(AipsError("Incorrect setting for keyword \"mode\".  "));

      if (!exprNode.isNull())
	{
	  selectedMS = MS(ms(exprNode));
	  spwid=msSelection.getSpwList();
	  fieldid=msSelection.getFieldList();
	}

      Bool compress=False;
      if (operation=="predict")
	{
	  useScratchColumns=True;
	  Int nFacets=1; Long cachesize=200000000; Int tilesize=16;
	  //Float paInc=1.0; // 1 deg.
	  Bool doPBCorr=true, applyPointingOffsets=true;
	  casa::nPBWProjectFT *pbwp_p;
	  String cfcacheDir=cfcache;
	  
	  pbwp_p = new casa::nPBWProjectFT(//*ms_p, 
				     nFacets, 
				     cachesize,
				     cfcacheDir,
				     applyPointingOffsets,  
				     doPBCorr,   //Bool do PB correction before prediction
				     tilesize,
				     paInc       //Float paSteps=1.0
				     );
	  casacore::Quantity patol(paInc,"deg");
	  pbwp_p->setPAIncrement(patol);


	  Block<int> sort(0);
	  sort.resize(5);
	  sort[0] = MS::FIELD_ID;
	  sort[1] = MS::FEED1;
	  sort[2] = MS::ARRAY_ID;
	  sort[3] = MS::DATA_DESC_ID;
	  sort[4] = MS::TIME;
	  Matrix<Int> noselection;
	  
	  casa::VisSet vs(selectedMS, sort, noselection);
	  casa::VisIter& vi(vs.iter());
	  casa::VisBuffer vb(vi), dAZVB, dELVB,r;

 	  
	  casa::EPJones *epJ = new casa::EPJones(vs,selectedMS);
	  casacore::TempImage<casacore::Complex> uvGrid;

	  setModel(vs, pbwp_p, uvGrid, models[0]);
	  //epJ->setModel(models[0]);
	  pbwp_p->setEPJones(epJ);
	  
	  // casacore::RecordDesc applyRecDesc;
	  // applyRecDesc.addField("table", TpString);
	  // applyRecDesc.addField("interp",TpString);
	  // casacore::Record applyRec(applyRecDesc);
	  // casacore::String epJTableName_p=pointingTable;
	  // applyRec.define("table",epJTableName_p);
	  // applyRec.define("interp", "nearest");
	  // epJ->setApply(applyRec);

	  // // casa::CalTable2 epjCalTab(epJTableName_p);
	  // // cerr << "Slots in EPJ table: " << epjCalTab.numberTimeSlots(0.01D) << " Max. ant. index: "
	  // //      << epjCalTab.maxAntenna() << endl;
	  // casa::CalSet<Float> fCS(epJTableName_p,"",1,4,27);

	  casacore::ProgressMeter pm(1.0, Double(selectedMS.nrow()),
				     "Predicting EPJ",
				     "", "", "", true,100);

	  Double rowsDone=0;
	  Cube<Float> pointingOffsets;

	  fillPointingOffsets(casacore::String(pointingTable), pointingOffsets);
	  cerr << "Offsets shape: " << pointingOffsets.shape() << endl;
	  for (vi.originChunks(); vi.moreChunks(); vi.nextChunk()) 
	    for (vi.origin(); vi.more(); vi++) 
	      {
	    	//	ve.collapse(vb);
	    	dAZVB = dELVB = vb;
	    	IPosition shp = vb.modelVisCube().shape();
	    	// Use the target VBs as temp. storage as well 
	    	// (reduce the max. mem. footprint)
	    	dAZVB.modelVisCube().resize(shp);
	    	dELVB.modelVisCube().resize(shp);
	    	vb.modelVisCube() = dAZVB.modelVisCube() = dELVB.modelVisCube() = Complex(0,0);
		
		// Predict model (but also the derivatives)
	    	pbwp_p->get(vb, dAZVB, dELVB, pointingOffsets);
		// Write the model to the MS
		vi.setVis(vb.modelVisCube(),casa::VisibilityIterator::Model);

		rowsDone+=shp[2];
		pm.update(rowsDone);
	      }

	  if (copydata)
	    {
	      if (copyboth)
		cerr << "###Info: Copying MODEL_DATA to DATA and CORRECTED_DATA columns." << endl;
	      else
		cerr << "###Info: Copying MODEL_DATA to CORRECTED_DATA columns." << endl;
		
	      copyMData2Data(selectedMS,copyboth);
	    }
	}

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
