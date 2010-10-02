//# Imager.cc: Implementation of Imager.h
//# Copyright (C) 1997-2008
//# Associated Universities, Inc. Washington DC, USA.
//#
//# This program is free software; you can redistribute it and/or modify it
//# under the terms of the GNU General Public License as published by the Free
//# Software Foundation; either version 2 of the License, or (at your option)
//# any later version.
//#
//# This program is distributed in the hope that it will be useful, but WITHOUT
//# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
//# more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with this program; if not, write to the Free Software Foundation, Inc.,
//# 675 Massachusetts Ave, Cambridge, MA 02139, USA.
//#
//# Correspondence concerning AIPS++ should be addressed as follows:
//#        Internet email: aips2-request@nrao.edu.
//#        Postal address: AIPS++ Project Office
//#                        National Radio Astronomy Observatory
//#                        520 Edgemont Road
//#                        Charlottesville, VA 22903-2475 USA
//#
//# $Id$

#include <casa/Exceptions/Error.h>
#include <casa/iostream.h>
#include <synthesis/MeasurementEquations/Imager.h>

#include <ms/MeasurementSets/MSHistoryHandler.h>

#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/ArrayLogical.h>

#include <casa/Logging.h>
#include <casa/Logging/LogIO.h>
#include <casa/Logging/LogMessage.h>

#include <casa/OS/DirectoryIterator.h>
#include <casa/OS/File.h>
#include <casa/OS/Path.h>

#include <casa/OS/HostInfo.h>
#include <tables/Tables/RefRows.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/SetupNewTab.h>
#include <tables/Tables/TableParse.h>
#include <tables/Tables/TableRecord.h>
#include <tables/Tables/TableDesc.h>
#include <tables/Tables/TableLock.h>
#include <tables/Tables/ExprNode.h>

#include <casa/BasicSL/String.h>
#include <casa/Utilities/Assert.h>
#include <casa/Utilities/Fallible.h>
#include <casa/Utilities/CompositeNumber.h>

#include <casa/BasicSL/Constants.h>

#include <casa/Logging/LogSink.h>
#include <casa/Logging/LogMessage.h>

#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/Slice.h>
#include <images/Images/ImageAnalysis.h>
#include <images/Images/ImageExpr.h>
#include <images/Images/ImagePolarimetry.h>
#include <synthesis/MeasurementEquations/ClarkCleanProgress.h>
#include <lattices/Lattices/LatticeCleanProgress.h>
#include <msvis/MSVis/VisSet.h>
#include <msvis/MSVis/VisSetUtil.h>
#include <msvis/MSVis/VisImagingWeight.h>
// Disabling Imager::correct() (gmoellen 06Nov20)
//#include <synthesis/MeasurementComponents/TimeVarVisJones.h>

#include <measures/Measures/Stokes.h>
#include <casa/Quanta/UnitMap.h>
#include <casa/Quanta/UnitVal.h>
#include <casa/Quanta/MVAngle.h>
#include <measures/Measures/MDirection.h>
#include <measures/Measures/MPosition.h>
#include <casa/Quanta/MVEpoch.h>
#include <casa/Quanta/MVTime.h>
#include <measures/Measures/MEpoch.h>
#include <measures/Measures/MeasTable.h>

#include <ms/MeasurementSets/MeasurementSet.h>
#include <ms/MeasurementSets/MSColumns.h>
#include <ms/MeasurementSets/MSSelection.h>
#include <ms/MeasurementSets/MSDataDescIndex.h>
#include <ms/MeasurementSets/MSDopplerUtil.h>
#include <ms/MeasurementSets/MSSourceIndex.h>
#include <ms/MeasurementSets/MSSummary.h>
#include <synthesis/MeasurementEquations/CubeSkyEquation.h>
// Disabling Imager::correct() (gmoellen 06Nov20)
//#include <synthesis/MeasurementEquations/VisEquation.h>
#include <synthesis/MeasurementComponents/ImageSkyModel.h>
#include <synthesis/MeasurementComponents/CEMemImageSkyModel.h>
#include <synthesis/MeasurementComponents/MFCEMemImageSkyModel.h>
#include <synthesis/MeasurementComponents/MFCleanImageSkyModel.h>
#include <synthesis/MeasurementComponents/CSCleanImageSkyModel.h>
#include <synthesis/MeasurementComponents/MFMSCleanImageSkyModel.h>
#include <synthesis/MeasurementComponents/HogbomCleanImageSkyModel.h>
#include <synthesis/MeasurementComponents/MSCleanImageSkyModel.h>
#include <synthesis/MeasurementComponents/NNLSImageSkyModel.h>
#include <synthesis/MeasurementComponents/WBCleanImageSkyModel.h>
#include <synthesis/MeasurementComponents/GridBoth.h>
#include <synthesis/MeasurementComponents/MosaicFT.h>
#include <synthesis/MeasurementComponents/WProjectFT.h>
#include <synthesis/MeasurementComponents/nPBWProjectFT.h>
#include <synthesis/MeasurementComponents/PBMosaicFT.h>
#include <synthesis/MeasurementComponents/PBMath.h>
#include <synthesis/MeasurementComponents/SimpleComponentFTMachine.h>
#include <synthesis/MeasurementComponents/SimpCompGridMachine.h>
#include <synthesis/MeasurementComponents/VPSkyJones.h>
#include <synthesis/MeasurementComponents/SynthesisError.h>
#include <synthesis/MeasurementComponents/HetArrayConvFunc.h>

#include <synthesis/DataSampling/SynDataSampling.h>
#include <synthesis/DataSampling/SDDataSampling.h>
#include <synthesis/DataSampling/ImageDataSampling.h>
#include <synthesis/DataSampling/PixonProcessor.h>

#include <synthesis/MeasurementEquations/StokesImageUtil.h>
#include <lattices/Lattices/LattRegionHolder.h>
#include <lattices/Lattices/TiledLineStepper.h> 
#include <lattices/Lattices/LatticeIterator.h> 
#include <lattices/Lattices/LatticeExpr.h> 
#include <lattices/Lattices/LatticeFFT.h>
#include <lattices/Lattices/LCEllipsoid.h>
#include <lattices/Lattices/LCRegion.h>
#include <lattices/Lattices/LCBox.h>
#include <lattices/Lattices/LCIntersection.h>
#include <lattices/Lattices/LCUnion.h>
#include <lattices/Lattices/LCExtension.h>

#include <images/Images/ImageRegrid.h>
#include <images/Regions/ImageRegion.h>
#include <images/Regions/RegionManager.h>
#include <images/Regions/WCBox.h>
#include <images/Regions/WCUnion.h>
#include <images/Regions/WCIntersection.h>
#include <synthesis/MeasurementComponents/PBMath.h>
#include <images/Images/PagedImage.h>
#include <images/Images/ImageInfo.h>
#include <images/Images/SubImage.h>
#include <images/Images/ImageMetaData.h>
#include <images/Images/ImageUtilities.h>
#include <coordinates/Coordinates/CoordinateSystem.h>
#include <coordinates/Coordinates/DirectionCoordinate.h>
#include <coordinates/Coordinates/SpectralCoordinate.h>
#include <coordinates/Coordinates/StokesCoordinate.h>
#include <coordinates/Coordinates/Projection.h>
#include <coordinates/Coordinates/ObsInfo.h>

#include <components/ComponentModels/ComponentList.h>
#include <components/ComponentModels/ConstantSpectrum.h>
#include <components/ComponentModels/TabularSpectrum.h>
#include <components/ComponentModels/Flux.h>
#include <components/ComponentModels/FluxStandard.h>
#include <components/ComponentModels/PointShape.h>
#include <components/ComponentModels/DiskShape.h>

#include <casadbus/viewer/ViewerProxy.h>
#include <casadbus/plotserver/PlotServerProxy.h>
#include <casadbus/utilities/BusAccess.h>
#include <casadbus/session/DBusSession.h>

#include <casa/OS/HostInfo.h>

#include <components/ComponentModels/ComponentList.h>

#include <measures/Measures/UVWMachine.h>

#include <casa/sstream.h>

#include <sys/types.h>
#include <unistd.h>

using namespace std;


#ifdef PABLO_IO
#include "PabloTrace.h"
#endif


namespace casa { //# NAMESPACE CASA - BEGIN

Imager::Imager() 
  :  msname_p(""), vs_p(0), rvi_p(0), wvi_p(0), ft_p(0), 
     cft_p(0), se_p(0),
     sm_p(0), vp_p(0), gvp_p(0), setimaged_p(False), nullSelect_p(False), 
     viewer_p(0), clean_panel_p(0), image_id_p(0), mask_id_p(0), prev_image_id_p(0), prev_mask_id_p(0)
{
  ms_p=0;
  mssel_p=0;
  lockCounter_p=0;
  defaults();
};


void Imager::defaults() 
{

#ifdef PABLO_IO
traceEvent(1,"Entering imager::defaults",25);
#endif

  setimaged_p=False;
  nullSelect_p=False;
  nx_p=128; ny_p=128; facets_p=1;
  wprojPlanes_p=1;
  mcellx_p=Quantity(1, "arcsec"); mcelly_p=Quantity(1, "arcsec");
  shiftx_p=Quantity(0.0, "arcsec"); shifty_p=Quantity(0.0, "arcsec");
  distance_p=Quantity(0.0, "m");
  stokes_p="I"; npol_p=1;
  nscales_p=5;
  ntaylor_p=1;
  reffreq_p=0.0;
  scaleMethod_p="nscales";  
  scaleInfoValid_p=False;
  dataMode_p="none";
  imageMode_p="MFS";
  dataNchan_p=0;
  imageNchan_p=0;
  doVP_p=False;
  doDefaultVP_p = True;
  parAngleInc_p=Quantity(360.,"deg");
  skyPosThreshold_p=Quantity(180.,"deg");
  telescope_p="";
  gridfunction_p="SF";
  doMultiFields_p=False;
  doWideBand_p=False;
  multiFields_p=False;
  // Use half the machine memory as cache. The user can override
  // this via the setoptions function().
  cache_p=(HostInfo::memoryTotal(true)/8)*1024;
  //On 32 bit machines with more than 2G of mem this can become negative
  // overriding it to 2 Gb.
  if(cache_p <=0 )
    cache_p=2000000000/8;
  tile_p=16;
  ftmachine_p="ft";
  wfGridding_p=False;
  padding_p=1.2;
  sdScale_p=1.0;
  sdWeight_p=1.0;
  sdConvSupport_p=-1;

  doShift_p=False;
  spectralwindowids_p.resize(1); 
  spectralwindowids_p=0;
  fieldid_p=0;
  dataspectralwindowids_p.resize(0); 
  datadescids_p.resize(0);
  datafieldids_p.resize(0);
  mImageStart_p=MRadialVelocity(Quantity(0.0, "km/s"), MRadialVelocity::LSRK);
  mImageStep_p=MRadialVelocity(Quantity(0.0, "km/s"), MRadialVelocity::LSRK);
  mDataStart_p=MRadialVelocity(Quantity(0.0, "km/s"), MRadialVelocity::LSRK);
  mDataStep_p=MRadialVelocity(Quantity(0.0, "km/s"), MRadialVelocity::LSRK);
  beamValid_p=False;
  bmaj_p=Quantity(0, "arcsec");
  bmin_p=Quantity(0, "arcsec");
  bpa_p=Quantity(0, "deg");
  images_p.resize(0);
  masks_p.resize(0);
  fluxMasks_p.resize(0);
  residuals_p.resize(0);
  componentList_p=0;

  cyclefactor_p = 1.5;
  cyclespeedup_p =  -1;
  stoplargenegatives_p = 2;
  stoppointmode_p = -1;
  fluxscale_p.resize(0);
  scaleType_p = "NONE";
  minPB_p = 0.1;
  constPB_p = 0.4;
  redoSkyModel_p=True;
  nmodels_p=0;
  useModelCol_p=False;
  freqFrameValid_p=False;
  doTrackSource_p=False;
  freqInterpMethod_p="nearest";
  pointingDirCol_p="DIRECTION";
  logSink_p=LogSink(LogMessage::NORMAL, False);
  imwgt_p=VisImagingWeight();
  smallScaleBias_p=0.6;
  freqFrame_p=MFrequency::LSRK;
  imageTileVol_p=0;
  singlePrec_p=False;
  spwchansels_p.resize();
#ifdef PABLO_IO
  traceEvent(1,"Exiting imager::defaults",24);
#endif

}


Imager::Imager(MeasurementSet& theMS,  Bool compress, Bool useModel)
  : msname_p(""), vs_p(0), rvi_p(0), wvi_p(0), 
    ft_p(0), cft_p(0), se_p(0),
    sm_p(0), vp_p(0), gvp_p(0), setimaged_p(False), nullSelect_p(False), 
    viewer_p(0), clean_panel_p(0), image_id_p(0), mask_id_p(0), prev_image_id_p(0), prev_mask_id_p(0)
{

  mssel_p=0;
  ms_p=0;
  lockCounter_p=0;
  LogIO os(LogOrigin("Imager", "Imager(MeasurementSet &theMS)", WHERE));
  if(!open(theMS, compress, useModel)) {
    os << LogIO::SEVERE << "Open of MeasurementSet failed" << LogIO::EXCEPTION;
  };

  defaults();
  latestObsInfo_p=ObsInfo();
}



Imager::Imager(MeasurementSet& theMS, Bool compress)
  :  msname_p(""),  vs_p(0), rvi_p(0), wvi_p(0), ft_p(0), cft_p(0), se_p(0),
    sm_p(0), vp_p(0), gvp_p(0), setimaged_p(False), nullSelect_p(False), 
    viewer_p(0), clean_panel_p(0), image_id_p(0), mask_id_p(0), prev_image_id_p(0), prev_mask_id_p(0)
{
  mssel_p=0;
  ms_p=0;
  lockCounter_p=0;
  LogIO os(LogOrigin("Imager", "Imager(MeasurementSet &theMS)", WHERE));
  if(!open(theMS, compress)) {
    os << LogIO::SEVERE << "Open of MeasurementSet failed" << LogIO::EXCEPTION;
  };

  defaults();

  latestObsInfo_p=ObsInfo();
}

Imager::Imager(const Imager & other)
  :  msname_p(""), vs_p(0), rvi_p(0), wvi_p(0), 
     ft_p(0), cft_p(0), se_p(0),
     sm_p(0), vp_p(0), gvp_p(0), setimaged_p(False), nullSelect_p(False), 
     viewer_p(0), clean_panel_p(0), image_id_p(0), mask_id_p(0), prev_image_id_p(0), prev_mask_id_p(0)
{
  mssel_p=0;
  ms_p=0;
  operator=(other);
}

Imager &Imager::operator=(const Imager & other)
{
  if (!ms_p.null() && this != &other) {
    *ms_p = *(other.ms_p);
  }
  //Equating the table and ms parameters
  antab_p=other.antab_p;
  datadesctab_p=other.datadesctab_p;
  feedtab_p=other.feedtab_p;
  fieldtab_p=other.fieldtab_p;
  obstab_p=other.obstab_p;
  pointingtab_p=other.pointingtab_p;
  poltab_p=other.poltab_p;
  proctab_p=other.proctab_p;
  spwtab_p=other.spwtab_p;
  statetab_p=other.statetab_p;
  latestObsInfo_p=other.latestObsInfo_p;
  parAngleInc_p=other.parAngleInc_p;
  skyPosThreshold_p=other.skyPosThreshold_p;
  doTrackSource_p=other.doTrackSource_p;
  trackDir_p=other.trackDir_p;
  smallScaleBias_p=other.smallScaleBias_p;
  if (!mssel_p.null() && this != &other) {
    *mssel_p = *(other.mssel_p);
  }
  if (vs_p && this != &other) {
    *vs_p = *(other.vs_p);
  }
  if (wvi_p && this != &other) {
    *wvi_p = *(other.wvi_p);
    rvi_p=wvi_p;
  }
  else if(rvi_p && this != &other){
    *rvi_p = *(other.rvi_p);
    wvi_p=NULL;
  }
  if (ft_p && this != &other) {
    *ft_p = *(other.ft_p);
  }
  if (cft_p && this != &other) {
    *cft_p = *(other.cft_p);
  }
  if (se_p && this != &other) {
    *se_p = *(other.se_p);
  }
  if (sm_p && this != &other) {
    *sm_p = *(other.sm_p);
  }
  if (vp_p && this != &other) {
    *vp_p = *(other.vp_p);
  }
  if (gvp_p && this != &other) {
    *gvp_p = *(other.gvp_p);
  }
  imageTileVol_p=other.imageTileVol_p;
  return *this;
}

Imager::~Imager()
{
  try{
    destroySkyEquation();
    this->unlock(); //unlock things if they are in a locked state
    
    //if (mssel_p) {
    //  delete mssel_p;
    // }
    mssel_p = 0;
    //if (ms_p) {
    //  delete ms_p;
    //}
    ms_p = 0;
    if (vs_p) {
      delete vs_p;
    }
    if(rvi_p)
      delete rvi_p;
    rvi_p=wvi_p=0;

    vs_p = 0;
    if (ft_p) {
      delete ft_p;
    }
    
    
    ft_p = 0;
    if (cft_p) {
      delete cft_p;
    }
    cft_p = 0;

    if ( viewer_p ) {
      // viewer_p->close( clean_panel_p );
      viewer_p->done();
      delete viewer_p;
    }

  }
  catch (AipsError x){
    String mess=x.getMesg();
    //This is a bug for wproject and facet together...
    //somebody is erasing a TempLattice before desturctor.
    //will keep this in place till i figure it out...its benign
    if(mess.contains("does not exist") && mess.contains("TempLattice")){
      String rootpath="/"+String(mess.after("/")).before("TempLattice");
      String pid=String(mess.after("TempLattice")).before("_");
      DirectoryIterator dir(rootpath, Regex::fromPattern("TempLattice"+pid+"*"));
      while(!dir.pastEnd()){
	  Directory ledir(rootpath+"/"+dir.name());
	  ledir.removeRecursive();
	  dir++;
	  
      }

    }
    else{
      throw(AipsError(x));

    }

  }


}


Bool Imager::open(MeasurementSet& theMs, Bool compress, Bool useModelCol)
{

#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::open",21);
#endif

  LogIO os(LogOrigin("Imager", "open()", WHERE));
  
  if (!ms_p.null()) {
    *ms_p = theMs;
  } else {
    ms_p = new MeasurementSet(theMs);
    AlwaysAssert(!ms_p.null(), AipsError);
  }
  

  try {
    this->openSubTables();
    this->lock();
    msname_p = ms_p->tableName();
    
    os << LogIO::DEBUG1
	   << "Opening MeasurementSet " << msname_p << LogIO::POST;

    // Check for DATA or FLOAT_DATA column
    if(!ms_p->tableDesc().isColumn("DATA") && 
       !ms_p->tableDesc().isColumn("FLOAT_DATA")) {
      ms_p->unlock();
      //delete ms_p; 
      ms_p=0;
      os << LogIO::SEVERE
	 << "Missing DATA or FLOAT_DATA column: imager cannot be run"
	 << LogIO::EXCEPTION;
      return False;
    }
    
    Bool initialize=(!ms_p->tableDesc().isColumn("CORRECTED_DATA"));
    
    /*if(vs_p) {
      delete vs_p; vs_p=0;
    }
    */
    if(rvi_p){
      delete rvi_p;
      rvi_p=0;
      wvi_p=0;
    }
    
    // Now open the selected MeasurementSet to be initially the
    // same as the original MeasurementSet

    mssel_p=new MeasurementSet(*ms_p);
    useModelCol_p=useModelCol;
    
    // Now create the VisSet
    this->makeVisSet(*mssel_p);
    AlwaysAssert(rvi_p, AipsError);
    
    // Polarization
    ROMSColumns msc(*mssel_p);
    Vector<String> polType=msc.feed().polarizationType()(0);
    if (polType(0)!="X" && polType(0)!="Y" &&
	polType(0)!="R" && polType(0)!="L") {
      this->unlock();
      os << LogIO::SEVERE << "Warning: Unknown stokes types in feed table: "
	 << polType(0) << endl
	 << "Results open to question!" << LogIO::POST;
    }

    this->unlock();

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::open",21);
#endif

    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught Exception: "<< x.getMesg() << LogIO::EXCEPTION;

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::open",21);
#endif

    return False;
  } 

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::open",21);
#endif

  return True;
}

Bool Imager::close()
{
  if(!valid()) return False;
  if (detached()) return True;
  LogIO os(LogOrigin("imager", "close()", WHERE));
  os << LogIO::NORMAL // Loglevel PROGRESS
     << "Closing MeasurementSet and detaching from imager"
     << LogIO::POST;
  this->unlock();
  if(ft_p) delete ft_p; ft_p = 0;
  if(cft_p) delete cft_p; cft_p = 0;
  if(vs_p) delete vs_p; vs_p = 0;
  if(rvi_p) delete rvi_p; 
  rvi_p=0;
  wvi_p=0;
  //if(mssel_p) delete mssel_p; 
  mssel_p = 0;
  //if(ms_p) delete ms_p; 
  ms_p = 0;

  if(se_p) delete se_p; se_p = 0;

  if(vp_p) delete vp_p; vp_p = 0;
  if(gvp_p) delete gvp_p; gvp_p = 0;

  destroySkyEquation();

  return True;
}

String Imager::name() const
{
  if (detached()) {
    return "none";
  }
  return msname_p;
}


String Imager::imageName()
{
  LogIO os(LogOrigin("imager", "imageName()", WHERE));
  try {
    lock();
    String name(msname_p);
    ROMSColumns msc(*ms_p);
    if(datafieldids_p.shape() !=0) {
      name=msc.field().name()(datafieldids_p(0));
    }
    else if(fieldid_p > -1) {
       name=msc.field().name()(fieldid_p);
    }
    unlock();
    return name;
  } catch (AipsError x) {
    unlock();
    os << LogIO::SEVERE << "Caught Exception: "<< x.getMesg() << LogIO::EXCEPTION; 
    return "";
  } 
  return String("imagerImage");
}

// Make standard choices for coordinates
Bool Imager::imagecoordinates(CoordinateSystem& coordInfo, const Bool verbose) 
{  
  if(!valid()) return False;
  if(!assertDefinedImageParameters()) return False;
  LogIO os(LogOrigin("Imager", "imagecoordinates()", WHERE));
  
  //===Adjust for some setimage defaults if imageNchan=-1 and spwids=-1
  if(imageNchan_p <=0){
    imageNchan_p=1;
  }
  if(spectralwindowids_p.nelements()==1){
    if(spectralwindowids_p[0]<0){
      spectralwindowids_p.resize();
      if(dataspectralwindowids_p.nelements()==0){
         Int nspwinms=ms_p->spectralWindow().nrow();
         dataspectralwindowids_p.resize(nspwinms);
         indgen(dataspectralwindowids_p);
      }
      spectralwindowids_p=dataspectralwindowids_p;
    }
    
  }
  if(fieldid_p < 0){
     if(datafieldids_p.shape() !=0) {
       fieldid_p=datafieldids_p[0];
     }
     else{
       fieldid_p=0; //best default if nothing is specified
     }
  }
  //===end of default
  Vector<Double> deltas(2);
  deltas(0)=-mcellx_p.get("rad").getValue();
  deltas(1)=mcelly_p.get("rad").getValue();
  
  ROMSColumns msc(*ms_p);
  MFrequency::Types obsFreqRef=MFrequency::DEFAULT;
  ROScalarColumn<Int> measFreqRef(ms_p->spectralWindow(),
				  MSSpectralWindow::columnName(MSSpectralWindow::MEAS_FREQ_REF));
  //using the first frame of reference; TO DO should do the right thing 
  //for different frames selected. 
  //Int eh = spectralwindowids_p(0);
  if(spectralwindowids_p.size() && measFreqRef(spectralwindowids_p(0)) >=0) {
     obsFreqRef=(MFrequency::Types)measFreqRef(spectralwindowids_p(0));
  }
			    

  MVDirection mvPhaseCenter(phaseCenter_p.getAngle());
  // Normalize correctly
  MVAngle ra=mvPhaseCenter.get()(0);
  ra(0.0);
  MVAngle dec=mvPhaseCenter.get()(1);
  Vector<Double> refCoord(2);
  refCoord(0)=ra.get().getValue();    
  refCoord(1)=dec;    
  
  Vector<Double> refPixel(2); 
  refPixel(0) = Double(nx_p / 2);
  refPixel(1) = Double(ny_p / 2);
  
  //defining observatory...needed for position on earth
  String telescop = msc.observation().telescopeName()(0);

  // defining epoch as begining time from timerange in OBSERVATION subtable
  // Using first observation for now
  //MEpoch obsEpoch = msc.observation().timeRangeMeas()(0)(IPosition(1,0));
  // modified to use main table's TIME column for better match with what
  // VisIter does.
  MEpoch obsEpoch = msc.timeMeas()(0);

  //Now finding the position of the telescope on Earth...needed for proper
  //frequency conversions

  MPosition obsPosition;
  if(!(MeasTable::Observatory(obsPosition, telescop))){
    os << LogIO::WARN << "Did not get the position of " << telescop 
       << " from data repository" << LogIO::POST;
    os << LogIO::WARN 
       << "Please contact CASA to add it to the repository."
       << LogIO::POST;
    os << LogIO::WARN << "Frequency conversion will not work " << LogIO::POST;
    freqFrameValid_p = False;
  }
  else{
    mLocation_p = obsPosition;
    freqFrameValid_p = True;
  }
  // Now find the projection to use: could probably also use
  // max(abs(w))=0.0 as a criterion
  Projection projection(Projection::SIN);
  if(telescop == "ATCASCP" || telescop == "WSRT" || telescop == "DRAO") {
    os << LogIO::NORMAL // Loglevel NORMAL
       << "Using SIN image projection adjusted for "
       << (telescop == "ATCASCP" ? 'S' : 'N') << "CP" 
       << LogIO::POST;
    Vector<Double> projectionParameters(2);
    projectionParameters(0) = 0.0;
    if(sin(dec) != 0.0){
      projectionParameters(1) = cos(dec)/sin(dec);
      projection=Projection(Projection::SIN, projectionParameters);
    }
    else {
      os << LogIO::WARN
         << "Singular projection for " << telescop << ": using plain SIN"
         << LogIO::POST;
      projection=Projection(Projection::SIN);
    }
  }
  else {
    os << LogIO::DEBUGGING << "Using SIN image projection" << LogIO::POST;
  }
  os << LogIO::NORMAL;
  
  Matrix<Double> xform(2,2);
  xform=0.0;xform.diagonal()=1.0;
  DirectionCoordinate
    myRaDec(MDirection::Types(phaseCenter_p.getRefPtr()->getType()),
	    projection,
	    refCoord(0), refCoord(1),
	    deltas(0), deltas(1),
	    xform,
	    refPixel(0), refPixel(1));
  
  // Now set up spectral coordinate
  SpectralCoordinate* mySpectral=0;
  Double refChan=0.0;
  
  // Spectral synthesis
  // For mfs band we set the window to include all spectral windows
  Int nspw=spectralwindowids_p.nelements();
  if (imageMode_p=="MFS") {
    Double fmin=C::dbl_max;
    Double fmax=-(C::dbl_max);
    Double fmean=0.0;
    for (Int i=0;i<nspw;++i) {
      Int spw=spectralwindowids_p(i);
      Vector<Double> chanFreq=msc.spectralWindow().chanFreq()(spw); 
      Vector<Double> freqResolution = msc.spectralWindow().chanWidth()(spw); 
      
      if(dataMode_p=="none"){
      
	if(i==0) {
	  fmin=min(chanFreq-abs(0.5*freqResolution));
	  fmax=max(chanFreq+abs(0.5*freqResolution));
	}
	else {
	  fmin=min(fmin,min(chanFreq-abs(0.5*freqResolution)));
	  fmax=max(fmax,max(chanFreq+abs(0.5*freqResolution)));
	}
      }
      else if(dataMode_p=="channel"){
	// This needs some careful thought about roundoff - it is likely 
	// still adding an extra half-channel at top and bottom but 
	// if the freqResolution is nonlinear, there are subtleties
	Int lastchan=dataStart_p[i]+ dataNchan_p[i]*dataStep_p[i];
        for(Int k=dataStart_p[i] ; k < lastchan ;  k+=dataStep_p[i]){
	  fmin=min(fmin,chanFreq[k]-abs(freqResolution[k]*(dataStep_p[i]-0.5)));
	  fmax=max(fmax,chanFreq[k]+abs(freqResolution[k]*(dataStep_p[i]-0.5)));
        }
      }
      else{
	this->unlock();
	os << LogIO::SEVERE 
	   << "setdata has to be in 'channel' or 'none' mode for 'mfs' imaging to work"
	   << LogIO::EXCEPTION;
      return False;
      }
 
    }

    fmean=(fmax+fmin)/2.0;
    Vector<Double> restFreqArray;
    Double restFreq=fmean;
    if(getRestFreq(restFreqArray, spectralwindowids_p(0))){
      restFreq=restFreqArray[0];
    }
    imageNchan_p=1;
    Double finc=(fmax-fmin); 
    mySpectral = new SpectralCoordinate(freqFrame_p,  fmean//-finc/2.0
					, finc,
      					refChan, restFreq);
    os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3) // Loglevel INFO
       << "Center frequency = "
       << MFrequency(Quantity(fmean, "Hz")).get("GHz").getValue()
       << " GHz, synthesized continuum bandwidth = "
       << MFrequency(Quantity(finc, "Hz")).get("GHz").getValue()
       << " GHz" << LogIO::POST;

    if(ntaylor_p>1 && reffreq_p==0.0) 
    {
	    reffreq_p = fmean;
	    os << "Setting center frequency as MS-MFS reference frequency" << LogIO::POST;
    }
  }
  
  else if(imageMode_p.contains("FREQ")) {
      if(imageNchan_p==0) {
	this->unlock();
	os << LogIO::SEVERE << "Must specify number of channels" 
	   << LogIO::EXCEPTION;
	return False;
      }
      Double restFreq=mfImageStart_p.get("Hz").getValue();
      Vector<Double> restFreqVec;
      if(getRestFreq(restFreqVec, spectralwindowids_p(0))){
	restFreq=restFreqVec[0];
      }
      MFrequency::Types mfreqref=(obsFreqRef==(MFrequency::REST)) ? MFrequency::REST : MFrequency::castType(mfImageStart_p.getRef().getType()) ; 
      //  mySpectral = new SpectralCoordinate(mfreqref,
      //					  mfImageStart_p.get("Hz").getValue()+
      //mfImageStep_p.get("Hz").getValue()/2.0,
      //					  mfImageStep_p.get("Hz").getValue(),
      //					  refChan, restFreq);
      mySpectral = new SpectralCoordinate(mfreqref,
      					  mfImageStart_p.get("Hz").getValue(),
      					  mfImageStep_p.get("Hz").getValue(),
      					  refChan, restFreq);
      os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
         << "Start frequency = " // Loglevel INFO
	 << mfImageStart_p.get("GHz").getValue()
	 << ", channel increment = "
	 << mfImageStep_p.get("GHz").getValue() 
	 << "GHz, frequency frame = "
         << MFrequency::showType(mfreqref)
         << endl;
      os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
         << "Rest frequency is "  // Loglevel INFO
	 << MFrequency(Quantity(restFreq, "Hz")).get("GHz").getValue()
	 << "GHz" << LogIO::POST;
      
  }
  

  else {
    //    if(nspw>1) {
    //      os << LogIO::SEVERE << "Line modes allow only one spectral window"
    //	 << LogIO::POST;
    //      return False;
    //    }
    Vector<Double> chanFreq;
    Vector<Double> freqResolution;
    //starting with a default rest frequency to be ref 
    //in case none is defined
    Double restFreq=
      msc.spectralWindow().refFrequency()(spectralwindowids_p(0));

    for (Int spwIndex=0; spwIndex < nspw; ++spwIndex){
 
      Int spw=spectralwindowids_p(spwIndex);
      Int origsize=chanFreq.shape()(0);
      Int newsize=origsize+msc.spectralWindow().chanFreq()(spw).shape()(0);
      chanFreq.resize(newsize, True);
      chanFreq(Slice(origsize, newsize-origsize))=msc.spectralWindow().chanFreq()(spw);
      freqResolution.resize(newsize, True);
     freqResolution(Slice(origsize, newsize-origsize))=
	msc.spectralWindow().chanWidth()(spw); 
      

     
      Vector<Double> restFreqArray;
      if(getRestFreq(restFreqArray, spw)){
	if(spwIndex==0){
	  restFreq = restFreqArray(0);
	}
	else{
	  if(restFreq != restFreqArray(0)){
	    os << LogIO::WARN << "Rest frequencies are different for  spectralwindows selected " 
	       << LogIO::POST;
	    os << LogIO::WARN 
	       <<"Will be using the restFreq defined in spectralwindow "
	       << spectralwindowids_p(0) << LogIO::POST;
	  }
	  
	}	
      }
    }
  

    if(imageMode_p=="CHANNEL") {
      if(imageNchan_p==0) {
	this->unlock();
	os << LogIO::SEVERE << "Must specify number of channels" 
	   << LogIO::EXCEPTION;
	return False;
      }
      Vector<Double> freqs;
      if(imageStep_p==0)
	imageStep_p=1;
//	TT: commented these out otherwise the case for multiple MSes would not work
//	Int nsubchans=
//	(chanFreq.shape()(0) - Int(imageStart_p)+1)/Int(imageStep_p)+1;
//      if((nsubchans >0) && (imageNchan_p>nsubchans)) imageNchan_p=nsubchans;

      os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
         << "Image spectral coordinate: "<< imageNchan_p
         << " channels, starting at visibility channel "
	 << imageStart_p << " stepped by " << imageStep_p << LogIO::POST;
      freqs.resize(imageNchan_p);
      for (Int chan=0;chan<imageNchan_p;chan++) {
	freqs(chan)=chanFreq(Int(imageStart_p)+Int(Float(chan)*Float(imageStep_p)));
      }
      // Use this next line when non-linear working
      //    mySpectral = new SpectralCoordinate(obsFreqRef, freqs,
      //					restFreq);
      // Since we are taking the frequencies as is, the frame must be
      // what is specified in the SPECTRAL_WINDOW table
      //      Double finc=(freqs(imageNchan_p-1)-freqs(0))/(imageNchan_p-1);
      Double finc;
      if(imageNchan_p > 1){
	finc=freqs(1)-freqs(0);
      }
      else if(imageNchan_p==1) {
	finc=freqResolution(IPosition(1,0))*imageStep_p;
      }

	  //in order to outframe to work need to set here original freq frame
      //mySpectral = new SpectralCoordinate(freqFrame_p, freqs(0)-finc/2.0, finc,
      mySpectral = new SpectralCoordinate(obsFreqRef, freqs(0)//-finc/2.0
					  , finc,
					  refChan, restFreq);
      os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
         << "Frequency = " // Loglevel INFO
	 << MFrequency(Quantity(freqs(0), "Hz")).get("GHz").getValue()
	 << ", channel increment = "
	 << MFrequency(Quantity(finc, "Hz")).get("GHz").getValue() 
	 << "GHz" << endl;
      os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
         << "Rest frequency is "  // Loglevel INFO
	 << MFrequency(Quantity(restFreq, "Hz")).get("GHz").getValue()
	 << "GHz" << LogIO::POST;
      
    }
    // Spectral channels resampled at equal increments in optical velocity
    // Here we compute just the first two channels and use increments for
    // the others
    else if (imageMode_p=="VELOCITY" || imageMode_p.contains("RADIO")) {
      if(imageNchan_p==0) {
	this->unlock();
	os << LogIO::SEVERE << "Must specify number of channels" 
	   << LogIO::EXCEPTION;
	return False;
      }
      {
	ostringstream oos;
	oos << "Image spectral coordinate:"<< imageNchan_p 
	    << " channels, starting at radio velocity " << mImageStart_p
	    << "  stepped by " << mImageStep_p << endl;
	os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
           << String(oos); // Loglevel INFO
      }
      Vector<Double> freqs(2);
      freqs=0.0;
      if(Double(mImageStep_p.getValue())!=0.0) {
	MRadialVelocity mRadVel=mImageStart_p;
	for (Int chan=0;chan<2;chan++) {
	  MDoppler mdoppler(mRadVel.getValue().get(), MDoppler::RADIO);
	  freqs(chan)=
	    MFrequency::fromDoppler(mdoppler, 
				    restFreq).getValue().getValue();
	  Quantity vel=mRadVel.get("m/s");
	  Quantity inc=mImageStep_p.get("m/s");
	  vel+=inc;
	  mRadVel=MRadialVelocity(vel, MRadialVelocity::LSRK);
	}
      }
      else {
	for (Int chan=0;chan<2;++chan) {
	  freqs(chan)=chanFreq(chan);
	}
      }

      MFrequency::Types mfreqref=MFrequency::LSRK;
      //Can't convert to frame in mImageStart
      if(!MFrequency::getType(mfreqref, (MRadialVelocity::showType(mImageStart_p.getRef().getType()))))
	mfreqref=freqFrame_p;
      mfreqref=(obsFreqRef==(MFrequency::REST)) ? MFrequency::REST : mfreqref; 
      mySpectral = new SpectralCoordinate(mfreqref, freqs(0)//-(freqs(1)-freqs(0))/2.0
					  , freqs(1)-freqs(0), refChan,
					  restFreq);
     
      {
	ostringstream oos;
	oos << "Reference Frequency = "
	    << MFrequency(Quantity(freqs(0), "Hz")).get("GHz")
	    << ", spectral increment = "
	    << MFrequency(Quantity(freqs(1)-freqs(0), "Hz")).get("GHz") 
	    << ", frequency frame = "
            << MFrequency::showType(mfreqref)
            << endl; 
	oos << "Rest frequency is " 
	    << MFrequency(Quantity(restFreq, "Hz")).get("GHz").getValue()
	    << " GHz" << endl;
	os << LogIO::NORMAL << String(oos) << LogIO::POST; // Loglevel INFO
      }
      
    }
    // Since optical/relativistic velocity is non-linear in frequency, we have to
    // pass in all the frequencies. For radio velocity we can use 
    // a linear axis.
    else if (imageMode_p=="OPTICALVELOCITY" || imageMode_p.contains("OPTICAL") || imageMode_p.contains("TRUE") 
	     || imageMode_p.contains("BETA") ||  imageMode_p.contains("RELATI") ) {
      if(imageNchan_p==0) {
	this->unlock();
	os << LogIO::SEVERE << "Must specify number of channels" 
	   << LogIO::EXCEPTION;
	return False;
      }
      {
	ostringstream oos;
	oos << "Image spectral coordinate: "<< imageNchan_p 
	    << " channels, starting at optical velocity " << mImageStart_p
	    << "  stepped by " << mImageStep_p << endl;
	os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
           << String(oos); // Loglevel INFO
      }
      Vector<Double> freqs(imageNchan_p);
      freqs=0.0;
      Double chanVelResolution=0.0;
      if(Double(mImageStep_p.getValue())!=0.0) {
	MRadialVelocity mRadVel=mImageStart_p;
	MDoppler mdoppler;
	for (Int chan=0;chan<imageNchan_p;++chan) {
	  if(imageMode_p.contains("OPTICAL")){
	    mdoppler=MDoppler(mRadVel.getValue().get(), MDoppler::OPTICAL);
	  } 
	  else{
	    mdoppler=MDoppler(mRadVel.getValue().get(), MDoppler::BETA);
	  }
	  freqs(chan)=
	    MFrequency::fromDoppler(mdoppler, restFreq).getValue().getValue();
	  mRadVel.set(mRadVel.getValue()+mImageStep_p.getValue());
	  if(imageNchan_p==1)
	    chanVelResolution=MFrequency::fromDoppler(mdoppler, restFreq).getValue().getValue()-freqs(0);
	}
      }
      else {
	for (Int chan=0;chan<imageNchan_p;++chan) {
	    freqs(chan)=chanFreq(chan);
	}
      }
      // Use this next line when non-linear is working
      // when selecting in velocity its specfied freqframe or REST 
      MFrequency::Types imfreqref=(obsFreqRef==MFrequency::REST) ? MFrequency::REST : freqFrame_p;
  
      if(imageNchan_p==1){
	if(chanVelResolution==0.0)
	  chanVelResolution=freqResolution(0);
	mySpectral = new SpectralCoordinate(imfreqref,
					    freqs(0)//-chanVelResolution/2.0,
					    ,
					    chanVelResolution,
					    refChan, restFreq);
      }
      else{
	mySpectral = new SpectralCoordinate(imfreqref, freqs,
					    restFreq);
      }
      // mySpectral = new SpectralCoordinate(MFrequency::DEFAULT, freqs(0),
      //				       freqs(1)-freqs(0), refChan,
      //				        restFreq);
      {
	ostringstream oos;
	oos << "Reference Frequency = "
	    << MFrequency(Quantity(freqs(0), "Hz")).get("GHz")
	    << " Ghz, " 
            <<" frequency frame= "<<MFrequency::showType(imfreqref)<<endl;
	os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
           << String(oos) << LogIO::POST; // Loglevel INFO
      }
    }
    else {
      this->unlock();
      os << LogIO::SEVERE << "Unknown mode " << imageMode_p
	 << LogIO::EXCEPTION;
      return False;
    }
        
    
  }
 
 
    //In FTMachine lsrk is used for channel matching with data channel 
    //hence we make sure that
    // we convert to lsrk when dealing with the channels
  freqFrameValid_p=freqFrameValid_p && (obsFreqRef !=MFrequency::REST);
  if(freqFrameValid_p){
      mySpectral->setReferenceConversion(MFrequency::LSRK, obsEpoch, 
					 obsPosition,
					 phaseCenter_p);
  }


 
  // Polarization
  Vector<String> polType=msc.feed().polarizationType()(0);
  if (polType(0)!="X" && polType(0)!="Y" &&
      polType(0)!="R" && polType(0)!="L") {
    os << LogIO::WARN << "Unknown stokes types in feed table: ["
       << polType(0) << ", " << polType(1) << "]" << endl
       << "Results open to question!" << LogIO::POST;
  }
  
  if (polType(0)=="X" || polType(0)=="Y") {
    polRep_p=SkyModel::LINEAR;
    os << LogIO::DEBUG1 
       << "Preferred polarization representation is linear" << LogIO::POST;
  }
  else {
    polRep_p=SkyModel::CIRCULAR;
    os << LogIO::DEBUG1
       << "Preferred polarization representation is circular" << LogIO::POST;
  }

  Vector<Int> whichStokes(npol_p);
  switch(npol_p) {
  case 1:
    whichStokes.resize(1);
    
    //possibilities
    if(stokes_p=="RR")
      whichStokes(0)=Stokes::RR;
    else if(stokes_p=="LL")
      whichStokes(0)=Stokes::LL;
    else if(stokes_p=="XX")
      whichStokes(0)=Stokes::XX;
    else if(stokes_p=="YY")
      whichStokes(0)=Stokes::YY;
    else
      whichStokes(0)=Stokes::I;
    os << LogIO::DEBUG1 << "Image polarization = Stokes "<< stokes_p << LogIO::POST;
    break;
  case 2:
    whichStokes.resize(2);
    //default to IQ or IV if not known
    if(stokes_p=="RRLL"){
      whichStokes(0)=Stokes::RR;
      whichStokes(1)=Stokes::LL;
    }
    else if(stokes_p=="XXYY"){
      whichStokes(0)=Stokes::XX;
      whichStokes(1)=Stokes::YY;
    }
    else if(stokes_p=="QU"){
      whichStokes(0)=Stokes::Q;
      whichStokes(1)=Stokes::U;
    }
    else{
      whichStokes(0)=Stokes::I;
      if (polRep_p==SkyModel::LINEAR) {
	whichStokes(1)=Stokes::Q;
	os << LogIO::DEBUG1 << "Image polarization = Stokes I,Q" << LogIO::POST;
      }
      else {
	whichStokes(1)=Stokes::V;
	os << LogIO::DEBUG1 << "Image polarization = Stokes I,V" << LogIO::POST;
      }
    }
    break;
  case 3:
    whichStokes.resize(3);
    whichStokes(0)=Stokes::I;
    whichStokes(1)=Stokes::Q;
    whichStokes(2)=Stokes::U;
    os << LogIO::DEBUG1 << "Image polarization = Stokes I,Q,U" << LogIO::POST;
    break;
  case 4:
    whichStokes.resize(4);
    whichStokes(0)=Stokes::I;
    whichStokes(1)=Stokes::Q;
    whichStokes(2)=Stokes::U;
    whichStokes(3)=Stokes::V;
    os << LogIO::DEBUG1 << "Image polarization = Stokes I,Q,U,V" << LogIO::POST;
    break;
  default:
    this->unlock();
    os << LogIO::SEVERE << "Illegal number of Stokes parameters: " << npol_p
       << LogIO::EXCEPTION;
    return False;
  };
  
  StokesCoordinate myStokes(whichStokes);
  

  //Set Observatory info
  ObsInfo myobsinfo;
  myobsinfo.setTelescope(telescop);
  myobsinfo.setPointingCenter(mvPhaseCenter);
  myobsinfo.setObsDate(obsEpoch);
  myobsinfo.setObserver(msc.observation().observer()(0));
  this->setObsInfo(myobsinfo);

  //Adding everything to the coordsystem
  coordInfo.addCoordinate(myRaDec);
  coordInfo.addCoordinate(myStokes);
  coordInfo.addCoordinate(*mySpectral);
  coordInfo.setObsInfo(myobsinfo);

  if(mySpectral) delete mySpectral;
  
  return True;
}

IPosition Imager::imageshape() const
{
  return IPosition(4, nx_p, ny_p, npol_p, imageNchan_p);
}

Bool Imager::summary() 
{
  if(!valid()) return False;
  LogOrigin OR("imager", "Imager::summary()", WHERE);
  
  LogIO los(OR);
  
  los << "Logging summary" << LogIO::POST;
  try {
    
    this->lock();
    MSSummary mss(*ms_p);
    mss.list(los, True);
    
    los << endl << state() << LogIO::POST;
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    los << LogIO::SEVERE << "Caught Exception: " << x.getMesg()
	<< LogIO::EXCEPTION;
    return False;
  } 
  
  return True;
}


String Imager::state() 
{
  ostringstream os;
  
  try {
    this->lock();
    os << "General: " << endl;
    os << "  MeasurementSet is " << ms_p->tableName() << endl;
    if(beamValid_p) {
      os << "  Beam fit: " << bmaj_p.get("arcsec").getValue() << " by "
	 << bmin_p.get("arcsec").getValue() << " (arcsec) at pa " 
	 << bpa_p.get("deg").getValue() << " (deg) " << endl;
    }
    else {
      os << "  Beam fit is not valid" << endl;
    }
    
    ROMSColumns msc(*ms_p);
    MDirection mDesiredCenter;
    if(setimaged_p) {
      os << "Image definition settings: "
	"(use setimage in Function Group <setup> to change)" << endl;
      os << "  nx=" << nx_p << ", ny=" << ny_p
	 << ", cellx=" << mcellx_p << ", celly=" << mcelly_p
	 << ", Stokes axes : " << stokes_p << endl;
      Int widthRA=20;
      Int widthDec=20;
      if(doShift_p) {
	os << "  doshift is True: Image phase center will be as specified "
	   << endl;
      }
      else {
	os << "  doshift is False: Image phase center will be that of field "
	   << fieldid_p << " :" << endl;
      }
      
      if(shiftx_p.get().getValue()!=0.0||shifty_p.get().getValue()!=0.0) {
	os << "  plus the shift: longitude: " << shiftx_p
	   << " / cos(latitude) : latitude: " << shifty_p << endl;
      }
      
      MVAngle mvRa=phaseCenter_p.getAngle().getValue()(0);
      MVAngle mvDec=phaseCenter_p.getAngle().getValue()(1);
      os << "     ";
      os.setf(ios::left, ios::adjustfield);
      os.width(widthRA);  os << mvRa(0.0).string(MVAngle::TIME,8);
      os.width(widthDec); os << mvDec.string(MVAngle::DIG2,8);
      os << "     " << MDirection::showType(phaseCenter_p.getRefPtr()->getType())
	 << endl;
      
      if(distance_p.get().getValue()!=0.0) {
	os << "  Refocusing to distance " << distance_p << endl;
      }
      
      if(imageMode_p=="MFS") {
	os << "  Image mode is mfs: Image will be frequency synthesised from spectral windows : ";
	for (uInt i=0;i<spectralwindowids_p.nelements();++i) {
	  os << spectralwindowids_p(i) << " ";
	}
	os << endl;
      }
      
      else {
	os << "  Image mode is " << imageMode_p
	   << "  Image number of spectral channels ="
	   << imageNchan_p << endl;
      }
      
    }
    else {
      os << "Image: parameters not yet set (use setimage "
	"in Function Group <setup> )" << endl;
    }
    
    os << "Data selection settings: (use setdata in Function Group <setup> "
      "to change)" << endl;
    if(dataMode_p=="none") {
      if(mssel_p->nrow() == ms_p->nrow()){
	os << "  All data selected" << endl;
      }
      else{
        os << " Number of rows of data selected= " << mssel_p->nrow() << endl;
      }
    }
    else {
      os << "  Data selection mode is " << dataMode_p << ": have selected "
	 << dataNchan_p << " channels";
      if(dataspectralwindowids_p.nelements()>0) {
	os << " spectral windows : ";
	for (uInt i=0;i<dataspectralwindowids_p.nelements();++i) {
	  os << dataspectralwindowids_p(i) << " ";
	}
      }
      if(datafieldids_p.nelements()>0) {
	os << "  Data selected includes fields : ";
	for (uInt i=0;i<datafieldids_p.nelements();++i) {
	  os << datafieldids_p(i) << " ";
	}
      }
      os << endl;
    }
    os << "Options settings: (use setoptions in Function Group <setup> "
      "to change) " << endl;
    os << "  Gridding cache has " << cache_p << " complex pixels, in tiles of "
       << tile_p << " pixels on a side" << endl;
    os << "  Gridding convolution function is ";
    
    if(gridfunction_p=="SF") {
      os << "Spheroidal wave function";
    }
    else if(gridfunction_p=="BOX") {
      os << "Box car convolution";
    }
    else if(gridfunction_p=="PB") {
      os << "Using primary beam for convolution";
    }
    else {
      os << "Unknown type : " << gridfunction_p;
    }
    os << endl;
    
    if(doVP_p) {
      os << "  Primary beam correction is enabled" << endl;
      //       Table vpTable( vpTableStr_p );   could fish out info and summarize
    }
    else {
      os << "  No primary beam correction will be made " << endl;
    }
    os << "  Image plane padding : " << padding_p << endl;
    
    this->unlock();
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;

  } 
  return String(os);
}

Bool Imager::setimage(const Int nx, const Int ny,
		      const Quantity& cellx, const Quantity& celly,
		      const String& stokes,
		      Bool doShift,
		      const MDirection& phaseCenter, 
		      const Quantity& shiftx, const Quantity& shifty,
		      const String& mode, const Int nchan,
		      const Int start, const Int step,
		      const MRadialVelocity& mStart, const MRadialVelocity& mStep,
		      const Vector<Int>& spectralwindowids,
		      const Int fieldid,
		      const Int facets,
		      const Quantity& distance)
{



#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::setimage",26);
#endif

  if(!valid())
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::setimage",25);
#endif

      return False;
    }

  //Clear the sink 
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "setimage()"), logSink_p);

  os << "nx=" << nx << " ny=" << ny
     << " cellx='" << cellx.getValue() << cellx.getUnit()
     << "' celly='" << celly.getValue() << celly.getUnit()
     << "' stokes=" << stokes << " doShift=" << doShift
     << " shiftx='" << shiftx.getValue() << shiftx.getUnit()
     << "' shifty='" << shifty.getValue() << shifty.getUnit()
     << "' mode=" << mode << " nchan=" << nchan
     << " start=" << start << " step=" << step
     << " spwids=" << spectralwindowids
     << " fieldid=" <<   fieldid << " facets=" << facets
     << " distance='" << distance.getValue() << distance.getUnit() <<"'";
  ostringstream clicom;
  clicom << " phaseCenter='" << phaseCenter;
  clicom << "' mStart='" << mStart << "' mStep='" << mStep << "'";
  os << String(clicom);
  
  try {
    
    this->lock();
    this->writeCommand(os);

    os << LogIO::NORMAL << "Defining image properties" << LogIO::POST; // Loglevel INFO
  
    /**** this check is not really needed here especially for SD imaging
    if(2*Int(nx/2)!=nx) {
      this->unlock();
      os << LogIO::SEVERE << "nx must be even" << LogIO::POST;
      return False;
    }
    if(2*Int(ny/2)!=ny) {
      this->unlock();
      os << LogIO::SEVERE << "ny must be even" << LogIO::POST;
      return False;
    }

    */
    {
      CompositeNumber cn(nx);
      if (! cn.isComposite(nx)) {
	Int nxc = (Int)cn.nextLargerEven(nx);
	Int nnxc = (Int)cn.nearestEven(nx);
	if (nxc == nnxc) {
	  os << LogIO::WARN << "nx = " << nx << " is not composite; nx = " 
	     << nxc << " will be more efficient" << LogIO::POST;
	} else {
	  os <<  LogIO::WARN << "nx = " << nx << " is not composite; nx = " 
	     << nxc <<  " or " << nnxc << " will be more efficient for FFTs" << LogIO::POST;
	}
      }
      if (! cn.isComposite(ny)) {
	Int nyc = (Int)cn.nextLargerEven(ny);
	Int nnyc = (Int)cn.nearestEven(ny);
	if (nyc == nnyc) {
	  os <<  LogIO::WARN << "ny = " << ny << " is not composite; ny = " 
	     << nyc << " will be more efficient" << LogIO::POST;
	} else {
	  os <<  LogIO::WARN << "ny = " << ny << " is not composite ; ny = " << nyc << 
	      " or " << nnyc << " will be more efficient for FFTs" << LogIO::POST;
	}
	os << LogIO::WARN 
	   << "You may safely ignore this message for single dish imaging" 
	   << LogIO::POST;

      }
      
    }

  
    nx_p=nx;
    ny_p=ny;
    mcellx_p=cellx;
    mcelly_p=celly;
    distance_p=distance;
    stokes_p=stokes;
    imageMode_p=mode;
    imageMode_p.upcase();
    imageNchan_p=nchan;
    imageStart_p=start;
    imageStep_p=step;
    mImageStart_p=mStart;
    mImageStep_p=mStep;
    spectralwindowids_p.resize(spectralwindowids.nelements());
    spectralwindowids_p=spectralwindowids;
    fieldid_p=fieldid;
    facets_p=facets;
    redoSkyModel_p=True;
    destroySkyEquation();    

    // Now make the derived quantities 
    if(stokes_p=="I" || stokes_p=="RR" ||stokes_p=="LL" || 
       stokes_p=="XX" || stokes_p=="YY") {
      npol_p=1;
    }
    else if(stokes_p=="IQ" || stokes_p=="RRLL" || stokes_p=="XXYY" || 
	    stokes_p=="QU") {
      npol_p=2;
    }
    else if(stokes_p=="IV") {
      npol_p=2;
    }
    else if(stokes_p=="IQU") {
      npol_p=3;
    }
    else if(stokes_p=="IQUV") {
      npol_p=4;
    }
    else {
      this->unlock();

      

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::setimage",25);
#endif
      os << LogIO::SEVERE << "Illegal Stokes string " << stokes_p
	 << LogIO::EXCEPTION;

      return False;
    };
    

    //THIS NEEDS TO GO
    this->setImageParam(nx_p, ny_p, npol_p, imageNchan_p);

    // Now do the shifts
    //    MSColumns msc(*ms_p);

    doShift_p=doShift;
    if(doShift_p) {
      phaseCenter_p=phaseCenter;
    }
    else {

      ROMSFieldColumns msfield(ms_p->field());
      phaseCenter_p=msfield.phaseDirMeas(fieldid_p);
      //    phaseCenter_p=msc.field().phaseDirMeas(fieldid_p);
    }
    
    // Now add the optional shifts
    shiftx_p=shiftx;
    shifty_p=shifty;
    if(shiftx_p.get().getValue()!=0.0||shifty_p.get().getValue()!=0.0) {
      Vector<Double> vPhaseCenter(phaseCenter_p.getAngle().getValue());
      if(cos(vPhaseCenter(1))!=0.0) {
	vPhaseCenter(0)+=shiftx_p.get().getValue()/cos(vPhaseCenter(1));
      }
      vPhaseCenter(1)+=shifty_p.get().getValue();
      phaseCenter_p.set(MVDirection(vPhaseCenter));
    }
    
    // Now we have set the image parameters
    setimaged_p=True;
    beamValid_p=False;
    
    this->unlock();

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::setimage",25);
#endif

    return True;
  } catch (AipsError x) {
    
    this->unlock();

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::setimage",25);
#endif
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::setimage",25);
#endif

  return True;
}


Bool Imager::defineImage(const Int nx, const Int ny,
			 const Quantity& cellx, const Quantity& celly,
			 const String& stokes,
			 const MDirection& phaseCenter, const Int fieldid,
			 const String& mode, const Int nchan,
			 const Int start, const Int step,
			 const MFrequency& mFreqStart,
			 const MRadialVelocity& mStart, 
			 const Quantity& qStep,
			 const Vector<Int>& spectralwindowids,
			 const Int facets,
			 const Quantity& restFreq,
                         const MFrequency::Types& mFreqFrame,
			 const Quantity& distance, const Bool dotrackDir, 
			 const MDirection& trackDir)
{




  //Clear the sink 
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "defineimage()"), logSink_p);


  os << LogIO::NORMAL << "Defining image properties:"; // Loglevel INFO
  os << "nx=" << nx << " ny=" << ny
     << " cellx='" << cellx.getValue() << cellx.getUnit()
     << "' celly='" << celly.getValue() << celly.getUnit()
     << "' stokes=" << stokes 
     << "' mode=" << mode << " nchan=" << nchan
     << " start=" << start << " step=" << step
     << " spwids=" << spectralwindowids
     << " fieldid=" <<   fieldid << " facets=" << facets
     << " frame=" << mFreqFrame 
     << " distance='" << distance.getValue() << distance.getUnit() <<"'";
  os << LogIO::POST;
  ostringstream clicom;
  clicom << " phaseCenter='"  ;
  if(fieldid < 0){
    MVAngle mvRA=phaseCenter.getAngle().getValue()(0);
    MVAngle mvDEC=phaseCenter.getAngle().getValue()(1);
    clicom << mvRA(0.0).string(MVAngle::TIME,8) << ", ";
    clicom << mvDEC(0.0).string(MVAngle::ANGLE_CLEAN,8) << ", ";
  }
  else{
    clicom << "field-" << fieldid<< " "; 
  }
  clicom << "' mStart='" << mStart << "' qStep='" << qStep << "'";
  clicom << "' mFreqStart='" << mFreqStart;
  os << String(clicom);
  os << LogIO::POST;
  
  try {
    
    this->lock();
    this->writeCommand(os);
  
    doTrackSource_p=dotrackDir;
    trackDir_p=trackDir;


    /**** this check is not really needed here especially for SD imaging
    if(2*Int(nx/2)!=nx) {
      this->unlock();
      os << LogIO::SEVERE << "nx must be even" << LogIO::POST;
      return False;
    }
    if(2*Int(ny/2)!=ny) {
      this->unlock();
      os << LogIO::SEVERE << "ny must be even" << LogIO::POST;
      return False;
    }

    */
    {
      CompositeNumber cn(nx);
      if (! cn.isComposite(nx)) {
	Int nxc = (Int)cn.nextLargerEven(nx);
	Int nnxc = (Int)cn.nearestEven(nx);
	if (nxc == nnxc) {
	  os << LogIO::POST << "nx = " << nx << " is not composite; nx = " 
	     << nxc << " will be more efficient" << LogIO::POST;
	} else {
	  os <<  LogIO::POST << "nx = " << nx << " is not composite; nx = " 
	     << nxc <<  " or " << nnxc << " will be more efficient" << LogIO::POST;
	}
      }
      if (! cn.isComposite(ny)) {
	Int nyc = (Int)cn.nextLargerEven(ny);
	Int nnyc = (Int)cn.nearestEven(ny);
	if (nyc == nnyc) {
	  os <<  LogIO::POST << "ny = " << ny << " is not composite; ny = " 
	     << nyc << " will be more efficient" << LogIO::POST;
	} else {
	  os <<  LogIO::POST << "ny = " << ny << " is not composite; ny = " << nyc << 
	      " or " << nnyc << " will be more efficient" << LogIO::POST;
	}
	os << LogIO::POST
	   << "You may safely ignore this message for single dish imaging" 
	   << LogIO::POST;

      }
      
    }


    if((abs(Double(nx)*cellx.getValue("rad")) > C::pi) || (abs(Double(ny)*celly.getValue("rad")) > C::pi))
      throw(AipsError("Cannot image the extent requested for this image;  more that PI ialong one or both of the axes " ));
    


    nx_p=nx;
    ny_p=ny;
    mcellx_p=cellx;
    mcelly_p=celly;
    distance_p=distance;
    stokes_p=stokes;
    imageMode_p=mode;
    imageMode_p.upcase();
    imageNchan_p=nchan;
    imageStart_p=start;
    imageStep_p=step;
    if(mode.contains("VEL")){
      mImageStart_p=mStart;
      mImageStep_p=MRadialVelocity(qStep);
    }
    if(mode.contains("FREQ")){
      mfImageStart_p=mFreqStart;
      mfImageStep_p=MFrequency(qStep);
    }
    restFreq_p=restFreq;
    freqFrame_p=mFreqFrame;
    spectralwindowids_p.resize(spectralwindowids.nelements());
    spectralwindowids_p=spectralwindowids;
    fieldid_p=fieldid;
    facets_p=facets;
    redoSkyModel_p=True;
    destroySkyEquation();    

    // Now make the derived quantities 
    if(stokes_p=="I" || stokes_p=="RR" ||stokes_p=="LL" || 
       stokes_p=="XX" || stokes_p=="YY") {
      npol_p=1;
    }
    else if(stokes_p=="IQ" || stokes_p=="RRLL" || stokes_p=="XXYY" ||
	    stokes_p=="QU") {
      npol_p=2;
    }
    else if(stokes_p=="IV") {
      npol_p=2;
    }
    else if(stokes_p=="IQU") {
      npol_p=3;
    }
    else if(stokes_p=="IQUV") {
      npol_p=4;
    }
    else {
      this->unlock();
     

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::setimage",25);
#endif
      os << LogIO::SEVERE << "Illegal Stokes string " << stokes_p
	 << LogIO::EXCEPTION;
      
      return False;
    };


    // nchan we need to get rid of one of these variables 
    nchan_p=imageNchan_p;
    
    if(fieldid < 0){      
      doShift_p=True;
      phaseCenter_p=phaseCenter;
    }
    else {
      ROMSFieldColumns msfield(ms_p->field());
      phaseCenter_p=msfield.phaseDirMeas(fieldid_p);
    }
    
    
    // Now we have set the image parameters
    setimaged_p=True;
    beamValid_p=False;
    
    this->unlock();


    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  return True;
}

Bool Imager::advise(const Bool takeAdvice, const Float amplitudeLoss,
		    const Quantity& fieldOfView, Quantity& cell,
		    Int& pixels, Int& facets, MDirection& phaseCenter)
{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "advise()", WHERE));
  
  try {
    
    os << "Advising image properties" << LogIO::POST;
    
    Float maxAbsUV=0.0;
    Float maxWtAbsUV=0.0;
    // To determine the number of facets, we need to fit w to
    // a.u + b.v. The misfit from this (i.e. the dispersion 
    // will determine the error beam due to the non-coplanar
    // baselines. We'll do both cases: where the position
    // errors are important and where they are not. We'll use
    // the latter.
    Double sumWt = 0.0;

    Double sumUU=0.0;
    Double sumUV=0.0;
    Double sumUW=0.0;
    Double sumVV=0.0;
    Double sumVW=0.0;
    Double sumWW=0.0;

    Double sumWtUU=0.0;
    Double sumWtUV=0.0;
    Double sumWtUW=0.0;
    Double sumWtVV=0.0;
    Double sumWtVW=0.0;
    Double sumWtWW=0.0;

    Double sum = 0.0;

    this->lock();
    ROVisIter& vi(*rvi_p);
    VisBuffer vb(vi);
    
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	Int nChan=vb.nChannel();
	for (Int row=0; row<nRow; ++row) {
	  for (Int chn=0; chn<nChan; ++chn) {
	    if(!vb.flag()(chn,row)) {
	      Float f=vb.frequency()(chn)/C::c;
	      Float u=vb.uvw()(row)(0)*f;
	      Float v=vb.uvw()(row)(1)*f;
	      Float w=vb.uvw()(row)(2)*f;
              Double wt=vb.imagingWeight()(chn,row);
	      if(wt>0.0) {
		if(abs(u)>maxWtAbsUV) maxWtAbsUV=abs(u);
		if(abs(v)>maxWtAbsUV) maxWtAbsUV=abs(v);
		sumWt += wt;
                sumWtUU += wt * u * u;
                sumWtUV += wt * u * v;
                sumWtUW += wt * u * w;
                sumWtVV += wt * v * v;
                sumWtVW += wt * v * w;
                sumWtWW += wt * w * w;
	      }
	      sum += 1;
	      if(abs(u)>maxAbsUV) maxAbsUV=abs(u);
	      if(abs(v)>maxAbsUV) maxAbsUV=abs(v);
	      sumUU += u * u;
	      sumUV += u * v;
	      sumUW += u * w;
	      sumVV += v * v;
	      sumVW += v * w;
	      sumWW += w * w;
	    }
	  }
	}
      }
    }
    
    if(sumWt==0.0) {
      os << LogIO::WARN << "Visibility data are not yet weighted: using unweighted values" << LogIO::POST;
      sumWt = sum;
    }
    else {
      sumUU = sumWtUU;
      sumUV = sumWtUV;
      sumUW = sumWtUW;
      sumVV = sumWtVV;
      sumVW = sumWtVW;
      sumWW = sumWtWW;
      maxAbsUV = maxWtAbsUV;
    }

    // First find the cell size
    if(maxAbsUV==0.0) {
      this->unlock();
      os << LogIO::SEVERE << "Maximum uv distance is zero" << LogIO::POST;
      return False;
    }
    else {
      cell=Quantity(0.5/maxAbsUV, "rad").get("arcsec");
      os << "Maximum uv distance = " << maxAbsUV << " wavelengths" << endl;
      os << "Recommended cell size < " << cell.get("arcsec").getValue()
	 << " arcsec" << LogIO::POST;
    }

    // Now we can find the number of pixels for the specified field of view
    pixels = 2*Int((fieldOfView.get("rad").getValue()/cell.get("rad").getValue())/2.0);
    {
      CompositeNumber cn(pixels);
      pixels = (Int) (cn.nextLargerEven(pixels));
    }
    if(pixels < 64) pixels = 64;
    os << "Recommended number of pixels = " << pixels << endl;

      // Rough rule for number of facets:
      // For the specified facet size, the loss in amplitude
      // due to the peeling of facets from the sphere should 
      // be equal to the amplitude error.
      Int worstCaseFacets=1;
      if(sumWt<=0.0||sumUU<=0.0||(sumUU+sumVV)<=0.0) {
	this->unlock();
	os << LogIO::SEVERE << "Sum of imaging weights is zero" << LogIO::POST;
	return False;
      }
      else {
	Double rmsUV  = sqrt((sumUU + sumVV)/sumWt);
	Double rmsW = sqrt(sumWW/sumWt);
	os << "Dispersion in uv, w distance = " << rmsUV << ", "<< rmsW
	   << " wavelengths" << endl;
	if(rmsW>0.0&&rmsUV>0.0&&amplitudeLoss>0.0) {
	  worstCaseFacets =
	    Int (pixels * (abs(cell.get("rad").getValue())*
				  sqrt(C::pi*rmsW/(sqrt(32.0*amplitudeLoss)))));
	}
	else {
	  os << LogIO::WARN << "Cannot calculate number of facets: using 1"
	     << LogIO::POST;
	  worstCaseFacets = 1;
	}
	// Solve for the parameters:
	Double Determinant = sumUU * sumVV - square(sumUV);
	Double rmsFittedW = rmsW;
	if(Determinant > 0.0) {
	  Double a = ( sumVV * sumUW - sumUV * sumVW)/Determinant;
	  Double b = (-sumUV * sumUW + sumUU * sumVW)/Determinant;
	  os << "Best fitting plane is w = " << a << " * u + "
	     << b << " * v" << endl;
	  Double FittedWW =
	    sumWW  + square(a) * sumUU + square(b) * sumVV +
	    + 2.0 * a * b * sumUV - 2.0 * (a * sumUW + b * sumVW);
	  rmsFittedW  = sqrt(FittedWW/sumWt);
	  os << "Dispersion in fitted w = " << rmsFittedW
	     << " wavelengths" << endl;
	  facets = Int (pixels * (abs(cell.get("rad").getValue())*
				  sqrt(C::pi*rmsFittedW/(sqrt(32.0*amplitudeLoss)))));
          if (facets<1) facets = 1;
	}
	else {
	  os << "Error in fitting plane to uvw data" << LogIO::POST;
	}
	if(worstCaseFacets<1) worstCaseFacets=1;
	if(worstCaseFacets>1) {
	  os << "imager recommends that you use the wide field clean" << endl
	     << "For accurate positions, use " << worstCaseFacets
	     << " facets on each axis" << endl
	     << "For accurate removal of sources, you only need "
	     << facets << " facets on each axis" << LogIO::POST;
	}
	else {
	  os << "Wide field cleaning is not necessary"
	     << LogIO::POST;
	}
      }

    ROMSColumns msc(*mssel_p);
    if(datafieldids_p.shape()!=0){
      //If setdata has been used prior to this
    phaseCenter=msc.field().phaseDirMeas(datafieldids_p(0));
    }
    else{
    phaseCenter=msc.field().phaseDirMeas(fieldid_p);   
    }

    
    // Now we have set the image parameters
    if(takeAdvice) {
      os << "Using advised image properties" << LogIO::POST;
      mcellx_p=cell;
      mcelly_p=cell;
      phaseCenter_p=phaseCenter;
      setimaged_p=True;
      beamValid_p=False;
      facets_p=facets;
      nx_p=ny_p=pixels;
    }
    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  
  return True;
}



Bool Imager::setDataPerMS(const String& msname, const String& mode, 
			  const Vector<Int>& nchan, 
			  const Vector<Int>& start,
			  const Vector<Int>& step,
			  const Vector<Int>& spectralwindowids,
			  const Vector<Int>& fieldids,
			  const String& msSelect, const String& timerng,
			  const String& fieldnames, 
			  const Vector<Int>& antIndex,
			  const String& antnames,
			  const String& spwstring,
                          const String& uvdist, const String& scan, const Bool useModelCol)
{
  LogIO os(LogOrigin("imager", "setdata()"), logSink_p);
  if(msname != ""){

    LogIO os(LogOrigin("imager", "setdata()"), logSink_p);
    os << LogIO::WARN
       << "Ignoring that ms" << msname << "specified here"
       << LogIO::POST;
    os << LogIO::WARN
       << "Imager was constructed with an ms "
       << LogIO::POST;
    os << LogIO::WARN
       << "if multi-ms are to be used please construct imager without parameters and use setdata to specify the ms's and selection"
       << LogIO::POST;

  }
  MRadialVelocity dummy;
  //Calling the old setdata
  return   setdata(mode, nchan, start, step, dummy, dummy, spectralwindowids, 
		   fieldids, msSelect, timerng, fieldnames, antIndex, 
                   antnames, spwstring, uvdist, scan, useModelCol);

}


Bool Imager::setdata(const String& mode, const Vector<Int>& nchan,
		     const Vector<Int>& start, const Vector<Int>& step,
		     const MRadialVelocity& mStart,
		     const MRadialVelocity& mStep,
		     const Vector<Int>& spectralwindowids,
		     const Vector<Int>& fieldids,
		     const String& msSelect, const String& timerng,
		     const String& fieldnames, const Vector<Int>& antIndex,
		     const String& antnames, const String& spwstring,
                     const String& uvdist, const String& scan,
                     const Bool useModelCol,
                     const Bool be_calm)
{
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "data selection"), logSink_p);

  if(ms_p.null()) {
    os << LogIO::SEVERE << "Program logic error: MeasurementSet pointer ms_p not yet set"
       << LogIO::EXCEPTION;
    return False;
  }

  os << "mode=" << mode << " nchan=" << nchan 
     <<  " start=" << start << " step=" << step;
  ostringstream clicom;
  clicom <<  " mstart='" << mStart << "' mstep='" << mStep;
  os << String(clicom) ;
  os <<  "' spectralwindowids=" << spectralwindowids
     << " fieldids=" << fieldids << " msselect=" << msSelect;

  try {
    
    this->lock();
    this->writeCommand(os);

    os << LogIO::NORMAL << "Selecting data" << LogIO::POST; // Loglevel PROGRESS
    //Need delete the ft machine as the channel flag may change
    if(ft_p)
      delete ft_p;
    ft_p=0;
    nullSelect_p=False;
    dataMode_p=mode;
    dataNchan_p.resize();
    dataStart_p.resize();
    dataStep_p.resize();
    dataNchan_p=nchan;
    dataStart_p=start;
    dataStep_p=step;
    mDataStart_p=mStart;
    mDataStep_p=mStep;
    dataspectralwindowids_p.resize(spectralwindowids.nelements());
    dataspectralwindowids_p=spectralwindowids;
    datafieldids_p.resize(fieldids.nelements());
    datafieldids_p=fieldids;
    //    useModelCol_p=useModelCol;
    
	if(rvi_p)
	  delete rvi_p; 
    rvi_p=0;
    wvi_p=0;
    // if(mssel_p) delete mssel_p; 
    mssel_p=0;
    
    // check that sorted table exists (it should), if not, make it now.
    //this->makeVisSet(*ms_p);
    
    //MeasurementSet sorted=ms_p->keywordSet().asTable("SORTED_TABLE");
    
    
    //Some MSSelection 
    MSSelection thisSelection;

    //MSSelection thisSelection (sorted, MSSelection::PARSE_NOW,timerng,antnames,
    //			       fieldnames, spwstring,uvdist, msSelect,"",
    //			       scan); 

    if(datafieldids_p.nelements() > 0){
      thisSelection.setFieldExpr(MSSelection::indexExprStr(datafieldids_p));
      os << LogIO::DEBUG1 << "Selecting on field ids" << LogIO::POST;
    }
    if(fieldnames != ""){
      thisSelection.setFieldExpr(fieldnames);
      os << LogIO::DEBUG1
		 << "Selecting on field names " << fieldnames << LogIO::POST;
    }
    
    if(dataspectralwindowids_p.nelements() > 0){
      thisSelection.setSpwExpr(MSSelection::indexExprStr(dataspectralwindowids_p));
      os << LogIO::DEBUG1 << "Selecting on spectral windows" << LogIO::POST;
    }
    else if(spwstring != ""){
      os << LogIO::DEBUG1
		 << "Selecting on spectral windows expression " << spwstring
		 << LogIO::POST;
      thisSelection.setSpwExpr(spwstring);
    }
    
    if(antIndex.nelements() >0){
      thisSelection.setAntennaExpr( MSSelection::indexExprStr(antIndex));
      os << LogIO::DEBUG1 << "Selecting on antenna ids" << LogIO::POST;	
    }
    if(antnames != ""){
      Vector<String>antNames(1, antnames);
      //       thisSelection.setAntennaExpr(MSSelection::nameExprStr(antNames));
      thisSelection.setAntennaExpr(antnames);
      os << LogIO::DEBUG1 << "Selecting on antenna names" << LogIO::POST; 
    } 
               
    if(timerng != ""){
      //	Vector<String>timerange(1, timerng);
	thisSelection.setTimeExpr(timerng);
	os << LogIO::DEBUG1 << "Selecting on time range" << LogIO::POST;	
    }
    
    
    if(uvdist != ""){
	thisSelection.setUvDistExpr(uvdist);
    }
    
    
    if(scan != ""){
      thisSelection.setScanExpr(scan);
    }
    if(msSelect != ""){
      thisSelection.setTaQLExpr(msSelect);
    }
    
    //***************
    
    TableExprNode exprNode=thisSelection.toTableExprNode(&(*ms_p));
    //TableExprNode exprNode=thisSelection.getTEN();
    //if(exprNode.isNull()){
      //      throw(AipsError("Selection failed...review ms and selection parameters"));
    //}
    datafieldids_p.resize();
    datafieldids_p=thisSelection.getFieldList();
    if(datafieldids_p.nelements()==0){
      Int nf=ms_p->field().nrow();
      datafieldids_p.resize(nf);
      indgen(datafieldids_p);
    }
    //Now lets see what was selected as spw and match it with datadesc
    //
    // getSpwList could return duplicated spw ids
    // when multiple channel ranges are specified.
    //dataspectralwindowids_p.resize();
    //dataspectralwindowids_p=thisSelection.getSpwList();
    //get channel selection in spw
    Matrix<Int> chansels=thisSelection.getChanList();
    //cout<<"chansels="<<chansels<<endl;
    //convert the selection into flag
    uInt nms = 1;
    uInt nrow = chansels.nrow();
    dataspectralwindowids_p.resize();
    const ROMSSpWindowColumns spwc(ms_p->spectralWindow());
    uInt nspw = spwc.nrow();
    const ROScalarColumn<Int> spwNchans(spwc.numChan());
    Vector<Int> nchanvec = spwNchans.getColumn();
    Int maxnchan = 0;
    for (uInt i=0;i<nchanvec.nelements();i++) {
      maxnchan=max(nchanvec[i],maxnchan);
    }	  
    
    spwchansels_p.resize(nms,nspw,maxnchan);
    spwchansels_p.set(0);
    uInt nselspw=0;
    if (nrow==0) {
      //no channel selection, select all channels
      spwchansels_p=1;
      dataspectralwindowids_p=thisSelection.getSpwList();
    }
    else {
      spwchansels_p=0; //deselect
      Int prvspwid=-1;
      Vector<Int> selspw;
      for (uInt i=0;i<nrow;i++) {
	Vector<Int> sel = chansels.row(i);
	Int spwid = sel[0];
	if (spwid != prvspwid){
	  nselspw++;
	  selspw.resize(nselspw,True);
	  selspw[nselspw-1]=spwid;
	}
	uInt minc= sel[1];
	uInt maxc = sel[2];
	uInt step = sel[3];
	// step as the same context as in im.selectvis
	// select channels 
	for (uInt k=minc;k<(maxc+1);k+=step) {
	  spwchansels_p(0,spwid,k)=1;
	}
	prvspwid=spwid;
      }
      dataspectralwindowids_p=selspw;
    }
    
    // Map the selected spectral window ids to data description ids
    if(dataspectralwindowids_p.nelements()==0){
      Int nspwinms=ms_p->spectralWindow().nrow();
      dataspectralwindowids_p.resize(nspwinms);
      indgen(dataspectralwindowids_p);
    }
    MSDataDescIndex msDatIndex(ms_p->dataDescription());
    datadescids_p.resize(0);
    datadescids_p=msDatIndex.matchSpwId(dataspectralwindowids_p);
    
    if (datafieldids_p.nelements() > 1) {
      os << LogIO::NORMAL4<< "Multiple fields specified" << LogIO::POST;
      multiFields_p = True;
    }
    
    
    if(mode=="none"){
      // Now channel selection from spw already stored in chansel,
      // no need for this- TT
      //check if we can find channel selection in the spw string
      //Matrix<Int> chanselmat=thisSelection.getChanList();
      //
      // This not correct for multiple channel ranges TT
      //if(chanselmat.nrow()==dataspectralwindowids_p.nelements()){
      if(nselspw==dataspectralwindowids_p.nelements()){
	
	dataMode_p="channel";
	dataStep_p.resize(dataspectralwindowids_p.nelements());
	dataStart_p.resize(dataspectralwindowids_p.nelements());
	dataNchan_p.resize(dataspectralwindowids_p.nelements());
	Cube<Int> spwchansels_tmp=spwchansels_p;
	
	for (uInt k =0 ; k < dataspectralwindowids_p.nelements(); ++k){
	  uInt curspwid=dataspectralwindowids_p[k];
	  //dataStep_p[k]=1;
	  if (nrow > 0) {
	    dataStep_p[k]=chansels.row(k)(3);
	  }
	  else {
	    dataStep_p[k]=1;
	  }
	  //dataStart_p[k]=chanselmat.row(k)(1);
	  dataStart_p[k]=0;
	  dataNchan_p[k]=nchanvec(curspwid);
	  //find start
	  Bool first =True;
	  uInt nchn = 0;
	  uInt lastchan = 0;
	  for (uInt j=0 ; j < uInt(nchanvec(curspwid)); j++) {
	    if (spwchansels_p(0,curspwid,j)==1) {
	      if (first) {
		dataStart_p[k]=j;
		first = False;
	      }
	      lastchan=j;
	      nchn++;
	    }	
	  }
	  dataNchan_p[k]=Int(ceil(Double(lastchan-dataStart_p[k])/Double(dataStep_p[k])))+1;
	  //dataNchan_p[k]=Int(ceil(Double(chanselmat.row(k)(2)-dataStart_p[k])/Double(dataStep_p[k])))+1;
	  
	  //if(dataNchan_p[k]<1)
	  //  dataNchan_p[k]=1;	  
	  
	  //cout<<"modified start="<<dataStart_p[k]<<endl;
	  //cout<<"modified nchan="<<dataNchan_p[k]<<endl;
	  //
	  //Since msselet will be applied to the data before flags from spwchansels_p
	  //are applied to the data in FTMachine, shift spwchansels_p by dataStart_p
	  //for (uInt j=0  ; j < nchanvec(k)-dataStart_p[k]; j++){
	  for (uInt j=0  ; j < uInt(nchanvec(curspwid)); j++){
	    if ( Int(j) < nchanvec(curspwid)-dataStart_p[k]) {
	      spwchansels_tmp(0,curspwid,j) = spwchansels_p(0,curspwid,j+dataStart_p[k]);
	    }
	    else {
	      spwchansels_tmp(0,curspwid,j) = 0;
	    }
	  }
	}
	spwchansels_p = spwchansels_tmp;
      }
    }
    if(!(exprNode.isNull())){
      mssel_p = new MeasurementSet((*ms_p)(exprNode));
    }
    else{
      // Null take all the ms ...setdata() blank means that
      mssel_p = new MeasurementSet(*ms_p);
    }
    AlwaysAssert(!mssel_p.null(), AipsError);
    if(mssel_p->nrow()==0) {
      //delete mssel_p; 
      mssel_p=0;
      os << (be_calm ? LogIO::NORMAL4 : LogIO::WARN)
         << "Selection is empty: reverting to sorted MeasurementSet"
	 << LogIO::POST;
      mssel_p=new MeasurementSet(*ms_p);
      nullSelect_p=True;
    }
    else {
      mssel_p->flush();
      nullSelect_p=False;
    }
    if (nullSelect_p) {
      Table mytab(msname_p+"/FIELD", Table::Old);
      if (mytab.nrow() > 1) {
	os << LogIO::NORMAL4 << "Multiple fields selected" << LogIO::POST;
	multiFields_p = True;
      } else {
	os << LogIO::NORMAL4 << "Single field selected" << LogIO::POST;
	multiFields_p = False;
      }
    }
    
    uInt nvis_all = ms_p->nrow();
    
    // Now create the VisSet
    this->makeVisSet(*mssel_p); 
    AlwaysAssert(rvi_p, AipsError);
    uInt nvis_sel = mssel_p->nrow();
    
    if(nvis_sel != nvis_all) {
      os << LogIO::NORMAL // Loglevel INFO
         << "Selected " << nvis_sel << " out of "
         << nvis_all << " visibilities."
	 << LogIO::POST;
    }
    else {
      os << (be_calm ? LogIO::NORMAL4 : LogIO::NORMAL)
         << "Selected all visibilities" << LogIO::POST; // Loglevel INFO
    }
    //    }
    
    // Now we do a selection to cut down the amount of information
    // passed around.
    
    this->selectDataChannel(dataspectralwindowids_p, dataMode_p,
			    dataNchan_p, dataStart_p, dataStep_p,
                            mDataStart_p, mDataStep_p);
    ///Tell iterator to use on the fly imaging weights if scratch columns is
    ///not in use
    imwgt_p=VisImagingWeight("natural");
    rvi_p->useImagingWeight(imwgt_p);
    
    // Guess that the beam is no longer valid
    beamValid_p=False;
    destroySkyEquation();
    if(!valid()){ 
      this->unlock();
      os << LogIO::SEVERE << "Check your data selection or Measurement set "
         << LogIO::EXCEPTION;
      return False;
    }
    this->unlock();
    return !nullSelect_p;
  } catch (AipsError x) {
    this->unlock();
    throw(x);
   
    return False;
  } 
  return !nullSelect_p;
}


Bool Imager::setmfcontrol(const Float cyclefactor,
			  const Float cyclespeedup,
			  const Int stoplargenegatives, 
			  const Int stoppointmode,
			  const String& scaleType,
			  const Float minPB,
			  const Float constPB,
			  const Vector<String>& fluxscale)
{  
  cyclefactor_p = cyclefactor;
  cyclespeedup_p =  cyclespeedup;
  stoplargenegatives_p = stoplargenegatives;
  stoppointmode_p = stoppointmode;
  fluxscale_p.resize( fluxscale.nelements() );
  fluxscale_p = fluxscale;
  scaleType_p = scaleType;
  minPB_p = minPB;

  constPB_p = constPB;
  return True;
}  


Bool Imager::setvp(const Bool dovp,
		   const Bool doDefaultVPs,
		   const String& vpTable,
		   const Bool doSquint,
		   const Quantity &parAngleInc,
		   const Quantity &skyPosThreshold,
		   String defaultTel,
                   const Bool verbose)
{

#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::setvp",23);
#endif

  //  if(!valid())
  //    {

  //#ifdef PABLO_IO
  //      traceEvent(1,"Exiting Imager::setvp",22);
  //#endif

  //     return False;
  //    }
  LogIO os(LogOrigin("Imager", "setvp()", WHERE));
  
  os << LogIO::NORMAL << "Setting voltage pattern parameters" << LogIO::POST; // Loglevel PROGRESS
  
  if(!dovp && !vp_p)
    delete vp_p;
  vp_p=0;
  doVP_p=dovp;
  doDefaultVP_p = doDefaultVPs;
  vpTableStr_p = vpTable;
  telescope_p= defaultTel;
  if (doSquint) {
    squintType_p = BeamSquint::GOFIGURE;
  } else {
    squintType_p = BeamSquint::NONE;
  }

  parAngleInc_p = parAngleInc;

  skyPosThreshold_p = skyPosThreshold;
  os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3) // Loglevel INFO
     <<"Sky position tolerance is "<<skyPosThreshold_p.getValue("deg")
     << " degrees" << LogIO::POST;

  if (doDefaultVP_p) {
    os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3) // Loglevel INFO
       << "Using system default voltage patterns for each telescope" << LogIO::POST;
  } else {
    os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3) // Loglevel INFO
       << "Using user defined voltage patterns in Table "
       <<  vpTableStr_p << LogIO::POST;
  }
  if (doSquint) {
    os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3) // Loglevel INFO
       << "Beam Squint will be included in the VP model" <<  LogIO::POST;
    os << (verbose ? LogIO::NORMAL : LogIO::NORMAL3)
       << "and the Parallactic Angle increment is "  // Loglevel INFO
       << parAngleInc_p.getValue("deg") << " degrees"  << LogIO::POST;
  }

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::setvp",22);
#endif

  // muddled with the state of SkyEquation..so redo it
  destroySkyEquation();
  return True;
}

Bool Imager::setoptions(const String& ftmachine, const Long cache, const Int tile,
			const String& gridfunction, const MPosition& mLocation,
                        const Float padding,
			const Int wprojplanes,
			const String& epJTableName,
			const Bool applyPointingOffsets,
			const Bool doPointingCorrection,
			const String& cfCacheDirName,const Float& paStep, 
			const Float& pbLimit, const String& interpMeth, const Int imageTileVol,
			const Bool singprec)
{

#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::setoptions",28);
#endif

  if(!valid()) 
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::setoptions",27);
#endif

      return False;
    }
  if(!assertDefinedImageParameters())
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::setoptions",27);
#endif

      return False;
    }
  LogIO os(LogOrigin("imager", "setoptions()", WHERE));
  
  os << LogIO::NORMAL << "Setting processing options" << LogIO::POST; // Loglevel PROGRESS

  ftmachine_p=downcase(ftmachine);
  if(ftmachine_p=="gridft") {
    os << LogIO::WARN
       << "FT machine gridft is now called ft - please use the new name in future"
       << LogIO::POST;
    ftmachine_p="ft";
  }

  if(ftmachine_p=="wfmemoryft"){
    wfGridding_p=True;
    ftmachine_p="ft";
  }

  wprojPlanes_p=wprojplanes;
  epJTableName_p = epJTableName;
  cfCacheDirName_p = cfCacheDirName;
  paStep_p = paStep;
  pbLimit_p = pbLimit;
  freqInterpMethod_p=interpMeth;
  imageTileVol_p=imageTileVol;
  singlePrec_p=singprec;

  if(cache>0) cache_p=cache;
  if(tile>0) tile_p=tile;
  gridfunction_p=downcase(gridfunction);
  mLocation_p=mLocation;
  if(padding>=1.0) {
    padding_p=padding;
  }
  // Destroy the FTMachine
  if(ft_p) {delete ft_p; ft_p=0;}
  if(gvp_p) {delete gvp_p; gvp_p=0;}
  if(cft_p) {delete cft_p; cft_p=0;}

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::setoptions",27);
#endif

  doPointing = applyPointingOffsets;
  doPBCorr = doPointingCorrection;

  return True;
}

Bool Imager::setsdoptions(const Float scale, const Float weight, 
			  const Int convsupport, String pointCol)
{

#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::setsdoptions",28);
#endif

  if(!valid()) 
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::setsdoptions",27);
#endif

      return False;
    }

  LogIO os(LogOrigin("imager", "setsdoptions()", WHERE));
  
  os << LogIO::NORMAL << "Setting single dish processing options" << LogIO::POST; // Loglevel PROGRESS
  
  sdScale_p=scale;
  sdWeight_p=weight;
  sdConvSupport_p=convsupport;
  pointingDirCol_p=pointCol;
  pointingDirCol_p.upcase();
  if( (pointingDirCol_p != "DIRECTION") &&(pointingDirCol_p != "TARGET") && (pointingDirCol_p != "ENCODER") && (pointingDirCol_p != "POINTING_OFFSET") && (pointingDirCol_p != "SOURCE_OFFSET")){
    os << LogIO::SEVERE
       << "No such direction column as "<< pointingDirCol_p
       << " in pointing table "<< LogIO::EXCEPTION;
  }
  // Destroy the FTMachine
  if(ft_p) {delete ft_p; ft_p=0;}
  if(gvp_p) {delete gvp_p; gvp_p=0;}
  if(cft_p) {delete cft_p; cft_p=0;}

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::setsdoptions",27);
#endif

  return True;
}

Bool Imager::mask(const String& mask, const String& image,
		  const Quantity& threshold) 
{
  //if(!valid()) return False;
  LogIO os(LogOrigin("imager", "mask()", WHERE));
  //if(!assertDefinedImageParameters()) return False;
  
  try {
    //this->lock();
    if(image=="") {
      //this->unlock();
      os << LogIO::SEVERE << "Need name for template image" << LogIO::EXCEPTION;
      return False;
    }
    String maskName(mask);
    if(maskName=="") {
      maskName=image+".mask";
    }
    if(!clone(image, maskName)) return False;
    PagedImage<Float> maskImage(maskName);
    maskImage.table().markForDelete();
    PagedImage<Float> imageImage(image);

    if(threshold.check(UnitVal(1.0, "Jy"))){
      os << LogIO::NORMAL // Loglevel INFO
         << "Making mask image " << maskName << ", applying threshold "
         << threshold.get("Jy").getValue() << "Jy, " << endl
         << "to template image " << image << LogIO::POST;
    
      StokesImageUtil::MaskFrom(maskImage, imageImage, threshold);
    }
    else{
      os << LogIO::NORMAL // Loglevel INFO
         << "Making mask image " << maskName << ", applying threshold "
         << threshold.getValue() << " " << threshold.getUnit() << endl
         << "to template image " << image << LogIO::POST;
    
      StokesImageUtil::MaskFrom(maskImage, imageImage, threshold.getValue());
    }
    maskImage.table().unmarkForDelete();

    //this->lock();
    return True;
  } catch (AipsError x) {
    //this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  //this->unlock();
  return True;
}

Bool Imager::boxmask(const String& mask, const Vector<Int>& blc,
		  const Vector<Int>& trc, const Float value) 
{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "boxmask()", WHERE));
  
  try {
    
    if(!assertDefinedImageParameters()) return False;
    
    if(mask=="") {
      os << LogIO::SEVERE << "Need name for mask image" << LogIO::EXCEPTION;
      return False;
    }
    if(!Table::isWritable(mask)) {
      make(mask);
      this->lock();
    }
    PagedImage<Float> maskImage(mask);
    maskImage.table().markForDelete();
    

    IPosition iblc(blc);
    IPosition itrc(trc);
    IPosition iinc(iblc.nelements(), 1);
    LCBox::verify(iblc, itrc, iinc, maskImage.shape());
    
    os << LogIO::DEBUG1
       << "Setting '" << mask << "' blc=" << iblc
       << " trc=" << itrc << " to " << value << LogIO::POST;
    
    StokesImageUtil::BoxMask(maskImage, iblc, itrc, value);
    
    maskImage.table().unmarkForDelete();

    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  return True;
}

  Bool Imager::regionmask(const String& maskimage, Record* imageRegRec, 
			  Matrix<Quantity>& blctrcs, Matrix<Float>& circles, 
			  const Float& value){

  //This function does not modify ms(s) so no need of lock 
  LogIO os(LogOrigin("imager", "regionmask()", WHERE));
  if(!Table::isWritable(maskimage)) {
    make(maskimage);
  }
  return Imager::regionToImageMask(maskimage, imageRegRec, blctrcs, circles, value);

}

  Bool Imager::regionToImageMask(const String& maskimage, Record* imageRegRec, Matrix<Quantity>& blctrcs, Matrix<Float>& circles, const Float& value){
  PagedImage<Float> maskImage(maskimage);
  CoordinateSystem cSys=maskImage.coordinates();
  maskImage.table().markForDelete();
  ImageRegion *boxregions=0;
  ImageRegion *circleregions=0;
  RegionManager regMan;
  regMan.setcoordsys(cSys);
  Vector<Quantum<Double> > blc(2);
  Vector<Quantum<Double> > trc(2);
  if(blctrcs.nelements() !=0){
    if(blctrcs.shape()(1) != 4)
      throw(AipsError("Need a list of 4 elements to define a box"));
    Int nrow=blctrcs.shape()(0);
    Vector<Int> absRel(2, RegionType::Abs); 
    PtrBlock<const WCRegion *> lesbox(nrow);
    for (Int k=0; k < nrow; ++k){
      blc(0) = blctrcs(k,0);
      blc(1) = blctrcs(k,1);
      trc(0) = blctrcs(k,2);
      trc(1) = blctrcs(k,3); 
      lesbox[k]= new WCBox (blc, trc, cSys, absRel);
    }
    boxregions=regMan.doUnion(lesbox);
    for (Int k=0; k < nrow; ++k){
      delete lesbox[k];
    }
  }
  if((circles.nelements()) > 0){
    if(circles.shape()(1) != 3)
      throw(AipsError("Need a list of 3 elements to define a circle"));
    Int nrow=circles.shape()(0);
    Vector<Float> cent(2); 
    cent(0)=circles(0,1); cent(1)=circles(0,2);
    Float radius=circles(0,0);
    IPosition xyshape(2,maskImage.shape()(0),maskImage.shape()(1));
    LCEllipsoid *circ= new LCEllipsoid(cent, radius, xyshape);
    //Tell LCUnion to delete the pointers
    LCUnion *elunion= new LCUnion(True, circ);
    //now lets do the remainder
    for (Int k=1; k < nrow; ++k){
      cent(0)=circles(k,1); cent(1)=circles(k,2);
      radius=circles(k,0);
      circ= new LCEllipsoid(cent, radius, xyshape); 
      elunion=new LCUnion(True, elunion, circ);
    }
    //now lets extend that to the whole image
    IPosition trc(2);
    trc(0)=maskImage.shape()(2)-1;
    trc(1)=maskImage.shape()(3)-1;
    LCBox lbox(IPosition(2,0,0), trc, 
	       IPosition(2,maskImage.shape()(2),maskImage.shape()(3)) );
    LCExtension linter(*elunion, IPosition(2,2,3),lbox);
    circleregions=new ImageRegion(linter);
    delete elunion;
  }


  ImageRegion* recordRegion=0;
  if(imageRegRec !=0){
    ImageRegion::tweakedRegionRecord(imageRegRec);
    TableRecord rec1;
    rec1.assign(*imageRegRec);
    recordRegion=ImageRegion::fromRecord(rec1,"");    
  }
  ImageRegion *unionReg=0;
  if(boxregions!=0 && recordRegion!=0){
    unionReg=regMan.doUnion(*boxregions, *recordRegion);
    delete boxregions; boxregions=0;
    delete recordRegion; recordRegion=0;
  }
  else if(boxregions !=0){
    unionReg=boxregions;
  }
  else if(recordRegion !=0){
    unionReg=recordRegion;
  }
      

 
  
  if(unionReg !=0){
    regionToMask(maskImage, *unionReg, value);
    delete unionReg; unionReg=0;
  }
  //As i can't unionize LCRegions and WCRegions;  do circles seperately
  if(circleregions !=0){
    regionToMask(maskImage, *circleregions, value);
    delete circleregions;
    circleregions=0;
  }
  maskImage.table().unmarkForDelete();
  return True;

}


Bool Imager::regionToMask(ImageInterface<Float>& maskImage, ImageRegion& imagreg, const Float& value){

  SubImage<Float> partToMask(maskImage, imagreg, True);
  LatticeRegion latReg=imagreg.toLatticeRegion(maskImage.coordinates(), maskImage.shape());
  ArrayLattice<Bool> pixmask(latReg.get());
  LatticeExpr<Float> myexpr(iif(pixmask, value, partToMask) );
  partToMask.copyData(myexpr);

  return True;
}


Bool Imager::clipimage(const String& image, const Quantity& threshold)
{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "clipimage()", WHERE));
  
  this->lock();
  try {
    
    if(!assertDefinedImageParameters()) return False;
    
    if(image=="") {
      this->unlock();
      os << LogIO::SEVERE << "Need name for image" << LogIO::EXCEPTION;
      return False;
    }
    PagedImage<Float> imageImage(image);
    os << LogIO::NORMAL // Loglevel PROGRESS
       << "Zeroing " << image << ", for all pixels where Stokes I < threshold "
       << threshold.get("Jy").getValue() << "Jy " << LogIO::POST;
    
    StokesImageUtil::MaskOnStokesI(imageImage, threshold);
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  
  return True;
}

// Add together low and high resolution images in the Fourier plane
Bool Imager::feather(const String& image, const String& highRes,
		     const String& lowRes, const String& lowPSF)
{
  // if(!valid()) return False;
  
  LoggerHolder lh (False);
  LogIO os = lh.logio();
  os << LogOrigin("imager", "feather()");
  
  try {
    Bool noStokes=False;
    String outLowRes=lowRes;
    String outHighRes=highRes;
    {
      if ( ! doVP_p ) {
	this->unlock();
	os << LogIO::SEVERE << 
	  "Must invoke setvp() first in order to apply the primary beam" 
	   << LogIO::EXCEPTION;
	return False;
      }
      
      os << LogIO::NORMAL // Loglevel PROGRESS
         << "\nFeathering together high and low resolution images.\n" << LogIO::POST;
      
     
      // Get initial images
      { //Drat lets deal with images that don't have stokes.
	PagedImage<Float> hightemp(highRes);
	PagedImage<Float> lowtemp(lowRes);
	if(hightemp.shape().nelements() != lowtemp.shape().nelements()){
	  this->unlock();
	  os << LogIO::SEVERE << 
	    "High res. image and low res. image donot have same number of axes" 
	     << LogIO::EXCEPTION;
	  return False;
	  
	}
	if ( (hightemp.coordinates().findCoordinate(Coordinate::STOKES) < 0) &&
	     (lowtemp.coordinates().findCoordinate(Coordinate::STOKES) < 0)){
	  noStokes=True;
	  os << LogIO::NORMAL // Loglevel PROGRESS
             << "Making some temporary images as the inputs have no Stokes axis.\n" 
             << LogIO::POST;
	  PtrHolder<ImageInterface<Float> > outImage1;
	  outHighRes= highRes+"_stokes";
	  ImageUtilities::addDegenerateAxes (os, outImage1, hightemp, outHighRes,
					     False, False,
					     "I", False, False,
					     False);

	  PtrHolder<ImageInterface<Float> > outImage2;
	  outLowRes= lowRes+"_stokes";
	  ImageUtilities::addDegenerateAxes (os, outImage2, lowtemp, outLowRes,
					     False, False,
					     "I", False, False,
					     False);
	  
	}
      }
      PagedImage<Float> high(outHighRes);
      PagedImage<Float> low0(outLowRes);
      
      Vector<Quantum<Double> > hBeam, lBeam;
      ImageInfo highInfo=high.imageInfo();
      hBeam=highInfo.restoringBeam();
      ImageInfo lowInfo=low0.imageInfo();
      lBeam=lowInfo.restoringBeam();
      if((hBeam.nelements()<3)||!((hBeam(0).get("arcsec").getValue()>0.0)
				  &&(hBeam(1).get("arcsec").getValue()>0.0))) {
	os << LogIO::WARN 
	   << "High resolution image does not have any resolution information - will be unable to scale correctly.\n" 
	   << LogIO::POST;
      }
      
      PBMath * myPBp = 0;
      if(lowPSF=="" &&((lBeam.nelements()==0) || 
	   (lBeam.nelements()>0)&&(lBeam(0).get("arcsec").getValue()==0.0)) ) {
	// create the low res's PBMath object, needed to apply PB 
	// to make high res Fourier weight image
	if (doDefaultVP_p) {
	  // look up the telescope in ObsInfo
	  ObsInfo oi = low0.coordinates().obsInfo();
	  String myTelescope = oi.telescope();
	  if (myTelescope == "") {
	    this->unlock();
	    os << LogIO::SEVERE << "No telescope imbedded in low res image" 
	       << LogIO::POST;
	    os << LogIO::SEVERE << "Create a PB description with the vpmanager"
	       << LogIO::EXCEPTION;
	    return False;
	  }
	  Quantity qFreq;
	  {
	    Int spectralIndex=low0.coordinates().findCoordinate(Coordinate::SPECTRAL);
	    AlwaysAssert(spectralIndex>=0, AipsError);
	    SpectralCoordinate
	      spectralCoord=
	      low0.coordinates().spectralCoordinate(spectralIndex);
	    Vector<String> units(1); units = "Hz";
	    spectralCoord.setWorldAxisUnits(units);	
	    Vector<Double> spectralWorld(1);
	    Vector<Double> spectralPixel(1);
	    spectralPixel(0) = 0;
	    spectralCoord.toWorld(spectralWorld, spectralPixel);  
	    Double freq  = spectralWorld(0);
	    qFreq = Quantity( freq, "Hz" );
	  }
	  String band;
	  PBMath::CommonPB whichPB;
	  String pbName;
	  // get freq from coordinates
	  PBMath::whichCommonPBtoUse (myTelescope, qFreq, band, whichPB, 
				      pbName);
	  if (whichPB  == PBMath::UNKNOWN) {
	    this->unlock();
	    os << LogIO::SEVERE << "Unknown telescope for PB type: " 
	       << myTelescope << LogIO::EXCEPTION;
	    return False;
	  }
	  myPBp = new PBMath(whichPB);
	} else {
	  // get the PB from the vpTable
	  Table vpTable( vpTableStr_p );
	  ROScalarColumn<TableRecord> recCol(vpTable, (String)"pbdescription");
	  myPBp = new PBMath(recCol(0));
	}
	AlwaysAssert((myPBp != 0), AipsError);
      }

      // regrid the single dish image
      TempImage<Float> low(high.shape(), high.coordinates());
      {
	IPosition axes(2,0,1);
	if(high.shape().nelements() >2){
	  Int spectralAxisIndex=high.coordinates().
	    findCoordinate(Coordinate::SPECTRAL);
	  if(spectralAxisIndex > -1){
	    axes.resize(3);
	    axes(0)=0;
	    axes(1)=1;
	    axes(2)=spectralAxisIndex+1;
	  }
	}
	ImageRegrid<Float> ir;
	ir.regrid(low, Interpolate2D::LINEAR, axes, low0);
      }
    
      // get image center direction (needed for SD PB, which is needed for
      // the high res Fourier weight image
      MDirection wcenter;  
      {
	Int directionIndex=
	  high.coordinates().findCoordinate(Coordinate::DIRECTION);
	AlwaysAssert(directionIndex>=0, AipsError);
	DirectionCoordinate
	  directionCoord=high.coordinates().directionCoordinate(directionIndex);
	Vector<Double> pcenter(2);
	pcenter(0) = high.shape()(0)/2;
	pcenter(1) = high.shape()(1)/2;    
	directionCoord.toWorld( wcenter, pcenter );
      }
      
      // make the weight image for high res Fourier plane:  1 - normalized(FT(sd_PB))
      IPosition myshap(high.shape());
      for( uInt k=2; k< myshap.nelements(); ++k){
	myshap(k)=1;
      }
      
      TempImage<Complex> cweight(myshap, high.coordinates());
      if(lowPSF=="") {
	os << LogIO::NORMAL // Loglevel INFO
           << "Using primary beam to determine weighting.\n" << LogIO::POST;
	if((lBeam.nelements()==0) || 
	   (lBeam.nelements()>0)&&(lBeam(0).get("arcsec").getValue()==0.0)) {
	  cweight.set(1.0);
	  if (myPBp != 0) {
	    myPBp->applyPB(cweight, cweight, wcenter, Quantity(0.0, "deg"), 
			   BeamSquint::NONE);
	  
	    TempImage<Float> lowpsf0(cweight.shape(), cweight.coordinates());
	    
	    os << LogIO::NORMAL // Loglevel INFO
               << "Determining scaling from SD Primary Beam.\n"
	       << LogIO::POST;
	    lBeam.resize(3);
	    StokesImageUtil::To(lowpsf0, cweight);
	    StokesImageUtil::FitGaussianPSF(lowpsf0, 
					    lBeam(0), lBeam(1), lBeam(2)); 
	  }
	  delete myPBp;
	}
	else{
	  os << LogIO::NORMAL // Loglevel INFO
             << "Determining scaling from SD restoring beam.\n"
	     << LogIO::POST;
	  TempImage<Float> lowpsf0(cweight.shape(), cweight.coordinates());
	  lowpsf0.set(0.0);
	  IPosition center(4, Int((cweight.shape()(0)/4)*2), 
			   Int((cweight.shape()(1)/4)*2),0,0);
	  lowpsf0.putAt(1.0, center);
	  StokesImageUtil::Convolve(lowpsf0, lBeam(0), lBeam(1),
				    lBeam(2), False);
	  StokesImageUtil::From(cweight, lowpsf0);

	}
      }
      else {
	os << LogIO::NORMAL // Loglevel INFO
           << "Using specified low resolution PSF to determine weighting.\n" 
	   << LogIO::POST;
	// regrid the single dish psf
	PagedImage<Float> lowpsfDisk(lowPSF);
	IPosition lshape(lowpsfDisk.shape());
	lshape.resize(4);
	lshape(2)=1; lshape(3)=1;
	TempImage<Float>lowpsf0(lshape,lowpsfDisk.coordinates());
	IPosition blc(lowpsfDisk.shape());
	IPosition trc(lowpsfDisk.shape());
	blc(0)=0; blc(1)=0;
	trc(0)=lowpsfDisk.shape()(0)-1;
	trc(1)=lowpsfDisk.shape()(1)-1;
	for( uInt k=2; k < lowpsfDisk.shape().nelements(); ++k){
	  blc(k)=0; trc(k)=0;	  	  
	}// taking first plane
	Slicer sl(blc, trc, Slicer::endIsLast);
	lowpsf0.copyData(SubImage<Float>(lowpsfDisk, sl, False));
	TempImage<Float> lowpsf(myshap, high.coordinates());
	{
	  ImageRegrid<Float> ir;
	  IPosition axes(2,0,1);   // if its a cube, regrid the spectral too
	  ir.regrid(lowpsf, Interpolate2D::LINEAR, axes, lowpsf0);
	}
	if((lBeam.nelements()==0) || 
	   (lBeam.nelements()>0)&&(lBeam(0).get("arcsec").getValue()==0.0)) {
	  os << LogIO::NORMAL // Loglevel INFO
             << "Determining scaling from low resolution PSF.\n" << LogIO::POST;
	  lBeam.resize(3);
	  StokesImageUtil::FitGaussianPSF(lowpsf0, lBeam(0), lBeam(1), lBeam(2));
	}
	StokesImageUtil::From(cweight, lowpsf);
      }
      LatticeFFT::cfft2d( cweight );
      LatticeExprNode node = max( cweight );
      Float fmax = abs(node.getComplex());
      cweight.copyData(  (LatticeExpr<Complex>)( 1.0f - cweight/fmax ) );
      
      // FT high res image
      TempImage<Complex> cimagehigh(high.shape(), high.coordinates() );
      StokesImageUtil::From(cimagehigh, high);
      LatticeFFT::cfft2d( cimagehigh );
      
      // FT low res image
      TempImage<Complex> cimagelow(high.shape(), high.coordinates() );
      StokesImageUtil::From(cimagelow, low);
      LatticeFFT::cfft2d( cimagelow );


      // This factor comes from the beam volumes
      if(sdScale_p!=1.0)
        os << LogIO::NORMAL // Loglevel INFO
           << "Multiplying single dish data by user specified factor"
           << sdScale_p << ".\n" << LogIO::POST;
      Float sdScaling  = sdScale_p;
      if((hBeam(0).get("arcsec").getValue()>0.0)
	 &&(hBeam(1).get("arcsec").getValue()>0.0)&&
       (lBeam(0).get("arcsec").getValue()>0.0)&&
	 (lBeam(1).get("arcsec").getValue()>0.0)) {
	Float beamFactor=
	  hBeam(0).get("arcsec").getValue()*hBeam(1).get("arcsec").getValue()/
	  (lBeam(0).get("arcsec").getValue()*lBeam(1).get("arcsec").getValue());
	os << LogIO::NORMAL // Loglevel INFO
           << "Applying additional scaling for ratio of the volumes of the high to the low resolution images : "
	   <<  beamFactor << ".\n" << LogIO::POST;
	sdScaling*=beamFactor;
      }
      else {
	os << LogIO::WARN << "Insufficient information to scale correctly.\n" 
	   << LogIO::POST;
      }

      // combine high and low res, appropriately normalized, in Fourier
      // plane. The vital point to remember is that cimagelow is already
      // multiplied by 1-cweight so we only need adjust for the ratio of beam
      // volumes
      Vector<Int> extraAxes(cimagehigh.shape().nelements()-2);
      if(extraAxes.nelements() > 0){
	
	if(extraAxes.nelements() ==2){
	  Int n3=cimagehigh.shape()(2);
	  Int n4=cimagehigh.shape()(3);
	  IPosition blc(cimagehigh.shape());
	  IPosition trc(cimagehigh.shape());
	  blc(0)=0; blc(1)=0;
	  trc(0)=cimagehigh.shape()(0)-1;
	  trc(1)=cimagehigh.shape()(1)-1;
	  for (Int j=0; j < n3; ++j){
	    for (Int k=0; k < n4 ; ++k){
	      blc(2)=j; trc(2)=j;
	      blc(3)=k; trc(3)=k;
	      Slicer sl(blc, trc, Slicer::endIsLast);
	      SubImage<Complex> cimagehighSub(cimagehigh, sl, True);
	      SubImage<Complex> cimagelowSub(cimagelow, sl, True);
	      cimagehighSub.copyData(  (LatticeExpr<Complex>)((cimagehighSub * cweight + cimagelowSub * sdScaling)));
	    }
	  }
	}
      }
      else{
	cimagehigh.copyData(  
			    (LatticeExpr<Complex>)((cimagehigh * cweight 
						    + cimagelow * sdScaling)));
      }
      // FT back to image plane
      LatticeFFT::cfft2d( cimagehigh, False);
    
      // write to output image
      PagedImage<Float> featherImage(high.shape(), high.coordinates(), image );
      StokesImageUtil::To(featherImage, cimagehigh);
      ImageUtilities::copyMiscellaneous(featherImage, high);

      try{
      { // write data processing history into image logtable
	LoggerHolder imagelog (False);
	LogSink& sink = imagelog.sink();
	LogOrigin lor(String("imager"), String("feather()"));
	LogMessage msg(lor);
	if (!ms_p.null()) {
	  String info = "MeasurementSet is " + ms_p->tableName() + "\n";
	  sink.postLocally(msg.message(info));
	  ROMSHistoryColumns msHis(ms_p->history());
	  if (msHis.nrow()>0) {
	    ostringstream oos;
	    uInt nmessages = msHis.time().nrow();
	    for (uInt i=0; i < nmessages; ++i) {
	      oos << frmtTime((msHis.time())(i))
		  << "  HISTORY " << (msHis.origin())(i);
	      oos << (msHis.message())(i)
		  << endl;
	    }
	    String historyline(oos);
	    sink.postLocally(msg.message(historyline));
	  }
	}
	ostringstream oos;
	oos << endl << "Imager::feather() input paramaters:" << endl
	    << "Feathered image =      '" << image   << "'" << endl
	    << "High resolution image ='" << highRes << "'" << endl
	    << "Low resolution image = '" << lowRes  << "'" << endl
	    << "Low resolution PSF =   '" << lowPSF  << "'" << endl << endl;
	String inputs(oos);
	sink.postLocally(msg.message(inputs));
	imagelog.flush();

	LoggerHolder& log = featherImage.logger();
	log.append(imagelog);
	lh.flush();
	log.append(lh);
	log.flush();
      }
      }catch(exception& x){

	os << LogIO::WARN << "Caught exception: " << x.what()
       << LogIO::POST;
	//os << LogIO::SEVERE << "Can it be that your MS/HISTORY table has gotten corrupted ? \n you may consider deleting all the rows from this table"
	// <<LogIO::POST; 
	//continue and wrap up

      }
      catch(...){
	os << LogIO::SEVERE << "your MS/HISTORY table may be corrupted;  you may consider deleting all the rows from this table" << LogIO::POST;

      }
    }
  
    if(noStokes){
      Table::deleteTable(outHighRes);
      Table::deleteTable(outLowRes);
    }
    return True;
  } catch (AipsError x) {
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  
  return True;
}




// Apply a primary beam or voltage pattern to an image
Bool Imager::pb(const String& inimage, 
		const String& outimage,
		const String& incomps,
		const String& outcomps,
		const String& operation, 
		const MDirection& pointingCenter,
		const Quantity& pa,
		const String& pborvp)

{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "pb()", WHERE));
  
  PagedImage<Float> * inImage_pointer = 0;
  PagedImage<Float> * outImage_pointer = 0;
  ComponentList * inComps_pointer = 0;
  ComponentList * outComps_pointer = 0;
  PBMath * myPBp = 0;
  try {

    if ( ! doVP_p ) {
      this->unlock();
      os << LogIO::SEVERE << 
	"Must invoke setvp() first in order to apply the primary beam" << LogIO::EXCEPTION;
      return False;
    }
    
    if (pborvp == "vp") {
      this->unlock();
      os << LogIO::SEVERE << "VP application is not yet implemented in DOimager" << LogIO::EXCEPTION;
      return False;
    }

    if (operation == "apply") {
      os << LogIO::DEBUG1
         << "function pb will apply " << pborvp << LogIO::POST;
    } else if (operation=="correct") {
      os << LogIO::DEBUG1
         << "function pb will correct for " << pborvp << LogIO::POST;
    } else {
      this->unlock();
      os << LogIO::SEVERE << "Unknown pb operation " << operation 
	 << LogIO::EXCEPTION;
      return False;
    }
    
    // Get initial image and/or SkyComponents

    if (incomps!="") {
      if(!Table::isReadable(incomps)) {
	this->unlock();
	os << LogIO::SEVERE << "ComponentList " << incomps
	   << " not readable" << LogIO::EXCEPTION;
	return False;
      }
      inComps_pointer = new ComponentList(incomps);
      outComps_pointer = new ComponentList( inComps_pointer->copy() );
    }
    if (inimage !="") {
      if(!Table::isReadable(inimage)) {
	this->unlock();
	os << LogIO::SEVERE << "Image " << inimage << " not readable" 
	   << LogIO::EXCEPTION;
	return False;
      }
      inImage_pointer = new PagedImage<Float>( inimage );
      if (outimage != "") {
	outImage_pointer = new PagedImage<Float>( TiledShape(inImage_pointer->shape(), inImage_pointer->niceCursorShape()), 
						  inImage_pointer->coordinates(), outimage);
      }
    }
    // create the PBMath object, needed to apply PB 
    // to make high res Fourier weight image
    Quantity qFreq;
    if (doDefaultVP_p && inImage_pointer==0) {
      this->unlock();
      os << LogIO::SEVERE << 
	"There is no default telescope associated with a componentlist" 
	 << LogIO::POST;
      os << LogIO::SEVERE << 
	"Either specify the PB/VP via a vptable or supply an image as well" 
	 << LogIO::EXCEPTION;
	return False;
    } else if (doDefaultVP_p && inImage_pointer!=0) {
      // look up the telescope in ObsInfo
      ObsInfo oi = inImage_pointer->coordinates().obsInfo();
      String myTelescope = oi.telescope();
      if (myTelescope == "") {
	this->unlock();
	os << LogIO::SEVERE << "No telescope imbedded in image" 
	   << LogIO::EXCEPTION;
	return False;
      }
      {
	Int spectralIndex=inImage_pointer->coordinates().findCoordinate(Coordinate::SPECTRAL);
	AlwaysAssert(spectralIndex>=0, AipsError);
	SpectralCoordinate
	  spectralCoord=inImage_pointer->coordinates().spectralCoordinate(spectralIndex);
	Vector<String> units(1); units = "Hz";
	spectralCoord.setWorldAxisUnits(units);	
	Vector<Double> spectralWorld(1);
	Vector<Double> spectralPixel(1);
	spectralPixel(0) = 0;
	spectralCoord.toWorld(spectralWorld, spectralPixel);  
	Double freq  = spectralWorld(0);
	qFreq = Quantity( freq, "Hz" );
      }
      String band;
      PBMath::CommonPB whichPB;
      String pbName;
      // get freq from coordinates
      PBMath::whichCommonPBtoUse (myTelescope, qFreq, band, whichPB, pbName);
      if (whichPB  == PBMath::UNKNOWN) {
	this->unlock();
	os << LogIO::SEVERE << "Unknown telescope for PB type: " 
	   << myTelescope << LogIO::EXCEPTION;
	return False;
      }
      myPBp = new PBMath(whichPB);
    } else {
      // get the PB from the vpTable
      Table vpTable( vpTableStr_p );
      ROScalarColumn<TableRecord> recCol(vpTable, (String)"pbdescription");
      myPBp = new PBMath(recCol(0));
    }
    AlwaysAssert((myPBp != 0), AipsError);


    // Do images (if indeed we have any)
    if (outImage_pointer!=0) {
      Vector<Int> whichStokes;
      CoordinateSystem cCoords;
      cCoords=StokesImageUtil::CStokesCoord(inImage_pointer->shape(),
					    inImage_pointer->coordinates(),
					    whichStokes,
					    polRep_p);
      TempImage<Complex> cIn(inImage_pointer->shape(),
			     cCoords);
      StokesImageUtil::From(cIn, *inImage_pointer);
      if (operation=="apply") {
	myPBp->applyPB(cIn, cIn, pointingCenter, 
		       pa, squintType_p, False);
      } else {
	myPBp->applyPB(cIn, cIn, pointingCenter, 
		       pa, squintType_p, True);
      }
      StokesImageUtil::To(*outImage_pointer, cIn);
    }
    // Do components (if indeed we have any)
    if (inComps_pointer!=0) {
      if (inImage_pointer==0) {
	this->unlock();
	os << LogIO::SEVERE << 
	  "No input image was given for the componentList to get the frequency from" 
	   << LogIO::EXCEPTION;
	return False;
      }
      Int ncomponents = inComps_pointer->nelements();
      for (Int icomp=0;icomp<ncomponents;++icomp) {
	SkyComponent component=outComps_pointer->component(icomp);
	if (operation=="apply") {
	  myPBp->applyPB(component, component, pointingCenter, 
			 qFreq, pa, squintType_p, False);
	} else {
	  myPBp->applyPB(component, component, pointingCenter, 
			 qFreq, pa, squintType_p, True);
	}
      }
    }
    if (myPBp) delete myPBp;
    if (inImage_pointer) delete inImage_pointer;
    if (outImage_pointer) delete outImage_pointer; 
    if (inComps_pointer) delete inComps_pointer; 
    if (outComps_pointer) delete outComps_pointer; 
    return True;
  } catch (AipsError x) {
    if (myPBp) delete myPBp;
    if (inImage_pointer) delete inImage_pointer;
    if (outImage_pointer) delete outImage_pointer; 
    if (inComps_pointer) delete inComps_pointer; 
    if (outComps_pointer) delete outComps_pointer; 
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  }
  return True;
}



Bool Imager::linearmosaic(const String& mosaic,
			  const String& fluxscale,
			  const String& sensitivity,
			  const Vector<String>& images,
			  const Vector<Int>& fieldids)

{
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "linearmosaic()", WHERE));
  if(mosaic=="") {
    os << LogIO::SEVERE << "Need name for mosaic image" << LogIO::POST;
    return False;
  }
  if(!Table::isWritable( mosaic )) {
    make( mosaic );
  }
  if (images.nelements() == 0) {
    os << LogIO::SEVERE << "Need names of images to mosaic" << LogIO::POST;
    return False;
  }
  if (images.nelements() != fieldids.nelements()) {
    os << LogIO::SEVERE << "number of fieldids doesn\'t match the" 
       << " number of images" << LogIO::POST;
    return False;
  }

  Double meminMB=Double(HostInfo::memoryTotal(true))/1024.0;
  PagedImage<Float> mosaicImage( mosaic );
  mosaicImage.set(0.0);
  TempImage<Float>  numerator( TiledShape(mosaicImage.shape(), mosaicImage.niceCursorShape()), mosaicImage.coordinates(), meminMB/2.0);
  numerator.set(0.0);
  TempImage<Float>  denominator( TiledShape(mosaicImage.shape(), mosaicImage.niceCursorShape()), mosaicImage.coordinates(), meminMB/2.0);
  numerator.set(0.0);
  IPosition iblcmos(mosaicImage.shape().nelements(),0);
  IPosition itrcmos(mosaicImage.shape());
  itrcmos=itrcmos-Int(1);
  LCBox lboxmos(iblcmos, itrcmos, mosaicImage.shape());
  ImageRegion imagregMos(WCBox(lboxmos, mosaicImage.coordinates()));
  ImageRegrid<Float> regridder;
  
  ROMSColumns msc(*ms_p);
  for (uInt i=0; i < images.nelements(); ++i) {
    if(!Table::isReadable(images(i))) {   
      os << LogIO::SEVERE << "Image " << images(i) << 
	" is not readable" << LogIO::POST;
      return False;
    }
   
    PagedImage<Float> smallImagedisk( images(i) );
    TempImage<Float> smallImage(smallImagedisk.shape(), smallImagedisk.coordinates(), meminMB/8.0);
    smallImage.copyData(smallImagedisk);
    CoordinateSystem fullimcoord=smallImage.coordinates();
     IPosition iblc(smallImage.shape().nelements(),0);
    IPosition itrc(smallImage.shape());
    itrc=itrc-Int(1);
      
    LCBox lbox(iblc, itrc, smallImage.shape());
    ImageRegion imagreg(WCBox(lbox, fullimcoord) );
    try{
      // accumulate the images
      SubImage<Float> subNum(numerator, imagreg, True);
     
      SubImage<Float> subDen(denominator, imagreg, True);




      TempImage<Float> fullImage(subNum.shape(), subNum.coordinates(), meminMB/8.0);
    
      os  << "Processing Image " << images(i)  << LogIO::POST;

      regridder.regrid( fullImage, Interpolate2D::LINEAR,
			IPosition(2,0,1), smallImage );

      TempImage<Float>  PB( subNum.shape(), subNum.coordinates(), meminMB/8.0);
      PB.set(1.0);

      MDirection pointingDirection = msc.field().phaseDirMeas( fieldids(i) );

      Quantity pa(0.0, "deg");
      pbguts ( PB, PB, pointingDirection, pa);

      fullImage.copyData( (LatticeExpr<Float>) (fullImage *  PB ) );
      subNum.copyData( (LatticeExpr<Float>) (subNum + fullImage) );
      subDen.copyData( (LatticeExpr<Float>) (subDen + (PB*PB)) );
      
    }
    catch(...){
	//most probably no overlap
	continue;
    }
  }
    
  LatticeExprNode LEN = max( denominator );
  Float dMax =  LEN.getFloat();


  if (scaleType_p == "SAULT") {

    // truncate denominator at ggSMin1
    denominator.copyData( (LatticeExpr<Float>) 
			  (iif(denominator < (dMax * constPB_p), dMax, 
			       denominator) ) );

    if (fluxscale != "") {
      clone( mosaic, fluxscale );
      
      PagedImage<Float> fluxscaleImage( fluxscale );
      fluxscaleImage.copyData( (LatticeExpr<Float>) 
			       (iif(denominator < (dMax*minPB_p), 0.0,
				    (dMax*minPB_p)/(denominator) )) );
      fluxscaleImage.copyData( (LatticeExpr<Float>) 
			       (iif(denominator > (dMax*constPB_p), 1.0,
				    (fluxscaleImage) )) );
      mosaicImage.copyData( (LatticeExpr<Float>)(iif(denominator > (dMax*minPB_p),
						   (numerator/denominator), 0)) );
    }
  } else {
    mosaicImage.copyData( (LatticeExpr<Float>)(iif(denominator > (dMax*minPB_p),
						   (numerator/denominator), 0)) );
    if (fluxscale != "") {
      clone(mosaic, fluxscale );
      PagedImage<Float> fluxscaleImage( fluxscale );
      fluxscaleImage.copyData( (LatticeExpr<Float>)( 1.0 ) );
    }
  }
  if (sensitivity != "") {
    clone(mosaic, sensitivity);
    PagedImage<Float> sensitivityImage( sensitivity );
    sensitivityImage.copyData( (LatticeExpr<Float>)( denominator/dMax ));
  }
  return True;
}



// Apply a primary beam or voltage pattern to an image
Bool Imager::pbguts(ImageInterface<Float>& inImage, 
		    ImageInterface<Float>& outImage,
		    const MDirection& pointingDirection,
		    const Quantity& pa)
{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "pbguts()", WHERE));
  
  try {
    if ( ! doVP_p ) {
      os << LogIO::SEVERE << 
	"Must invoke setvp() first in order to apply the primary beam" << LogIO::POST;
      return False;
    }
    String operation = "apply";  // could have as input in the future!

    // create the PBMath object, needed to apply PB 
    // to make high res Fourier weight image
    Quantity qFreq;
    PBMath * myPBp = 0;

    if (doDefaultVP_p) {
      // look up the telescope in ObsInfo
      ObsInfo oi = inImage.coordinates().obsInfo();
      String myTelescope = oi.telescope();
      if (myTelescope == "") {
	os << LogIO::SEVERE << "No telescope imbedded in image" << LogIO::POST;
	return False;
      }
      {
	Int spectralIndex=inImage.coordinates().findCoordinate(Coordinate::SPECTRAL);
	AlwaysAssert(spectralIndex>=0, AipsError);
	SpectralCoordinate
	  spectralCoord=inImage.coordinates().spectralCoordinate(spectralIndex);
	Vector<String> units(1); units = "Hz";
	spectralCoord.setWorldAxisUnits(units);	
	Vector<Double> spectralWorld(1);
	Vector<Double> spectralPixel(1);
	spectralPixel(0) = 0;
	spectralCoord.toWorld(spectralWorld, spectralPixel);  
	Double freq  = spectralWorld(0);
	qFreq = Quantity( freq, "Hz" );
      }
      String band;
      PBMath::CommonPB whichPB;
      String pbName;
      // get freq from coordinates
      PBMath::whichCommonPBtoUse (myTelescope, qFreq, band, whichPB, pbName);
      if (whichPB  == PBMath::UNKNOWN) {
	os << LogIO::SEVERE << "Unknown telescope for PB type: " << myTelescope << LogIO::POST;
	return False;
      }
      myPBp = new PBMath(whichPB);
    } else {
      // get the PB from the vpTable
      Table vpTable( vpTableStr_p );
      ROScalarColumn<TableRecord> recCol(vpTable, (String)"pbdescription");
      myPBp = new PBMath(recCol(0));
    }
    AlwaysAssert((myPBp != 0), AipsError);

    Vector<Int> whichStokes;
    CoordinateSystem cCoords;
    cCoords=StokesImageUtil::CStokesCoord(inImage.shape(),
					  inImage.coordinates(),
					  whichStokes,
					  polRep_p);
    TempImage<Complex> cIn(inImage.shape(),
			   cCoords);
    StokesImageUtil::From(cIn, inImage);
    if (operation=="apply") {
      myPBp->applyPB(cIn, cIn, pointingDirection, 
		     pa, squintType_p, False);
    } else {
      myPBp->applyPB(cIn, cIn, pointingDirection, 
		     pa, squintType_p, True);
    }
    StokesImageUtil::To(outImage, cIn);
    delete myPBp;    
    return True;
  } catch (AipsError x) {
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::POST;
    return False;
  } 
  
  return True;
}







// Weight the MeasurementSet
Bool Imager::weight(const String& type, const String& rmode,
                 const Quantity& noise, const Double robust,
                 const Quantity& fieldofview,
		    const Int npixels, const Bool multiField)
{
  if(!valid()) return False;
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "weight()"),logSink_p);
  
  this->lock();
  try {
    
    os << LogIO::NORMAL // Loglevel INFO
       << "Weighting MS: Imaging weights will be changed" << LogIO::POST;
    
    if (type=="natural") {
      os << LogIO::NORMAL // Loglevel INFO
         << "Natural weighting" << LogIO::POST;
      imwgt_p=VisImagingWeight("natural");
    }
    else if(type=="superuniform"){
      if(!assertDefinedImageParameters()) return False;
      Int actualNpix=npixels;
      if(actualNpix <=0)
	actualNpix=3;
      os << LogIO::NORMAL // Loglevel INFO
         << "SuperUniform weighting over a square cell spanning [" 
	 << -actualNpix 
	 << ", " << actualNpix << "] in the uv plane" << LogIO::POST;
      imwgt_p=VisImagingWeight(*rvi_p, rmode, noise, robust, nx_p, 
                               ny_p, mcellx_p, mcelly_p, actualNpix, 
                               actualNpix, multiField);
    }
    else if ((type=="robust")||(type=="uniform")||(type=="briggs")) {
      if(!assertDefinedImageParameters()) return False;
      Quantity actualFieldOfView(fieldofview);
      Int actualNPixels(npixels);
      String wtype;
      if(type=="briggs") {
        wtype = "Briggs";
      }
      else {
        wtype = "Uniform";
      }
      if(actualFieldOfView.get().getValue()==0.0&&actualNPixels==0) {
        actualNPixels=nx_p;
        actualFieldOfView=Quantity(actualNPixels*mcellx_p.get("rad").getValue(),
								   "rad");
        os << LogIO::NORMAL // Loglevel INFO
           << wtype
           << " weighting: sidelobes will be suppressed over full image"
           << LogIO::POST;
      }
      else if(actualFieldOfView.get().getValue()>0.0&&actualNPixels==0) {
        actualNPixels=nx_p;
        os << LogIO::NORMAL // Loglevel INFO
           << wtype
           << " weighting: sidelobes will be suppressed over specified field of view: "
           << actualFieldOfView.get("arcsec").getValue() << " arcsec" << LogIO::POST;
      }
      else if(actualFieldOfView.get().getValue()==0.0&&actualNPixels>0) {
        actualFieldOfView=Quantity(actualNPixels*mcellx_p.get("rad").getValue(),
								   "rad");
        os << LogIO::NORMAL // Loglevel INFO
           << wtype
           << " weighting: sidelobes will be suppressed over full image field of view: "
           << actualFieldOfView.get("arcsec").getValue() << " arcsec" << LogIO::POST;
      }
      else {
        os << LogIO::NORMAL // Loglevel INFO
           << wtype
           << " weighting: sidelobes will be suppressed over specified field of view: "
           << actualFieldOfView.get("arcsec").getValue() << " arcsec" << LogIO::POST;
      }
      os << LogIO::DEBUG1
         << "Weighting used " << actualNPixels << " uv pixels."
         << LogIO::POST;
      Quantity actualCellSize(actualFieldOfView.get("rad").getValue()/actualNPixels, "rad");

      imwgt_p=VisImagingWeight(*rvi_p, rmode, noise, robust, 
                               actualNPixels, actualNPixels, actualCellSize, 
                               actualCellSize, 0, 0, multiField);
      
    }
    else if (type=="radial") {
      os << "Radial weighting" << LogIO::POST;
      imwgt_p=VisImagingWeight("radial");
    }
    else {
      this->unlock();
      os << LogIO::SEVERE << "Unknown weighting " << type
         << LogIO::EXCEPTION;    
      return False;
    }
    
      rvi_p->useImagingWeight(imwgt_p);
    
    // Beam is no longer valid
    beamValid_p=False;
    destroySkyEquation();
    this->writeHistory(os);
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::EXCEPTION;
    return False;
  } 
  
  return True;
}




// Filter the MeasurementSet
Bool Imager::filter(const String& type, const Quantity& bmaj,
		 const Quantity& bmin, const Quantity& bpa)
{
  if(!valid()) return False;
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "filter()"),logSink_p);
  
  this->lock();
  try {
      
    os << LogIO::NORMAL // Loglevel INFO
       << "Imaging weights will be tapered" << LogIO::POST;
    imwgt_p.setFilter(type, bmaj, bmin, bpa);
    rvi_p->useImagingWeight(imwgt_p);
      
    // Beam is no longer valid
    beamValid_p=False;
    destroySkyEquation();
    this->writeHistory(os);
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    throw(x);
    return False;
  } 
  
  return True;
}


// Implement a uv range
Bool Imager::uvrange(const Double& uvmin, const Double& uvmax)
{
  if(!valid()) return False;
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "uvrange()"),logSink_p);
  
  try {
    os << LogIO::NORMAL // Loglevel INFO
       << "Selecting data according to  uvrange: setdata will reset this selection"
       << LogIO::POST;

    Double auvmin(uvmin);
    Double auvmax(uvmax);

    if(auvmax<=0.0) auvmax=1e10;
    if(auvmax>auvmin&&(auvmin>=0.0)) {
      os << LogIO::NORMAL // Loglevel INFO
         << "Allowed uv range: " << auvmin << " to " << auvmax
	 << " wavelengths" << LogIO::POST;
    }
    else {
      os << LogIO::SEVERE << "Invalid uvmin and uvmax: "
	 << auvmin << ", " << auvmax
	 << LogIO::EXCEPTION;
      return False;
    }
    Vector<Double> freq;
    ostringstream strUVmax, strUVmin, ostrInvLambda;

    this->lock();
      
    if(mssel_p.null()){ os << "Please setdata first before using uvrange " << LogIO::POST; return False; }


     // use the average wavelength for the selected windows to convert
     // uv-distance from lambda to meters
     ostringstream spwsel;
     spwsel << "select from $1 where ROWID() IN [";
     for(uInt i=0; i < dataspectralwindowids_p.nelements(); ++i) {
	 if (i > 0) spwsel << ", ";
	 spwsel << dataspectralwindowids_p(i);
     }
     spwsel << "]";

     MSSpectralWindow msspw(tableCommand(spwsel.str(), 
					 mssel_p->spectralWindow()));
     ROMSSpWindowColumns spwc(msspw);

     // This averaging scheme will work even if the spectral windows are
     // of different sizes.  Note, however, that using an average wavelength
     // may not be a good choice when the total range in frequency is 
     // large (e.g. mfs across double sidebands).
     uInt nrows = msspw.nrow();
     Double ftot = 0.0;
     Int nchan = 0;
     for(uInt i=0; i < nrows; ++i) {
	 nchan += (spwc.numChan())(i);
	 ftot += sum((spwc.chanFreq())(i));
     }
     Double invLambda=ftot/(nchan*C::c);

     // This is message may not be helpful as mfs is set with setimage()
     // which may sometimes get called after uvrange()
     if (nrows > 1 && imageMode_p=="MFS") {
 	 os << LogIO::WARN 
 	    << "When using mfs over a broad range of frequencies, It is more "
 	    << "accurate to " << endl 
 	    << "constrain uv-ranges using setdata(); try: " << endl 
 	    << "  msselect='(SQUARE(UVW[1]) + SQUARE(UVW[2])) > uvmin && "
 	    << "(SQUARE(UVW[1]) + SQUARE(UVW[2])) < uvmax'" << endl
 	    << "where [uvmin, uvmax] is the range given in meters." 
 	    << LogIO::POST;
     }

     invLambda=invLambda*invLambda;
     auvmax=auvmax*auvmax;
     auvmin=auvmin*auvmin;
     strUVmax << auvmax; 
     strUVmin << auvmin;
     ostrInvLambda << invLambda; 
     String strInvLambda=ostrInvLambda;
     MeasurementSet* mssel_p2;

     // Apply the TAQL selection string, to remake the selected MS
     String parseString="select from $1 where (SQUARE(UVW[1]) + SQUARE(UVW[2]))*" + strInvLambda + " > " + strUVmin + " &&  (SQUARE(UVW[1]) + SQUARE(UVW[2]))*" + strInvLambda + " < " + strUVmax ;

     mssel_p2=new MeasurementSet(tableCommand(parseString,*mssel_p));
     AlwaysAssert(mssel_p2, AipsError);
     // Rename the selected MS as */SELECTED_UVRANGE
     //mssel_p2->rename(msname_p+"/SELECTED_UVRANGE", Table::Scratch);
      
     if (mssel_p2->nrow()==0) {
	 os << LogIO::WARN
	    << "Selection string results in empty MS: "
	    << "reverting to sorted MeasurementSet"
	    << LogIO::POST;
	 delete mssel_p2;
     } else {
       if (!mssel_p.null()) {
	     os << LogIO::NORMAL // Loglevel INFO
                << "By UVRANGE selection previously selected number of rows "
                << mssel_p->nrow() << "  are now reduced to "
                << mssel_p2->nrow() << LogIO::POST; 
	     //delete mssel_p; 
	     mssel_p=mssel_p2;
	     mssel_p->flush();
	 }
     }
      
     
     this->makeVisSet(*mssel_p);
     AlwaysAssert(rvi_p, AipsError);

     // NOW WE HAVE TO REDO THE VELOCITY INFO FOR visiter AS IN SETDATA

     this->selectDataChannel(dataspectralwindowids_p, dataMode_p,
                             dataNchan_p, dataStart_p, dataStep_p,
                             mDataStart_p, mDataStep_p);

     this->writeHistory(os);
     this->unlock();
     
     // Beam is no longer valid
     beamValid_p=False;
     return True;    
  } catch (AipsError x) {
    this->unlock();
    throw(x);
    return False;
  } 
  return True;
}

// Find the sensitivity
Bool Imager::sensitivity(Quantity& pointsourcesens, Double& relativesens,
		      Double& sumwt)
{
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "sensitivity()", WHERE));
  
  try {
    
    os << LogIO::NORMAL // Loglevel INFO
       << "Calculating sensitivity from imaging weights and from SIGMA column"
       << LogIO::POST;
    os << LogIO::NORMAL // Loglevel INFO
       << "(assuming that SIGMA column is correct, otherwise scale appropriately)"
       << LogIO::POST;
    
    this->lock();
    VisSetUtil::Sensitivity(*rvi_p, pointsourcesens, relativesens, sumwt);
    os << LogIO::NORMAL << "RMS Point source sensitivity  : " // Loglevel INFO
       << pointsourcesens.get("Jy").getValue() << " Jy/beam"
       << LogIO::POST;
    os << LogIO::NORMAL // Loglevel INFO
       << "Relative to natural weighting : " << relativesens << LogIO::POST;
    os << LogIO::NORMAL // Loglevel INFO
       << "Sum of weights                : " << sumwt << LogIO::POST;
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    throw(x);
    return False;
  } 
  return True;
}

// Calculate various sorts of image. Only one image
// can be calculated at a time. The complex Image make
// be retained if a name is given. This does not use
// the SkyEquation.
Bool Imager::makeimage(const String& type, const String& image,
                       const String& compleximage, const Bool verbose)
{
#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::makeimage",23);
#endif
  
  if(!valid()) 
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::makeimage",22);
#endif

      return False;
    }
  LogIO os(LogOrigin("imager", "makeimage()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters())
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::makeimage",22);
#endif

	return False;
      }
    
    os << LogIO::NORMAL // Loglevel INFO
       << "Calculating image (without full skyequation)" << LogIO::POST;
    
    FTMachine::Type seType(FTMachine::OBSERVED);
    Bool doSD(False);

    if(type=="observed") {
      seType=FTMachine::OBSERVED;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making dirty image from " << type << " data "
	 << LogIO::POST;
    }
    else if (type=="model") {
      if(rvi_p->msColumns().modelData().isNull())
	os << LogIO::SEVERE
           << "Cannot make model image without scratch model-data column "
	   << LogIO::EXCEPTION;
      seType=FTMachine::MODEL;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making dirty image from " << type << " data "
	 << LogIO::POST;
    }
    else if (type=="corrected") {
      seType=FTMachine::CORRECTED;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making dirty image from " << type << " data "
	 << LogIO::POST;
    }
    else if (type=="psf") {
      seType=FTMachine::PSF;
      if(rvi_p->msColumns().modelData().isNull())
	os << "Cannot make psf image without scratch model-data column for now"
	   << LogIO::EXCEPTION;
      os << "Making point spread function "
	 << LogIO::POST;
    }
    else if (type=="residual") {
      if(rvi_p->msColumns().modelData().isNull())
	os << LogIO::SEVERE
           << "Cannot make residual image without scratch model-data column "
	   << LogIO::EXCEPTION;
      seType=FTMachine::RESIDUAL;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making dirty image from " << type << " data "
	 << LogIO::POST;
    }
    else if (type=="singledish-observed") {
      doSD = True;
      seType=FTMachine::OBSERVED;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making single dish image from observed data" << LogIO::POST;
    }
    else if (type=="singledish") {
      doSD = True;
      seType=FTMachine::CORRECTED;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making single dish image from corrected data" << LogIO::POST;
    }
    else if (type=="coverage") {
      doSD = True;
      seType=FTMachine::COVERAGE;
      os << LogIO::NORMAL // Loglevel PROGRESS
         << "Making single dish coverage function "
	 << LogIO::POST;
    }
    else if (type=="holography") {
      doSD = True;
      seType=FTMachine::CORRECTED;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making complex holographic image from corrected data "
	 << LogIO::POST;
    }
    else if (type=="holography-observed") {
      doSD = True;
      seType=FTMachine::OBSERVED;
      os << LogIO::NORMAL // Loglevel INFO
         << "Making complex holographic image from observed data "
	 << LogIO::POST;
    }
    else if (type=="pb"){
      if ( ! doVP_p ) {
        if( ftmachine_p == "pbwproject" ){
	   os << LogIO::WARN << "Using pb from ft-machines" << LogIO::POST;
	}
	else{
	  this->unlock();
	  os << LogIO::SEVERE << 
	    "Must invoke setvp() first in order to make its image" 
	     << LogIO::EXCEPTION;
	  return False;
	}
      }
      CoordinateSystem coordsys;
      imagecoordinates(coordsys, verbose);
      if (doDefaultVP_p) {
	if(telescope_p!=""){
	  ObsInfo myobsinfo=this->latestObsInfo();
	  myobsinfo.setTelescope(telescope_p);
	  coordsys.setObsInfo(myobsinfo);
	  
	}
	else{
	  telescope_p=coordsys.obsInfo().telescope();
	}
	this->unlock();
	ROMSAntennaColumns ac(ms_p->antenna());
	Double dishDiam=ac.dishDiameter()(0);
        return this->makePBImage(coordsys, telescope_p, image, False, dishDiam);
      }
      else{
	Table vpTable(vpTableStr_p);
	this->unlock();
        return this->makePBImage(coordsys, vpTable, image);	
      }

    }
    else {
      this->unlock();
      os << LogIO::SEVERE << "Unknown image type " << type << LogIO::EXCEPTION;

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::makeimage",22);
#endif  

      return False;
    }

    if(doSD && (ftmachine_p == "ft")){
      os << LogIO::SEVERE
         << "To make single dish images, ftmachine in setoptions must be set to either sd or both"
	 << LogIO::EXCEPTION;
    }
    
    // Now make the images. If we didn't specify the names then
    // delete on exit.
    String imageName(image);
    if(image=="") {
      imageName=Imager::imageName()+".image";
    }
    os << LogIO::NORMAL << "Image is : " << imageName << LogIO::POST; // Loglevel INFO
    Bool keepImage=(image!="");
    Bool keepComplexImage=(compleximage!="")||(type=="holography")||(type=="holography-observed");
    String cImageName(compleximage);

    if(compleximage=="") {
      cImageName=imageName+".compleximage";
    }

    if(keepComplexImage) {
      os << "Retaining complex image: " << compleximage << LogIO::POST;
    }

    CoordinateSystem imagecoords;
    if(!imagecoordinates(imagecoords, false))
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::makeimage",22);
#endif  

	return False;
      }
    make(imageName);
    PagedImage<Float> imageImage(imageName);
    imageImage.set(0.0);
    imageImage.table().markForDelete();
    
    // Now set up the tile size, here we guess only
    IPosition cimageShape(imageshape());
    Int tilex=32;
    if(imageTileVol_p >0){
      tilex=static_cast<Int>(ceil(sqrt(imageTileVol_p/min(4,
                                                          cimageShape(2))/min(32,
                                                                              cimageShape(3)))));
      if(tilex >0){
	if(tilex > min(Int(cimageShape(0)), Int(cimageShape(1))))
	  tilex=min(Int(cimageShape(0)), Int(cimageShape(1)));
	else
	  tilex=cimageShape(0)/Int(cimageShape(0)/tilex);
      }
      //Not too small in x-y tile
      if(tilex < 10)
	tilex=10;
   
    }
    IPosition tileShape(4, min(tilex, cimageShape(0)), min(tilex, cimageShape(1)),
			min(4, cimageShape(2)), min(32, cimageShape(3)));
    CoordinateSystem cimagecoords;
    if(!imagecoordinates(cimagecoords, false))
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::makeimage",22);
#endif

	return False;
      }
    PagedImage<Complex> cImageImage(TiledShape(cimageShape, tileShape),
				    cimagecoords,
				    cImageName);
    cImageImage.set(Complex(0.0));
    cImageImage.setMaximumCacheSize(cache_p/2);
    cImageImage.table().markForDelete();
    //
    // Add the distance to the object: this is not nice. We should define the
    // coordinates properly.
    //
    Record info(imageImage.miscInfo());
    info.define("distance", distance_p.get("m").getValue());
    cImageImage.setMiscInfo(info);

    
    String ftmachine(ftmachine_p);
    if (!ft_p)
      createFTMachine();
    
    // Now make the required image
    Matrix<Float> weight;
    ft_p->makeImage(seType, *rvi_p, cImageImage, weight);
    StokesImageUtil::To(imageImage, cImageImage);
    imageImage.setUnits(Unit("Jy/beam"));
    cImageImage.setUnits(Unit("Jy/beam"));
    
    if(keepImage) {
      imageImage.table().unmarkForDelete();
    }
    if(keepComplexImage) {
      cImageImage.table().unmarkForDelete();
    }
    this->unlock();

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::makeimage",22);
#endif

    return True;
  } catch (AipsError x) {
    this->unlock();
   
#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::makeimage",22);
#endif
    throw(x);
    return False;
  }
  catch(...){
    //Unknown exception...
    throw(AipsError("Unknown exception caught ...imager/casa may need to be exited"));
  }
  this->unlock();

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::makeimage",22);
#endif

  return True;
}  

// Restore: at least one model must be supplied
Bool Imager::restore(const Vector<String>& model,
		     const String& complist,
		     const Vector<String>& image,
		     const Vector<String>& residual)
{
  
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "restore()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) return False;
    
    if(image.nelements()>model.nelements()) {
      this->unlock();
      os << LogIO::SEVERE << "Cannot specify more output images than models"
	 << LogIO::EXCEPTION;
      return False;
    }
    else {
      os << LogIO::NORMAL // Loglevel PROGRESS
         << "Restoring " << model.nelements() << " models" << LogIO::POST;
    }

    ///if the skymodel is already set...no need to get rid of the psf and ftmachine state
    //as long as the images match
    if(!redoSkyModel_p){
      Bool coordMatch=True; 
      for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
	CoordinateSystem cs=(sm_p->image(thismodel)).coordinates();
	coordMatch= coordMatch || checkCoord(cs, model(thismodel));
      }
      if(!coordMatch)
	destroySkyEquation();
    }
    
    if(redoSkyModel_p){
      Vector<String> imageNames(image);
      if(image.nelements()<model.nelements()) {
	imageNames.resize(model.nelements());
	for(Int i=Int(image.nelements());i<Int(model.nelements()); ++i) {
	  imageNames(i)="";
	}
      }
      
      for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
	if(imageNames(thismodel)=="") {
	  imageNames(thismodel)=model(thismodel)+".restored";
	}
	removeTable(imageNames(thismodel));
	if(imageNames(thismodel)=="") {
	  this->unlock();
	  os << LogIO::SEVERE << "Illegal name for output image "
	     << imageNames(thismodel) << LogIO::EXCEPTION;
	  return False;
	}
	if(!clone(model(thismodel), imageNames(thismodel))) return False;
      }
      
      Vector<String> residualNames(residual);
      if(residual.nelements()<model.nelements()) {
	residualNames.resize(model.nelements());
	for(Int i=Int(residual.nelements());i<Int(model.nelements());++i) {
	  residualNames(i)="";
	}
      }

      for (Int thismodel=0;thismodel<Int(model.nelements()); ++thismodel) {
	if(residualNames(thismodel)=="")
	  residualNames(thismodel)=model(thismodel)+".residual";
	removeTable(residualNames(thismodel));
	if(residualNames(thismodel)=="") {
	  this->unlock();
	  os << LogIO::SEVERE << "Illegal name for output residual "
	     << residualNames(thismodel) << LogIO::EXCEPTION;
	  return False;
	}
	if(!clone(model(thismodel), residualNames(thismodel))) return False;
      }
    
      if(beamValid_p) {
	os << LogIO::NORMAL << "Using previous beam fit" << LogIO::POST; // Loglevel INFO
      }
      else {
	os << LogIO::NORMAL // Loglevel INFO
           << "Calculating PSF using current parameters" << LogIO::POST;
	String psf;
	psf=imageNames(0)+".psf";
	if(!clone(imageNames(0), psf)) return False;
	Imager::makeimage("psf", psf);
	fitpsf(psf, bmaj_p, bmin_p, bpa_p);
	beamValid_p=True;
      }
    
      //      if (!se_p)
      if(!createSkyEquation(model, complist)) return False;
      
      addResidualsToSkyEquation(residualNames);
    }
    sm_p->solveResiduals(*se_p);
    
    restoreImages(image);
    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    throw(x);
    return False;
  } 
  this->unlock();
  return True;
}

// Residual
Bool Imager::residual(const Vector<String>& model,
		      const String& complist,
		      const Vector<String>& image)
{
  
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "residual()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) return False;
    os << LogIO::NORMAL // Loglevel INFO
       << "Calculating residual image using full sky equation" << LogIO::POST;
    Vector<String> theModels=model;

    Bool deleteModel=False;

    if(model.nelements()==1 && model[0]=="" && complist != "" 
       && image.nelements()==1){

      //      A component list with no model passed...
      theModels.resize(1);
      theModels[0]="Imager_Scratch_model";
      make(theModels[0]);
      deleteModel=True;
    }

    if(image.nelements()>theModels.nelements()) {
      this->unlock();
      os << LogIO::SEVERE << "Cannot specify more output images than models"
	 << LogIO::EXCEPTION;
      return False;
    }
    else {
      os << LogIO::NORMAL << "Finding residuals for " << theModels.nelements() // Loglevel INFO
	 << " models" << LogIO::POST;
    }
    
    Vector<String> imageNames(image);
    if(image.nelements()<theModels.nelements()) {
      imageNames.resize(model.nelements());
      for(Int i=Int(image.nelements());i<Int(theModels.nelements());++i) {
	imageNames(i)="";
      }
    }

    for (Int thismodel=0;thismodel<Int(theModels.nelements()); ++thismodel) {
      if(imageNames(thismodel)=="")
	imageNames(thismodel)=model(thismodel)+".residual";
      removeTable(imageNames(thismodel));
      if(imageNames(thismodel)=="") {
	this->unlock();
	os << LogIO::SEVERE << "Illegal name for output image "
	   << imageNames(thismodel) << LogIO::EXCEPTION;
	return False;
      }
      if(!clone(theModels(thismodel), imageNames(thismodel))) return False;
    }
    destroySkyEquation();
    if(!createSkyEquation(theModels, complist)) return False;
    
    addResidualsToSkyEquation(imageNames);
    
    sm_p->solveResiduals(*se_p);
    destroySkyEquation();
    if(deleteModel) 
      removeTable(theModels[0]);
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    throw(x);
    return False;
  } 
  this->unlock();
  return True;
}

// Residual
Bool Imager::approximatepsf(const String& psf)
{
  
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "approximatepsfs()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) return False;
    os << LogIO::NORMAL // Loglevel INFO
       << "Calculating approximate PSFs using full sky equation" << LogIO::POST;
    
 
    if(psf==""){
      this->unlock();
      os << LogIO::SEVERE << "Illegal name for output psf "
	 << psf << LogIO::EXCEPTION;
      return False;
    } 
    removeTable(psf);
    make(psf);
    
    Vector<String>onepsf(1,psf);
    // Previous SkyEquation if they exist is not useful
    destroySkyEquation();
    //    if (!se_p)
    // As we are not going to make any use of a useful model and to economize 
    // temporary image...using the psf itself as model...
    // need to change this if you donot destroy the skyequation after you're done.
    if(!createSkyEquation(onepsf)) return False;
    
    sm_p->makeApproxPSFs(*se_p);
    

    PagedImage<Float> elpsf(psf);
    elpsf.copyData(sm_p->PSF(0));
    Quantity mbmaj, mbmin, mbpa;
    StokesImageUtil::FitGaussianPSF(elpsf, mbmaj, mbmin, mbpa);
    LatticeExprNode sumPSF = sum(elpsf);
    Float volume=sumPSF.getFloat();
    os << LogIO::NORMAL << "Approximate PSF  "  << ": size " // Loglevel INFO
       << mbmaj.get("arcsec").getValue() << " by "
       << mbmin.get("arcsec").getValue() << " (arcsec) at pa " 
       << mbpa.get("deg").getValue() << " (deg)" << endl
       << "and volume = " << volume << " pixels " << LogIO::POST;
    
    
    destroySkyEquation();
    if(ft_p)
      delete ft_p;
    ft_p=0;

    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    throw(x);
    return False;
  } 
  this->unlock();
  return True;
}

Bool Imager::smooth(const Vector<String>& model, 
		    const Vector<String>& image, Bool usefit, 
		    Quantity& mbmaj, Quantity& mbmin, Quantity& mbpa,
		    Bool normalizeVolume)
{
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "smooth()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) return False;
    
    os << LogIO::NORMAL << "Smoothing image" << LogIO::POST; // Loglevel PROGRESS
    
    if(model.nelements()>0) {
      for ( uInt thismodel=0;thismodel<model.nelements(); ++thismodel) {
	if(model(thismodel)=="") {
	  this->unlock();
	  os << LogIO::SEVERE << "Need a name for model " << thismodel << LogIO::POST;
	  return False;
	}
      }
    }
    
    if(image.nelements()>model.nelements()) {
      this->unlock();
      os << LogIO::SEVERE << "Cannot specify more output images than models" << LogIO::POST;
      return False;
    }
    
    if(usefit) {
      if(beamValid_p) {
	os << LogIO::NORMAL << "Using previous beam" << LogIO::POST; // Loglevel INFO
	mbmaj=bmaj_p;
	mbmin=bmin_p;
	mbpa=bpa_p;
      }
      else {
	os << LogIO::NORMAL // Loglevel INFO
           << "Calculating PSF using current parameters" << LogIO::POST;
	String psf;
	psf=model(0)+".psf";
	if(!clone(model(0), psf)) return False;
	Imager::makeimage("psf", psf);
	fitpsf(psf, mbmaj, mbmin, mbpa);
	bmaj_p=mbmaj;
	bmin_p=mbmin;
	bpa_p=mbpa;
	beamValid_p=True;
      }
    }
    
    // Smooth all the images
    Vector<String> imageNames(image);
    for (Int thismodel=0;thismodel<Int(image.nelements()); ++thismodel) {
      if(imageNames(thismodel)=="") {
        imageNames(thismodel)=model(thismodel)+".smoothed";
      }
      PagedImage<Float> modelImage(model(thismodel));
      PagedImage<Float> imageImage(TiledShape(modelImage.shape(), 
					      modelImage.niceCursorShape()),
				   modelImage.coordinates(),
				   imageNames(thismodel));
      imageImage.table().markForDelete();
      imageImage.copyData(modelImage);
      StokesImageUtil::Convolve(imageImage, mbmaj, mbmin, mbpa,
				normalizeVolume);
      
      ImageInfo ii = imageImage.imageInfo();
      ii.setRestoringBeam(mbmaj, mbmin, mbpa); 
      imageImage.setImageInfo(ii);
      imageImage.setUnits(Unit("Jy/beam"));
      imageImage.table().unmarkForDelete();
    }
    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  this->unlock();
  return True;
}

// Clean algorithm
Bool Imager::clean(const String& algorithm,
		   const Int niter, 
		   const Float gain,
		   const Quantity& threshold, 
		   const Bool displayProgress, 
		   const Vector<String>& model, const Vector<Bool>& fixed,
		   const String& complist,
		   const Vector<String>& mask,
		   const Vector<String>& image,
		   const Vector<String>& residual,
		   const Vector<String>& psfnames,
                   const Bool firstrun)
{
#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::clean",22);
#endif
  Bool converged=True; 

 
  if(!valid())
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::clean",21);
#endif

      return False;
    }
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "clean()"),logSink_p);
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) 
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::clean",21);
#endif

	return False;
      }
    
    Int nmodels=model.nelements();
    os << LogIO::DEBUG1
       << "Found " << nmodels << " specified model images" << LogIO::POST;
    
    if(model.nelements()>0) {
      for (uInt thismodel=0;thismodel<model.nelements(); ++thismodel) {
	if(model(thismodel)=="") {
	  this->unlock();
	  os << LogIO::SEVERE << "Need a name for model "
	     << thismodel << LogIO::POST;

#ifdef PABLO_IO
	  traceEvent(1,"Exiting Imager::clean",21);
#endif

	  return False;
	}
      }
    }
    
    Vector<String> modelNames=model;
    // Make first image with the required shape and coordinates only if
    // it doesn't exist yet. Otherwise we'll throw an exception later
    if(modelNames(0)=="") modelNames(0)=imageName()+".clean";
    if(!Table::isWritable(modelNames(0))) {
      make(modelNames(0));
    }
    else{
      Bool coordMatch=False;
      CoordinateSystem coordsys;
      imagecoordinates(coordsys, firstrun);
      for (uInt modelNum=0; modelNum < modelNames.nelements(); ++modelNum){
	if(Table::isWritable(modelNames(modelNum))){
	  coordMatch= coordMatch || 
	    (this->checkCoord(coordsys, modelNames(modelNum)));
				     
	}
	  
      } 
      if(!coordMatch){
	os << LogIO::WARN << "The model(s) image exists on disk " 
	   << LogIO::POST;
	os << LogIO::WARN 
	   << "The coordinates or shape were found not to match the one "
	   << "defined by setimage " 
	   << LogIO::POST;

	os << LogIO::WARN 
	   << "Cleaning process is going to ignore setimage parameters and "
	   << "continue cleaning from from model on disk " 
	   << LogIO::POST;
      }
    }
    Vector<String> maskNames(nmodels);
    if(Int(mask.nelements())==nmodels) {
      maskNames=mask;
    }
    else {
      maskNames="";
    }

    if(sm_p){
      if( sm_p->getAlgorithm() != "clean") destroySkyEquation();
      if(images_p.nelements() != uInt(nmodels)){
	destroySkyEquation();
      }
      else{
	for (Int k=0; k < nmodels ; ++k){
	  if(!(images_p[k]->name().contains(modelNames[k]))) destroySkyEquation();
	}
      }
    }

    // Always fill in the residual images
    Vector<String> residualNames(nmodels);
    if(redoSkyModel_p){
 
      if(Int(residual.nelements())==nmodels) {
	residualNames=residual;
      }
      else {
	residualNames="";
      }
      for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
	if(residualNames(thismodel)=="") {
	  residualNames(thismodel)=modelNames(thismodel)+".residual";
	}
	removeTable(residualNames(thismodel));
	if(!clone(model(thismodel), residualNames(thismodel)))
	  {
	    
#ifdef PABLO_IO
	    traceEvent(1,"Exiting Imager::clean",21);
#endif

	    return False;
	  }
      }
    }
    
    // Make an ImageSkyModel with the specified polarization representation
    // (i.e. circular or linear)

    if( redoSkyModel_p || !sm_p){
      if(sm_p) delete sm_p;
      if(algorithm.substr(0,5)=="clark") {
	// Support serial and parallel specializations
	setClarkCleanImageSkyModel();
	if(algorithm.contains("stokes"))
          sm_p->setJointStokesClean(False);
	os << LogIO::NORMAL // Loglevel INFO.        Stating the algo is more for
           << "Using Clark clean" << LogIO::POST; // the logfile than the window.
      }
      else if (algorithm=="hogbom") {
	sm_p = new HogbomCleanImageSkyModel();
	os << LogIO::NORMAL // Loglevel INFO.         Stating the algo is more for
           << "Using Hogbom clean" << LogIO::POST; // the logfile than the window.
      }
      else if (algorithm=="wfhogbom") {
	setWFCleanImageSkyModel();
	sm_p->setSubAlgorithm("hogbom");
	doMultiFields_p = True;
	doMultiFields_p = False;
	os << LogIO::NORMAL // Loglevel INFO
           << "Using wide-field algorithm with Hogbom clean" << LogIO::POST;
      }
      else if (algorithm=="multiscale") {
	if (!scaleInfoValid_p) {
	  this->unlock();
	  os << LogIO::SEVERE << "Scales not yet set" << LogIO::POST;
	  return False;
	}
	if (scaleMethod_p=="uservector") {	
	  sm_p = new MSCleanImageSkyModel(userScaleSizes_p, stoplargenegatives_p, 
					    stoppointmode_p, smallScaleBias_p);
	} else {
	  sm_p = new MSCleanImageSkyModel(nscales_p, stoplargenegatives_p, 
					    stoppointmode_p, smallScaleBias_p);
	}
	if(ftmachine_p=="mosaic" ||ftmachine_p=="wproject" )
	  sm_p->setSubAlgorithm("full");
	os << LogIO::NORMAL // Loglevel INFO.             Stating the algo is more for
           << "Using multiscale clean" << LogIO::POST; // the logfile than the window.
      }
      else if (algorithm.substr(0,7)=="mfclark" || algorithm=="mf") {
	sm_p = new MFCleanImageSkyModel();
	sm_p->setSubAlgorithm("clark");
	if(algorithm.contains("stokes"))
          sm_p->setJointStokesClean(False);

	doMultiFields_p = True;
	os << LogIO::NORMAL << "Using multifield Clark clean" << LogIO::POST; // Loglevel INFO
      }
      else if (algorithm=="csclean" || algorithm=="cs") {
	sm_p = new CSCleanImageSkyModel();
	doMultiFields_p = True;
	os << LogIO::NORMAL << "Using Cotton-Schwab Clean" << LogIO::POST; // Loglevel INFO
      }
      else if (algorithm=="csfast" || algorithm=="csf") {
	sm_p = new CSCleanImageSkyModel();
	sm_p->setSubAlgorithm("fast");
	doMultiFields_p = True;
	os << LogIO::NORMAL // Loglevel INFO
           << "Using Cotton-Schwab Clean (optimized)" << LogIO::POST;
      }
      else if (algorithm=="mfhogbom") {
	sm_p = new MFCleanImageSkyModel();
	sm_p->setSubAlgorithm("hogbom");
	doMultiFields_p = True;
	os << LogIO::NORMAL << "Using multifield Hogbom clean" << LogIO::POST; // Loglevel INFO
      }
      else if (algorithm=="mfmultiscale") {
	if (!scaleInfoValid_p) {
	  this->unlock();
	  os << LogIO::SEVERE << "Scales not yet set" << LogIO::POST;
	  return False;
	}
	if (scaleMethod_p=="uservector") {
	  sm_p = new MFMSCleanImageSkyModel(userScaleSizes_p, 
					    stoplargenegatives_p, 
					    stoppointmode_p,
                                            smallScaleBias_p);
	} else {
	  sm_p = new MFMSCleanImageSkyModel(nscales_p, 
					    stoplargenegatives_p, 
					    stoppointmode_p,
                                            smallScaleBias_p);
	}
	//	if(ftmachine_p=="mosaic"|| ftmachine_p=="wproject")
	// For some reason  this does not seem to work without full
	sm_p->setSubAlgorithm("full");

	doMultiFields_p = True;
	os << LogIO::NORMAL << "Using multifield multi-scale clean"  // Loglevel INFO
	   << LogIO::POST;
      } 
      else if (algorithm=="wfclark" || algorithm=="wf") {
	// Support serial and parallel specializations
	setWFCleanImageSkyModel();
	sm_p->setSubAlgorithm("clark");
	doMultiFields_p = False;
	os << LogIO::NORMAL // Loglevel INFO
           << "Using wide-field algorithm with Clark clean" << LogIO::POST;
      }
      else if (algorithm=="wfhogbom") {
	// Support serial and parallel specializations
	setWFCleanImageSkyModel();
	sm_p->setSubAlgorithm("hogbom");
	doMultiFields_p = False;
	os << LogIO::NORMAL // Loglevel INFO
           << "Using wide-field algorithm with Hogbom clean" << LogIO::POST;
      }
      else if (algorithm=="msmfs") {
	doMultiFields_p = False;
	doWideBand_p = True;
	if ( ftmachine_p != "ft" && ftmachine_p != "wproject") {
	  os << LogIO::SEVERE
             << "Multi-scale Multi-frequency Clean currently works only with the default ftmachine and wproject"
             << LogIO::POST;
	  return False;
	}
	if (!scaleInfoValid_p) {
          this->unlock();
          os << LogIO::WARN << "Scales not yet set, using power law" << LogIO::POST;
          sm_p = new WBCleanImageSkyModel(ntaylor_p, 1 ,reffreq_p);
	}
	if (scaleMethod_p=="uservector") {	
          sm_p = new WBCleanImageSkyModel(ntaylor_p,userScaleSizes_p,reffreq_p);
	} else {
          sm_p = new WBCleanImageSkyModel(ntaylor_p,nscales_p,reffreq_p);
	}
	os << LogIO::NORMAL // Loglevel INFO
           << "Using multi frequency synthesis algorithm" << LogIO::POST;
	((WBCleanImageSkyModel*)sm_p)->imageNames = Vector<String>(image);
	/* Check masks. Should be only one per field. Duplicate the name ntaylor_p times 
	   Note : To store taylor-coefficients, msmfs uses the same data structure as for
	          multi-field imaging. In the case of multifield and msmfs, the list of 
		  images is nested and follows a field-major ordering.
		  All taylor-coeffs for a single field should have the same mask (for now).
	   For now, since only single-field is allowed for msmfs, we have the following.*/
        if(Int(mask.nelements()) != nmodels && Int(mask.nelements())>0) 
	{
            for(Int tay=0;tay<nmodels;tay++)
		   maskNames[tay] = mask[0];
	}
      }
      else {
	this->unlock();
	os << LogIO::SEVERE << "Unknown algorithm: " << algorithm 
	   << LogIO::POST;

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::clean",21);
#endif
	
	return False;
      }
    
      AlwaysAssert(sm_p, AipsError);
      sm_p->setAlgorithm("clean");

      //    if (!se_p)
      if(!createSkyEquation(modelNames, fixed, maskNames, complist)) 
        {
	
#ifdef PABLO_IO
          traceEvent(1,"Exiting Imager::clean",21);
#endif
	
          return False;
        }
      os << LogIO::NORMAL3 << "Created Sky Equation" << LogIO::POST;

      addResidualsToSkyEquation(residualNames);
    }
    else{
      //adding or modifying mask associated with skyModel
      addMasksToSkyEquation(maskNames,fixed);
    }

    // The old plot that showed how much flux was being incorporated in each
    // scale.   No longer available, slated for removal.
    // if (displayProgress) {
    //   sm_p->setDisplayProgress(True);
    //   sm_p->setPGPlotter( getPGPlotter() );
    // }

    sm_p->setGain(gain);
    sm_p->setNumberIterations(niter);
    sm_p->setThreshold(threshold.get("Jy").getValue());
    sm_p->setCycleFactor(cyclefactor_p);
    sm_p->setCycleSpeedup(cyclespeedup_p);
    {
      ostringstream oos;
      oos << "Clean gain = " <<gain<<", Niter = "<<niter<<", Threshold = "
	  << threshold;
      os << LogIO::NORMAL << String(oos) << LogIO::POST; // More for the
                                                         // logfile than the
                                                         // log window.
    }

#ifdef PABLO_IO
    traceEvent(1,"Starting Deconvolution",23);
#endif

    os << LogIO::NORMAL << (firstrun ? "Start" : "Continu")
       << "ing deconvolution" << LogIO::POST; // Loglevel PROGRESS
    if(se_p->solveSkyModel()) {
      os << LogIO::NORMAL
         << (niter == 0 ? "Image OK" : "Successfully deconvolved image")
         << LogIO::POST; // Loglevel PROGRESS
    }
    else {
      converged=False;
      os << LogIO::NORMAL << "Threshhold not reached yet." << LogIO::POST; // Loglevel PROGRESS
    }

#ifdef PABLO_IO
    traceEvent(1,"Exiting Deconvolution",21);
#endif

    printbeam(sm_p, os, firstrun);
    
    if(((algorithm.substr(0,5)=="clark") || algorithm=="hogbom" ||
        algorithm=="multiscale") && (niter != 0))
      //write the model visibility to ms for now 
      sm_p->solveResiduals(*se_p, True);

    savePSF(psfnames);
    redoSkyModel_p=False;
    restoreImages(image);
    writeFluxScales(fluxscale_p);
    for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
      if(residuals_p[thismodel]->ok()){
	residuals_p[thismodel]->table().relinquishAutoLocks(True);
	residuals_p[thismodel]->table().unlock();
      }
    }
    this->writeHistory(os);
    try{
     // write data processing history into image logtable
      LoggerHolder imagelog (False);
      LogSink& sink = imagelog.sink();
      LogOrigin lor( String("imager"), String("clean()") );
      LogMessage msg(lor);
      sink.postLocally(msg);
      
      ROMSHistoryColumns msHis(ms_p->history());
      const ROScalarColumn<Double> &time_col = msHis.time();
      const ROScalarColumn<String> &origin_col = msHis.origin();
      const ROArrayColumn<String> &cli_col = msHis.cliCommand();
      const ROScalarColumn<String> &message_col = msHis.message();
      if (msHis.nrow()>0) {
	ostringstream oos;
	uInt nmessages = time_col.nrow();
	for (uInt i=0; i < nmessages; i++) {
	  String tmp=frmtTime(time_col(i));
	  oos << tmp
	      << "  HISTORY " << origin_col(i);
	  oos << " " << cli_col(i) << " ";
	  oos << message_col(i)
	      << endl;
	}
	// String historyline(oos);
	sink.postLocally(msg.message(oos.str()));
      }
      for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
	if(Table::isWritable(image(thismodel))){
	  PagedImage<Float> restoredImage(image(thismodel),
					  TableLock(TableLock::AutoNoReadLocking));
	  LoggerHolder& log = restoredImage.logger();
	  log.append(imagelog);
	  log.flush();
	  restoredImage.table().relinquishAutoLocks(True);
	}
      }
    }
    catch(exception& x){
      
      this->unlock();
      os << LogIO::WARN << "Caught exception: " << x.what()
	 << LogIO::POST;
      os << LogIO::SEVERE << "This means your MS/HISTORY table may be corrupted;  you may consider deleting all the rows from this table"
	 <<LogIO::POST; 
      //continue and wrap up this function
      
    }
    catch(...){
      this->unlock();
      os << LogIO::WARN << "Caught unknown exception" <<  LogIO::POST;
      os << LogIO::SEVERE << "The MS/HISTORY table may be corrupted;  you may consider deleting all the rows from this table"
	 <<LogIO::POST;

    }
    
    this->unlock();

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::clean",21);
#endif

    return converged;
  }
  catch (PSFZero&  x)
    {
      //os << LogIO::WARN << x.what() << LogIO::POST;
      savePSF(psfnames);
      this->unlock();
      throw(AipsError(String("PSFZero  ")+ x.getMesg()));
      return True;
    }  
  catch (exception &x) { 
    this->unlock();
    throw(AipsError(x.what()));

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::clean",21);
#endif

    return False;
  } 

  catch(...){
    this->unlock();
    //Unknown exception...
    throw(AipsError("Unknown exception caught ...imager/casa may need to be exited"));
  }
  this->unlock();

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::clean",21);
#endif  

  os << LogIO::NORMAL << "Exiting Imager::clean" << LogIO::POST; // Loglevel PROGRESS
  return converged;
}

void Imager::printbeam(CleanImageSkyModel *sm_p, LogIO &os, const Bool firstrun)
{
  //Use predefined beam for restoring or find one by fitting
  Bool printBeam = false;
  if(!beamValid_p){
    Vector<Float> beam(3);
    beam=sm_p->beam(0);
    if(beam[0] > 0.0){
      bmaj_p=Quantity(abs(beam(0)), "arcsec"); 
      bmin_p=Quantity(abs(beam(1)), "arcsec");
      bpa_p=Quantity(beam(2), "deg");
      beamValid_p=True;
      printBeam = true;
      os << LogIO::NORMAL << "Fitted beam used in restoration: " ;	 // Loglevel INFO
    }
  }
  else if(firstrun){
    printBeam = true;
    os << LogIO::NORMAL << "Beam used in restoration: "; // Loglevel INFO
  }
  if(printBeam)
    os << LogIO::NORMAL << bmaj_p.get("arcsec").getValue() << " by " // Loglevel INFO
       << bmin_p.get("arcsec").getValue() << " (arcsec) at pa " 
       << bpa_p.get("deg").getValue() << " (deg) " << LogIO::POST;
}

// Mem algorithm
Bool Imager::mem(const String& algorithm,
		 const Int niter, 
		 const Quantity& sigma, 
		 const Quantity& targetFlux,
		 const Bool constrainFlux,
		 const Bool displayProgress, 
		 const Vector<String>& model, 
		 const Vector<Bool>& fixed,
		 const String& complist,
		 const Vector<String>& prior,
		 const Vector<String>& mask,
		 const Vector<String>& image,
		 const Vector<String>& residual)
{
   if(!valid())
    {
      return False;
    }
   logSink_p.clearLocally();
   LogIO os(LogOrigin("imager", "mem()"), logSink_p);
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) 
      {
	return False;
      }
    os << LogIO::NORMAL << "Deconvolving images with MEM" << LogIO::POST; // Loglevel PROGRESS
    
    Int nmodels=model.nelements();
    os << LogIO::NORMAL  // Loglevel INFO
       << "Found " << nmodels << " specified model images" << LogIO::POST;
    
    if(model.nelements()>0) {
      for (uInt thismodel=0;thismodel<model.nelements();++thismodel) {
	if(model(thismodel)=="") {
	  this->unlock();
	  os << LogIO::SEVERE << "Need a name for model "
	     << thismodel << LogIO::POST;
	  return False;
	}
      }
    }
    
    Vector<String> modelNames=model;
    // Make first image with the required shape and coordinates only if
    // it doesn't exist yet. Otherwise we'll throw an exception later
    if(modelNames(0)=="") modelNames(0)=imageName()+".mem";
    if(!Table::isWritable(modelNames(0))) {
      make(modelNames(0));
    }
    
    Vector<String> maskNames(nmodels);
    if(Int(mask.nelements())==nmodels) {
      maskNames=mask;
      for(Int k=0; k < nmodels; ++k){
	if(mask(k)!=""&& !Table::isReadable(mask(k))) {
	  os << LogIO::WARN 
	     << "Mask" << mask(k) 
	     << " is unreadable; ignoring masks altogether " 
	     << LogIO::POST;
	  maskNames.resize(1);
	  maskNames(0)="";
	}
      }
    }
    else {
      maskNames.resize(1);
      maskNames(0)="";
    }
    
    // Always fill in the residual images
    Vector<String> residualNames(nmodels);
    if(Int(residual.nelements())==nmodels) {
      residualNames=residual;
    }
    else {
      residualNames="";
    }
    for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
      if(residualNames(thismodel)=="") {
	residualNames(thismodel)=modelNames(thismodel)+".residual";
      }
      removeTable(residualNames(thismodel));
      if(!clone(model(thismodel), residualNames(thismodel)))
	{
	  return False;
	}
    }
    
    // Make an ImageSkyModel with the specified polarization representation
    // (i.e. circular or linear)
    if(algorithm=="entropy") {
      sm_p = new CEMemImageSkyModel(sigma.get("Jy").getValue(),
				    targetFlux.get("Jy").getValue(),
				    constrainFlux,
				    prior,
				    algorithm);
      os << LogIO::NORMAL // Loglevel INFO
         << "Using single-field algorithm with Maximum Entropy" << LogIO::POST;
      if(ftmachine_p=="mosaic" ||ftmachine_p=="wproject" )
	sm_p->setSubAlgorithm("full");
    }
    else if (algorithm=="emptiness") {
      sm_p = new CEMemImageSkyModel(sigma.get("Jy").getValue(),
				    targetFlux.get("Jy").getValue(),
				    constrainFlux,
				    prior,
				    algorithm);
      os << LogIO::NORMAL // Loglevel INFO
         << "Using single-field algorithm with Maximum Emptiness" << LogIO::POST;
      if(ftmachine_p=="mosaic" ||ftmachine_p=="wproject" )
	sm_p->setSubAlgorithm("full");
    }
    else if (algorithm=="mfentropy") {
      sm_p = new MFCEMemImageSkyModel(sigma.get("Jy").getValue(),
				      targetFlux.get("Jy").getValue(),
				      constrainFlux,
				      prior,
				      algorithm);
      doMultiFields_p = True;
      os << LogIO::NORMAL << "Using Maximum Entropy" << LogIO::POST; // Loglevel INFO
      //   if(ftmachine_p=="mosaic" ||ftmachine_p=="wproject" )
      sm_p->setSubAlgorithm("full");
    } else if (algorithm=="mfemptiness") {
      sm_p = new MFCEMemImageSkyModel(sigma.get("Jy").getValue(),
				      targetFlux.get("Jy").getValue(),
				      constrainFlux,
				      prior,
				      algorithm);
      doMultiFields_p = True;
      os << LogIO::NORMAL << "Using Maximum Emptiness" << LogIO::POST; // Loglevel INFO
      // if(ftmachine_p=="mosaic" ||ftmachine_p=="wproject" )
      sm_p->setSubAlgorithm("full");
    } else {
      this->unlock();
      os << LogIO::SEVERE << "Unknown algorithm: " << algorithm << LogIO::POST;
      return False;
    }
    AlwaysAssert(sm_p, AipsError);
    sm_p->setAlgorithm("mem");
    if (displayProgress) {
      sm_p->setDisplayProgress(True);
    }
    sm_p->setNumberIterations(niter);
    sm_p->setCycleFactor(cyclefactor_p);   // used by mf algs
    sm_p->setCycleSpeedup(cyclespeedup_p); // used by mf algs

    {
      ostringstream oos;
      oos << "MEM algorithm = " <<algorithm<<", Niter = "<<niter<<", Sigma = "
	  <<sigma << ", Target Flux = " << targetFlux;
      os << LogIO::DEBUG1 << String(oos) << LogIO::POST;
    }
    
    //    if (!se_p)
    if(!createSkyEquation(modelNames, fixed, maskNames, complist)) 
      {
	return False;
      }
    os << LogIO::NORMAL3 << "Created Sky Equation" << LogIO::POST;
    
    addResidualsToSkyEquation(residualNames);

    os << LogIO::NORMAL << "Starting deconvolution" << LogIO::POST; // Loglevel PROGRESS
    if(se_p->solveSkyModel()) {
      os << LogIO::NORMAL << "Successfully deconvolved image" << LogIO::POST; // Loglevel INFO
    }
    else {
      os << LogIO::NORMAL << "Nominally failed deconvolution" << LogIO::POST; // Loglevel INFO
    }

    // Get the PSF fit while we are here
    if(!beamValid_p){
      Vector<Float> beam(3);
      beam=sm_p->beam(0);
      if(beam[0] > 0){
	bmaj_p=Quantity(abs(beam(0)), "arcsec"); 
	bmin_p=Quantity(abs(beam(1)), "arcsec");
	bpa_p=Quantity(beam(2), "deg");
	beamValid_p=True;
      }
    }
    if(algorithm=="entropy" || algorithm=="emptiness" )
      sm_p->solveResiduals(*se_p, True);
    restoreImages(image);
    writeFluxScales(fluxscale_p);
    destroySkyEquation();  
    this->writeHistory(os);
    try{
      { // write data processing history into image logtable
	LoggerHolder imagelog (False);
	LogSink& sink = imagelog.sink();
	LogOrigin lor( String("imager"), String("mem()") );
	LogMessage msg(lor);
	sink.postLocally(msg);
	
	ROMSHistoryColumns msHis(ms_p->history());
	if (msHis.nrow()>0) {
	  ostringstream oos;
	  uInt nmessages = msHis.time().nrow();
	  for (uInt i=0; i < nmessages; ++i) {
	    oos << frmtTime((msHis.time())(i))
		<< "  HISTORY " << (msHis.origin())(i);
	    oos << " " << (msHis.cliCommand())(i) << " ";
	    oos << (msHis.message())(i)
		<< endl;
	  }
	  String historyline(oos);
	  sink.postLocally(msg.message(historyline));
	}
	
	for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
	  PagedImage<Float> restoredImage(image(thismodel),
					  TableLock(TableLock::UserLocking));
	  LoggerHolder& log = restoredImage.logger();
	  log.append(imagelog);
	  log.flush();
	}
      }
    }
    catch(exception& x){
      
      os << LogIO::WARN << "Caught exception: " << x.what()
	 << LogIO::POST;
      os << LogIO::SEVERE << "This means your MS/HISTORY table may be corrupted; you may consider deleting all the rows from this table"
	 <<LogIO::POST; 
      //continue and wrap up this function as normal
      
    }
    catch(...){
      //Unknown exception...
      throw(AipsError("Unknown exception caught ...imager/casa may need to be exited"));
    }
    
    this->unlock();

    return True;
  } catch (exception& x) {
    this->unlock();
    throw(AipsError(x.what()));

    return False;
  } 
  this->unlock();
  return True;

}

Bool Imager::pixon(const String& algorithm,
		   const Quantity& sigma, 
		   const String& model)
{
  if(!valid()) {
    return False;
  }
  LogIO os(LogOrigin("imager", "pixon()", WHERE));
  
  this->lock();

  try {

    if(algorithm=="singledish") {

      if(!assertDefinedImageParameters()) {
	return False;
      }
      String modelName=model;
      if(modelName=="") modelName=imageName()+".pixon";
      make(modelName);
      
      PagedImage<Float> modelImage(modelName);
      
      os << LogIO::NORMAL << "Single dish pixon processing" << LogIO::POST; // Loglevel PROGRESS
      os << LogIO::NORMAL // Loglevel INFO
         << "Using defaults for primary beams in pixon processing" << LogIO::POST;
      gvp_p=new VPSkyJones(*mssel_p, True, parAngleInc_p, squintType_p,
                           skyPosThreshold_p);
      os << LogIO::NORMAL << "Calculating data sampling, etc." << LogIO::POST; // Loglevel PROGRESS
      SDDataSampling ds(*mssel_p, *gvp_p, modelImage.coordinates(),
			modelImage.shape(), sigma);
      
      os << LogIO::NORMAL << "Finding pixon solution" << LogIO::POST; // Loglevel PROGRESS
      PixonProcessor pp;

      IPosition zero(4, 0, 0, 0, 0);
      Array<Float> result;
      if(pp.calculate(ds, result)) {
	os << LogIO::NORMAL << "Pixon solution succeeded" << LogIO::POST; // Loglevel INFO
	modelImage.putSlice(result, zero);
      }
      else {
	os << LogIO::WARN << "Pixon solution failed" << LogIO::POST;
      }
    }
    else if(algorithm=="synthesis") {

      if(!assertDefinedImageParameters()) {
	return False;
      }
      String modelName=model;
      if(modelName=="") modelName=imageName()+".pixon";
      make(modelName);
      
      PagedImage<Float> modelImage(modelName);
      
      os << LogIO::NORMAL << "Synthesis pixon processing" << LogIO::POST; // Loglevel INFO
      os << LogIO::NORMAL << "Calculating data sampling, etc." << LogIO::POST; // Loglevel PROGRESS
      SynDataSampling ds(*mssel_p, modelImage.coordinates(),
			 modelImage.shape(), sigma);
      
      os << LogIO::NORMAL << "Finding pixon solution" << LogIO::POST; // Loglevel PROGRESS
      PixonProcessor pp;
      
      IPosition zero(4, 0, 0, 0, 0);
      Array<Float> result;
      if(pp.calculate(ds, result)) {
	os << LogIO::NORMAL << "Pixon solution succeeded" << LogIO::POST; // Loglevel INFO
	modelImage.putSlice(result, zero);
      }
      else {
	os << LogIO::WARN << "Pixon solution failed" << LogIO::POST;
      }
    }
    else if(algorithm=="synthesis-image") {

      if(!assertDefinedImageParameters()) {
	return False;
      }
      String modelName=model;
      if(modelName=="") modelName=imageName()+".pixon";
      make(modelName);
      
      PagedImage<Float> modelImage(modelName);
      
      os << LogIO::NORMAL << "Synthesis image pixon processing" << LogIO::POST; // Loglevel PROGRESS
      String dirtyName=modelName+".dirty";
      Imager::makeimage("corrected", dirtyName);
      String psfName=modelName+".psf";
      Imager::makeimage("psf", psfName);
      PagedImage<Float> dirty(dirtyName);
      PagedImage<Float> psf(psfName);
      os << "Calculating data sampling, etc." << LogIO::POST;
      ImageDataSampling imds(dirty, psf, sigma.getValue());
      os << "Finding pixon solution" << LogIO::POST;
      PixonProcessor pp;
      IPosition zero(4, 0, 0, 0, 0);
      Array<Float> result;
      if(pp.calculate(imds, result)) {
	os << "Pixon solution succeeded" << LogIO::POST;
	modelImage.putSlice(result, zero);
      }
      else {
	os << LogIO::WARN << "Pixon solution failed" << LogIO::POST;
      }
    }

    else if(algorithm=="test") {

      os << LogIO::NORMAL << "Pixon standard test" << LogIO::POST; // Loglevel INFO
      PixonProcessor pp;

      return pp.standardTest();
      
    } else {
      this->unlock();
      os << LogIO::SEVERE << "Unknown algorithm: " << algorithm << LogIO::POST;
      return False;

    }

    this->unlock();

    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;

    return False;
  } 
  this->unlock();
  return True;

}

Bool Imager::restoreImages(const Vector<String>& restoredNames)
{

  LogIO os(LogOrigin("imager", "restoreImages()", WHERE));
  try{
    // It's important that we use the congruent images in both
    // cases. This means that we must use the residual image as
    // passed to the SkyModel and not the one returned. This 
    // distinction is important only (currently) for WFCleanImageSkyModel
    // which has a different representation for the image internally.
    Vector<String> residualNames(images_p.nelements());
    Vector<String> modelNames(images_p.nelements());
    for(uInt k=0; k < modelNames.nelements() ; ++k){
      residualNames[k]=residuals_p[k]->name();
      modelNames[k]=images_p[k]->name();
    }

    /*
    if((nx_p*ny_p > 7000*7000) && (ft_p->name() != "MosaicFT")){
      // very large for convolution ...destroy Skyequations to release memory
      // need to fix the convolution to be leaner
      destroySkyEquation();

    }
    */
    Bool dorestore=False;
    if( (bmaj_p.getValue() >0.0) && (bmin_p.getValue() > 0.0))
      dorestore=True;
    
    if(restoredNames.nelements()>0) {
      for (Int thismodel=0;thismodel<Int(restoredNames.nelements()); 
	   ++thismodel) {
	if(restoredNames(thismodel)!="") {
	  PagedImage<Float> modelIm(modelNames[thismodel]);
	  PagedImage<Float> residIm(residualNames[thismodel]);
	  TempImage<Float> restored;
	  if(dorestore){
	    restored=TempImage<Float>(modelIm.shape(),
				modelIm.coordinates());
	    restored.copyData(modelIm);
	   
	    StokesImageUtil::Convolve(restored, bmaj_p, bmin_p, bpa_p);
	  }
	  // We can work only if the residual image was defined.
	  if(residIm.name() != "") {
	    if(dorestore){
	      LatticeExpr<Float> le(restored+(residIm)); 
	      restored.copyData(le);
	    }
	    if(freqFrameValid_p){
	      CoordinateSystem cs=residIm.coordinates();
	      String errorMsg;
	      if (CoordinateUtil::setSpectralConversion (errorMsg, cs,MFrequency::showType(freqFrame_p))) {
		residIm.setCoordinateInfo(cs);
		if(dorestore)
		  restored.setCoordinateInfo(cs);
	      }
	    } 
	    
	    //should be able to do that only on testing dofluxscale
	    // ftmachines or sm_p should tell us that
	    
	    // this should go away...e
	    //special casing like this gets hard to maintain
	    // need to redo how interactive is done so that it is outside 
	    //of this code 


	    //
	    // Using minPB_p^2 below to make it consistent with the normalization in SkyEquation.
	    //
	    Float cutoffval=minPB_p;
	    if(ft_p->name()=="MosaicFT")
	      cutoffval=minPB_p*minPB_p;
	    
	    if (sm_p->doFluxScale(thismodel)) {
	      TempImage<Float> cover(modelIm.shape(),modelIm.coordinates());
	      if(ft_p->name()=="MosaicFT")
		se_p->getCoverageImage(thismodel, cover);
              else
                  cover.copyData(sm_p->fluxScale(thismodel));
	      if(scaleType_p=="NONE"){
		if(dorestore){
		  LatticeExpr<Float> le(iif(cover < minPB_p, 
					    0.0,(restored/(sm_p->fluxScale(thismodel)))));
		  restored.copyData(le);
		}
		LatticeExpr<Float> le1(iif(cover < minPB_p, 
					   0,(residIm/(sm_p->fluxScale(thismodel)))));
		residIm.copyData(le1);
		
	      }
	      
	      //Setting the bit-mask for mosaic image
	      LatticeExpr<Bool> lemask(iif((cover < cutoffval) , 
					   False, True));
	      if(dorestore){
		ImageRegion outreg=restored.makeMask("mask0", False, True);
		LCRegion& outmask=outreg.asMask();
		outmask.copyData(lemask);
		restored.defineRegion("mask0", outreg, RegionHandler::Masks, True);
		restored.setDefaultMask("mask0");
	  
	      }
	      
	    }
	    if(dorestore){
	      PagedImage<Float> diskrestore(restored.shape(), restored.coordinates(), 
					    restoredNames(thismodel));
	      diskrestore.copyData(restored);
	      ImageInfo ii = modelIm.imageInfo();
	      ii.setRestoringBeam(bmaj_p, bmin_p, bpa_p); 
	      diskrestore.setImageInfo(ii);
	      diskrestore.setUnits(Unit("Jy/beam"));
	      copyMask(diskrestore, restored, "mask0");
		 
	    }
	  }
	  else {
	    os << LogIO::SEVERE << "No residual image for model "
	       << thismodel << ", cannot restore image" << LogIO::POST;
	  }
	  
	}
    
      }

    }
  }
  catch (exception &x) { 
    os << LogIO::SEVERE << "Exception: " << x.what() << LogIO::POST;
    return False;
  }
  return True;
}

Bool Imager::copyMask(ImageInterface<Float>& out, const ImageInterface<Float>& in, String maskname, Bool setdef){
  try{
    if(in.hasRegion(maskname) && !out.hasRegion(maskname)){
      ImageRegion outreg=out.makeMask(maskname, False, True);
      LCRegion& outmask=outreg.asMask();
      outmask.copyData(in.getRegion("mask0").asLCRegion());
      LatticeExpr<Float> myexpr(iif(outmask, out, 0.0) );
      out.copyData(myexpr);
      out.defineRegion(maskname, outreg, RegionHandler::Masks, True);
      if(setdef)
	out.setDefaultMask(maskname);
    }
    else
      return False;
    
  }
  catch(exception &x){
    throw(AipsError(x.what()));
    return False;
  }


  return True;
}

Bool Imager::writeFluxScales(const Vector<String>& fluxScaleNames)
{
  LogIO os(LogOrigin("imager", "writeFluxScales()", WHERE));
  Bool answer = False;
  ImageInterface<Float> *cover;
  if(fluxScaleNames.nelements()>0) {
    for (Int thismodel=0;thismodel<Int(fluxScaleNames.nelements());++thismodel) {
      if(fluxScaleNames(thismodel)!="") {
        PagedImage<Float> fluxScale(images_p[thismodel]->shape(),
                                    images_p[thismodel]->coordinates(),
                                    fluxScaleNames(thismodel));
	PagedImage<Float> coverimage(images_p[thismodel]->shape(),
				     images_p[thismodel]->coordinates(),
				     fluxScaleNames(thismodel)+".pbcoverage");
        coverimage.table().markForDelete();
        if (sm_p->doFluxScale(thismodel)) {
	  cover=&(sm_p->fluxScale(thismodel));
	  answer = True;
          fluxScale.copyData(sm_p->fluxScale(thismodel));
	  Float cutoffval=minPB_p;
	  if(ft_p->name()=="MosaicFT"){
	    cutoffval=minPB_p*minPB_p;
	    se_p->getCoverageImage(thismodel, coverimage);
            cover=&(coverimage);
	    //Do the sqrt
	    coverimage.copyData(( LatticeExpr<Float> )(iif(coverimage > 0.0, sqrt(coverimage), 0.0)));
            coverimage.table().unmarkForDelete();
	    LatticeExpr<Bool> lemask(iif((*cover) < sqrt(cutoffval), 
				       False, True));
	    ImageRegion outreg=coverimage.makeMask("mask0", False, True);
	    LCRegion& outmask=outreg.asMask();
	    outmask.copyData(lemask);
	    coverimage.defineRegion("mask0", outreg,RegionHandler::Masks, True); 
	    coverimage.setDefaultMask("mask0");
	  }
	  LatticeExpr<Bool> lemask(iif((*cover) < minPB_p, 
				       False, True));
	  ImageRegion outreg=fluxScale.makeMask("mask0", False, True);
	  LCRegion& outmask=outreg.asMask();
	  outmask.copyData(lemask);
	  fluxScale.defineRegion("mask0", outreg,RegionHandler::Masks, True); 
	  fluxScale.setDefaultMask("mask0");


        } else {
	  answer = False;
          os << LogIO::NORMAL // Loglevel INFO
             << "No flux scale available (or required) for model " << thismodel
             << LogIO::POST;
          os << LogIO::NORMAL // Loglevel INFO
             << "(This is only pertinent to mosaiced images)" << LogIO::POST;
          os << LogIO::NORMAL // Loglevel INFO
             << "Writing out image of constant 1.0" << LogIO::POST;
          fluxScale.set(1.0);
        }
      }
    }
  }
  return answer;
}
    
// NNLS algorithm
Bool Imager::nnls(const String&,  const Int niter, const Float tolerance, 
		  const Vector<String>& model, const Vector<Bool>& fixed,
		  const String& complist,
		  const Vector<String>& fluxMask,
		  const Vector<String>& dataMask,
		  const Vector<String>& residual,
		  const Vector<String>& image)
{
  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "nnls()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) return False;
    
    os << LogIO::NORMAL << "Performing NNLS deconvolution" << LogIO::POST; // Loglevel PROGRESS
    
    if(niter<0) {
      this->unlock();
      os << LogIO::SEVERE << "Number of iterations must be positive" << LogIO::POST;
      return False;
    }
    if(tolerance<0.0) {
      this->unlock();
      os << LogIO::SEVERE << LogIO::SEVERE << "Tolerance must be positive" << LogIO::POST;
      return False;
    }
    
    // Add the images to the ImageSkyModel
    Int nmodels=model.nelements();
    if(nmodels>1) os<< "Can only process one model" << LogIO::POST;
    
    if(model(0)=="") {
      this->unlock();
      os << LogIO::SEVERE << "Need a name for model " << LogIO::POST;
      return False;
    }
    
    if(!Table::isWritable(model(0))) {
      make(model(0));
      this->lock();
    }
    
    // Always fill in the residual images
    Vector<String> residualNames(nmodels);
    if(Int(residual.nelements())==nmodels) {
      residualNames=residual;
    }
    else {
      residualNames="";
    }
    for (Int thismodel=0;thismodel<Int(model.nelements());++thismodel) {
      if(residualNames(thismodel)=="") {
	residualNames(thismodel)=model(thismodel)+".residual";
      }
      removeTable(residualNames(thismodel));
      if(!clone(model(thismodel), residualNames(thismodel))) return False;
    }
    
    // Now make the NNLS ImageSkyModel
    sm_p= new NNLSImageSkyModel();
    sm_p->setNumberIterations(niter);
    sm_p->setTolerance(tolerance);
    sm_p->setAlgorithm("nnls");
    os << LogIO::DEBUG1
       << "NNLS Niter = " << niter << ", Tolerance = " << tolerance << LogIO::POST;
    
    //    if (!se_p)
    if(!createSkyEquation(model, fixed, dataMask, fluxMask, complist)) return False;

    addResidualsToSkyEquation(residualNames);
    
    os << LogIO::NORMAL << "Starting deconvolution" << LogIO::POST; // Loglevel PROGRESS

    if(se_p->solveSkyModel()) {
      os << LogIO::NORMAL << "Successfully deconvolved image" << LogIO::POST; // Loglevel INFO
    }
    else {
      os << LogIO::NORMAL << "Nominally failed deconvolution" << LogIO::POST; // Loglevel INFO
    }
    
    // Get the PSF fit while we are here
    StokesImageUtil::FitGaussianPSF(sm_p->PSF(0), bmaj_p, bmin_p, bpa_p);
    beamValid_p=True;
   
    
    // Restore the image
    restoreImages(image);

    destroySkyEquation();
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  this->unlock();
  return True;
}

// Fourier transform the model and componentlist
Bool Imager::ft(const Vector<String>& model, const String& complist,
		const Bool incremental)
{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "ft()", WHERE));

  if (useModelCol_p == False)
    os << LogIO::WARN << "Please start the imager tool with \"usescratch=true\" when using Imager::ft" << LogIO::EXCEPTION;
  
  this->lock();
  try {
    
    if(sm_p) destroySkyEquation();
    
    if(incremental) {
      os << LogIO::NORMAL // Loglevel INFO
         << "Fourier transforming: adding to MODEL_DATA column" << LogIO::POST;
    }
    else {
      os << LogIO::NORMAL // Loglevel INFO
         << "Fourier transforming: replacing MODEL_DATA column" << LogIO::POST;
    }
    
    //    if (!se_p)
    if(!createSkyEquation(model, complist)) return False;
    if(incremental){
      for (Int mod=0; mod < (sm_p->numberOfModels()); ++mod){
	(sm_p->deltaImage(mod)).copyData(sm_p->image(mod));
      }
    }
    se_p->predict(incremental);
    
    destroySkyEquation();
    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  this->unlock();
  return True;
}

Bool Imager::setjy(const Int fieldid, 
		   const Int spectralwindowid,
		   const Vector<Double>& fluxDensity, const String& standard)
{
  // the old interface to new interface
  String fieldnames="";
  String spwstring="";
  Vector<Int>fldids(1,fieldid);
  Vector<Int>spwids(1,spectralwindowid);
  return setjy(fldids, spwids, fieldnames, spwstring, fluxDensity, standard);

}

Bool Imager::setjy(const Vector<Int>& fieldid, 
		   const Vector<Int>& spectralwindowid,
		   const String& fieldnames, const String& spwstring,
		   const Vector<Double>& fluxDensity, const String& standard)
{
  if(!valid()) return False;
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "setjy()"), logSink_p);
  this->lock();

  String tempCL;
  try {
    Bool precompute=(fluxDensity(0) <= 0);

    // Figure out which fields/spws to treat
    Record selrec=ms_p->msseltoindex(spwstring, fieldnames);
    Vector<Int> fldids(selrec.asArrayInt("field"));
    Vector<Int> spwids(selrec.asArrayInt("spw"));

    expand_blank_sel(spwids, ms_p->spectralWindow().nrow());
    expand_blank_sel(fldids, ms_p->field().nrow());

    // Loop over field id. and spectral window id.
    Vector<Double> fluxUsed(4);
    String fluxScaleName;
    Bool matchedScale=False;
    Int spwid, fldid;
    ROMSColumns msc(*ms_p);
    ConstantSpectrum cspectrum;

    for (uInt kk=0; kk<fldids.nelements(); ++kk) {
      fldid=fldids[kk];
      // Extract field name and field center position 
      MDirection position=msc.field().phaseDirMeas(fldid);
      String fieldName=msc.field().name()(fldid);

      for (uInt jj=0; jj< spwids.nelements(); ++jj) {
	spwid=spwids[jj];

	// Determine spectral window center frequency
	IPosition ipos(1,0);
	MFrequency mfreq=msc.spectralWindow().chanFreqMeas()(spwid)(ipos);
	Array<Double> freqArray;
	msc.spectralWindow().chanFreq().get(spwid, freqArray, True);
	Double medianFreq=median(freqArray);
	mfreq.set(MVFrequency(medianFreq));

	fluxUsed=fluxDensity;
	fluxScaleName="user-specified";
	if (precompute) {
	  // Pre-compute flux density for standard sources if not specified
	  // using the specified flux scale standard or catalog.


	  FluxStandard::FluxScale fluxScaleEnum;
	  matchedScale=FluxStandard::matchStandard(standard, fluxScaleEnum, 
						   fluxScaleName);
	  FluxStandard fluxStd(fluxScaleEnum);
	  Flux<Double> returnFlux, returnFluxErr;

	  if (fluxStd.compute(fieldName, mfreq, returnFlux, returnFluxErr)) {
	    // Standard reference source identified
	    returnFlux.value(fluxUsed);
	  } 

	  // dgoscha, NCSA, 02 May, 2002
	  // this else condtion is to handle the case where the user
	  // specifies standard='SOURCE' in the setjy argument.  This will
	  // then look into the SOURCE_MODEL column of the SOURCE subtable
	  // for a table-record entry that points to a component list with the
	  // model information in it.


	  else if (standard==String("SOURCE")) {
		// Look in the SOURCE_MODEL column of the SOURCE subtable for 
		// the name of the CL which contains the model.

		// First test to make sure the SOURCE_MODEL column exists.
		if (ms_p->source().tableDesc().isColumn("SOURCE_MODEL")) {
			TableRecord modelRecord;
			msc.source().sourceModel().get(0, modelRecord);
	
			// Get the name of the model component list from the table record
			Table modelRecordTable = 
				modelRecord.asTable(modelRecord.fieldNumber(String ("model")));
			String modelCLName = modelRecordTable.tableName();
			modelRecord.closeTable(modelRecord.fieldNumber(String ("model")));

			// Now grab the flux from the model component list and use.
			ComponentList modelCL = ComponentList(Path(modelCLName), True);
			SkyComponent fluxComponent = modelCL.component(fldid);

			fluxUsed = 0;
			fluxUsed = real(fluxComponent.flux().value());
			fluxScaleName = modelCLName;
		}
		else {
			os << LogIO::SEVERE << "Missing SOURCE_MODEL column."
			   << LogIO::SEVERE << "Using default, I=1.0"
			   << LogIO::POST;
			fluxUsed = 0;
			fluxUsed(0) = 1.0;
		}
	  }

	  else {
	    // Source not found; use Stokes I=1.0 Jy for now
	    fluxUsed=0;
	    fluxUsed(0)=1.0;
	    fluxScaleName="default";
	  };
	}

	// Set the component flux density
	Flux<Double> fluxval;
	fluxval.setValue(fluxUsed);

	// Create a point component at the field center
	// with the specified flux density
	PointShape point(position);
	SkyComponent skycomp(fluxval, point, cspectrum);

	// Create a component list containing this entry
	String baseString=msname_p + "." + fieldName + ".spw" +
	  String::toString(spwid);
	tempCL=baseString + ".tempcl";

	// Force a call to the ComponentList destructor
	// using scoping rules.
	{ 
	  ComponentList cl;
	  cl.add(skycomp);
	  cl.rename(tempCL, Table::New);
	}

	// Select the uv-data for this field and spw. id.;
	// all frequency channels selected.
	Vector<Int> selectSpw(1), selectField(1);
	selectSpw(0)=spwid;
	selectField(0)=fldid;
	String msSelectString = "";
	Vector<Int> numDeChan(1);
	numDeChan[0]=0;
	Vector<Int> begin(1);
	begin[0]=0;
	Vector<Int> stepsize(1);
	stepsize[0]=1;
	setdata("channel", numDeChan, begin, stepsize, MRadialVelocity(), 
		MRadialVelocity(),
		selectSpw, selectField, msSelectString, "", "", Vector<Int>(), 
		"", "", "", "", True);

	if (!nullSelect_p) {

	  // Transform the component model table
	  Vector<String> model;
	  ft(model, tempCL, False);

	  // Log flux density used for this field and spectral window
	  os.output().width(12);
	  os << fieldName << "  spwid=";
	  os.output().width(3);
	  os << (spwid) << "  ";
	  os.output().width(0);
	  os.output().precision(4);
	  os << LogIO::NORMAL << "[I=" << fluxUsed(0) << ", "; // Loglevel INFO
	  os << "Q=" << fluxUsed(1) << ", ";
	  os << "U=" << fluxUsed(2) << ", ";
	  os << "V=" << fluxUsed(3) << "] Jy, ";
	  os << ("(" + fluxScaleName + ")") << LogIO::POST;
	};
	  
	// Delete the temporary component list and image tables
	Table::deleteTable(tempCL);

      }
    }
    this->writeHistory(os);
    this->unlock();
    return True;

  } catch (AipsError x) {
    this->unlock();
    if(Table::canDeleteTable(tempCL)) Table::deleteTable(tempCL);
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  return True;
}

// Supports the "[] or -1 => everything" convention using the rule:
// If v is empty or only has 1 element, and it is < 0, 
//     replace v with 0, 1, ..., nelem - 1.
// Returns whether or not it modified v.
//   If so, v is modified in place.
Bool Imager::expand_blank_sel(Vector<Int>& v, const uInt nelem)
{
  if((v.nelements() == 1 && v[0] < 0) || v.nelements() == 0){
    v.resize(nelem);
    indgen(v);
    return true;
  }
  return false;
}

// This is the one used by im.setjy() (because it has a model arg).
Bool Imager::setjy(const Vector<Int>& fieldid, 
                   const Vector<Int>& spectralwindowid,
                   const String& fieldnames, const String& spwstring,
                   const String& model,
                   const Vector<Double>& fluxDensity, 
                   const String& standard, const Bool chanDep)
{

  if(!valid()) return False;
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "setjy()"), logSink_p);
  this->lock();

  Vector<Double> fluxdens=fluxDensity;
  if (fluxDensity.nelements()<4) {
    fluxdens.resize(4,True);
    for (Int i=fluxDensity.nelements();i<4;++i)
      fluxdens(i)=0.0;
  }

  Vector<String> tempCLs;
  TempImage<Float> *tmodimage(NULL);

  try {
    Bool precompute = (fluxdens(0) < 0 || (model != ""));

    // Figure out which fields/spws to treat
    Record selrec=ms_p->msseltoindex(spwstring, fieldnames);
    Vector<Int> fldids(selrec.asArrayInt("field"));
    Vector<Int> spwids(selrec.asArrayInt("spw"));

    expand_blank_sel(spwids, ms_p->spectralWindow().nrow());
    expand_blank_sel(fldids, ms_p->field().nrow());

    // Forbid multiple fields for some circumstances
    if (fldids.nelements()>1) {

      if (model!="") {
	String errmsg("setjy cannot apply a model image to multiple fields");
	os << LogIO::SEVERE
	   << errmsg << endl
	   << "run setjy once per field, or use ft"
	   << LogIO::POST;
	throw(AipsError(errmsg));
      }

      if (!precompute) {
	String errmsg("setjy cannot apply user flux density to multiple fields");
	os << LogIO::SEVERE
	   << errmsg << endl
	   << "run setjy once per field"
	   << LogIO::POST;
	throw(AipsError(errmsg));
      }
    }

    // Ignore user-polarization if using an image:
    if (model!="") {
      if (fluxdens(1)!=0.0 ||
	  fluxdens(2)!=0.0 ||
	  fluxdens(3)!=0.0 )
	os << LogIO::WARN
	   << "Using model image, so zeroing user QUV flux densities."
	   << LogIO::POST;
      fluxdens(1)=fluxdens(2)=fluxdens(3)=0.0;
    }

    // Loop over field id. and spectral window id.
    Vector<Double> fluxUsed(4);
    Matrix<Double> fluxUsedPerChan; // 4 rows nchan col ...will resize when needed
    String fluxScaleName("user-specified");
    FluxStandard::FluxScale fluxScaleEnum;
    if(!FluxStandard::matchStandard(standard, fluxScaleEnum, fluxScaleName))
      throw(AipsError(standard + " is not a recognized flux density scale"));

    FluxStandard fluxStd(fluxScaleEnum);
    Int spwid, fldid;
    ROMSColumns msc(*ms_p);
    ConstantSpectrum cspectrum;

    uInt nspws = spwids.nelements();
    Vector<Flux<Double> > returnFluxes(nspws), returnFluxErrs(nspws);
    Vector<MFrequency> mfreqs(nspws);
    Array<Double> freqArray;
    // .getUnits() is a little confusing - it seems to return a Vector which is
    // a list of all the units, not the unit for each row.
    const Unit freqUnit(msc.spectralWindow().chanFreqQuant().getUnits()[0]);
    
    tempCLs.resize(nspws);
      
    IPosition ipos(1,0);

    for(Int fldInd = fldids.nelements(); fldInd--;){
      fldid = fldids[fldInd];
      // Extract field name and field center position 
      MDirection fieldDir=msc.field().phaseDirMeas(fldid);
      String fieldName=msc.field().name()(fldid);
      Bool foundSrc = false;
     
      for(uInt spwInd = 0; spwInd < nspws; ++spwInd){
	spwid = spwids[spwInd];
	Double medianFreq=median(msc.spectralWindow().chanFreq()(spwid));
	// Determine spectral window center frequency
	mfreqs[spwInd] = msc.spectralWindow().chanFreqMeas()(spwid)(ipos);
	mfreqs[spwInd].set(MVFrequency(Quantum<Double>(medianFreq,
                                                       freqUnit)));
      }

      fluxUsed=fluxdens;
      if(precompute){
        // Pre-compute flux density for standard sources if not specified
        // using the specified flux scale standard or catalog.

        if(model != ""){
          // Just get the fluxes and their uncertainties for scaling the image.
          foundSrc = fluxStd.compute(fieldName, mfreqs, returnFluxes,
                                     returnFluxErrs);
        }
        else{
          // Go ahead and get FluxStandard to make the ComponentList, since
          // it knows what type of component to use. 

          // This is _a_ time.  It would be more accurate and safer, but
          // slower, to use the weighted average of the times at which this
          // source was observed, and to use the range of times in the
          // estimate of the error introduced by using a single time.
          //
          // Obviously that would be overkill if the source does not vary.
          //
          MEpoch mtime = msc.field().timeMeas()(fldid);
            
          foundSrc = fluxStd.computeCL(fieldName, mfreqs, mtime, fieldDir,
                                       cspectrum, returnFluxes, returnFluxErrs,
                                       tempCLs);
	  if(chanDep){
	    for (uInt kk =0; kk < nspws; ++kk){
	      spwid=spwids[kk];
	      
	      Vector<Double> freqArray=msc.spectralWindow().chanFreq()(spwid);
	      IPosition whichChan(1,0);
	      MFrequency::Ref myRef=msc.spectralWindow().chanFreqMeas()(spwid)(whichChan).getRef();
	      Vector<Flux<Double> >returnFlux(freqArray.nelements());
	      Vector<MVFrequency> freqvals(freqArray.nelements());
	      Flux<Double> returnFluxErr;
	      for (uInt jj =0 ; jj < freqArray.nelements(); ++jj){
		whichChan[0]=jj;
		freqvals[jj]=msc.spectralWindow().chanFreqMeas()(spwid)(whichChan).getValue();
		fluxStd.compute(fieldName, msc.spectralWindow().chanFreqMeas()(spwid)(whichChan), returnFlux[jj], returnFluxErr);
	      }
	      {
		TabularSpectrum newspec(mfreqs[kk], freqvals, returnFlux, myRef);
		ComponentList cl(tempCLs[kk], False);
		for (uInt uu=0; uu < cl.nelements(); ++uu){
		  cl.component(uu).setSpectrum(newspec);
		}
	      }//This should save that componentlist back
	    }


	  }

       


        }
        if(!foundSrc){
          if (standard==String("SOURCE")) {
            // dgoscha, NCSA, 02 May, 2002
            // this else condtion is to handle the case where the user
            // specifies standard='SOURCE' in the setjy argument.  This will
            // then look into the SOURCE_MODEL column of the SOURCE subtable
            // for a table-record entry that points to a component list with the
            // model information in it.

            // Look in the SOURCE_MODEL column of the SOURCE subtable for 
            // the name of the CL which contains the model.

            // First test to make sure the SOURCE_MODEL column exists.
            if (ms_p->source().tableDesc().isColumn("SOURCE_MODEL")) {
              TableRecord modelRecord;
              msc.source().sourceModel().get(0, modelRecord);
	
              // Get the name of the model component list from the table record
              Table modelRecordTable = 
                modelRecord.asTable(modelRecord.fieldNumber(String ("model")));
              String modelCLName = modelRecordTable.tableName();
              modelRecord.closeTable(modelRecord.fieldNumber(String ("model")));

              // Now grab the flux from the model component list and use.
              ComponentList modelCL = ComponentList(Path(modelCLName), True);
              SkyComponent fluxComponent = modelCL.component(fldid);

              fluxUsed = 0;
              fluxUsed = real(fluxComponent.flux().value());
              fluxScaleName = modelCLName;
            }
            else {
              os << LogIO::SEVERE << "Missing SOURCE_MODEL column."
                 << LogIO::SEVERE << "Using default, I=1.0"
                 << LogIO::POST;
              fluxUsed = 0;
              fluxUsed(0) = 1.0;
            }
	  }
	  else {
	    // Source not found; use Stokes I=1.0 Jy for now
	    fluxUsed=0;
	    fluxUsed(0)=1.0;
	    fluxScaleName="default";
	  };

          // Currently, if !foundSrc, then the flux density is the same for all
          // spws.
          // Log the flux density found for this field.
          os.output().width(12);
          os << fieldName << "  ";
          os.output().width(0);
          os.output().precision(4);
          os << LogIO::NORMAL << "[I=" << fluxUsed(0) << ", "; // Loglevel INFO
          os << "Q=" << fluxUsed(1) << ", ";
          os << "U=" << fluxUsed(2) << ", ";
          os << "V=" << fluxUsed(3) << "] Jy, ";
          os << ("(" + fluxScaleName + ")") << LogIO::POST;
	}
      }
      
      for(uInt spwInd = 0; spwInd < nspws; ++spwInd){
        spwid = spwids[spwInd];

        if(foundSrc){
          // Read this as fluxUsed = returnFluxes[spwInd].value().
          returnFluxes[spwInd].value(fluxUsed);
            
          // Log flux density found for this field and spectral window
          os.output().width(12);
          os << fieldName << "  spwid=";
          os.output().width(3);
          os << spwid << "  ";
          os.output().width(0);
          os.output().precision(4);
          os << LogIO::NORMAL << "[I=" << fluxUsed(0) << ", "; // Loglevel INFO
          os << "Q=" << fluxUsed(1) << ", ";
          os << "U=" << fluxUsed(2) << ", ";
          os << "V=" << fluxUsed(3) << "] Jy, ";
          os << ("(" + fluxScaleName + ")") << LogIO::POST;
        }
          
        // If a model image has been specified, 
        //  rescale it according to the I f.d. determined above

        Bool useimage(False);
        if (model!="") {
	  
	  Vector<Double> freqArray=msc.spectralWindow().chanFreq()(spwid);
	  Vector<Double> freqInc=msc.spectralWindow().chanWidth()(spwid);
	  Double medianFreq=median(freqArray);
	  Double freqWidth=(max(freqArray)-min(freqArray)) + 2*max(freqInc); // 2 channel extra
	  if(chanDep){
	    IPosition whichChan(1,0);
	    Flux<Double> returnFlux;
	    Flux<Double> returnFluxErr;
	    fluxUsedPerChan.resize(4, freqArray.nelements());
	    for (uInt k =0 ; k < freqArray.nelements(); ++k){
	      whichChan[0]=k;
	      fluxStd.compute(fieldName, msc.spectralWindow().chanFreqMeas()(spwid)(whichChan), returnFlux, returnFluxErr);
	      returnFlux.value(fluxUsed);
	      fluxUsedPerChan.column(k)=fluxUsed;
	    }
	  }

          useimage=True;

          PagedImage<Float> modimage(model);
          modimage.table().unmarkForDelete();
	  IPosition imshape=modimage.shape();
	  CoordinateSystem csys(modimage.coordinates());
	  Int freqAxis=CoordinateUtil::findSpectralAxis(csys);
	  Vector<Stokes::StokesTypes> whichPols;
	  Int polAxis=CoordinateUtil::findStokesAxis(whichPols, csys);
	  Int icoord(csys.findCoordinate(Coordinate::SPECTRAL));
	  SpectralCoordinate spcsys=csys.spectralCoordinate(icoord);
	  spcsys.setReferenceValue(Vector<Double>(1,medianFreq));
	  spcsys.setReferencePixel(Vector<Double>(1,0.0));
	  spcsys.setWorldAxisUnits(Vector<String>(1,mfreqs[spwInd].getUnit().getName()));
	  spcsys.setIncrement(Vector<Double>(1,freqWidth));
	  // make a cube model
	  if(chanDep && (fluxUsedPerChan.ncolumn() > 1)){
	    spcsys=SpectralCoordinate(MFrequency::castType(mfreqs[spwInd].getRef().getType()), freqArray, spcsys.restFrequency());
	    if(freqAxis < 2 || polAxis < 2)
	      throw(AipsError("Cannot setjy with a model that has spectral or stokes axis before direction axes.\n Please reorder the axes of the image"));
	    if(freqAxis ==2) {//pol and freq are swapped
	      imshape(3)=freqArray.nelements();
	      Vector<Int> trans(4);
	      trans[0]=0; trans[1]=1; trans[2]=3; trans[3]=2;
	      csys.transpose(trans, trans);
	    }
	    else{
	      imshape(freqAxis)=freqArray.nelements();
	    }
	  } 

	  csys.replaceCoordinate(spcsys,icoord);
	  tmodimage = new TempImage<Float>(imshape, csys);
	  tmodimage->set(0.0f);
	  

          os << LogIO::DEBUG1
             << "freqUnit.getName() = " << freqUnit.getName()
             << LogIO::POST;
          os << LogIO::DEBUG1
             << "mfreqs[spwInd].get(freqUnit).getValue() = "
             << mfreqs[spwInd].get(freqUnit).getValue()
             << LogIO::POST;
          
        	  

          // Check direction consistency (reported in log message below)
	  String err;
	  if(!CoordinateUtil::setDirectionConversion(err, csys, fieldDir.getRefString())){
	    os << "LogIO::WARN " 
	       << "Could not set direction conversion between flux image and " 
	       << fieldDir.getRefString() << LogIO::POST;
	  }
          Int dircoord(csys.findCoordinate(Coordinate::DIRECTION));
          DirectionCoordinate dircsys=csys.directionCoordinate(dircoord);
          MVDirection mvd;
          dircsys.toWorld(mvd,dircsys.referencePixel());
          Double sep=fieldDir.getValue().separation(mvd,"\"").getValue();
	  
          if(fluxdens[0] != 0.0){

            Float sumI=1.0;
	    if(whichPols.nelements() >1)
	      sumI=sum(ImagePolarimetry(modimage).stokesI()).getFloat();
	    else
	      sumI=sum(modimage).getFloat();

            if(spwInd == 0)
              os << LogIO::NORMAL
                 << "Using model image " << modimage.name() // Loglevel INFO
                 << LogIO::POST;
	    // scale the image
	    if(imshape(3)>1){
	      if(modimage.shape()(freqAxis) ==1){
		IPosition blc(imshape.nelements(), 0);
		IPosition trc=imshape-1;
		os << LogIO::NORMAL << "Scaling channels model image to I= " 
		   << fluxUsedPerChan.row(0) 
		   << " Jy for visibility prediction."
		   << LogIO::POST;
		for (uInt k =0; k < fluxUsedPerChan.ncolumn() ; ++k){
		  Float scale=fluxUsedPerChan.column(k)(0)/sumI;
		  blc[3]=k;
		  trc[3]=k;
		  Slicer sl(blc, trc, Slicer::endIsLast);
		  SubImage<Float> subim(*tmodimage, sl, True);
		  subim.copyData((LatticeExpr<Float>)(modimage*scale));
		}
	      }
	      else{// model image is a cube...just regrid it then
		if(freqAxis != 3)
		  throw(AipsError("Cannot setjy with a cube model that does not have the spectral axis as the last one.\n Please reorder the axes of the image"));
		ImageRegrid<Float> ir;
		IPosition axes(1,freqAxis);   // regrid the spectral only
		ir.regrid(*tmodimage, Interpolate2D::LINEAR, axes, modimage);
	      }
	    }
	    else{
	      // Scale factor
		Float scale=fluxUsed(0)/sumI;
		tmodimage->copyData( (LatticeExpr<Float>)(modimage*scale) );	
		os << LogIO::NORMAL << "Scaling model image to I=" << fluxUsed(0) // Loglevel INFO
		   << " Jy for visibility prediction."
		   << LogIO::POST;
	    }
	  
	  }
          else{
            os << LogIO::NORMAL                                  // Loglevel INFO
               << "Using the model image's original unscaled flux density for visibility prediction."
               << LogIO::POST;
            tmodimage->copyData( (LatticeExpr<Float>)(modimage) );
          }
            

          if(spwInd == 0)
            os << LogIO::NORMAL // Loglevel INFO
               << "The model image's reference pixel is " << sep 
               << " arcsec from " << fieldName << "'s phase center."
               << LogIO::POST;
        }
        else{
          useimage=False;

          if(!precompute){
            // fluxUsed was supplied by the user instead of FluxStandard, so
            // make a component list for it now, for use in ft.

            // Set the component flux density
            Flux<Double> fluxval;
            Flux<Double> fluxerr;
            fluxval.setValue(fluxUsed);
	            
            // Create a point component at the field center
            // with the specified flux density
            PointShape point(fieldDir);

            // No worries about varying fluxes or sizes here, so any time will do.
            MEpoch mtime = msc.field().timeMeas()(fldid);

            tempCLs[spwInd] = FluxStandard::makeComponentList(fieldName, mfreqs[spwInd],
                                                              mtime, fluxval, point,
                                                              cspectrum);
          }
        }

        // Select the uv-data for this field and spw. id.;
        // all frequency channels selected.
        Vector<Int> selectSpw(1), selectField(1);
        selectSpw(0)=spwid;
        selectField(0)=fldid;
        String msSelectString = "";
        Vector<Int> numDeChan(1);
        numDeChan[0]=0;
        Vector<Int> begin(1);
        begin[0]=0;
        Vector<Int> stepsize(1);
        stepsize[0]=1;
        setdata("channel", numDeChan, begin, stepsize, MRadialVelocity(), 
                MRadialVelocity(),
                selectSpw, selectField, msSelectString, "", "",
                Vector<Int>(), "", "", "", "", True, true);

        if (!nullSelect_p) {

          // Use ft to form visibilities
          Vector<String> modelv;
          if (useimage) {
	    if(sm_p) destroySkyEquation();
	    if(!ft_p) createFTMachine();
	    sm_p = new CleanImageSkyModel();
	    sm_p->add(*tmodimage,1);
	    useModelCol_p=True;
	    setSkyEquation();
	    se_p->predict(False);
	    destroySkyEquation();
          }
          else
            ft(modelv, tempCLs[spwInd], False);

        };
	
        // Delete the temporary component list and image tables
        if (tempCLs[spwInd] != "")
          if(Table::canDeleteTable(tempCLs[spwInd]))Table::deleteTable(tempCLs[spwInd]);
        if (tmodimage) delete tmodimage;
        tmodimage=NULL;
        //	if (Table::canDeleteTable("temp.setjy.image")) Table::deleteTable("temp.setjy.image");
      }
    }
    this->writeHistory(os);
    this->unlock();
    return True;

  } catch (AipsError x) {
    this->unlock();
    for(Int i = tempCLs.nelements(); i--;){
      if(tempCLs[i] != "")
        Table::deleteTable(tempCLs[i]);
    }
    if (tmodimage) delete tmodimage; tmodimage=NULL;
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  return True;
}

// Temporary copy of setjy() while flux calibration with Solar System objects
// is being tested.
Bool Imager::ssoflux(const Vector<Int>& fieldid, 
                     const Vector<Int>& spectralwindowid,
                     const String& fieldnames, const String& spwstring,
                     const String& model,
                     const Vector<Double>& fluxDensity, 
                     const String& standard)
{
  if(!valid()) return False;
  logSink_p.clearLocally();
  LogIO os(LogOrigin("imager", "ssoflux()"), logSink_p);
  this->lock();

  Vector<Double> fluxdens=fluxDensity;
  if (fluxDensity.nelements()<4) {
    fluxdens.resize(4,True);
    for (Int i=fluxDensity.nelements();i<4;++i)
      fluxdens(i)=0.0;
  }

  Vector<String> tempCLs;
  PagedImage<Float> *tmodimage(NULL);

  try {
    Bool precompute = (fluxdens(0) < 0 || (fluxdens(0) == 0.0 && model != ""));

    // Figure out which fields/spws to treat
    Record selrec=ms_p->msseltoindex(spwstring, fieldnames);
    Vector<Int> fldids(selrec.asArrayInt("field"));
    Vector<Int> spwids(selrec.asArrayInt("spw"));

    expand_blank_sel(spwids, ms_p->spectralWindow().nrow());
    expand_blank_sel(fldids, ms_p->field().nrow());

    // Forbid multiple fields for some circumstances
    if (fldids.nelements()>1) {

      if (model!="") {
	String errmsg("ssoflux cannot apply a model image to multiple fields");
	os << LogIO::SEVERE
	   << errmsg << endl
	   << "run ssoflux once per field, or use ft"
	   << LogIO::POST;
	throw(AipsError(errmsg));
      }

      if (!precompute) {
	String errmsg("ssoflux cannot apply user flux density to multiple fields");
	os << LogIO::SEVERE
	   << errmsg << endl
	   << "run ssoflux once per field"
	   << LogIO::POST;
	throw(AipsError(errmsg));
      }
    }

    // Ignore user-polarization if using an image:
    if (model!="") {
      if (fluxdens(1)!=0.0 ||
	  fluxdens(2)!=0.0 ||
	  fluxdens(3)!=0.0 )
	os << LogIO::WARN
	   << "Using model image, so zeroing user QUV flux densities."
	   << LogIO::POST;
      fluxdens(1)=fluxdens(2)=fluxdens(3)=0.0;
    }

    // Loop over field id. and spectral window id.
    Vector<Double> fluxUsed(4);
    String fluxScaleName("user-specified");
    FluxStandard::FluxScale fluxScaleEnum;
    if(!FluxStandard::matchStandard(standard, fluxScaleEnum, fluxScaleName))
      throw(AipsError(standard + " is not a recognized flux density scale"));

    FluxStandard fluxStd(fluxScaleEnum);
    Int spwid, fldid;
    ROMSColumns msc(*ms_p);
    ConstantSpectrum cspectrum;

    uInt nspws = spwids.nelements();
    Vector<Flux<Double> > returnFluxes(nspws), returnFluxErrs(nspws);
    Vector<MFrequency> mfreqs(nspws);
    Array<Double> freqArray;
    // .getUnits() is a little confusing - it seems to return a Vector which is
    // a list of all the units, not the unit for each row.
    const Unit freqUnit(msc.spectralWindow().chanFreqQuant().getUnits()[0]);
    
    tempCLs.resize(nspws);
      
    IPosition ipos(1,0);

    for(Int fldInd = fldids.nelements(); fldInd--;){
      fldid = fldids[fldInd];
      // Extract field name and field center position 
      MDirection position=msc.field().phaseDirMeas(fldid);
      String fieldName=msc.field().name()(fldid);
      Bool foundSrc = false;
     
      for(uInt spwInd = 0; spwInd < nspws; ++spwInd){
	spwid = spwids[spwInd];

	// Determine spectral window center frequency
	mfreqs[spwInd] = msc.spectralWindow().chanFreqMeas()(spwid)(ipos);
	msc.spectralWindow().chanFreq().get(spwid, freqArray, True);
	Double medianFreq=median(freqArray);
	mfreqs[spwInd].set(MVFrequency(Quantum<Double>(medianFreq,
                                                       freqUnit)));
      }

      fluxUsed=fluxdens;
      if(precompute){
        // Pre-compute flux density for standard sources if not specified
        // using the specified flux scale standard or catalog.

        if(model != ""){
          // Just get the fluxes and their uncertainties for scaling the image.
          foundSrc = fluxStd.compute(fieldName, mfreqs, returnFluxes,
                                     returnFluxErrs);
        }
        else{
          // Go ahead and get FluxStandard to make the ComponentList, since
          // it knows what type of component to use. 

          // This is _a_ time.  It would be more accurate and safer, but
          // slower, to use the weighted average of the times at which this
          // source was observed, and to use the range of times in the
          // estimate of the error introduced by using a single time.
          //
          // Obviously that would be overkill if the source does not vary.
          //
          MEpoch mtime = msc.field().timeMeas()(fldid);
            
          foundSrc = fluxStd.computeCL(fieldName, mfreqs, mtime, position,
                                       cspectrum, returnFluxes, returnFluxErrs,
                                       tempCLs);
        }
        if(!foundSrc){
          if (standard==String("SOURCE")) {
            // dgoscha, NCSA, 02 May, 2002
            // this else condtion is to handle the case where the user
            // specifies standard='SOURCE' in the ssoflux argument.  This will
            // then look into the SOURCE_MODEL column of the SOURCE subtable
            // for a table-record entry that points to a component list with the
            // model information in it.

            // Look in the SOURCE_MODEL column of the SOURCE subtable for 
            // the name of the CL which contains the model.

            // First test to make sure the SOURCE_MODEL column exists.
            if (ms_p->source().tableDesc().isColumn("SOURCE_MODEL")) {
              TableRecord modelRecord;
              msc.source().sourceModel().get(0, modelRecord);
	
              // Get the name of the model component list from the table record
              Table modelRecordTable = 
                modelRecord.asTable(modelRecord.fieldNumber(String ("model")));
              String modelCLName = modelRecordTable.tableName();
              modelRecord.closeTable(modelRecord.fieldNumber(String ("model")));

              // Now grab the flux from the model component list and use.
              ComponentList modelCL = ComponentList(Path(modelCLName), True);
              SkyComponent fluxComponent = modelCL.component(fldid);

              fluxUsed = 0;
              fluxUsed = real(fluxComponent.flux().value());
              fluxScaleName = modelCLName;
            }
            else {
              os << LogIO::SEVERE << "Missing SOURCE_MODEL column."
                 << LogIO::SEVERE << "Using default, I=1.0"
                 << LogIO::POST;
              fluxUsed = 0;
              fluxUsed(0) = 1.0;
            }
	  }
	  else {
	    // Source not found; use Stokes I=1.0 Jy for now
	    fluxUsed=0;
	    fluxUsed(0)=1.0;
	    fluxScaleName="default";
	  };

          // Currently, if !foundSrc, then the flux density is the same for all
          // spws.
          // Log the flux density found for this field.
          os.output().width(12);
          os << fieldName << "  ";
          os.output().width(0);
          os.output().precision(4);
          os << LogIO::NORMAL << "[I=" << fluxUsed(0) << ", "; // Loglevel INFO
          os << "Q=" << fluxUsed(1) << ", ";
          os << "U=" << fluxUsed(2) << ", ";
          os << "V=" << fluxUsed(3) << "] Jy, ";
          os << ("(" + fluxScaleName + ")") << LogIO::POST;
	}
      }
      
      for(uInt spwInd = 0; spwInd < nspws; ++spwInd){
        spwid = spwids[spwInd];

        if(foundSrc){
          // Read this as fluxUsed = returnFluxes[spwInd].value().
          returnFluxes[spwInd].value(fluxUsed);
            
          // Log flux density found for this field and spectral window
          os.output().width(12);
          os << fieldName << "  spwid=";
          os.output().width(3);
          os << spwid << "  ";
          os.output().width(0);
          os.output().precision(4);
          os << LogIO::NORMAL << "[I=" << fluxUsed(0) << ", "; // Loglevel INFO
          os << "Q=" << fluxUsed(1) << ", ";
          os << "U=" << fluxUsed(2) << ", ";
          os << "V=" << fluxUsed(3) << "] Jy, ";
          os << ("(" + fluxScaleName + ")") << LogIO::POST;
        }
          
        // If a model image has been specified, 
        //  rescale it according to the I f.d. determined above

        Bool useimage(False);
        if (model!="") {
	  
          useimage=True;

          PagedImage<Float> modimage(model);
          modimage.table().unmarkForDelete();
          String tempmodname=File::newUniqueName("./", "ssofluximage").baseName();
          tmodimage = new PagedImage<Float>(modimage.shape(),modimage.coordinates(),tempmodname);
          tmodimage->table().markForDelete();
          tmodimage->set(0.0f);
	  
          CoordinateSystem csys(tmodimage->coordinates());

          Int icoord(csys.findCoordinate(Coordinate::SPECTRAL));
          SpectralCoordinate spcsys=csys.spectralCoordinate(icoord);

          os << LogIO::DEBUG1
             << "freqUnit.getName() = " << freqUnit.getName()
             << LogIO::POST;
          os << LogIO::DEBUG1
             << "mfreqs[spwInd].get(freqUnit).getValue() = "
             << mfreqs[spwInd].get(freqUnit).getValue()
             << LogIO::POST;
          
          spcsys.setReferenceValue(Vector<Double>(1,
                                                  mfreqs[spwInd].get(freqUnit).getValue()));
          spcsys.setWorldAxisUnits(Vector<String>(1, freqUnit.getName()));
          csys.replaceCoordinate(spcsys,icoord);
          tmodimage->setCoordinateInfo(csys);
	  

          // Check direction consistency (reported in log message below)
          Int dircoord(csys.findCoordinate(Coordinate::DIRECTION));
          DirectionCoordinate dircsys=csys.directionCoordinate(dircoord);
          MVDirection mvd;
          dircsys.toWorld(mvd,dircsys.referencePixel());
          Double sep=position.getValue().separation(mvd,"\"").getValue();
	  
          if(fluxUsed[0] != 0.0){
            os << LogIO::NORMAL                                  // Loglevel INFO
               << "Scaling model image to I=" << fluxUsed(0)
               << " Jy for visibility prediction."
               << LogIO::POST;

            Float sumI=sum(modimage.get());
            
            // Scale factor
            Float scale=fluxUsed(0)/sumI;

            // scale the image
            tmodimage->copyData( (LatticeExpr<Float>)(modimage*scale) );
          }
          else{
            os << LogIO::NORMAL                                  // Loglevel INFO
               << "Using the model image's original unscaled flux density for visibility prediction."
               << LogIO::POST;
            tmodimage->copyData( (LatticeExpr<Float>)(modimage) );
          }
            
          if(spwInd == 0){
            os << LogIO::NORMAL
               << "Using model image " << modimage.name() // Loglevel INFO
               << LogIO::POST;
            os << LogIO::NORMAL // Loglevel INFO
               << "Its reference pixel is " << sep 
               << " arcsec from " << fieldName << "'s phase center."
               << LogIO::POST;
          }
        }
        else{
          useimage=False;

          if(!precompute){
            // fluxUsed was supplied by the user instead of FluxStandard, so
            // make a component list for it now, for use in ft.

            // Set the component flux density
            Flux<Double> fluxval;
            Flux<Double> fluxerr;
            fluxval.setValue(fluxUsed);
	            
            // Create a point component at the field center
            // with the specified flux density
            PointShape point(position);

            // No worries about varying fluxes or sizes here, so any time will do.
            MEpoch mtime = msc.field().timeMeas()(fldid);

            tempCLs[spwInd] = FluxStandard::makeComponentList(fieldName, mfreqs[spwInd],
                                                              mtime, fluxval, point,
                                                              cspectrum);
          }
        }

        // Select the uv-data for this field and spw. id.;
        // all frequency channels selected.
        Vector<Int> selectSpw(1), selectField(1);
        selectSpw(0)=spwid;
        selectField(0)=fldid;
        String msSelectString = "";
        Vector<Int> numDeChan(1);
        numDeChan[0]=0;
        Vector<Int> begin(1);
        begin[0]=0;
        Vector<Int> stepsize(1);
        stepsize[0]=1;
        setdata("channel", numDeChan, begin, stepsize, MRadialVelocity(), 
                MRadialVelocity(),
                selectSpw, selectField, msSelectString, "", "",
                Vector<Int>(), "", "", "", "", True, true);

        if (!nullSelect_p) {

          // Use ft to form visibilities
          Vector<String> modelv;
          if (useimage) {
            modelv.resize(1);
            modelv(0)=tmodimage->name();
            ft(modelv, "", False);
          }
          else
            ft(modelv, tempCLs[spwInd], False);

        };
	
        // Delete the temporary component list and image tables
        if (tempCLs[spwInd] != "")
          Table::deleteTable(tempCLs[spwInd]);
        if (tmodimage) delete tmodimage;
        tmodimage=NULL;
        //	if (Table::canDeleteTable("temp.setjy.image")) Table::deleteTable("temp.setjy.image");
      }
    }
    this->writeHistory(os);
    this->unlock();
    return True;

  } catch (AipsError x) {
    this->unlock();
    for(Int i = tempCLs.nelements(); i--;){
      if(tempCLs[i] != "")
        Table::deleteTable(tempCLs[i]);
    }
    if (tmodimage) delete tmodimage; tmodimage=NULL;
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  return True;
}

Bool Imager::clone(const String& imageName, const String& newImageName)
{
  //if(!valid()) return False;
  // This is not needed if(!assertDefinedImageParameters()) return False;
  LogIO os(LogOrigin("imager", "clone()", WHERE));
  try {
    PagedImage<Float> oldImage(imageName);
    PagedImage<Float> newImage(TiledShape(oldImage.shape(), 
					  oldImage.niceCursorShape()), oldImage.coordinates(),
			       newImageName);
    newImage.set(0.0);
    newImage.table().flush(True, True);
  } catch (AipsError x) {
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  return True;
}

// Make an empty image
Bool Imager::make(const String& model)
{

#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::make",21);
#endif

  if(!valid())
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::make",20);
#endif

      return False;
    }
  LogIO os(LogOrigin("imager", "make()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters())
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::make",20);
#endif

	return False;
      }
    
    // Make an image with the required shape and coordinates
    String modelName(model);
    if(modelName=="") modelName=imageName()+".model";
    os << LogIO::DEBUG1
       << "Making empty image: " << modelName << LogIO::POST;
    
    removeTable(modelName);
    CoordinateSystem coords;
    if(!imagecoordinates(coords, false)) 
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::make",20);
#endif
	this->unlock();
	return False;
      }
    this->makeEmptyImage(coords, modelName, fieldid_p);
    this->unlock();
    
#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::make",20);
#endif

    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;

#ifdef PABLO_IO
    traceEvent(1,"Exiting Imager::make",20);
#endif

    return False;    

  } 
  this->unlock();

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::make",20);
#endif

  return True;
}

// Fit the psf. If psf is blank then make the psf first.
Bool Imager::fitpsf(const String& psf, Quantity& mbmaj, Quantity& mbmin,
		    Quantity& mbpa)
{

#ifdef PABLO_IO
  traceEvent(1,"Entering Imager::fitpsf",23);
#endif

  if(!valid()) 
    {

#ifdef PABLO_IO
      traceEvent(1,"Exiting Imager::fitps",22);
#endif

      return False;
    }
  LogIO os(LogOrigin("imager", "fitpsf()", WHERE));
  
  this->lock();
  try {
    if(!assertDefinedImageParameters()) 
      {

#ifdef PABLO_IO
	traceEvent(1,"Exiting Imager::fitps",22);
#endif
	this->unlock();
	return False;
      }
    
    os << LogIO::NORMAL << "Fitting to psf" << LogIO::POST; // Loglevel PROGRESS
    
    String lpsf; lpsf=psf;
    if(lpsf=="") {
      lpsf=imageName()+".psf";
      makeimage("psf", lpsf);
    }

    if(!Table::isReadable(lpsf)) {
      this->unlock();
      os << LogIO::SEVERE << "PSF image " << lpsf << " does not exist"
	 << LogIO::POST;

#ifdef PABLO_IO
     traceEvent(1,"Exiting Imager::fitpsf",22);
#endif

      return False;
    }

    PagedImage<Float> psfImage(lpsf);
    StokesImageUtil::FitGaussianPSF(psfImage, mbmaj, mbmin, mbpa);
    bmaj_p=mbmaj;
    bmin_p=mbmin;
    bpa_p=mbpa;
    beamValid_p=True;
    
    os << LogIO::NORMAL // Loglevel INFO
       << "  Beam fit: " << bmaj_p.get("arcsec").getValue() << " by "
       << bmin_p.get("arcsec").getValue() << " (arcsec) at pa " 
       << bpa_p.get("deg").getValue() << " (deg) " << endl;

    this->unlock();
    
#ifdef PABLO_IO
     traceEvent(1,"Exiting Imager::fitps",22);
#endif

return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;

#ifdef PABLO_IO
     traceEvent(1,"Exiting Imager::fitps",22);
#endif

     return False;
  } 
  this->unlock();

#ifdef PABLO_IO
  traceEvent(1,"Exiting Imager::fitps",22);
#endif

  return True;
}


Bool Imager::setscales(const String& scaleMethod,
			    const Int inscales,
			    const Vector<Float>& userScaleSizes)
{
  scaleMethod_p = scaleMethod;
  userScaleSizes_p.resize(userScaleSizes.nelements());
  userScaleSizes_p = userScaleSizes;
  if (scaleMethod_p == "uservector") {
    nscales_p =  userScaleSizes.nelements();
  } else {
    nscales_p = inscales;
  }
  //Force the creation of a new sm_p with the new scales
  destroySkyEquation();
  scaleInfoValid_p = True;  
  return True;
};

Bool Imager::setSmallScaleBias(const Float inbias)
{ 
  smallScaleBias_p = inbias;
  return True;
}

// Added for wb algo.
Bool Imager::settaylorterms(const Int intaylor,const Double inreffreq)
{
  ntaylor_p = intaylor;
  reffreq_p = inreffreq;
  return True;
};

// Set the beam
Bool Imager::setbeam(const Quantity& mbmaj, const Quantity& mbmin,
		     const Quantity& mbpa)
{
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "setbeam()", WHERE));
  
  bmaj_p=mbmaj;
  bmin_p=mbmin;
  bpa_p=mbpa;
  beamValid_p=True;
    
  return True;
}

// Correct the data using a plain VisEquation.
// This just moves data from observed to corrected.
// Eventually we should pass in a calibrater
// object to do the work.
Bool Imager::correct(const Bool doparallactic, const Quantity& t) 
{
  if(!valid()) return False;

  LogIO os(LogOrigin("imager", "correct()", WHERE));
  
  // (gmoellen 06Nov20)
  // VisEquation/VisJones have been extensively revised, so
  //  so we are disabling this method.  Will probably soon
  /// delete it entirely, since this is a calibrater responsibility....

  this->lock();
  try {

    // Warn users we are disabled.
    throw(AipsError("Imager::correct() has been disabled (gmoellen 06Nov20)."));

 /* Commenting out old VisEquation/VisJones usage
    os << "Correcting data: CORRECTED_DATA column will be replaced"
       << LogIO::POST;
    
    
    if(doparallactic) {
      os<<"Correcting parallactic angle variation"<<LogIO::POST;
      VisEquation ve(*vs_p);
      Float ts=t.get("s").getValue();
      Float dt=ts/10.0;
      PJones pj(*vs_p, ts, dt);
      ve.setVisJones(pj);
      ve.correct();
    }
    else {
      VisEquation ve(*vs_p);
      ve.correct();
    }
    this->unlock();
    return True;
 */


  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  this->unlock();
  
  return True;
}

uInt Imager::count_visibilities(ROVisibilityIterator *rvi,
                                const Bool unflagged_only,
                                const Bool must_have_imwt)
{
  if(!valid()) return 0;
  LogIO os(LogOrigin("imager", "count_visibilities()", WHERE));
  
  this->lock();
  ROVisIter& vi(*rvi);
  VisBuffer vb(vi);
    
  uInt nVis = 0;
  for(vi.originChunks(); vi.moreChunks(); vi.nextChunk()){
    for(vi.origin(); vi.more(); ++vi){
      uInt nRow = vb.nRow();
      uInt nChan = vb.nChannel();
      for(uInt row = 0; row < nRow; ++row)
        for(uInt chn = 0; chn < nChan; ++chn)
          if((!unflagged_only || !vb.flag()(chn,row)) &&
             (!must_have_imwt || vb.imagingWeight()(chn,row) > 0.0))
            ++nVis;
    }
  }
  this->unlock();
  return nVis;
}

// Plot the uv plane
Bool Imager::plotuv(const Bool rotate) 
{

  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "plotuv()", WHERE));
  
  this->lock();
  try {
    os << LogIO::NORMAL // Loglevel PROGRESS
       << "Plotting uv coverage for currently selected data" << LogIO::POST;
    
    ROVisIter& vi(*rvi_p);
    VisBuffer vb(vi);
    
    uInt nVis = count_visibilities(rvi_p, true, true);
    
    if(nVis==0) {
      this->unlock();
      os << LogIO::SEVERE << "No unflagged visibilities" << LogIO::POST;
      return False;
    }
    
    if(rotate) {
      os << LogIO::NORMAL // Loglevel INFO
         << "UVW will be rotated to specified phase center" << LogIO::POST;    
    }
    
    
    Vector<Float> u(nVis); u=0.0;
    Vector<Float> v(nVis); v=0.0;
    Vector<Float> uRotated(nVis); uRotated=0.0;
    Vector<Float> vRotated(nVis); vRotated=0.0;
    Float maxAbsUV=0.0;
    
    Int iVis=0;
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	Int nChan=vb.nChannel();
	Vector<Double> uvwRotated(3);
	MeasFrame mFrame((MEpoch(Quantity(vb.time()(0), "s"))), mLocation_p);
	UVWMachine uvwMachine(phaseCenter_p, vb.phaseCenter(), mFrame);
	for (Int row=0; row<nRow; ++row) {
	  if(rotate) {
	    for (Int dim=0;dim<3;++dim) {
	      uvwRotated(dim)=vb.uvw()(row)(dim);
	    }
	    uvwMachine.convertUVW(uvwRotated);
	  }
	  
	  for (Int chn=0; chn<nChan; ++chn) {
	    if(!vb.flag()(chn,row)&&vb.imagingWeight()(chn,row)>0.0) {
	      Float f=vb.frequency()(chn)/C::c;
	      u(iVis)=vb.uvw()(row)(0)*f;
	      v(iVis)=vb.uvw()(row)(1)*f;
	      if(abs(u(iVis))>maxAbsUV) maxAbsUV=abs(u(iVis));
	      if(abs(v(iVis))>maxAbsUV) maxAbsUV=abs(v(iVis));
	      if(rotate) {
		uRotated(iVis)=uvwRotated(0)*f;
		vRotated(iVis)=uvwRotated(1)*f;
		if(abs(uRotated(iVis))>maxAbsUV) maxAbsUV=abs(uRotated(iVis));
		if(abs(vRotated(iVis))>maxAbsUV) maxAbsUV=abs(vRotated(iVis));
	      }
	      ++iVis;
	    }
	  }
	}
      }
    }
    
    if(maxAbsUV==0.0) {
      this->unlock();
      os << LogIO::SEVERE << "Maximum uv distance is zero" << LogIO::POST;
      return False;
    }
    else {
      Quantity cell(0.5/maxAbsUV, "rad");
      os << LogIO::NORMAL // Loglevel INFO
         << "Maximum uv distance = " << maxAbsUV << " wavelengths" << endl;
      os << LogIO::NORMAL // Loglevel INFO
         << "Recommended cell size < " << cell.get("arcsec").getValue()
	 << " arcsec" << LogIO::POST;
    }
    
   
    if(rotate) {
    
      PlotServerProxy *plotter = dbus::launch<PlotServerProxy>( );
      dbus::variant panel_id = plotter->panel( "UV-Coverage for "+imageName(), "U (wavelengths)", "V (wavelengths)", "UV-Plot", "right");

      if ( panel_id.type( ) != dbus::variant::INT ) {
	os << LogIO::SEVERE << "failed to start plotter" << LogIO::POST;
	return False;
      }

      plotter->release( panel_id.getInt( ) );
      plotter->scatter(dbus::af(u),dbus::af(v),"blue","unrotated","hexagon",6,-1,panel_id.getInt( ));
      plotter->scatter(dbus::af(uRotated),dbus::af(vRotated),"red","rotated","ellipse",6,-1,panel_id.getInt( ));

    }
    else {
       PlotServerProxy *plotter = dbus::launch<PlotServerProxy>( );
      dbus::variant panel_id = plotter->panel( "UV-Coverage for "+imageName(), "U (wavelengths)", "V (wavelengths)", "UV-Plot" ,"right");

      if ( panel_id.type( ) != dbus::variant::INT ) {
	os << LogIO::SEVERE << "failed to start plotter" << LogIO::POST;
	return False;
      }

      plotter->release( panel_id.getInt( ) );
      plotter->scatter(dbus::af(u),dbus::af(v),"blue","uv in data","rect",6,-1,panel_id.getInt( ));
      u=u*Float(-1.0);
      v=v*Float(-1.0);
      plotter->scatter(dbus::af(u),dbus::af(v),"red","conjugate","ellipse",6,-1,panel_id.getInt( ));

    }
    
    this->unlock();
  
  } 
  catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
    
  } 
  catch (...) {
    this->unlock();  
  } 
  this->unlock();
  
  

  return True;
}

// Plot the visibilities
Bool Imager::plotvis(const String& type, const Int increment) 
{

  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "plotvis()", WHERE));
  
  this->lock();
  try {
    
    os << LogIO::NORMAL // Loglevel PROGRESS
       << "Plotting Stokes I visibility for currently selected data"
       << LogIO::POST;
    
    
    ROMSColumns msc(*mssel_p);
    Bool hasCorrected=!(msc.correctedData().isNull());
    Bool hasModel= !(msc.modelData().isNull());

    Bool twoPol=True;
    Vector<String> polType=msc.feed().polarizationType()(0);
    if (polType(0)!="X" && polType(0)!="Y" &&
	polType(0)!="R" && polType(0)!="L") {
      twoPol=False;
    }
    
    ROVisIter& vi(*rvi_p);
    VisBuffer vb(vi);
    
    Int nVis=0;
    Int counter=0;
    Float maxWeight=0;
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	Int nChan=vb.nChannel();
	maxWeight=max(maxWeight, max(vb.imagingWeight()));
	for (Int row=0; row<nRow; ++row) {
	  for (Int chn=0; chn<nChan; ++chn) {
	    if(!vb.flag()(chn,row)&&vb.imagingWeight()(chn,row)>0.0) {
	      ++counter;
	      if(counter==increment) {
		counter=0;
		++nVis;
	      }
	    }
	  }
	}
      }
    }
    
    if(nVis==0) {
      os << LogIO::SEVERE << "No unflagged visibilities" << LogIO::POST;
      if(maxWeight <=0){
	os << LogIO::SEVERE << "Max of imaging-weight is " << maxWeight 
	   << LogIO::POST;
	os << LogIO::SEVERE << "Try setting it with the function weight"  
	   << LogIO::POST;
      }
      this->unlock();
      return False;
    }
    
    if(increment>1) {
      os << LogIO::NORMAL << "For increment = " << increment << ", found " << nVis // Loglevel INFO
	 << " points for plotting" << endl;
    }
    else {
      os << LogIO::NORMAL << "Found " << nVis << " points for plotting" << endl; // Loglevel INFO
    }
    Vector<Float> amp(nVis); amp=0.0;
    Vector<Float> correctedAmp(nVis); correctedAmp=0.0;
    Vector<Float> modelAmp(nVis); modelAmp=0.0;
    Vector<Float> residualAmp(nVis); residualAmp=0.0;
    Vector<Float> uvDistance(nVis); uvDistance=0.0;
   
    if(!hasModel)
      modelAmp.resize();
    if(!hasCorrected)
      correctedAmp.resize();
    if(!hasCorrected || !hasModel)
      residualAmp.resize();
    
    Float maxuvDistance=0.0;
    Float maxAmp=0.0;
    Float maxCorrectedAmp=0.0;
    Float maxModelAmp=0.0;
    Float maxResidualAmp=0.0;
    Int iVis=0;
    counter=0;
    vi.originChunks();
     vi.origin();
     uInt numCorrPol=vb.modelVisCube().shape()(0)-1;
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	Int nChan=vb.nChannel();
	for (Int row=0; row<nRow; ++row) {
	  for (Int chn=0; chn<nChan; ++chn) {
	    if(!vb.flag()(chn,row)&&vb.imagingWeight()(chn,row)>0.0) {
	      ++counter;
	      if(counter==increment) {
		counter=0;
		Float f=vb.frequency()(chn)/C::c;
		Float u=vb.uvw()(row)(0)*f; 
		Float v=vb.uvw()(row)(1)*f;
		uvDistance(iVis)=sqrt(square(u)+square(v));
		if(twoPol) {
		  amp(iVis)=sqrt((square(abs(vb.visCube()(0,chn,row)))+
				  square(abs(vb.visCube()(numCorrPol,chn,row))))/2.0);
		  if(hasCorrected)
		    correctedAmp(iVis)=
		      sqrt((square(abs(vb.correctedVisCube()(0,chn,row)))+
			    square(abs(vb.correctedVisCube()(numCorrPol,chn,row))))/2.0);
		  if(hasModel)
		    modelAmp(iVis)=
		      sqrt((square(abs(vb.modelVisCube()(0,chn,row)))+
			    square(abs(vb.modelVisCube()(numCorrPol,chn,row))))/2.0);
		  if(hasCorrected && hasModel)
		    residualAmp(iVis)=
		      sqrt((square(abs(vb.modelVisCube()(0,chn,row)-
				       vb.correctedVisCube()(0,chn,row)))+
			    square(abs(vb.modelVisCube()(numCorrPol,chn,row)-
				       vb.correctedVisCube()(numCorrPol,chn,row))))/2.0);
		}
		else {
		  amp(iVis)=abs(vb.visCube()(0,chn,row));
		  if(hasCorrected)
		    correctedAmp(iVis)=abs(vb.correctedVisCube()(0,chn,row));
		   if(hasModel)
		     modelAmp(iVis)=abs(vb.modelVisCube()(0,chn,row));
		  if(hasCorrected && hasModel) 
		    residualAmp(iVis)=
		      abs(vb.modelVisCube()(0,chn,row)-
			  vb.correctedVisCube()(0,chn,row));
		}
		if(uvDistance(iVis)>maxuvDistance) {
		  maxuvDistance=uvDistance(iVis);
		}
		if(amp(iVis)>maxAmp) {
		  maxAmp=amp(iVis);
		}
		if(hasCorrected && (correctedAmp(iVis)>maxCorrectedAmp)) {
		  maxCorrectedAmp=correctedAmp(iVis);
		}
		if(hasModel && (modelAmp(iVis)>maxModelAmp)) {
		  maxModelAmp=modelAmp(iVis);
		}
		if((hasModel&&hasCorrected) && (residualAmp(iVis)>maxResidualAmp)) {
		  maxResidualAmp=residualAmp(iVis);
		}
		++iVis;
	      }
	    }
	  }
	}
      }
    }
    
    if(maxuvDistance==0.0) {
      os << LogIO::SEVERE << "Maximum uv distance is zero" << LogIO::POST;
      this->unlock();
      return False;
    }
    


    

    Float Ymax(0.0);

    if (type.contains("corrected") && hasCorrected)
      if(maxCorrectedAmp>Ymax) Ymax = maxCorrectedAmp;

    if (type.contains("model") && hasModel)
      if(maxModelAmp>Ymax)     Ymax = maxModelAmp;

    if (type.contains("residual") && (hasModel && hasCorrected))
      if(maxResidualAmp>Ymax)  Ymax = maxResidualAmp;

    if (type.contains("observed"))
      if(maxAmp>Ymax)          Ymax = maxAmp;

    if ((type=="all") || (type == ""))
      {
	if (maxAmp > Ymax)       Ymax = maxAmp;
	if(hasCorrected && (maxCorrectedAmp>Ymax)) Ymax = maxCorrectedAmp;
	if(hasModel && (maxModelAmp>Ymax))     Ymax = maxModelAmp;
	if((hasModel && hasCorrected) && maxResidualAmp>Ymax)  Ymax = maxResidualAmp;
      }
    PlotServerProxy *plotter = dbus::launch<PlotServerProxy>( );
    dbus::variant panel_id = plotter->panel( "Stokes I Visibility for "+imageName(),"UVDistance (wavelengths)" , "Amplitude", "Vis-Plot", "right");
    
    if ( panel_id.type( ) != dbus::variant::INT ) {
      os << LogIO::SEVERE << "failed to start plotter" << LogIO::POST;
      return False;
    }

    plotter->release( panel_id.getInt( ) );

    if(type=="all"||type==""||type.contains("observed")) {
      plotter->scatter(dbus::af(uvDistance),dbus::af(amp),"blue","observed","hexagon",6,-1,panel_id.getInt( ));
      
    }
    if((type=="all"||type==""||type.contains("corrected")) && hasCorrected) {
      plotter->scatter(dbus::af(uvDistance),dbus::af(correctedAmp),"red","corrected","ellipse",6,-1,panel_id.getInt( ));
    }
    if((type=="all"||type==""||type.contains("model")) && hasModel) {
      plotter->scatter(dbus::af(uvDistance),dbus::af(modelAmp),"green","model","rect",6,-1,panel_id.getInt( ));
    }
    if((type=="all"||type==""||type.contains("residual")) && (hasCorrected && hasModel)) {
      plotter->scatter(dbus::af(uvDistance),dbus::af(residualAmp),"yellow","residual","cross",6,-1,panel_id.getInt( ));
    }

   


    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  this->unlock();

  return True;
}

// Plot the weights
Bool Imager::plotweights(const Bool gridded, const Int increment) 
{

  if(!valid()) return False;
  LogIO os(LogOrigin("imager", "plotweights()", WHERE));
  
  this->lock();
  try {
    
    
    os << LogIO::NORMAL // Loglevel PROGRESS
       << "Plotting imaging weights for currently selected data"
       << LogIO::POST;
    
    ROVisIter& vi(*rvi_p);
    VisBuffer vb(vi);
    
    if(gridded) {
      if(!assertDefinedImageParameters()) {this->unlock(); return False;}
      // First find the gridded weights
      Float uscale, vscale;
      Int uorigin, vorigin;
      uscale=(nx_p*mcellx_p.get("rad").getValue())/2.0;
      vscale=(ny_p*mcelly_p.get("rad").getValue())/2.0;
      uorigin=nx_p/2;
      vorigin=ny_p/2;
      
      // Simply declare a big matrix 
      Float maxWeight=0.0;
      Matrix<Float> gwt(nx_p,ny_p);
      gwt=0.0;
      
      Float u, v;
      Float sumwt=0.0;
      for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
	for (vi.origin();vi.more();vi++) {
	  Int nRow=vb.nRow();
	  Int nChan=vb.nChannel();
	  for (Int row=0; row<nRow; ++row) {
	    for (Int chn=0; chn<nChan; ++chn) {
	      if(!vb.flag()(chn,row)&&vb.imagingWeight()(chn,row)>0.0) {
		Float f=vb.frequency()(chn)/C::c;
		u=vb.uvw()(row)(0)*f; 
		v=vb.uvw()(row)(1)*f;
		Int ucell=Int(uscale*u+uorigin);
		Int vcell=Int(vscale*v+vorigin);
		if((ucell>0)&&(ucell<nx_p)&&(vcell>0)&&(vcell<ny_p)) {
		  gwt(ucell,vcell)+=vb.imagingWeight()(chn,row);
		  sumwt+=vb.imagingWeight()(chn,row);
		  if(vb.imagingWeight()(chn,row)>maxWeight) {
		    maxWeight=vb.imagingWeight()(chn,row);
		  }
		}
		ucell=Int(-uscale*u+uorigin);
		vcell=Int(-vscale*v+vorigin);
		if((ucell>0)&&(ucell<nx_p)&&(vcell>0)&&(vcell<ny_p)) {
		  gwt(ucell,vcell)+=vb.imagingWeight()(chn,row);
		}
	      }
	    }
	  }
	}
      }
      
      if(sumwt>0.0) {
	os << LogIO::NORMAL << "Sum of weights = " << sumwt << endl; // Loglevel INFO
      }
      else {
	this->unlock();
	os << LogIO::SEVERE << "Sum of weights is zero: perhaps you need to weight the data"
	   << LogIO::POST;
	  return False;
      }
     
      
      //Float umax=Float(nx_p/2)/uscale;
      //Float vmax=Float(ny_p/2)/vscale;


      PlotServerProxy *plotter = dbus::launch<PlotServerProxy>( );
      dbus::variant panel_id = plotter->panel( "Gridded weights for "+imageName(), "U (wavelengths)", "V (wavelengths)", "ImagingWeight-plot" );
      if ( panel_id.type() != dbus::variant::INT ) {
	  os << "failed to create plot panel" << LogIO::WARN << LogIO::POST;
	  return False;
      }

      plotter->release( panel_id.getInt( ) );

      gwt=Float(0xFFFFFF)-gwt*(Float(0xFFFFFF)/maxWeight);
      IPosition shape = gwt.shape( );
      //bool deleteit = false;
      std::vector<double> data(shape[0] * shape[1]);
      int off = 0;
      for ( int column=0; column < shape[1]; ++column ) {
	for ( int row=0; row < shape[0]; ++row ) {
	  data[off++] = gwt(row,column);
	}
      }

      plotter->raster( data, (int) shape[1], (int) shape[0] );


    }
    else {
      
      // Now do the points plot
      Int nVis=0;
      Int counter=0;
      Float maxWeight=0.0;
      for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
	for (vi.origin();vi.more();vi++) {
	  Int nRow=vb.nRow();
	  Int nChan=vb.nChannel();
	  for (Int row=0; row<nRow; ++row) {
	    for (Int chn=0; chn<nChan; ++chn) {
	      if(!vb.flag()(chn,row)&&vb.imagingWeight()(chn,row)>0.0) {
		++counter;
		if(counter==increment) {
		  counter=0;
		  ++nVis;
		}
	      }
	    }
	  }
	}
      }
      
      if(increment>1) {
	os << LogIO::NORMAL // Loglevel INFO
           << "For increment = " << increment << ", found " << nVis
	   << " points for plotting" << endl;
      }
      else {
	os << LogIO::NORMAL // Loglevel INFO
           << "Found " << nVis << " points for plotting" << endl;
      }
      
      Float maxuvDistance=0.0;
      Vector<Float> weights(nVis);
      Vector<Float> uvDistance(nVis);
      weights=0.0;
      uvDistance=0.0;
      
      Int iVis=0;
      for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
	for (vi.origin();vi.more();vi++) {
	  Int nRow=vb.nRow();
	  Int nChan=vb.nChannel();
	  for (Int row=0; row<nRow; ++row) {
	    for (Int chn=0; chn<nChan; ++chn) {
	      if(!vb.flag()(chn,row)&&vb.imagingWeight()(chn,row)>0.0) {
		++counter;
		if(counter==increment) {
		  Float f=vb.frequency()(chn)/C::c;
		  Float u=vb.uvw()(row)(0)*f; 
		  Float v=vb.uvw()(row)(1)*f;
		  uvDistance(iVis)=sqrt(square(u)+square(v));
		  weights(iVis)=vb.imagingWeight()(chn,row);
		  if(vb.imagingWeight()(chn,row)>maxWeight) {
		    maxWeight=vb.imagingWeight()(chn,row);
		  }
		  if(uvDistance(iVis)>maxuvDistance) {
		    maxuvDistance=uvDistance(iVis);
		  }
		  counter=0;
		  ++iVis;
		}
	      }
	    }
	  }
	}
      }
      
      if(maxuvDistance==0.0) {
	this->unlock();
	os << LogIO::SEVERE << "Maximum uv distance is zero" << LogIO::POST;
	return False;
      }


      PlotServerProxy *plotter = dbus::launch<PlotServerProxy>( );
      dbus::variant panel_id = plotter->panel( "Weights for "+imageName(), "UVDistance (wavelengths)", "Weights", "ImagingWeight-plot" );

      if ( panel_id.type( ) != dbus::variant::INT ) {
	os << LogIO::SEVERE << "failed to start plotter" << LogIO::POST;
	return False;
      }

      plotter->release( panel_id.getInt( ) );
      plotter->scatter(dbus::af(uvDistance),dbus::af(weights),"blue","","hexagon",4,-1,panel_id.getInt( ));

    }
    

    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
  this->unlock();

  return True;
}

Bool Imager::clipvis(const Quantity& threshold) 
{

  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "clipvis()", WHERE));
  
  this->lock();
  try {
    
    Float thres=threshold.get("Jy").getValue();
    
    os << LogIO::NORMAL // Loglevel PROGRESS
       << "Clipping visibilities where residual visibility > "
       << thres << " Jy" << LogIO::POST;
    if(!wvi_p){
      os << LogIO::WARN
         << "Cannot clip visibilities in read only mode of ms" 
	 << LogIO::POST;
      return False;
    }
    VisIter& vi(*wvi_p);
    VisBuffer vb(vi);
    

    vi.originChunks();
    vi.origin();
// Making sure picking LL for [RR RL LR LL] correlations or [RR LL] 
    uInt numCorrPol=vb.modelVisCube().shape()(0) - 1 ; 
    Int nBad=0;
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	Int nChan=vb.nChannel();
	for (Int row=0; row<nRow; ++row) {
	  for (Int chn=0; chn<nChan; ++chn) {
	    if(!vb.flag()(chn,row)) {
	      Float residualAmp=
		sqrt((square(abs(vb.modelVisCube()(0,chn,row)-
				 vb.correctedVisCube()(0,chn,row)))+
		      square(abs(vb.modelVisCube()(numCorrPol,chn,row)-
				 vb.correctedVisCube()(numCorrPol,chn,row))))/2.0);
	      if(residualAmp>thres) {
		vb.flag()(chn,row)=True;
		++nBad;
	      }
	    }
	  }
	}
	vi.setFlag(vb.flag());
      }
    }
    
    os << LogIO::NORMAL << "Flagged " << nBad << " points" << LogIO::POST; // Loglevel INFO
    
    this->unlock();
    return True;
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
  } 
  this->unlock();

  return True;
}

// Plot various ids
Bool Imager::plotsummary() 
{
  
  if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "plotsummary()", WHERE));
  
  os << LogIO::WARN << "NOT implemented " << LogIO::POST;
  return False;

  this->lock();
  try {
    /*
    os << "Plotting field and spectral window ids for currently selected data" << LogIO::POST;
    
    ROVisIter& vi(*rvi_p);
    VisBuffer vb(vi);
    
    Int nVis=0;
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	for (Int row=0; row<nRow; ++row) {
	  ++nVis;
	}
      }
    }
    
    os << "Found " << nVis << " selected records" << LogIO::POST;
    
    Vector<Float> fieldId(nVis);
    Vector<Float> spectralWindowId(nVis);
    Vector<Double> t(nVis);
    
    Int maxFieldId=0;
    Int maxSpectralWindowId=0;
    Int iVis=0;
    for (vi.originChunks();vi.moreChunks();vi.nextChunk()) {
      for (vi.origin();vi.more();vi++) {
	Int nRow=vb.nRow();
	for (Int row=0; row<nRow; ++row) {
	  t(iVis)=vb.time()(row);
	  fieldId(iVis)=vb.fieldId()+1.0;
	  spectralWindowId(iVis)=vb.spectralWindow()+1.003;
	  if(Int(fieldId(iVis))>maxFieldId) maxFieldId=Int(fieldId(iVis));
	  if(Int(spectralWindowId(iVis))>maxSpectralWindowId)
	    maxSpectralWindowId=Int(spectralWindowId(iVis));
	  ++iVis;
	}
      }
    }
    
    Double tStart=t(0);
    Vector<Float> timeFloat(nVis);
    for(Int i=0;i<nVis;++i) {
      timeFloat(i)=Float(t(i)-tStart);
    }
    
    ROMSColumns msc(*ms_p);
    PGPlotter plotter=getPGPlotter();
    plotter.subp(1, 2);
    plotter.page();
    plotter.swin(timeFloat(0), timeFloat(nVis-1)*1.20, 0, Float(maxFieldId)*1.1);
    plotter.tbox("BCSNTZHFO", 0.0, 0, "ABCNTS", 0.0, 0);
    String xLabel="Time (offset from " + MVTime(tStart/86400.0).string() + ")";
    plotter.lab(xLabel, "ID", "Field IDs for " +imageName());
    plotter.sci(1);
    for (Int fid=0;fid<maxFieldId;++fid) {
      String fieldName=msc.field().name()(fid);
      plotter.text(1.02*timeFloat(nVis-1), Float(fid+1), fieldName);
    }
    plotter.pt(timeFloat,fieldId,-1);
    plotter.page();
    plotter.swin(timeFloat(0), timeFloat(nVis-1)*1.20, 0,
		 Float(maxSpectralWindowId)*1.1);
    plotter.tbox("BCSNTZHFO", 0.0, 0, "ABCNTS", 0.0, 0);
    xLabel="Time (offset from " + MVTime(tStart/86400.0).string() + ")";
    plotter.lab(xLabel, "ID", "Spectral Window IDs for " +imageName());
    plotter.sci(1);
    for(Int spwId=0;spwId<maxSpectralWindowId;++spwId) {
      Vector<Double> chanFreq=msc.spectralWindow().chanFreq()(spwId); 
      ostringstream spwString;
      spwString<<chanFreq(0)/1.0e9<<" GHz";
      plotter.text(1.02*timeFloat(nVis-1), Float(spwId+1),
		   spwString);
    }
    plotter.pt(timeFloat,spectralWindowId,-1);
    plotter.iden();
    this->unlock();
    return True;
    */
  } catch (AipsError x) {
    this->unlock();
    os << LogIO::SEVERE << "Exception: " << x.getMesg() << LogIO::POST;
    return False;
  } 
    
  this->unlock();

  return True;
}


Bool Imager::detached() const
{
  if (ms_p.null()) {
    LogIO os(LogOrigin("imager", "detached()", WHERE));
    os << LogIO::SEVERE << 
      "imager is detached - cannot perform operation." << endl <<
      "Call imager.open('filename') to reattach." << LogIO::POST;
    return True;
  }
  return False;
}

// Create the FTMachine as late as possible
Bool Imager::createFTMachine()
{
  

  if(ft_p) {delete ft_p; ft_p=0;}
  if(gvp_p) {delete gvp_p; gvp_p=0;}
  if(cft_p) {delete cft_p; cft_p=0;}

  //For ftmachines that can use double precision gridders 
  Bool useDoublePrecGrid=False;
  //few channels use Double precision
  //till we find a better algorithm to determine when to use Double prec gridding
  if((imageNchan_p < 5) && !(singlePrec_p))
    useDoublePrecGrid=True;

  LogIO os(LogOrigin("imager", "createFTMachine()", WHERE));
  
  // This next line is only a guess
  Int numberAnt=((MeasurementSet&)*ms_p).antenna().nrow();
  
  Int imageVolume=imageshape().product()/square(facets_p);
  if(imageVolume) {
    //    Int minimumCachesize=
    //      tile_p*tile_p*npol_p*imageNchan_p*numberAnt*(numberAnt-1)/2;
    if(imageVolume>cache_p/2) {
      os<<"Cache too small to fit entire image, will use tiled gridding"
	<< endl;  
      os<<"MS has at most "<<numberAnt<<" antennas"<<endl;
      os<<"Optimum cache size to do all baselines is "<< imageVolume
	<< " (complex words)" << LogIO::POST;
    }
  }
  

  Float padding;
  padding=1.0;
  if(doMultiFields_p||(facets_p>1)) {
    padding = padding_p;
    os << LogIO::NORMAL // Loglevel INFO
       << "Multiple fields or facets: transforms will be padded by a factor "
       << padding << LogIO::POST;
  }

  if(ftmachine_p=="sd") {
    os << LogIO::NORMAL // Loglevel INFO
       << "Performing single dish gridding..." << LogIO::POST;
    os << LogIO::NORMAL1 // gridfunction_p is too cryptic for most users.
       << "with convolution function " << gridfunction_p << LogIO::POST;

    // Now make the Single Dish Gridding
    os << LogIO::NORMAL // Loglevel INFO
       << "Gridding will use specified common tangent point:" << LogIO::POST;
    os << LogIO::NORMAL << tangentPoint() << LogIO::POST; // Loglevel INFO
    if(gridfunction_p=="pb") {
      if(!gvp_p) {
	if (doDefaultVP_p) {
	  os << LogIO::NORMAL // Loglevel INFO
             << "Using defaults for primary beams used in gridding" << LogIO::POST;
	  gvp_p=new VPSkyJones(*ms_p, True, parAngleInc_p, squintType_p,
                               skyPosThreshold_p);
	} else {
	  os << LogIO::NORMAL // Loglevel INFO
             << "Using VP as defined in " << vpTableStr_p <<  LogIO::POST;
	  Table vpTable( vpTableStr_p ); 
	  gvp_p=new VPSkyJones(*ms_p, vpTable, parAngleInc_p, squintType_p,
                               skyPosThreshold_p);
	}
      } 
      ft_p = new SDGrid(mLocation_p, *gvp_p, cache_p/2, tile_p, gridfunction_p,
                        sdConvSupport_p);
    }
    else {
      ft_p = new SDGrid(mLocation_p, cache_p/2, tile_p, gridfunction_p,
                        sdConvSupport_p);
    }
    ft_p->setPointingDirColumn(pointingDirCol_p);

    ROVisIter& vi(*rvi_p);
    // Get bigger chunks o'data: this should be tuned some time
    // since it may be wrong for e.g. spectral line
    vi.setRowBlocking(100);
    
    AlwaysAssert(ft_p, AipsError);
    
    cft_p = new SimpleComponentGridMachine();
    AlwaysAssert(cft_p, AipsError);
  }
  else if(ftmachine_p=="mosaic") {
    os << LogIO::NORMAL << "Performing mosaic gridding" << LogIO::POST; // Loglevel PROGRESS
   
    setMosaicFTMachine(useDoublePrecGrid);

    // VisIter& vi(vs_p->iter());
    //   vi.setRowBlocking(100);
    
    AlwaysAssert(ft_p, AipsError);
    
    cft_p = new SimpleComponentFTMachine();
    AlwaysAssert(cft_p, AipsError);
  }
  //
  // Make WProject FT machine (for non co-planar imaging)
  //
  else if (ftmachine_p == "wproject"){
    os << LogIO::NORMAL << "Performing w-plane projection" // Loglevel PROGRESS
       << LogIO::POST;
    if(wprojPlanes_p<64) {
      os << LogIO::WARN
	 << "No of WProjection planes set too low for W projection - recommend at least 128"
	 << LogIO::POST;
    }
    
    ft_p = new WProjectFT(wprojPlanes_p,  mLocation_p,
			  cache_p/2, tile_p, True, padding_p, useDoublePrecGrid);
    AlwaysAssert(ft_p, AipsError);
    cft_p = new SimpleComponentFTMachine();
    AlwaysAssert(cft_p, AipsError);
  }
  //
  // Make PBWProject FT machine (for non co-planar imaging with
  // antenna based PB corrections)
  //
  else if (ftmachine_p == "pbwproject"){
    if (wprojPlanes_p<=1)
      {
	os << LogIO::NORMAL
	   << "You are using wprojplanes=1. Doing co-planar imaging (no w-projection needed)" 
	   << LogIO::POST;
	os << LogIO::NORMAL << "Performing pb-projection" << LogIO::POST; // Loglevel PROGRESS
      }
    if((wprojPlanes_p>1)&&(wprojPlanes_p<64)) 
      {
	os << LogIO::WARN
	   << "No. of w-planes set too low for W projection - recommend at least 128"
	   << LogIO::POST;
	os << LogIO::NORMAL << "Performing pb + w-plane projection" // Loglevel PROGRESS
	   << LogIO::POST;
      }


    if(!gvp_p) 
      {
	os << LogIO::NORMAL // Loglevel INFO
           << "Using defaults for primary beams used in gridding" << LogIO::POST;
	gvp_p = new VPSkyJones(*ms_p, True, parAngleInc_p, squintType_p,
                               skyPosThreshold_p);
      }

    ft_p = new nPBWProjectFT(//*ms_p, 
			     wprojPlanes_p, cache_p/2,
                            cfCacheDirName_p, doPointing, doPBCorr,
                             tile_p, paStep_p, pbLimit_p, True);
    //
    // Explicit type casting of ft_p does not look good.  It does not
    // pick up the setPAIncrement() method of PBWProjectFT without
    // this
    //
    ((nPBWProjectFT *)ft_p)->setPAIncrement(parAngleInc_p);
    if (doPointing)
      {
        try
          {
	    //            epJ = new EPJones(*vs_p, *ms_p);
	    VisSet elVS(*rvi_p);
            epJ = new EPJones(elVS);
            RecordDesc applyRecDesc;
            applyRecDesc.addField("table", TpString);
            applyRecDesc.addField("interp",TpString);
            Record applyRec(applyRecDesc);
            applyRec.define("table",epJTableName_p);
            applyRec.define("interp", "nearest");
            epJ->setApply(applyRec);
            ((nPBWProjectFT *)ft_p)->setEPJones(epJ);
          }
        catch(AipsError& x)
          {
            //
            // Add some more useful info. to the message and translate
            // the generic AipsError exception object to a more specific
            // SynthesisError object.
            //
            String mesg = x.getMesg();
            mesg += ". Error in loading pointing offset table.";
            SynthesisError err(mesg);
            throw(err);
          }
      }

    AlwaysAssert(ft_p, AipsError);
    cft_p = new SimpleComponentFTMachine();
    AlwaysAssert(cft_p, AipsError);
  }
  else if (ftmachine_p == "pbmosaic"){

    if (wprojPlanes_p<=1)
      {
	os << LogIO::NORMAL
	   << "You are using wprojplanes=1. Doing co-planar imaging (no w-projection needed)" 
	   << LogIO::POST;
	os << LogIO::NORMAL << "Performing pb-mosaic" << LogIO::POST; // Loglevel PROGRESS
      }
    if((wprojPlanes_p>1)&&(wprojPlanes_p<64)) 
      {
	os << LogIO::WARN
	   << "No. of w-planes set too low for W projection - recommend at least 128"
	   << LogIO::POST;
	os << LogIO::NORMAL << "Performing pb + w-plane projection" << LogIO::POST; // Loglevel PROGRESS
      }

    if(!gvp_p) 
      {
	os << LogIO::NORMAL // Loglevel INFO
           << "Using defaults for primary beams used in gridding" << LogIO::POST;
	gvp_p = new VPSkyJones(*ms_p, True, parAngleInc_p, squintType_p,
                               skyPosThreshold_p);
      }
    ft_p = new PBMosaicFT(*ms_p, wprojPlanes_p, cache_p/2, 
			  cfCacheDirName_p, True /*doPointing*/, doPBCorr, 
			  tile_p, paStep_p, pbLimit_p, True);
    ((PBMosaicFT *)ft_p)->setObservatoryLocation(mLocation_p);
    //
    // Explicit type casting of ft_p does not look good.  It does not
    // pick up the setPAIncrement() method of PBWProjectFT without
    // this
    //
    os << LogIO::NORMAL << "Setting PA increment to " << parAngleInc_p.getValue("deg") << " deg" << endl;
    ((nPBWProjectFT *)ft_p)->setPAIncrement(parAngleInc_p);

    if (doPointing) 
      {
	try
	  {
	    // Warn users we are have temporarily disabled pointing cal
	    //	    throw(AipsError("Pointing calibration temporarily disabled (gmoellen 06Nov20)."));
	    //  TBD: Bring this up-to-date with new VisCal mechanisms
	    VisSet elVS(*rvi_p);
	    epJ = new EPJones(elVS, *ms_p);
	    RecordDesc applyRecDesc;
	    applyRecDesc.addField("table", TpString);
	    applyRecDesc.addField("interp",TpString);
	    Record applyRec(applyRecDesc);
	    applyRec.define("table",epJTableName_p);
	    applyRec.define("interp", "nearest");
	    epJ->setApply(applyRec);
	    ((nPBWProjectFT *)ft_p)->setEPJones(epJ);
	  }
	catch(AipsError& x)
	  {
	    //
	    // Add some more useful info. to the message and translate
	    // the generic AipsError exception object to a more specific
	    // SynthesisError object.
	    //
	    String mesg = x.getMesg();
	    mesg += ". Error in loading pointing offset table.";
	    SynthesisError err(mesg);
	    throw(err);
	  }
      }
    AlwaysAssert(ft_p, AipsError);
    cft_p = new SimpleComponentFTMachine();
    AlwaysAssert(cft_p, AipsError);

  }
  else if(ftmachine_p=="both") {
      
    os << LogIO::NORMAL // Loglevel INFO
       << "Performing single dish gridding with convolution function "
       << gridfunction_p << LogIO::POST;
    os << LogIO::NORMAL // Loglevel INFO
       << "and interferometric gridding with the prolate spheroidal convolution function"
       << LogIO::POST;
    
    // Now make the Single Dish Gridding
    os << LogIO::NORMAL // Loglevel INFO
       << "Gridding will use specified common tangent point:" << LogIO::POST;
    os << LogIO::NORMAL << tangentPoint() << LogIO::POST; // Loglevel INFO
    if(!gvp_p) {
      os << LogIO::NORMAL // Loglevel INFO
         << "Using defaults for primary beams used in gridding" << LogIO::POST;
      gvp_p = new VPSkyJones(*ms_p, True, parAngleInc_p, squintType_p,
                             skyPosThreshold_p);
    }
    if(sdScale_p != 1.0)
      os << LogIO::NORMAL // Loglevel INFO
         << "Multiplying single dish data by factor " << sdScale_p << LogIO::POST;
    if(sdWeight_p != 1.0)
      os << LogIO::NORMAL // Loglevel INFO
         << "Multiplying single dish weights by factor " << sdWeight_p
         << LogIO::POST;
    ft_p = new GridBoth(*gvp_p, cache_p/2, tile_p,
			mLocation_p, phaseCenter_p,
			gridfunction_p, "SF", padding,
			sdScale_p, sdWeight_p);
    
    //VisIter& vi(vs_p->iter());
    // Get bigger chunks o'data: this should be tuned some time
    // since it may be wrong for e.g. spectral line
    rvi_p->setRowBlocking(100);
    
    AlwaysAssert(ft_p, AipsError);
    
    cft_p = new SimpleComponentFTMachine();
    AlwaysAssert(cft_p, AipsError);
    
  }  
  else {
    os << LogIO::NORMAL // Loglevel INFO
       << "Performing interferometric gridding..."
       << LogIO::POST;
    os << LogIO::NORMAL1 // gridfunction_p is too cryptic for most users.
       << "...with convolution function " << gridfunction_p << LogIO::POST;
    // Now make the FTMachine
    if(facets_p>1) {
      os << LogIO::NORMAL // Loglevel INFO
         << "Multi-facet Fourier transforms will use specified common tangent point:"
	 << LogIO::POST;
      os << LogIO::NORMAL << tangentPoint() << LogIO::POST; // Loglevel INFO
      ft_p = new GridFT(cache_p / 2, tile_p, gridfunction_p, mLocation_p,
                        phaseCenter_p, padding, False, useDoublePrecGrid);
      
    }
    else {
      os << LogIO::DEBUG1
         << "Single facet Fourier transforms will use image center as tangent points"
	 << LogIO::POST;
      ft_p = new GridFT(cache_p/2, tile_p, gridfunction_p, mLocation_p,
			padding, False, useDoublePrecGrid);

    }
    AlwaysAssert(ft_p, AipsError);
    
    cft_p = new SimpleComponentFTMachine();
    AlwaysAssert(cft_p, AipsError);
    
  }
  ft_p->setSpw(dataspectralwindowids_p, freqFrameValid_p);
  ft_p->setFreqInterpolation(freqInterpMethod_p);
  if(doTrackSource_p){
    ft_p->setMovingSource(trackDir_p);
  }
  ft_p->setSpwChanSelection(spwchansels_p);

  return True;
}

String Imager::tangentPoint()
{
  MVAngle mvRa=phaseCenter_p.getAngle().getValue()(0);
  MVAngle mvDec=phaseCenter_p.getAngle().getValue()(1);
  ostringstream oos;
  oos << "     ";
  Int widthRA=20;
  Int widthDec=20;
  oos.setf(ios::left, ios::adjustfield);
  oos.width(widthRA);  oos << mvRa(0.0).string(MVAngle::TIME,8);
  oos.width(widthDec); oos << mvDec.string(MVAngle::DIG2,8);
  oos << "     "
      << MDirection::showType(phaseCenter_p.getRefPtr()->getType());
  return String(oos);
}

Bool Imager::removeTable(const String& tablename) {
  
  LogIO os(LogOrigin("imager", "removeTable()", WHERE));
  
  if(Table::isReadable(tablename)) {
    if (! Table::isWritable(tablename)) {
      os << LogIO::SEVERE << "Table " << tablename
	 << " is not writable!: cannot alter it" << LogIO::POST;
      return False;
    }
    else {
      if (Table::isOpened(tablename)) {
	os << LogIO::SEVERE << "Table " << tablename
	   << " is already open in the process. It needs to be closed first"
	   << LogIO::POST;
	  return False;
      } else {
	Table table(tablename, Table::Update);
	if (table.isMultiUsed()) {
	  os << LogIO::SEVERE << "Table " << tablename
	     << " is already open in another process. It needs to be closed first"
	     << LogIO::POST;
	    return False;
	} else {
	  Table table(tablename, Table::Delete);
	}
      }
    }
  }
  return True;
}

Bool Imager::createSkyEquation(const String complist) 
{
  Vector<String> image;
  Vector<String> mask;
  Vector<String> fluxMask;
  Vector<Bool> fixed;
  return createSkyEquation(image, fixed, mask, fluxMask, complist);
}

Bool Imager::createSkyEquation(const Vector<String>& image,
			       const String complist) 
{
  Vector<Bool> fixed(image.nelements()); fixed=False;
  Vector<String> mask(image.nelements()); mask="";
  Vector<String> fluxMask(image.nelements()); fluxMask="";
  return createSkyEquation(image, fixed, mask, fluxMask, complist);
}

Bool Imager::createSkyEquation(const Vector<String>& image, const Vector<Bool>& fixed,
			       const String complist) 
{
  Vector<String> mask(image.nelements()); mask="";
  Vector<String> fluxMask(image.nelements()); fluxMask="";
  return createSkyEquation(image, fixed, mask, fluxMask, complist);
}

Bool Imager::createSkyEquation(const Vector<String>& image, const Vector<Bool>& fixed,
			       const Vector<String>& mask,
			       const String complist) 
{
  Vector<String> fluxMask(image.nelements()); fluxMask="";
  return createSkyEquation(image, fixed, mask, fluxMask, complist);
}

Bool Imager::createSkyEquation(const Vector<String>& image,
			       const Vector<Bool>& fixed,
			       const Vector<String>& mask,
			       const Vector<String>& fluxMask,
			       const String complist)
{
 
  if(!valid()) return False;

  LogIO os(LogOrigin("imager", "createSkyEquation()", WHERE));
  
  // If there is no sky model, we'll make one:

  if(sm_p==0) {
    if((facets_p >1)){
      // Support serial and parallel specializations
      setWFCleanImageSkyModel();
    }
    else if(ntaylor_p>1){
      // Init the msmfs sky-model so that Taylor weights are triggered in CubeSkyEqn
      sm_p = new WBCleanImageSkyModel(ntaylor_p, 1 ,reffreq_p);
    }
    else {
      sm_p = new CleanImageSkyModel();
    }
  }
  AlwaysAssert(sm_p, AipsError);

  // Add the componentlist
  if(complist!="") {
    if(!Table::isReadable(complist)) {
      os << LogIO::SEVERE << "ComponentList " << complist
	 << " not readable" << LogIO::POST;
      return False;
    }
    componentList_p=new ComponentList(complist, True);
    if(componentList_p==0) {
      os << LogIO::SEVERE << "Cannot create ComponentList from " << complist
	 << LogIO::POST;
      return False;
    }
    if(!sm_p->add(*componentList_p)) {
      os << LogIO::SEVERE << "Cannot add ComponentList " << complist
	 << " to SkyModel" << LogIO::POST;
      return False;
    }
    os << LogIO::NORMAL // Loglevel INFO
       << "Processing after subtracting componentlist " << complist << LogIO::POST;
  }
  else {
    componentList_p=0;
  }
 
  // Make image with the required shape and coordinates only if
  // they don't exist yet
  nmodels_p=image.nelements();

  // Remove degenerate case (due to user interface?)
  if((nmodels_p==1)&&(image(0)=="")) {
    nmodels_p=0;
  }
  if(nmodels_p>0) {
    images_p.resize(nmodels_p); 
    masks_p.resize(nmodels_p);  
    fluxMasks_p.resize(nmodels_p); 
    residuals_p.resize(nmodels_p); 
    for (Int model=0;model<Int(nmodels_p);++model) {
      if(image(model)=="") {
	os << LogIO::SEVERE << "Need a name for model "
	   << model << LogIO::POST;
	return False;
      }
      else {
	if(!Table::isReadable(image(model))) {
	  if(!assertDefinedImageParameters()) return False;
	  make(image(model));
	}
      }
      images_p[model]=0;
      images_p[model]=new PagedImage<Float>(image(model));
      AlwaysAssert(!images_p[model].null(), AipsError);

      //Determining the number of XFR
      Int numOfXFR=nmodels_p+1;
      if(datafieldids_p.nelements() >0)
	numOfXFR=datafieldids_p.nelements()*nmodels_p + 1;
      if(squintType_p != BeamSquint::NONE){
	if(parAngleInc_p.getValue("deg") >0 ){
	  numOfXFR= numOfXFR* Int(360/parAngleInc_p.getValue("deg"));
	}	
	else{
	numOfXFR= numOfXFR*10;
	}
      }
      if((sm_p->add(*images_p[model], numOfXFR))!=model) {
	os << LogIO::SEVERE << "Error adding model " << model << LogIO::POST;
	return False;
      }
 
      fluxMasks_p[model]=0;
      if(fluxMask(model)!=""&&Table::isReadable(fluxMask(model))) {
	fluxMasks_p[model]=new PagedImage<Float>(fluxMask(model));
	AlwaysAssert(!fluxMasks_p[model].null(), AipsError);
        if(!sm_p->addFluxMask(model, *fluxMasks_p[model])) {
	  os << LogIO::SEVERE << "Error adding flux mask " << model
	     << " : " << fluxMask(model) << LogIO::POST;
	  return False;
	}
      }
      residuals_p[model]=0;
    }
    addMasksToSkyEquation(mask, fixed);
  }
  
  // Always need a VisSet and an FTMachine
  if (!ft_p)
    createFTMachine();
  
  // Now set up the SkyEquation
  AlwaysAssert(sm_p, AipsError);
  // AlwaysAssert(vs_p, AipsError);
  AlwaysAssert(rvi_p, AipsError);
  AlwaysAssert(ft_p, AipsError);
  AlwaysAssert(cft_p, AipsError);

  // This block determines which SkyEquation is to be used.
  // We are using a mf* algorithm and there is more than one image
  if (doMultiFields_p && multiFields_p) {
    // Mosaicing case
    if(doVP_p){
      //bypassing the minimum size FT stuff as its broken till its fixed
      //se_p=new MosaicSkyEquation(*sm_p, *vs_p, *ft_p, *cft_p);
      
      setSkyEquation();
      if(ft_p->name() != "MosaicFT") 
	sm_p->mandateFluxScale(0);
      os << LogIO::NORMAL // Loglevel INFO
         << "Mosaicing multiple fields with simple sky equation" << LogIO::POST;
    }
    // mosaicing with no vp correction
    else{
      setSkyEquation();
      os << LogIO::NORMAL // Loglevel INFO
         << "Processing multiple fields with simple sky equation" << LogIO::POST;
      os << LogIO::WARN
         << "Voltage Pattern is not set: will not correct for primary beam"
	 << LogIO::POST;
      doMultiFields_p=False;
    }
  }
  // We are not using an mf* algorithm or there is only one image
  else {
    // Support serial and parallel specializations
    if((facets_p >1)){
	setSkyEquation();
	//se_p=new SkyEquation(*sm_p, *vs_p, *ft_p, *cft_p);
	os << LogIO::NORMAL // Loglevel INFO
           << "Processing multiple facets with simple sky equation" << LogIO::POST;
    }
    // Mosaicing
    else if(doVP_p) {
      //Bypassing the mosaicskyequation to the slow version for now.
      //      se_p=new MosaicSkyEquation(*sm_p, *vs_p, *ft_p, *cft_p);
      
      setSkyEquation();
      if(ft_p->name() != "MosaicFT") 
	sm_p->mandateFluxScale(0);
      os << LogIO::NORMAL // Loglevel PROGRESS
         << "Mosaicing single field with simple sky equation" << LogIO::POST;      
    }
    // Default
    else {
      setSkyEquation();
      os << LogIO::DEBUG1
         << "Processing single field with simple sky equation" << LogIO::POST;    
    } 
  }
  //os.localSink().flush();
  //For now force none sault weighting with mosaic ft machine
  
  String scaleType=scaleType_p;
  if(ft_p->name()=="MosaicFT")
    scaleType="NONE";
  
  se_p->setImagePlaneWeighting(scaleType, minPB_p, constPB_p);

  AlwaysAssert(se_p, AipsError);

  // Now add any SkyJones that are needed
  if(doVP_p && (ft_p->name()!="MosaicFT")) {
    if (doDefaultVP_p) {
      vp_p=new VPSkyJones(*ms_p, True, parAngleInc_p, squintType_p, skyPosThreshold_p);
    } else { 
      Table vpTable( vpTableStr_p ); 
      vp_p=new VPSkyJones(*ms_p, vpTable, parAngleInc_p, squintType_p, skyPosThreshold_p);
    }
    se_p->setSkyJones(*vp_p);
  }
  else {
    vp_p=0;
  }
  return True;  
}

// Tell the sky model to use the specified images as the residuals    
Bool Imager::addResidualsToSkyEquation(const Vector<String>& imageNames) {
  
  residuals_p.resize(imageNames.nelements());
  for (Int thismodel=0;thismodel<Int(imageNames.nelements());++thismodel) {
    if(imageNames(thismodel)!="") {
      removeTable(imageNames(thismodel));
      residuals_p[thismodel]=
	new PagedImage<Float> (TiledShape(images_p[thismodel]->shape(), 
images_p[thismodel]->niceCursorShape()),
			       images_p[thismodel]->coordinates(),
			       imageNames(thismodel));
      AlwaysAssert(!residuals_p[thismodel].null(), AipsError);
      residuals_p[thismodel]->setUnits(Unit("Jy/beam"));
      sm_p->addResidual(thismodel, *residuals_p[thismodel]);
    }
  }
  return True;
} 

void Imager::destroySkyEquation() 
{
  if(se_p) delete se_p; se_p=0;
  if(sm_p) delete sm_p; sm_p=0;
  if(vp_p) delete vp_p; vp_p=0;
  if(gvp_p) delete gvp_p; gvp_p=0;
  
  if(componentList_p) delete componentList_p; componentList_p=0;
 
  
  for (Int model=0;model<Int(nmodels_p); ++model) {
    //As these are CountedPtrs....just assigning them to NULL 
    //get the objects out of context
    if(!images_p[model].null())  
      images_p[model]=0;
    
    if(!masks_p[model].null()) 
      masks_p[model]=0;
 
   if(!fluxMasks_p[model].null()) 
     fluxMasks_p[model]=0;
    
   if(!residuals_p[model].null()) 
     residuals_p[model]=0;
  }
  
  redoSkyModel_p=True;
}

Bool Imager::assertDefinedImageParameters() const
{
  LogIO os(LogOrigin("imager", "if(!assertDefinedImageParameters()", WHERE));
  if(!setimaged_p) { 
    os << LogIO::SEVERE << "Image parameters not yet set: use im.defineimage."
       << LogIO::POST;
    return False;
  }
  return True;
}

Bool Imager::valid() const {
  LogIO os(LogOrigin("imager", "if(!valid()) return False", WHERE));
  if(ms_p.null()) {
    os << LogIO::SEVERE << "Program logic error: MeasurementSet pointer ms_p not yet set"
       << LogIO::POST;
    return False;
  }
  if(mssel_p.null()) {
    os << LogIO::SEVERE << "Program logic error: MeasurementSet pointer mssel_p not yet set"
       << LogIO::POST;
    return False;
  }
  if(!rvi_p) {
    os << LogIO::SEVERE << "Program logic error: VisibilityIterator not yet set"
       << LogIO::POST;
    return False;
  }
  return True;
}


Bool Imager::addMasksToSkyEquation(const Vector<String>& mask, const Vector<Bool>& fixed){
  LogIO os(LogOrigin("imager", "addMasksToSkyEquation()", WHERE));

  for(Int model=0 ;model < nmodels_p; ++model){

    
    if((Int(fixed.nelements())>model) && fixed(model)) {
      os << LogIO::NORMAL // Loglevel INFO
         << "Model " << model << " will be held fixed" << LogIO::POST;
      sm_p->fix(model);
    }     
    /*
    if(!(masks_p[model].null())) delete masks_p[model];
    masks_p[model]=0;
    */
      if(mask(model)!=""&&Table::isReadable(mask(model))) {
	masks_p[model]=new PagedImage<Float>(mask(model));
	AlwaysAssert(!masks_p[model].null(), AipsError);
        if(!sm_p->addMask(model, *masks_p[model])) {
	  os << LogIO::SEVERE << "Error adding mask " << model
	     << " : " << mask(model) << LogIO::POST;
	  return False;
	}
      }
  }
  return True;
}
Bool Imager::makemodelfromsd(const String& sdImage, const String& modelImage, 
			     const String& lowPSF, String& maskImage)
{

 if(!valid()) return False;
  
  LogIO os(LogOrigin("imager", "makemodelfromsd()", WHERE));
  
  try {
    
    if(!Table::isReadable(sdImage)){
      os << LogIO::SEVERE  << "Single Dish " << sdImage 
	 << "  image is not readable" << LogIO::EXCEPTION;
      
      return False;
    }

    os << LogIO::NORMAL << "Creating an initial model image " << modelImage  // Loglevel INFO
       << " from single dish image " << sdImage << LogIO::POST;
    
    CoordinateSystem coordsys;
    imagecoordinates(coordsys);
    String modelName=modelImage;
    this->makeEmptyImage(coordsys, modelName, fieldid_p);
    
    PagedImage<Float> model(modelImage);
    PagedImage<Float> low0(sdImage);
    String sdObs=low0.coordinates().obsInfo().telescope();

    Vector<Quantum<Double> > lBeam;
    ImageInfo lowInfo=low0.imageInfo();
    lBeam=lowInfo.restoringBeam();
  
    Float beamFactor=-1.0;

    
    // regrid the single dish image
    {
      ImageRegrid<Float> ir;
      IPosition axes(3,0,1,3);   // if its a cube, regrid the spectral too
      ir.regrid(model, Interpolate2D::LINEAR, axes, low0);
    }
    
  

    // Will need to make a complex image to apply the beam
    TempImage<Complex> ctemp(model.shape(), model.coordinates());
    if(lowPSF=="") {
      os << LogIO::NORMAL // Loglevel INFO
         << "Using primary beam of single dish to determine flux scale"
         << LogIO::POST;

      TempImage<Float> beamTemp(model.shape(), model.coordinates());
      //Make the PB accordingly
      if((lBeam.nelements()==0) || 
	 (lBeam.nelements()>0)&&(lBeam(0).get("arcsec").getValue()==0.0)) {
      
	if (doDefaultVP_p) { 
	  if(telescope_p!=""){
	    ObsInfo myobsinfo=this->latestObsInfo();
	    myobsinfo.setTelescope(telescope_p);
	    coordsys.setObsInfo(myobsinfo);
	    
	  }
	  else{
	    if(sdObs != ""){
	      telescope_p=sdObs;
	      ObsInfo myobsinfo=this->latestObsInfo();
	      myobsinfo.setTelescope(telescope_p);
	      coordsys.setObsInfo(myobsinfo);
	    }
	    else{
	      telescope_p=coordsys.obsInfo().telescope();
	    }
	  }
	  beamTemp.setCoordinateInfo(coordsys);
	  this->makePBImage(beamTemp);
	 
	}
	else{
	  Table vpTable(vpTableStr_p);
	  this->makePBImage(vpTable, beamTemp);	
	}
	lBeam.resize(3);
	StokesImageUtil::FitGaussianPSF(beamTemp, lBeam(0), lBeam(1), lBeam(2));
	LatticeExprNode sumImage = sum(beamTemp);
	beamFactor=sumImage.getFloat();
	
      }
      
      
    }
    else {
      os << LogIO::NORMAL // Loglevel INFO
         << "Using specified low resolution PSF to determine sd flux scale"
         << LogIO::POST;
      // regrid the single dish psf
      PagedImage<Float> lowpsf0(lowPSF);
      TempImage<Float> lowpsf(model.shape(), model.coordinates());
      {
	ImageRegrid<Float> ir;
	IPosition axes(2,0,1);   //
	ir.regrid(lowpsf, Interpolate2D::LINEAR, axes, lowpsf0);
      }
      LatticeExprNode sumImage = sum(lowpsf);
      beamFactor=sumImage.getFloat();
      if((lBeam.nelements()>0)&&(lBeam(0).get("arcsec").getValue()==0.0)) {
	os << LogIO::NORMAL << "Finding SD beam from given PSF" << LogIO::POST; // Loglevel PROGRESS
	lBeam.resize(3);
	StokesImageUtil::FitGaussianPSF(lowpsf0, lBeam(0), lBeam(1), lBeam(2));
      }
    }
    

    // This factor comes from the beam volumes
    if(sdScale_p!=1.0)
      os << LogIO::DEBUG1
         << "Multiplying single dish data by user specified factor "
         << sdScale_p << LogIO::POST;
    Float sdScaling  = sdScale_p;
    if((lBeam(0).get("arcsec").getValue()>0.0)&&
       (lBeam(1).get("arcsec").getValue()>0.0)) {
      Int directionIndex=model.coordinates().findCoordinate(Coordinate::DIRECTION);
      DirectionCoordinate
	directionCoord=model.coordinates().directionCoordinate(directionIndex);
      Vector<String> units(2); units.set("arcsec");
      directionCoord.setWorldAxisUnits(units); 
      Vector<Double> incr= directionCoord.increment();
      if(beamFactor >0.0) {
	beamFactor=1.0/beamFactor;
      }
      else{
	//	beamFactor=
	//	  abs(incr(0)*incr(1))/(lBeam(0).get("arcsec").getValue()*lBeam(1).get("arcsec").getValue()*1.162);
	//Brute Force for now.
	IPosition imshape(4, nx_p, ny_p, 1, 1);
	TempImage<Float> lowpsf(imshape, coordsys);
	lowpsf.set(0.0);
	IPosition center(4, Int((nx_p/4)*2), Int((ny_p/4)*2),0,0);
        lowpsf.putAt(1.0, center);
	StokesImageUtil::Convolve(lowpsf, lBeam(0), lBeam(1),lBeam(2), False);
	LatticeExprNode sumImage = sum(lowpsf);
	beamFactor=1.0/sumImage.getFloat();

	
      }
      os << LogIO::NORMAL << "Beam volume factor  " // Loglevel INFO
	 <<  beamFactor << LogIO::POST;
      sdScaling*=beamFactor;
    }
    else {
      os << LogIO::WARN << "Insufficient information to scale correctly" << LogIO::POST;
    }
 
    //Convert to Jy/pixel
    model.copyData(  (LatticeExpr<Float>)((model * sdScaling)));
    model.setUnits(Unit("Jy/pixel"));
    
    //make a mask image
    this->makeEmptyImage(coordsys, maskImage, fieldid_p);
    PagedImage<Float> mask(maskImage);
    mask.set(1.0);
    ArrayLattice<Bool> sdMask(model.getMask());
    mask.copyData( LatticeExpr<Float> (mask* ntrue(sdMask)*model));
    StokesImageUtil::MaskFrom(mask, mask, Quantity(0.0, "Jy"));
    model.copyData( LatticeExpr<Float> (mask*model));
    return True;
  } catch (AipsError x) {
    os << LogIO::SEVERE << "Caught exception: " << x.getMesg()
       << LogIO::POST;
    return False;
  } 
  
  return True;


}


Bool Imager::openSubTables(){

 antab_p=Table(ms_p->antennaTableName(),
	       TableLock(TableLock::UserNoReadLocking));
 datadesctab_p=Table(ms_p->dataDescriptionTableName(),
	       TableLock(TableLock::UserNoReadLocking));
 feedtab_p=Table(ms_p->feedTableName(),
		 TableLock(TableLock::UserNoReadLocking));
 fieldtab_p=Table(ms_p->fieldTableName(),
		  TableLock(TableLock::UserNoReadLocking));
 obstab_p=Table(ms_p->observationTableName(),
		TableLock(TableLock::UserNoReadLocking));
 poltab_p=Table(ms_p->polarizationTableName(),
		TableLock(TableLock::UserNoReadLocking));
 proctab_p=Table(ms_p->processorTableName(),
		TableLock(TableLock::UserNoReadLocking));
 spwtab_p=Table(ms_p->spectralWindowTableName(),
		TableLock(TableLock::UserNoReadLocking));
 statetab_p=Table(ms_p->stateTableName(),
		TableLock(TableLock::UserNoReadLocking));

 if(Table::isReadable(ms_p->dopplerTableName()))
   dopplertab_p=Table(ms_p->dopplerTableName(),
		      TableLock(TableLock::UserNoReadLocking));

 if(Table::isReadable(ms_p->flagCmdTableName()))
   flagcmdtab_p=Table(ms_p->flagCmdTableName(),
		      TableLock(TableLock::UserNoReadLocking));
 if(Table::isReadable(ms_p->freqOffsetTableName()))
   freqoffsettab_p=Table(ms_p->freqOffsetTableName(),
			 TableLock(TableLock::UserNoReadLocking));

 if(ms_p->isWritable()){
   if(!(Table::isReadable(ms_p->historyTableName()))){
     // setup a new table in case its not there
     TableRecord &kws = ms_p->rwKeywordSet();
     SetupNewTable historySetup(ms_p->historyTableName(),
				MSHistory::requiredTableDesc(),Table::New);
     kws.defineTable(MS::keywordName(MS::HISTORY), Table(historySetup));
     
   }
   historytab_p=Table(ms_p->historyTableName(),
		      TableLock(TableLock::UserNoReadLocking), Table::Update);
 }
 if(Table::isReadable(ms_p->pointingTableName()))
   pointingtab_p=Table(ms_p->pointingTableName(), 
		       TableLock(TableLock::UserNoReadLocking));
 
 if(Table::isReadable(ms_p->sourceTableName()))
   sourcetab_p=Table(ms_p->sourceTableName(),
		     TableLock(TableLock::UserNoReadLocking));

 if(Table::isReadable(ms_p->sysCalTableName()))
 syscaltab_p=Table(ms_p->sysCalTableName(),
		   TableLock(TableLock::UserNoReadLocking));
 if(Table::isReadable(ms_p->weatherTableName()))
   weathertab_p=Table(ms_p->weatherTableName(),
		      TableLock(TableLock::UserNoReadLocking));
 if(ms_p->isWritable()){
   hist_p= new MSHistoryHandler(*ms_p, "imager");
 }
return True;

}

Bool Imager::lock(){

  Bool ok; 
  ok=True;
  if(lockCounter_p == 0){

    ok= ok && (ms_p->lock());
    ok= ok && antab_p.lock(False);
    ok= ok && datadesctab_p.lock(False);
    ok= ok && feedtab_p.lock(False);
    ok= ok && fieldtab_p.lock(False);
    ok= ok && obstab_p.lock(False);
    ok= ok && poltab_p.lock(False);
    ok= ok && proctab_p.lock(False);
    ok= ok && spwtab_p.lock(False);
    ok= ok && statetab_p.lock(False);
    if(!dopplertab_p.isNull())
      ok= ok && dopplertab_p.lock(False);
    if(!flagcmdtab_p.isNull())
      ok= ok && flagcmdtab_p.lock(False);
    if(!freqoffsettab_p.isNull())
      ok= ok && freqoffsettab_p.lock(False);
    if(!historytab_p.isNull())
      ok= ok && historytab_p.lock(False);
    if(!pointingtab_p.isNull())
      ok= ok && pointingtab_p.lock(False);
    if(!sourcetab_p.isNull())
      ok= ok && sourcetab_p.lock(False);
    if(!syscaltab_p.isNull())
      ok= ok && syscaltab_p.lock(False);
    if(!weathertab_p.isNull())
      ok= ok && weathertab_p.lock(False);
 
  }
  ++lockCounter_p;

  return ok ; 
}

Bool Imager::unlock(){

  if(lockCounter_p==1){
    ms_p->unlock();
    antab_p.unlock();
    datadesctab_p.unlock();
    feedtab_p.unlock();
    fieldtab_p.unlock();
    obstab_p.unlock();
    poltab_p.unlock();
    proctab_p.unlock();
    spwtab_p.unlock();
    statetab_p.unlock();
    if(!dopplertab_p.isNull())
      dopplertab_p.unlock();
    if(!flagcmdtab_p.isNull())
      flagcmdtab_p.unlock();
    if(!freqoffsettab_p.isNull())
    freqoffsettab_p.unlock();
    if(!historytab_p.isNull())
      historytab_p.unlock();
    if(!pointingtab_p.isNull())
      pointingtab_p.unlock();
    if(!sourcetab_p.isNull())
      sourcetab_p.unlock();
    if(!syscaltab_p.isNull())
      syscaltab_p.unlock();
    if(!weathertab_p.isNull())
      weathertab_p.unlock();
  }
  for (Int thismodel=0;thismodel<Int(images_p.nelements());++thismodel) {
    if ((images_p.nelements() > uInt(thismodel)) && (!images_p[thismodel].null())) {
      images_p[thismodel]->table().relinquishAutoLocks(True);
      images_p[thismodel]->table().unlock();
    }
    if ((residuals_p.nelements()> uInt(thismodel)) && (!residuals_p[thismodel].null())) {
      residuals_p[thismodel]->table().relinquishAutoLocks(True);
      residuals_p[thismodel]->table().unlock();
    }
  }
  if(lockCounter_p > 0 )
    --lockCounter_p;
  return True ; 
}

Bool Imager::selectDataChannel(Vector<Int>& spectralwindowids, 
			       String& dataMode, 
			       Vector<Int>& dataNchan, 
			       Vector<Int>& dataStart, Vector<Int>& dataStep,
			       MRadialVelocity& mDataStart, 
			       MRadialVelocity& mDataStep){



  LogIO os(LogOrigin("Imager", "selectDataChannel()", WHERE));



  if(dataMode=="channel") {
      if (dataNchan.nelements() != spectralwindowids.nelements()){
	if(dataNchan.nelements()==1){
	  dataNchan.resize(spectralwindowids.nelements(), True);
	  for(uInt k=1; k < spectralwindowids.nelements(); ++k){
	    dataNchan[k]=dataNchan[0];
	  }
	}
	else{
	  os << LogIO::SEVERE 
	     << "Vector of nchan has to be of size 1 or be of the same shape as spw " 
	     << LogIO::POST;
	  return False; 
	}
      }
      if (dataStart.nelements() != spectralwindowids.nelements()){
	if(dataStart.nelements()==1){
	  dataStart.resize(spectralwindowids.nelements(), True);
	  for(uInt k=1; k < spectralwindowids.nelements(); ++k){
	    dataStart[k]=dataStart[0];
	  }
	}
	else{
	  os << LogIO::SEVERE 
	     << "Vector of start has to be of size 1 or be of the same shape as spw " 
	     << LogIO::POST;
	  return False; 
	}
      }
      if (dataStep.nelements() != spectralwindowids.nelements()){
	if(dataStep.nelements()==1){
	  dataStep.resize(spectralwindowids.nelements(), True);
	  for(uInt k=1; k < spectralwindowids.nelements(); ++k){
	    dataStep[k]=dataStep[0];
	  }
	}
	else{
	  os << LogIO::SEVERE 
	     << "Vector of step has to be of size 1 or be of the same shape as spw " 
	     << LogIO::POST;
	  return False; 
	}
      }

      if(spectralwindowids.nelements()>0) {
	Int nch=0;
	for(uInt i=0;i<spectralwindowids.nelements();++i) {
	  Int spwid=spectralwindowids(i);
	  Int numberChan=rvi_p->msColumns().spectralWindow().numChan()(spwid);
	  if(dataStart[i]<0) {
	    os << LogIO::SEVERE << "Illegal start pixel = " 
	       << dataStart[i]  << " for spw " << spwid
	       << LogIO::POST;
	    return False;
	  }
	 
	  if(dataNchan[i]<=0){ 
	    if(dataStep[i] <= 0)
	      dataStep[i]=1;
	    nch=(numberChan-dataStart[i])/Int(dataStep[i])+1;
	  }
	  else nch = dataNchan[i];
	  while((nch*dataStep[i]+dataStart[i]) > numberChan){
	    --nch;
	  }
	  Int end = Int(dataStart[i]) + Int(nch-1) * Int(dataStep[i]);
	  if(end < 0 || end > (numberChan)-1) {
	    os << LogIO::SEVERE << "Illegal step pixel = " << dataStep[i]
	       << " for spw " << spwid
	       << "\n end channel " << end 
	       << " is out of range " << dataStart[i] << " to " 
	       << (numberChan-1)
	       << LogIO::POST;
	    return False;
	  }

          os << LogIO::NORMAL // Too contentious for DEBUG1
             << "Selecting ";
          if(nch > 1)
            os << nch << " channels, starting at "
               << dataStart[i]  << ", stepped by " << dataStep[i] << ",";
          else
            os << "channel " << dataStart[i];
          os << " for spw " << spwid << LogIO::POST;
	  rvi_p->selectChannel(1, Int(dataStart[i]), Int(nch),
				     Int(dataStep[i]), spwid);
	  dataNchan[i]=nch;
	}
      }	else {
	Int numberChan=rvi_p->msColumns().spectralWindow().numChan()(0);
	if(dataNchan[0]<=0){
	  if(dataStep[0] <=0)
	    dataStep[0]=1;
	  dataNchan[0]=(numberChan-dataStart[0])/Int(dataStep[0])+1;
	  
	}
	while((dataNchan[0]*dataStep[0]+dataStart[0]) > numberChan)
	  --dataNchan[0];

	Int end = Int(dataStart[0]) + Int(dataNchan[0]-1) 
	  * Int(dataStep[0]);
	if(end < 0 || end > (numberChan)-1) {
	  os << LogIO::SEVERE << "Illegal step pixel = " << dataStep[0]
	     << "\n end channel " << end << " is out of range 1 to " 
	     << (numberChan-1)
	     << LogIO::POST;
	  return False;
	}
	os << LogIO::NORMAL << "Selecting "<< dataNchan[0] // Loglevel INFO
	   << " channels, starting at visibility channel "
	 << dataStart[0]  << " stepped by "
	   << dataStep[0] << LogIO::POST;
      }
  }
  else if (dataMode=="velocity") {
      MVRadialVelocity mvStart(mDataStart.get("m/s"));
      MVRadialVelocity mvStep(mDataStep.get("m/s"));
      MRadialVelocity::Types
	vType((MRadialVelocity::Types)mDataStart.getRefPtr()->getType());
      os << LogIO::NORMAL << "Selecting "<< dataNchan[0] // Loglevel INFO
	 << " channels, starting at radio velocity " << mvStart
	 << " stepped by " << mvStep << ", reference frame is "
	 << MRadialVelocity::showType(vType) << LogIO::POST;
      rvi_p->selectVelocity(Int(dataNchan[0]), mvStart, mvStep,
				  vType, MDoppler::RADIO);
  }
  else if (dataMode=="opticalvelocity") {
      MVRadialVelocity mvStart(mDataStart.get("m/s"));
      MVRadialVelocity mvStep(mDataStep.get("m/s"));
      MRadialVelocity::Types
	vType((MRadialVelocity::Types)mDataStart.getRefPtr()->getType());
      os << LogIO::NORMAL << "Selecting "<< dataNchan[0] // Loglevel INFO
	 << " channels, starting at optical velocity " << mvStart
	 << " stepped by " << mvStep << ", reference frame is "
	 << MRadialVelocity::showType(vType) << LogIO::POST;
      rvi_p->selectVelocity(Int(dataNchan[0]), mvStart, mvStep,
				  vType, MDoppler::OPTICAL);
  }

  return True;

}


Bool Imager::checkCoord(const CoordinateSystem& coordsys,  
			const String& imageName){ 

  PagedImage<Float> image(imageName);
  CoordinateSystem imageCoord= image.coordinates();
  Vector<Int> imageShape= image.shape().asVector();

  if(imageShape.nelements() > 3){
    if(imageShape(3) != imageNchan_p)
      return False;
  }
  else{
    if(imageNchan_p >1)
      return False;
  }

  if(imageShape.nelements() > 2){
    if(imageShape(2) != npol_p)
      return False;
  } 
  else{
    if(npol_p > 1)
      return False;
  }
  if(imageShape(0) != nx_p)
    return False;
  if(imageShape(1) != ny_p)
    return False;

  if (imageCoord.nCoordinates() != coordsys.nCoordinates())
    return False;
  DirectionCoordinate dir1(coordsys.directionCoordinate(0));
  DirectionCoordinate dir2(imageCoord.directionCoordinate(0));
  if(dir1.increment()(0) != dir2.increment()(0))
    return False;
  if(dir1.increment()(1) != dir2.increment()(1))
    return False;
  SpectralCoordinate sp1(coordsys.spectralCoordinate(2));
  SpectralCoordinate sp2(imageCoord.spectralCoordinate(2));
  if(sp1.increment()(0) != sp2.increment()(0))
    return False;

  return True;
}

void Imager::setImageParam(Int& nx, Int& ny, Int& npol, Int& nchan){

  nx_p=nx;
  ny_p=ny;
  npol_p=npol;
  nchan_p=nchan;

}

void Imager::makeVisSet(MeasurementSet& ms, 
			Bool compress, Bool mosaicOrder){

  if(rvi_p) {
    delete rvi_p;
    rvi_p=0;
    wvi_p=0;
  }

  Block<Int> sort(0);
  if(mosaicOrder){
    sort.resize(4);
    sort[0] = MS::FIELD_ID;
    sort[1] = MS::ARRAY_ID;
    sort[2] = MS::DATA_DESC_ID;
    sort[3] = MS::TIME;
 
  }
  //else use default sort order
  else{
    sort.resize(4);
    sort[0] = MS::ARRAY_ID;
    sort[1] = MS::FIELD_ID;
    sort[2] = MS::DATA_DESC_ID;
    sort[3] = MS::TIME;
  }
  Matrix<Int> noselection;
  Double timeInterval=0;
  //if you want to use scratch col...make sure they are there
  if(useModelCol_p)
    VisSet(ms,sort,noselection,useModelCol_p,timeInterval,compress);
  
  if(imwgt_p.getType()=="none"){
      imwgt_p=VisImagingWeight("natural");
  }

  if(!useModelCol_p){
    rvi_p=new ROVisibilityIterator(ms, sort, timeInterval);
  }
  else{
    wvi_p=new VisibilityIterator(ms, sort, timeInterval);
    rvi_p=wvi_p;    
  }
  rvi_p->useImagingWeight(imwgt_p);

}
/*
void Imager::makeVisSet(MeasurementSet& ms, 
			Bool compress, Bool mosaicOrder){


  Block<Int> sort(0);
  if(mosaicOrder){
    sort.resize(4);
    sort[0] = MS::FIELD_ID;
    sort[1] = MS::ARRAY_ID;
    sort[2] = MS::DATA_DESC_ID;
    sort[3] = MS::TIME;
  }
  //else use default sort order
  else{
  
    sort.resize(4);
    sort[0] = MS::ARRAY_ID;
    sort[1] = MS::FIELD_ID;
    sort[2] = MS::DATA_DESC_ID;
    sort[3] = MS::TIME;
  }
  Matrix<Int> noselection;
  Double timeInterval=0;

  VisSet vs(ms,sort,noselection,useModelCol_p,timeInterval,compress);

}
*/

void Imager::writeHistory(LogIO& os){

  LogIO oslocal(LogOrigin("Imager", "writeHistory"));
  try{

    os.postLocally();
    if(!hist_p.null())
      hist_p->addMessage(os);
  }catch (AipsError x) {
    oslocal << LogIO::SEVERE << "Caught exception: " << x.getMesg()
	    << LogIO::POST;
  } 
}

void Imager::writeCommand(LogIO& os){

  LogIO oslocal(LogOrigin("Imager", "writeCommand"));
  try{
    os.postLocally();
    if(!hist_p.null())
      hist_p->cliCommand(os);
  }catch (AipsError x) {
    oslocal << LogIO::SEVERE << "Caught exception: " << x.getMesg()
	    << LogIO::POST;
  } 
}

Bool Imager::makePBImage(ImageInterface<Float>& pbImage, 
			 Bool useSymmetricBeam){

  LogIO os(LogOrigin("Imager", "makePBImage()", WHERE));
  CoordinateSystem imageCoord=pbImage.coordinates();
   Int spectralIndex=imageCoord.findCoordinate(Coordinate::SPECTRAL);
  SpectralCoordinate
    spectralCoord=imageCoord.spectralCoordinate(spectralIndex);
  Vector<String> units(1); units = "Hz";
  spectralCoord.setWorldAxisUnits(units);	
  Vector<Double> spectralWorld(1);
  Vector<Double> spectralPixel(1);
  spectralPixel(0) = 0;
  spectralCoord.toWorld(spectralWorld, spectralPixel);  
  Double freq  = spectralWorld(0);
  Quantity qFreq( freq, "Hz" );
  String telName=imageCoord.obsInfo().telescope();
  if(telName=="UNKNOWN"){
    os << LogIO::SEVERE << "Telescope encoded in image in not known " 
       << LogIO::POST;
	  return False;
  }

    
  PBMath myPB(telName, useSymmetricBeam, qFreq);
  return makePBImage(myPB, pbImage);

}

Bool Imager::makePBImage(const CoordinateSystem& imageCoord, 
			 const String& telescopeName, 
			 const String& diskPBName, 
			 Bool useSymmetricBeam, Double diam){

  LogIO os(LogOrigin("Imager", "makePBImage()", WHERE));
  Int spectralIndex=imageCoord.findCoordinate(Coordinate::SPECTRAL);
  SpectralCoordinate
    spectralCoord=imageCoord.spectralCoordinate(spectralIndex);
  Vector<String> units(1); units = "Hz";
  spectralCoord.setWorldAxisUnits(units);	
  Vector<Double> spectralWorld(1);
  Vector<Double> spectralPixel(1);
  spectralPixel(0) = 0;
  spectralCoord.toWorld(spectralWorld, spectralPixel);  
  Double freq  = spectralWorld(0);
  Quantity qFreq( freq, "Hz" );
  String telName=telescopeName;
  PBMath::CommonPB whichPB;
  PBMath::enumerateCommonPB(telName, whichPB);  
  PBMath myPB;
  if(whichPB!=PBMath::UNKNOWN && whichPB!=PBMath::NONE){
    
    myPB=PBMath(telName, useSymmetricBeam, qFreq);
  }
  else if(diam > 0.0){
    myPB=PBMath(diam,useSymmetricBeam, qFreq);
  }
  else{
    os << LogIO::WARN << "Telescope " << telName << " is not known\n "
       << "Not making the PB  image" 
       << LogIO::POST;
    return False; 
  }
  return makePBImage(imageCoord, myPB, diskPBName);

}

Bool Imager::makePBImage(const CoordinateSystem& imageCoord, 
			 const Table& vpTable, const String& diskPBName){
  ROScalarColumn<TableRecord> recCol(vpTable, (String)"pbdescription");
  PBMath myPB(recCol(0));
  return makePBImage(imageCoord, myPB, diskPBName);

}


Bool Imager::makePBImage(const Table& vpTable, ImageInterface<Float>& pbImage){
  ROScalarColumn<TableRecord> recCol(vpTable, (String)"pbdescription");
  PBMath myPB(recCol(0));
  return makePBImage(myPB, pbImage);

}

Bool Imager::makePBImage(const CoordinateSystem& imageCoord, PBMath& pbMath, 
			 const String& diskPBName){

  make(diskPBName);
  PagedImage<Float> pbImage(diskPBName);
  return makePBImage(pbMath, pbImage);
}


Bool Imager::makePBImage(PBMath& pbMath, ImageInterface<Float>& pbImage){

  CoordinateSystem imageCoord=pbImage.coordinates();
  pbImage.set(0.0);
  MDirection wcenter;  
  Int directionIndex=imageCoord.findCoordinate(Coordinate::DIRECTION);
  DirectionCoordinate
    directionCoord=imageCoord.directionCoordinate(directionIndex);

  IPosition imShape=pbImage.shape();
  //Vector<Double> pcenter(2);
  // pcenter(0) = imShape(0)/2;
  // pcenter(1) = imShape(1)/2;    
  //directionCoord.toWorld( wcenter, pcenter );
  VisBuffer vb(*rvi_p);
  Int fieldCounter=0;
  Vector<Int> fieldsDone;
  
  for(rvi_p->originChunks(); rvi_p->moreChunks(); rvi_p->nextChunk()){
    Bool fieldDone=False;
    for (uInt k=0;  k < fieldsDone.nelements(); ++k)
      fieldDone=fieldDone || (vb.fieldId()==fieldsDone(k));
    if(!fieldDone){
      ++fieldCounter;
      fieldsDone.resize(fieldCounter, True);
      fieldsDone(fieldCounter-1)=vb.fieldId();
      wcenter=vb.direction1()(0);
      TempImage<Float> pbTemp(imShape, imageCoord);
      TempImage<Complex> ctemp(imShape, imageCoord);
      ctemp.set(1.0);
      pbMath.applyPB(ctemp, ctemp, wcenter, Quantity(0.0, "deg"), BeamSquint::NONE);
      StokesImageUtil::To(pbTemp, ctemp);
      pbImage.copyData(  (LatticeExpr<Float>)(pbImage+pbTemp) );
    }
  }
  LatticeExprNode elmax= max( pbImage );
  Float fmax = abs(elmax.getFloat());
  //If there are multiple overlap of beam such that the peak is larger than 1 then normalize
  //otherwise leave as is
  if(fmax>1.0)
    pbImage.copyData((LatticeExpr<Float>)(pbImage/fmax));

  Float cutoffval=minPB_p;
  LatticeExpr<Bool> lemask(iif(pbImage < cutoffval, 
			       False, True));
  ImageRegion outreg=pbImage.makeMask("mask0", False, True);
  LCRegion& outmask=outreg.asMask();
  outmask.copyData(lemask);
  pbImage.defineRegion("mask0", outreg,RegionHandler::Masks, True); 
  pbImage.setDefaultMask("mask0");



  return True;
}
void Imager::setObsInfo(ObsInfo& obsinfo){

  latestObsInfo_p=obsinfo;
}

ObsInfo& Imager::latestObsInfo(){
  return latestObsInfo_p;
}

Bool Imager::makeEmptyImage(CoordinateSystem& coords, String& name, Int fieldID){

  Int tilex=32;
  if(imageTileVol_p >0){
    tilex=static_cast<Int>(ceil(sqrt(imageTileVol_p/min(4, npol_p)/min(32, imageNchan_p))));
    if(tilex >0){
      if(tilex > min(nx_p, ny_p))
	tilex=min(nx_p, ny_p);
      else
	tilex=nx_p/Int(nx_p/tilex);
    }    
    //Not too small in x-y tile
    if(tilex < 10)
      tilex=10;
  }
  IPosition tileShape(4, min(tilex, nx_p), min(tilex, ny_p),
		     min(4, npol_p), min(32, imageNchan_p));
  IPosition imageShape(4, nx_p, ny_p, npol_p, imageNchan_p);
  PagedImage<Float> modelImage(TiledShape(imageShape, tileShape), coords, name);
  modelImage.set(0.0);
  modelImage.table().markForDelete();
    
  // Fill in miscellaneous information needed by FITS
  ROMSColumns msc(*ms_p);
  Record info;
  String object=msc.field().name()(fieldID)
;  //defining object name
  String objectName=msc.field().name()(fieldID);
  ImageInfo iinfo=modelImage.imageInfo();
  iinfo.setObjectName(objectName);
  modelImage.setImageInfo(iinfo);
  String telescop=msc.observation().telescopeName()(0);
  info.define("OBJECT", object);
  info.define("TELESCOP", telescop);
  info.define("INSTRUME", telescop);
  info.define("distance", 0.0);
  modelImage.setMiscInfo(info);
  modelImage.table().tableInfo().setSubType("GENERIC");
  modelImage.setUnits(Unit("Jy/beam"));
  modelImage.table().unmarkForDelete();
  return True;
  
}

String Imager::frmtTime(const Double time) {
  MVTime mvtime(Quantity(time, "s"));
  Time t=mvtime.getTime();
  ostringstream os;
  os << t;
  return os.str();
}

Bool Imager::getRestFreq(Vector<Double>& restFreq, const Int& spw){
  // MS Doppler tracking utility
  MSDopplerUtil msdoppler(*ms_p);
  restFreq.resize();
  if(restFreq_p.getValue() > 0){// User defined restfrequency
    restFreq.resize(1);
    restFreq[0]=restFreq_p.getValue("Hz");
  }
  else{
    // Look up first rest frequency found (for now)

    Int fieldid = (datafieldids_p.nelements()>0 ? datafieldids_p(0) : 
		   fieldid_p);
    try{
      msdoppler.dopplerInfo(restFreq ,spw,fieldid);
    }
    catch(...){
      restFreq.resize();
    }
  }
  if(restFreq.nelements() >0) 
    return True;
  return False;
}


class interactive_clean_callback {
    public:
	interactive_clean_callback( ) { }
	casa::dbus::variant result( ) { return casa::dbus::toVariant(result_); }
	bool callback( const DBus::Message & msg );
    private:
	DBus::Variant result_;
};

bool interactive_clean_callback::callback( const DBus::Message &msg ) {
    if (msg.is_signal("edu.nrao.casa.viewer","interact")) {
	DBus::MessageIter ri = msg.reader( );
	::operator >>(ri,result_);
	casa::DBusSession::instance( ).dispatcher( ).leave( );
    }
    return true;
}

Int Imager::interactivemask(const String& image, const String& mask, 
			    Int& niter, Int& ncycles, String& thresh, const Bool forceReload){

  LogIO os(LogOrigin("Imager", "interactivemask()", WHERE));
   if(Table::isReadable(mask)) {
    if (! Table::isWritable(mask)) {
      os << LogIO::WARN << "Mask image is not modifiable " << LogIO::POST;
      return False;
    }
    //we should regrid here if image and mask do not match
   }
   else{
     clone(image, mask);
   }

   if ( viewer_p == 0 ) {
     viewer_p = dbus::launch<ViewerProxy>( );
     if ( viewer_p == 0 ) {
       os << LogIO::WARN << "failed to launch viewer gui" << LogIO::POST;
       return False;
     }
   }
   if ( clean_panel_p == 0) {
     dbus::variant panel_id = viewer_p->panel( "clean" );
     if ( panel_id.type() != dbus::variant::INT ) {
       os << LogIO::WARN << "failed to create clean panel" << LogIO::POST;
       return False;
     }
     clean_panel_p = panel_id.getInt( );
   }

   if ( image_id_p == 0 || mask_id_p == 0 || forceReload ) {
     //Make sure image left after a "no more" is pressed is cleared
     if(forceReload && image_id_p !=0)
       prev_image_id_p=image_id_p;
     if(forceReload && mask_id_p !=0)
       prev_mask_id_p=mask_id_p;
     if(prev_image_id_p){
       viewer_p->unload( prev_image_id_p );
     }
     if(prev_mask_id_p)
       viewer_p->unload( prev_mask_id_p );
     prev_image_id_p=0;
     prev_mask_id_p=0;
     dbus::variant image_id = viewer_p->load(image, "raster", clean_panel_p);
     if ( image_id.type() != dbus::variant::INT ) {
       os << LogIO::WARN << "failed to load image" << LogIO::POST;
       return False;
     }
     image_id_p = image_id.getInt( );
     
     dbus::variant mask_id = viewer_p->load(mask, "contour", clean_panel_p);
      if ( mask_id.type() != dbus::variant::INT ) {
	os << "failed to load mask" << LogIO::WARN << LogIO::POST;
	return False;
      }
      mask_id_p = mask_id.getInt( );
   } else {
     //viewer_p->reload( clean_panel_p );
     viewer_p->reload(image_id_p);
     viewer_p->reload(mask_id_p);
   }

   
   casa::dbus::record options;
   options.insert("niter", niter);
   options.insert("ncycle", ncycles);
   options.insert("threshold", thresh);  
   viewer_p->setoptions(options, clean_panel_p);
   
    interactive_clean_callback *mycb = new interactive_clean_callback( );
    DBus::MessageSlot filter;
    filter = new DBus::Callback<interactive_clean_callback,bool,const DBus::Message &>( mycb, &interactive_clean_callback::callback );
    casa::DBusSession::instance( ).connection( ).add_filter( filter );
    casa::dbus::variant res = viewer_p->start_interact( dbus::variant(), clean_panel_p);
    casa::DBusSession::instance( ).dispatcher( ).enter( );
    casa::DBusSession::instance( ).connection( ).remove_filter( filter );
    casa::dbus::variant interact_result = mycb->result( );
    delete mycb;


    int result = 0;
    if ( interact_result.type() == dbus::variant::RECORD ) {
      const dbus::record  &rec = interact_result.getRecord( );
      for ( dbus::record::const_iterator iter = rec.begin(); iter != rec.end(); ++iter ) {
	if ( iter->first == "action" ) {
	  if ( iter->second.type( ) != dbus::variant::STRING ) {
	    os << "ill-formed action result" << LogIO::WARN << LogIO::POST;
	    return False;
	  } else {
	    const std::string &action = iter->second.getString( );
	    if ( action == "stop" )
	      result = 2;
	    else if ( action == "no more" )
	      result = 1;
	    else if ( action == "continue" )
	      result = 0;
	    else {
	      os << "ill-formed action result" << LogIO::WARN << LogIO::POST;
	      return False;
	    }
	  }
	} else if ( iter->first == "ncycle" ) {
	  if ( iter->second.type( ) != dbus::variant::INT ) {
	    os << "ill-formed ncycle result" << LogIO::WARN << LogIO::POST;
	    return False;
	  } else {
	    ncycles = iter->second.getInt( );
	  }
	} else if ( iter->first == "niter" ) {
	  if ( iter->second.type( ) != dbus::variant::INT ) {
	    os << "ill-formed niter result" << LogIO::WARN << LogIO::POST;
	    return False;
	  } else {
	    niter = iter->second.getInt( );
	  }
	} else if ( iter->first == "threshold" ) {
	  if ( iter->second.type( ) != dbus::variant::STRING ) {
	    os << "ill-formed threshold result" << LogIO::WARN << LogIO::POST;
	    return False;
	  } else {
	    thresh = iter->second.getString( );
	  }
	}
      }
    } else {
      os << "failed to get a vaild result for viewer" << LogIO::WARN << LogIO::POST;
      return False;
    }
    prev_image_id_p=image_id_p;
    prev_mask_id_p=mask_id_p;

    if(result==1){
      //Keep the image up but clear the next time called
      image_id_p=0;
      mask_id_p=0;
    }
    if(result==2){
      //clean up
      //viewer_p->close(clean_panel_p);
      //viewer_p->done();
      //delete viewer_p;
      //viewer_p=0;
      //viewer_p->unload(image_id_p);
      //viewer_p->unload(mask_id_p);
      //Setting clean_panel_p to 0 seems to do the trick...the above stuff 
      // like done causes a crash after a call again...have to understand that
      viewer_p->close(clean_panel_p);
      clean_panel_p=0;
      image_id_p=0;
      mask_id_p=0;
    }

    // return 0 if "continue"
    // return 1 if "no more interaction"
    // return 2 if "stop"
    return result;
}

void Imager::setSkyEquation(){
  /*  if(sm_p->nmodels() >0){
    Long npix=0;
    for (Int model=0; model < sm_p->nmodels(); ++model){
      Long pixmod=sm_p->image(model).product();
      npix=max(pixmod, npix);
    }
    Long pixInMem=(HostInfo::memoryTotal()/8)*1024;
    if(npix > (pixInMem/2)){
      se_p = new CubeSkyEquation(*sm_p, *vs_p, *ft_p, *cft_p, !useModelCol_p); 
      return;
    }
    //else lets make the base SkyEquation for now
  }
  
  se_p = new SkyEquation(*sm_p, *vs_p, *ft_p, *cft_p, !useModelCol_p);

  */
//  if (ft_p->name() == "PBWProjectFT")
//    {
//      logSink_p.clearLocally();
//      LogIO os(LogOrigin("imager", "setSkyEquation()"), logSink_p);
//      os << "Creating SkyEquation for PBWProjectFT" << LogIO::POST;
//     se_p = new SkyEquation(*sm_p, *vs_p, *ft_p, *cft_p, !useModelCol_p);
//    }
//  else
  se_p = new CubeSkyEquation(*sm_p, *rvi_p, *ft_p, *cft_p, !useModelCol_p);
  return;
}

void Imager::savePSF(const Vector<String>& psf){

  if( (psf.nelements() > 0) && (Int(psf.nelements()) <= sm_p->numberOfModels())){

    for (Int thismodel=0;thismodel<Int(psf.nelements());++thismodel) {
      if(removeTable(psf(thismodel))) {
	PagedImage<Float> psfimage(images_p[thismodel]->shape(),
				   images_p[thismodel]->coordinates(),
				   psf(thismodel));
	psfimage.copyData(sm_p->PSF(thismodel));
      }
    }



  }  

}

void Imager::setMosaicFTMachine(Bool useDoublePrec){
  LogIO os(LogOrigin("Imager", "setmosaicftmachine", WHERE));
  ROMSColumns msc(*ms_p);
  String telescop=msc.observation().telescopeName()(0);
  PBMath::CommonPB kpb;
  PBMath::enumerateCommonPB(telescop, kpb);
  if(!((kpb == PBMath::UNKNOWN) || 
       (kpb==PBMath::OVRO) || (kpb==PBMath::ALMA) || (kpb==PBMath::ACA))){
    
    if(!gvp_p) {
      if (doDefaultVP_p) {
	os << LogIO::NORMAL // Loglevel INFO
           << "Using defaults for primary beams used in gridding" << LogIO::POST;
	gvp_p=new VPSkyJones(*ms_p, True, parAngleInc_p, squintType_p,
                             skyPosThreshold_p);
      } else {
	os << LogIO::NORMAL // Loglevel INFO
           << "Using VP as defined in " << vpTableStr_p <<  LogIO::POST;
	Table vpTable( vpTableStr_p ); 
	gvp_p=new VPSkyJones(*ms_p, vpTable, parAngleInc_p, squintType_p,
                             skyPosThreshold_p);
      }
    } 
    gvp_p->setThreshold(minPB_p);
  }
  
  ft_p = new MosaicFT(gvp_p, mLocation_p, stokes_p, cache_p/2, tile_p, True, 
		      useDoublePrec);

  if((kpb == PBMath::UNKNOWN) || (kpb==PBMath::OVRO) || (kpb==PBMath::ACA)
     || (kpb==PBMath::ALMA)){
    os << LogIO::NORMAL // Loglevel INFO
       << "Using antenna diameters for determining beams for gridding"
       << LogIO::POST;
    CountedPtr<SimplePBConvFunc> mospb=new HetArrayConvFunc();
    static_cast<MosaicFT &>(*ft_p).setConvFunc(mospb);
  }
  
}

  Bool Imager::iClean(const String& algorithm, const Int niter, const Double gain, 
		      const Quantity& threshold,
		      const Bool displayprogress,
		      const Vector<String>& model,
		      const Vector<Bool>& keepfixed, const String& complist,
		      const Vector<String>& mask,
		      const Vector<String>& image,
		      const Vector<String>& residual,
		      const Vector<String>& psfnames,
		      const Bool interactive, const Int npercycle,
		      const String& masktemplate)
  {
    Bool rstat(False);
      
    logSink_p.clearLocally();
    LogIO os(LogOrigin("imager", "iClean()"), logSink_p);

    if(!ms_p.null()) {
      //try
       {
       
	Vector<String> amodel(model);
	Vector<Bool>   fixed(keepfixed);
	Vector<String> amask(mask);
	Vector<String> aimage(image);
	Vector<String> aresidual(residual);
	Vector<String> apsf(psfnames);
	
	if(String(algorithm) != "msmfs") ntaylor_p=1; /* masks increment by ntaylor_p only for msmfs */

	if( (apsf.nelements()==1) && apsf[0]==String(""))
	  apsf.resize();
	if(!interactive){
	  rstat = clean(String(algorithm), niter, gain,  
			threshold, displayprogress, 
			amodel, fixed, String(complist), amask,  
			aimage, aresidual, apsf);
	}
	else{
	  if((amask.nelements()==0) || (amask[0]==String(""))){
	    amask.resize(amodel.nelements());
	    for (uInt k=0; k < amask.nelements(); ++k){
	      amask[k]=amodel[k]+String(".mask");
	    }
	  }
	  Vector<Bool> nointerac(amodel.nelements());
	  nointerac.set(False);
	  if(fixed.nelements() != amodel.nelements()){
	    fixed.resize(amodel.nelements());
	    fixed.set(False);
	  }
	  Bool forceReload=True;
	  Int nloop=0;
	  if(npercycle != 0)
	    nloop=niter/npercycle;
	  Int continter=0;
	  Int elniter=npercycle;
	  ostringstream oos;
	  threshold.print(oos);
	  String thresh=String(oos);
	  if(String(masktemplate) != String("")){
	    continter=interactivemask(masktemplate, amask[0], 
						 elniter, nloop, thresh);
	  }
	  else {
	    // do a zero component clean to get started
	    clean(String(algorithm), 0, gain, 
		  threshold, displayprogress,
		  amodel, fixed, String(complist), amask,  
		  aimage, aresidual, Vector<String>(0), false);
	    for (uInt nIm=0; nIm < aresidual.nelements(); nIm+=ntaylor_p){
	      if(Table::isReadable(aimage[nIm]) && Table::isWritable(aresidual[nIm]) ){
		PagedImage<Float> rest(aimage[nIm]);
		PagedImage<Float> resi(aresidual[nIm]);
		copyMask(resi, rest, "mask0");
	      }
	      forceReload=forceReload || (aresidual.nelements() >1);
	      continter=interactivemask(aresidual[nIm], amask[nIm], 
						   elniter, nloop,thresh, forceReload);
	      forceReload=False;
	      if(continter>=1)
	        nointerac(nIm)=True;
	      if(continter==2)
	        fixed(nIm)=True;
	      
	    }
	    if(allEQ(nointerac, True)){
	      elniter=niter;
	      //make it do one more loop/clean but with all niter 
	      nloop=1;
	    }
	  }
	  for (Int k=0; k < nloop; ++k){
	    
	    casa::Quantity thrsh;
	    if(!casa::Quantity::read(thrsh, thresh)){
	      os << LogIO::WARN << "Error interpreting threshold" 
		      << LogIO::POST;
	      thrsh=casa::Quantity(0, "Jy");
	      thresh="0.0Jy";
	    }
	    Vector<String> elpsf(0);
	    //Need to save psfs in interactive only once and lets do it the 
	    //first time
	    if(k==0)
	      elpsf=apsf;
	    if(anyEQ(fixed, False)){
	      rstat = clean(String(algorithm), elniter, gain, 
			    thrsh, 
			    displayprogress,
			    amodel, fixed, String(complist), 
			    amask,  
			    aimage, aresidual, elpsf, k == 0);
	      //if clean converged... equivalent to stop
	      if(rstat){
		continter=2;
		fixed.set(True);
	      }
	      if(anyEQ(fixed, False) && anyEQ(nointerac,False)){
		Int remainloop=nloop-k-1;
		for (uInt nIm=0; nIm < aresidual.nelements(); nIm+=ntaylor_p){
		  if(!nointerac(nIm)){
		    continter=interactivemask(aresidual[nIm], amask[nIm],
					      
					      elniter, remainloop, 
					      thresh, (aresidual.nelements() >1));
		    if(continter>=1)
		      nointerac(nIm)=True;
		    if(continter==2)
		      fixed(nIm)=True;
		  }
		}
		k=nloop-remainloop-1;
		if(allEQ(nointerac,True)){
		  elniter=niter-(k+1)*npercycle;
		  //make it do one more loop/clean but with remaining niter 
		  k=nloop-2;
		}
	      } 
	    }
	  }
	  //Unset the mask in the residual 
	  // Cause as requested in CAS-1768...
	  for (uInt nIm=0; nIm < aresidual.nelements(); ++nIm){
	    if(Table::isWritable(aresidual[nIm]) ){
	      PagedImage<Float> resi(aresidual[nIm]);
	      if(resi.hasRegion("mask0", RegionHandler::Masks)){
		resi.setDefaultMask("");
	      }
	    }
	  }
	}
       } //catch  (AipsError x) {
       //os << LogIO::SEVERE << "Exception Reported: " << x.getMesg() << LogIO::POST;
       //	RETHROW(x);
       //  }
    } else {
      os << LogIO::SEVERE << "No MeasurementSet has been assigned, please run open." << LogIO::POST;
    }
    return rstat;
  }

} //# NAMESPACE CASA - END

