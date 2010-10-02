//# VisBuffer.cc: buffer for iterating through MS in large blocks
//# Copyright (C) 1996,1997,1998,1999,2000,2002,2003
//# Associated Universities, Inc. Washington DC, USA.
//#
//# This library is free software; you can redistribute it and/or modify it
//# under the terms of the GNU Library General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or (at your
//# option) any later version.
//#
//# This library is distributed in the hope that it will be useful, but WITHOUT
//# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
//# License for more details.
//#
//# You should have received a copy of the GNU Library General Public License
//# along with this library; if not, write to the Free Software Foundation,
//# Inc., 675 Massachusetts Ave, Cambridge, MA 02139, USA.
//#
//# Correspondence concerning AIPS++ should be addressed as follows:
//#        Internet email: aips2-request@nrao.edu.
//#        Postal address: AIPS++ Project Office
//#                        National Radio Astronomy Observatory
//#                        520 Edgemont Road
//#                        Charlottesville, VA 22903-2475 USA
//#
//# $Id$

#include <msvis/MSVis/VisibilityIterator.h>
#include <msvis/MSVis/VisBuffer.h>
#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/ArrayLogical.h>
#include <casa/Arrays/MaskedArray.h>
#include <casa/Arrays/MaskArrMath.h>
#include <casa/Arrays/ArrayUtil.h>
#include <casa/Utilities/Assert.h>
#include <casa/Utilities/GenSort.h>
#include <ms/MeasurementSets/MSColumns.h>

namespace casa { //# NAMESPACE CASA - BEGIN

VisBuffer::VisBuffer():visIter_p(static_cast<ROVisibilityIterator*>(0)),
twoWayConnection_p(False),corrSorted_p(False),This(this),nChannel_p(0),nRow_p(0)
{validate(); oldMSId_p=-1;}

VisBuffer::VisBuffer(ROVisibilityIterator& iter):visIter_p(&iter),This(this)
{ 
  iter.attachVisBuffer(*this); 
  twoWayConnection_p=True;
  oldMSId_p=-1;
  corrSorted_p=False;
}

VisBuffer::VisBuffer(const VisBuffer& vb):visIter_p(static_cast<ROVisibilityIterator*>(0)),
This(this)
{
  corrSorted_p=False;
  operator=(vb);
}

VisBuffer& VisBuffer::operator=(const VisBuffer& other)
{
  if (this!=&other) {
    assign(other);
    oldMSId_p=-1;
  }
  return *this;
}

VisBuffer& VisBuffer::assign(const VisBuffer& other, Bool copy)
{
  if (other.corrSorted_p)
    throw(AipsError("Cannot assign a VisBuffer that has had correlations sorted!"));
  
  if (this!=&other) {
    if (visIter_p!=static_cast<ROVisibilityIterator*>(0) && twoWayConnection_p) 
      visIter_p->detachVisBuffer(*this);
    visIter_p=other.visIter_p;
    oldMSId_p=other.oldMSId_p;
    twoWayConnection_p=False;
    if (visIter_p == static_cast<ROVisibilityIterator*>(0)) {
      validate();
    } else if (copy) {
      nCorrOK_p=other.nCorrOK_p;
      nChannelOK_p=other.nChannelOK_p;
      nRowOK_p=other.nRowOK_p;
      channelOK_p=other.channelOK_p;
      ant1OK_p=other.ant1OK_p;
      ant2OK_p=other.ant2OK_p;
      feed1OK_p=other.feed1OK_p;
      feed2OK_p=other.feed2OK_p;
      feed1_paOK_p=other.feed1_paOK_p;
      feed2_paOK_p=other.feed2_paOK_p;
      direction1OK_p=other.direction1OK_p;
      direction2OK_p=other.direction2OK_p;
      arrayIdOK_p=other.arrayIdOK_p;
      corrTypeOK_p=other.corrTypeOK_p;
      cjonesOK_p=other.cjonesOK_p;
      fieldIdOK_p=other.fieldIdOK_p;
      flagOK_p=other.flagOK_p;
      flagCubeOK_p=other.flagCubeOK_p;
      flagRowOK_p=other.flagRowOK_p;
      scanOK_p=other.scanOK_p;
      freqOK_p=other.freqOK_p;
      lsrFreqOK_p=other.lsrFreqOK_p;
      phaseCenterOK_p=other.phaseCenterOK_p;
      polFrameOK_p=other.polFrameOK_p;
      sigmaOK_p=other.sigmaOK_p;
      sigmaMatOK_p=other.sigmaMatOK_p;
      spwOK_p=other.spwOK_p;
      timeOK_p=other.timeOK_p;
      timeIntervalOK_p=other.timeIntervalOK_p;
      uvwOK_p=other.uvwOK_p;
      uvwMatOK_p=other.uvwMatOK_p;
      visOK_p=other.visOK_p;
      modelVisOK_p=other.modelVisOK_p;
      correctedVisOK_p=other.correctedVisOK_p;
      visCubeOK_p=other.visCubeOK_p;
      modelVisCubeOK_p=other.modelVisCubeOK_p;
      correctedVisCubeOK_p=other.correctedVisCubeOK_p;
      weightOK_p=other.weightOK_p;
      weightMatOK_p=other.weightMatOK_p;
      weightSpectrumOK_p=other.weightSpectrumOK_p;
      imagingWeightOK_p=other.imagingWeightOK_p;
      msOK_p=other.msOK_p;
      rowIdsOK_p=other.rowIdsOK_p;
    } else {
      invalidate();
    }
    if (copy) {
      if (nCorrOK_p) nCorr_p=other.nCorr_p;
      if (nChannelOK_p) nChannel_p=other.nChannel_p;
      if (nRowOK_p) nRow_p=other.nRow_p;
      if (channelOK_p) {
	channel_p.resize(other.channel_p.nelements()); 
	channel_p=other.channel_p;
      }
      if (ant1OK_p) {
	antenna1_p.resize(other.antenna1_p.nelements());
	antenna1_p=other.antenna1_p;
      }
      if (ant2OK_p) {
	antenna2_p.resize(other.antenna2_p.nelements());
	antenna2_p=other.antenna2_p;
      }
      if (feed1OK_p) {
	feed1_p.resize(other.feed1_p.nelements());
	feed1_p=other.feed1_p;
      }
      if (feed2OK_p) {
	feed2_p.resize(other.feed2_p.nelements());
	feed2_p=other.feed2_p;
      }
      if (feed1_paOK_p) {
        feed1_pa_p.resize(other.feed1_pa_p.nelements());
	feed1_pa_p=other.feed1_pa_p;
      }
      if (feed2_paOK_p) {
        feed2_pa_p.resize(other.feed2_pa_p.nelements());
	feed2_pa_p=other.feed2_pa_p;
      }
      if (direction1OK_p) {
	  direction1_p.resize(other.direction1_p.nelements());
	  direction1_p=other.direction1_p;
      }
      if (direction2OK_p) {
	  direction2_p.resize(other.direction2_p.nelements());
	  direction2_p=other.direction2_p;
      }
      if (corrTypeOK_p) {
	corrType_p.resize(other.corrType_p.nelements()); 
	corrType_p=other.corrType_p;
      }
      if (cjonesOK_p) {
	cjones_p.resize(other.cjones_p.nelements()); 
	cjones_p=other.cjones_p;
      }
      if (fieldIdOK_p) fieldId_p=other.fieldId_p;
      if (arrayIdOK_p) arrayId_p=other.arrayId_p;
      if (flagOK_p) {
	flag_p.resize(other.flag_p.shape()); 
	flag_p=other.flag_p;
      }
      if (flagCubeOK_p) {
	flagCube_p.resize(other.flagCube_p.shape()); 
	flagCube_p=other.flagCube_p;
      }
      if (flagRowOK_p) {
	flagRow_p.resize(other.flagRow_p.nelements());
	flagRow_p=other.flagRow_p;
      }
      if (scanOK_p) {
	scan_p.resize(other.scan_p.nelements());
	scan_p=other.scan_p;
      }
      if (freqOK_p) {
	frequency_p.resize(other.frequency_p.nelements()); 
	frequency_p=other.frequency_p;
      }
      if (lsrFreqOK_p) {
	lsrFrequency_p.resize(other.lsrFrequency_p.nelements()); 
	lsrFrequency_p=other.lsrFrequency_p;
      }
      if (phaseCenterOK_p) phaseCenter_p=other.phaseCenter_p;
      if (polFrameOK_p) polFrame_p=other.polFrame_p;
      if (sigmaOK_p) {
	sigma_p.resize(other.sigma_p.shape()); 
	sigma_p=other.sigma_p;
      }
      if (sigmaMatOK_p) {
	sigmaMat_p.resize(other.sigmaMat_p.shape()); 
	sigmaMat_p=other.sigmaMat_p;
      }
      if (spwOK_p) spectralWindow_p=other.spectralWindow_p;
      if (timeOK_p) {
	time_p.resize(other.time_p.nelements()); 
	time_p=other.time_p;
      }
      if (timeIntervalOK_p) {
	timeInterval_p.resize(other.timeInterval_p.nelements()); 
	timeInterval_p=other.timeInterval_p;
      }
      if (uvwOK_p) {
	uvw_p.resize(other.uvw_p.nelements()); 
	uvw_p=other.uvw_p;
      }
      if (uvwMatOK_p) {
	uvwMat_p.resize(other.uvwMat_p.shape()); 
	uvwMat_p=other.uvwMat_p;
      }
      if (visOK_p) {
	visibility_p.resize(other.visibility_p.shape());
	visibility_p=other.visibility_p;
      }
      if (modelVisOK_p) {
	modelVisibility_p.resize(other.modelVisibility_p.shape());
	modelVisibility_p=other.modelVisibility_p;
      }
      if (correctedVisOK_p) {
	correctedVisibility_p.resize(other.correctedVisibility_p.shape());
	correctedVisibility_p=other.correctedVisibility_p;
      }
      if (visCubeOK_p) {
	visCube_p.resize(other.visCube_p.shape());
	visCube_p=other.visCube_p;
      }
      if (modelVisCubeOK_p) {
	modelVisCube_p.resize(other.modelVisCube_p.shape());
	modelVisCube_p=other.modelVisCube_p;
      }
      if (correctedVisCubeOK_p) {
	correctedVisCube_p.resize(other.correctedVisCube_p.shape());
	correctedVisCube_p=other.correctedVisCube_p;
      }
      if (weightOK_p) {
	weight_p.resize(other.weight_p.nelements()); 
	weight_p=other.weight_p;
      }
      if (weightMatOK_p) {
	weightMat_p.resize(other.weightMat_p.shape()); 
	weightMat_p=other.weightMat_p;
      }
      if (weightSpectrumOK_p) {
	weightSpectrum_p.resize(other.weightSpectrum_p.shape()); 
	weightSpectrum_p=other.weightSpectrum_p;
      }
      if (imagingWeightOK_p) {
	imagingWeight_p.resize(other.imagingWeight_p.shape()); 
	imagingWeight_p=other.imagingWeight_p;
      }
      if (rowIdsOK_p){
	rowIds_p.resize(other.rowIds_p.shape());
	rowIds_p=other.rowIds_p;
      }
    }
  }
  return *this;
}
  
VisBuffer::~VisBuffer()
{ 
  if (visIter_p!=static_cast<ROVisibilityIterator*>(0) && twoWayConnection_p) 
    visIter_p->detachVisBuffer(*this);
}

VisBuffer& VisBuffer::operator-=(const VisBuffer& vb)
{
  // check the shapes
  AlwaysAssert(nRow_p==vb.nRow(),AipsError);
  AlwaysAssert(nChannel_p==vb.nChannel(),AipsError);
  AlwaysAssert(nCorr_p==vb.nCorr(),AipsError);
  // make sure flag and flagRow are current
  flag(); flagRow();

  // do the subtraction, or'ing the flags
  for (Int row=0; row<nRow_p; row++) {
    if (vb.flagRow()(row)) flagRow_p(row)=True;
    if (!flagRow_p(row)) {
      for (Int chn=0; chn<nChannel_p; chn++) {
	if (vb.flag()(chn,row)) flag_p(chn,row)=True;
	if (!flag_p(chn,row)) {
	  visibility_p(chn,row)-=vb.visibility()(chn,row);
	}
      }
    }
  }
  return *this;
}

void VisBuffer::attachToVisIter(ROVisibilityIterator& iter)
{ 
  if (visIter_p!=static_cast<ROVisibilityIterator*>(0) && twoWayConnection_p) 
    visIter_p->detachVisBuffer(*this);
  visIter_p=&iter; 
  iter.attachVisBuffer(*this);
  twoWayConnection_p=True;
}

void VisBuffer::invalidate()
{
  nChannelOK_p=channelOK_p=nRowOK_p=ant1OK_p=ant2OK_p=feed1OK_p=feed2OK_p=
    arrayIdOK_p=cjonesOK_p=fieldIdOK_p=flagOK_p=flagRowOK_p=scanOK_p=freqOK_p=
    lsrFreqOK_p=phaseCenterOK_p=polFrameOK_p=sigmaOK_p=sigmaMatOK_p=spwOK_p=
    timeOK_p=timeIntervalOK_p=uvwOK_p=uvwMatOK_p=visOK_p=weightOK_p=
    weightMatOK_p=weightSpectrumOK_p=corrTypeOK_p=nCorrOK_p=    False;
  flagCubeOK_p=visCubeOK_p=imagingWeightOK_p=msOK_p=False;
  modelVisOK_p=correctedVisOK_p=modelVisCubeOK_p=correctedVisCubeOK_p=False;
  feed1_paOK_p=feed2_paOK_p=direction1OK_p=direction2OK_p=rowIdsOK_p=False;
}

void VisBuffer::validate()
{
  nChannelOK_p=channelOK_p=nRowOK_p=ant1OK_p=ant2OK_p=feed1OK_p=feed2OK_p=
    arrayIdOK_p=cjonesOK_p=fieldIdOK_p=flagOK_p=flagRowOK_p=scanOK_p=freqOK_p=
    lsrFreqOK_p=phaseCenterOK_p=polFrameOK_p=sigmaOK_p=sigmaMatOK_p=spwOK_p=
    timeOK_p=timeIntervalOK_p=uvwOK_p=uvwMatOK_p=visOK_p=weightOK_p=
    weightMatOK_p=weightSpectrumOK_p=corrTypeOK_p=nCorrOK_p=    True;
  flagCubeOK_p=visCubeOK_p=imagingWeightOK_p=msOK_p=True;  
  modelVisOK_p=correctedVisOK_p=modelVisCubeOK_p=correctedVisCubeOK_p=True;
  feed1_paOK_p=feed2_paOK_p=direction1OK_p=direction2OK_p=rowIdsOK_p=True;
}

void VisBuffer::freqAverage() 
{
  Matrix<CStokesVector> newVisibility(1,nRow()); 
  Matrix<Bool> newFlag(1,nRow()); newFlag=True;
  Double newFrequency; newFrequency=0;
  Int nfreq=0;
  Int nChan=nChannel();
  for (Int row=0; row<nRow(); row++) {
    if (!flagRow()(row)) {
      Int n=0;
      for (Int chn=0; chn<nChan; chn++) {
	if (!flag()(chn,row)) {
	  newVisibility(0,row)+=visibility()(chn,row);
	  newFlag(0,row)=False;
	  newFrequency+=frequency()(chn);
	  n++; nfreq++;
	}
      }
      if (n==0) flagRow()(row)=True;
      if (n>0) newVisibility(0,row)*=1.0f/n;
    }
  }
  // Average frequency for this buffer (should really be row based)
  if (nfreq>0) newFrequency/=Double(nfreq);
  nChannel_p=1;
  flag_p.reference(newFlag);
  visibility_p.reference(newVisibility);
  frequency_p.resize(1); frequency_p(0)=newFrequency;
}

void VisBuffer::freqAveCubes() 
{
  // TBD: Use weightSpec, if present, and update weightMat accordingly
  // TBD: Provide partial decimation option

  // Ensure visCube filled, at least
  visCube();

  // Freq-averaged shape
  IPosition csh=visCube().shape();
  csh(1)=1;   // One channel in output

  Cube<Complex> newVisCube(csh); newVisCube=Complex(0.0);
  Matrix<Bool> newFlag(1,nRow()); newFlag=True;
  Double newFrequency; newFrequency=0;
  Int nfreq=0;
  Int nChan=nChannel();
  Int nCor=nCorr();
  for (Int row=0; row<nRow(); row++) {
    if (!flagRow()(row)) {
      Int n=0;
      for (Int chn=0; chn<nChan; chn++) {
	if (!flag()(chn,row)) {
	  newFlag(0,row)=False;
	  newFrequency+=frequency()(chn);
	  for (Int cor=0;cor<nCor;cor++) 
	    newVisCube(cor,0,row)+=visCube()(cor,chn,row);
	  n++; nfreq++;

	  if (row==-1)
	    cout << "V: " 
		 << chn << " " << n << " "
		 << visCube()(0,chn,row) << " "
		 << newVisCube(0,0,row) << " "
		 << endl;


	}
      }
      if (n==0) flagRow()(row)=True;
      if (n>0) {
	Matrix<Complex> nVC;
	nVC.reference(newVisCube.xyPlane(row));
	nVC*=Complex(1.0f/n);

	if (row==-1)
	  cout << "V:-----> " << n << " " << newVisCube(0,0,row) << endl;
	

      }
    }
  }
  visCube_p.reference(newVisCube);
  
  // Now do model, if present
  if (modelVisCubeOK_p) {

    Cube<Complex> newModelVisCube(csh); newModelVisCube=Complex(0.0);
    for (Int row=0; row<nRow(); row++) {
      if (!flagRow()(row)) {
	Int n=0;
	for (Int chn=0; chn<nChan; chn++) {
	  if (!flag()(chn,row)) {

	    n++; 
	    for (Int cor=0;cor<nCor;cor++) 
	      newModelVisCube(cor,0,row)+=modelVisCube()(cor,chn,row);

	    if (row==-1)
	      cout << "M: " 
		   << chn << " " << n << " "
		   << modelVisCube()(0,chn,row) << " "
		   << newModelVisCube(0,0,row) << " "
		   << endl;

	  }
	}
	if (n==0) flagRow()(row)=True;
	if (n>0) {
	  Matrix<Complex> nMVC;
	  nMVC.reference(newModelVisCube.xyPlane(row));
	  nMVC*=Complex(1.0f/n);

	  if (row==-1)
	    cout << "M:-----> " << n << " " << newModelVisCube(0,0,row) << endl;
	}
      }
    }
    modelVisCube_p.reference(newModelVisCube);
  }    

  // Use averaged flags
  flag_p.reference(newFlag);

  // Average frequency for this buffer 
  //  (Strictly, this should really be row based, but doing this
  //   average here suggests frequency precision isn't so important)
  if (nfreq>0) newFrequency/=Double(nfreq);
  nChannel_p=1;
  frequency_p.resize(1); frequency_p(0)=newFrequency;

}

void VisBuffer::formStokes() {

  // We must form the weights and flags correctly
  formStokesWeightandFlag();

  // Now do whatever data is present
  if (visCubeOK_p)
    formStokes(visCube_p);

  if (modelVisCubeOK_p)
    formStokes(modelVisCube_p);

  if (correctedVisCubeOK_p)
    formStokes(correctedVisCube_p);

}

void VisBuffer::formStokesWeightandFlag() {

  // Ensure corrType, weightMat and flagCube are filled 
  corrType();
  weightMat();
  flagCube();

  switch (nCorr()) {
  case 4: {

    Slice all=Slice();
    Slice pp(0,1,1),pq(1,1,1),qp(2,1,1),qq(3,1,1);
    Slice a(0,1,1),b(1,1,1),c(2,1,1),d(3,1,1);
    
    // Sort for linears
    if (polFrame()==MSIter::Linear) {
      d=Slice(1,1,1);  // Q
      b=Slice(2,1,1);  // U
      c=Slice(3,1,1);  // V
    }
    
    Matrix<Float> newWtMat(weightMat_p.shape());
    newWtMat(a,all)=newWtMat(d,all)=(weightMat_p(pp,all)+weightMat_p(qq,all));
    newWtMat(b,all)=newWtMat(c,all)=(weightMat_p(pq,all)+weightMat_p(qp,all));
    weightMat_p.reference(newWtMat);

    Cube<Bool> newFlagCube(flagCube_p.shape());
    newFlagCube(a,all,all)=newFlagCube(d,all,all)=(flagCube_p(pp,all,all)|flagCube_p(qq,all,all));
    newFlagCube(b,all,all)=newFlagCube(c,all,all)=(flagCube_p(pq,all,all)|flagCube_p(qp,all,all));
    flagCube_p.reference(newFlagCube);

    corrType_p(0)=Stokes::I;
    corrType_p(1)=Stokes::Q;
    corrType_p(2)=Stokes::U;
    corrType_p(3)=Stokes::V;

    break;
  }
  case 2: {
    // parallel hands only
    Slice all=Slice();
    Slice pp(0,1,1),qq(1,1,1);
    Slice a(0,1,1),d(1,1,1);
    
    Matrix<Float> newWtMat(weightMat_p.shape());
    newWtMat(a,all)=newWtMat(d,all)=weightMat_p(pp,all)+weightMat_p(qq,all);
    weightMat_p.reference(newWtMat);

    Cube<Bool> newFlagCube(flagCube_p.shape());
    newFlagCube(a,all,all)=newFlagCube(d,all,all)=flagCube_p(pp,all,all)|flagCube_p(qq,all,all);
    flagCube_p.reference(newFlagCube);

    corrType_p(0)=Stokes::I;
    corrType_p(1)=((polFrame()==MSIter::Circular) ? Stokes::V : Stokes::Q);

    break;
  }
  case 1: {

    // Just need to re-label as I
    corrType_p(0)=Stokes::I;

  }
  default: {
    cout << "Insufficient correlations to form Stokes" << endl;
    break;
  }
  }

}



void VisBuffer::formStokes(Cube<Complex>& vis) {

  Cube<Complex> newvis(vis.shape());
  newvis.set(0.0);
  Slice all=Slice();

  switch (nCorr()) {
  case 4: {

    Slice pp(0,1,1),pq(1,1,1),qp(2,1,1),qq(3,1,1);
    Slice a(0,1,1),b(1,1,1),c(2,1,1),d(3,1,1);
    
    if (polFrame()==MSIter::Linear) {
      d=Slice(1,1,1);  // Q
      b=Slice(2,1,1);  // U
      c=Slice(3,1,1);  // V
    }
    
    newvis(a,all,all)=(vis(pp,all,all)+vis(qq,all,all)); //  I / I
    newvis(d,all,all)=(vis(pp,all,all)-vis(qq,all,all)); //  V / Q
    
    newvis(b,all,all)=(vis(pq,all,all)+vis(qp,all,all)); //  Q / U
    newvis(c,all,all)=(vis(pq,all,all)-vis(qp,all,all))/Complex(0.0,1.0); //  U / V
    newvis/=Complex(2.0);

    vis.reference(newvis);

    break;
  }
  case 2: {
    // parallel hands only
    Slice pp(0,1,1),qq(1,1,1);
    Slice a(0,1,1),d(1,1,1);
    
    newvis(a,all,all)=(vis(pp,all,all)+vis(qq,all,all)); //  I / I
    newvis(d,all,all)=(vis(pp,all,all)-vis(qq,all,all)); //  V / Q
    newvis/=Complex(2.0);

    vis.reference(newvis);

    break;
  }
  case 1: {
    // need do nothing for single correlation case
    break;
  }
  default: {
    cout << "Insufficient correlations to form Stokes" << endl;
    break;
  }
  }
}

void VisBuffer::channelAve(const Matrix<Int>& chanavebounds) 
{
  // TBD: Use/update weightSpec, if present, and update weightMat accordingly

  //  Only do something if there is something to do
  if ( chanavebounds.nelements()>0 ) {

    // refer to the supplied chanavebounds
    chanAveBounds_p.reference(chanavebounds);

    //  cout << "chanAveBounds = " << chanAveBounds_p << endl;
    
    Int nChanOut(chanAveBounds_p.nrow());
    
    // Apply averaging to whatever data is present
    if (visCubeOK_p)          chanAveVisCube(visCube(),nChanOut);
    if (modelVisCubeOK_p)     chanAveVisCube(modelVisCube(),nChanOut);
    if (correctedVisCubeOK_p) chanAveVisCube(correctedVisCube(),nChanOut);
    if (flagCubeOK_p)         chanAveFlagCube(flagCube(),nChanOut);
    
    // Finally, collapse the frequency values themselves
    // TBD: move this up to bounds calculation loop?
    Vector<Double> newFreq(nChanOut,0.0);
    Vector<Int> newChan(nChanOut,0);
    frequency(); // Ensure frequencies pre-filled
    Vector<Int> chans(channel()); // Ensure channels pre-filled
    Int nChan0(chans.nelements());
    Int ichan=0;
    for (Int ochan=0;ochan<nChanOut;++ochan) {
      Int n=0;
      while (chans(ichan)>=chanAveBounds_p(ochan,0) &&
	     chans(ichan)<=chanAveBounds_p(ochan,1) &&
	     ichan<nChan0) {
	
	newFreq(ochan)+=frequency()(ichan);
	newChan(ochan)+=chans(ichan);
	ichan++;
	n++;
      }
      if (n>0) {
	newFreq(ochan)/=Double(n);
	newChan(ochan)/=n;
      }
    }
    
    // Install the new values
    frequency().reference(newFreq);
    channel().reference(newChan);
    nChannel()=nChanOut;

  }

}

void VisBuffer::chanAveVisCube(Cube<Complex>& data,Int nChanOut) {

  IPosition csh(data.shape());
  Int nChan0=csh(1);
  csh(1)=nChanOut;

  Vector<Int>& chans(channel());

  Cube<Complex> newCube(csh); newCube=Complex(0.0);
  Int nCor=nCorr();
  Int ichan(0);
  Vector<Int> ngood(nCor,0);
  for (Int row=0; row<nRow(); row++) {
    if (!flagRow()(row)) {
      ichan=0;
      for (Int ochan=0;ochan<nChanOut;++ochan) {
	ngood=0;
	while (chans(ichan)>=chanAveBounds_p(ochan,0) &&
	       chans(ichan)<=chanAveBounds_p(ochan,1) &&
	       ichan<nChan0) {
	  for (Int icor=0;icor<nCor;++icor) 
	    if (!flagCube()(icor,ichan,row)) {
	      newCube(icor,ochan,row)+=data(icor,ichan,row);
	      ngood(icor)++;
	    }
	  ichan++;
	}
	for (Int icor=0;icor<nCor;++icor) {
	  if (ngood(icor)>0) {
	    newCube(icor,ochan,row)*=Complex(1.0f/ngood(icor));
	  }
	}
      }
    }
  }

  // Install averaged info
  data.reference(newCube);

}

void VisBuffer::chanAveFlagCube(Cube<Bool>& flagcube,Int nChanOut) {

  IPosition csh(flagcube.shape());
  Int nChan0=csh(1);
  csh(1)=nChanOut;

  Vector<Int>& chans(channel());

  Cube<Bool> newFlag(csh); newFlag=True;
  Int nCor=nCorr();
  Int ichan(0);
  for (Int row=0; row<nRow(); row++) {
    if (!flagRow()(row)) {
      ichan=0;
      for (Int ochan=0;ochan<nChanOut;++ochan) {
	while (chans(ichan)>=chanAveBounds_p(ochan,0) &&
	       chans(ichan)<=chanAveBounds_p(ochan,1) &&
	       ichan<nChan0) {
	  for (Int icor=0;icor<nCor;++icor) 
	    if (!flagcube(icor,ichan,row)) 
	      newFlag(icor,ochan,row)=False;
	  ichan++;
	}
      }
    }
  }
  // Use averaged flags
  flagcube.reference(newFlag);

}




// Sort correlations: (PP,QQ,PQ,QP) -> (PP,PQ,QP,QQ)
void VisBuffer::sortCorr() {
 
  // This method is for temporarily sorting the correlations
  //  into canonical order if the MS data is out-of-order
  // NB: Always sorts the weightMat()
  // NB: Only works on the visCube-style data
  // NB: It only sorts the data columns which are already present
  //     (so make sure the ones you need are already read!)
  // NB: It is the user's responsibility to run unSortCorr
  //     after using the sorted data to put it back in order
  // NB: corrType_p is NOT changed to match the sorted
  //     correlations (it is expected that this sort is
  //     temporary, and that we will run unSortCorr
  // NB: This method does nothing if no sort required

  // If nominal order is non-canonical (only for nCorr=4)
  //   and data not yet sorted
  if (nonCanonCorr() && !corrSorted_p) {


    // First, do weightMat
    {
      weightMat();    // (ensures it is filled)
      
      Vector<Float> wtmp(nRow());
      Vector<Float> w1,w2,w3;
      IPosition wblc(1,0,0), wtrc(3,0,nRow()-1), vec(1,nRow());
      
      wblc(0)=wtrc(0)=1;
      w1.reference(weightMat_p(wblc,wtrc).reform(vec));
      wblc(0)=wtrc(0)=2;
      w2.reference(weightMat_p(wblc,wtrc).reform(vec));
      wblc(0)=wtrc(0)=3;
      w3.reference(weightMat_p(wblc,wtrc).reform(vec));
      wtmp=w1;
      w1=w2;
      w2=w3;
      w3=wtmp;
    }

    // Now do data:
    
    // Work space, references, coords
    Matrix<Complex> tmp(nChannel(),nRow());
    Matrix<Complex> p1,p2,p3;
    IPosition blc(3,0,0,0), trc(3,0,nChannel()-1,nRow()-1), mat(2,nChannel(),nRow());

    // Do visCube if present
    if (visCubeOK_p && visCube_p.nelements()>0) {
      blc(0)=trc(0)=1;
      p1.reference(visCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=2;
      p2.reference(visCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=3;
      p3.reference(visCube_p(blc,trc).reform(mat));
      tmp=p1;
      p1=p2;
      p2=p3;
      p3=tmp;
    }
    // Do modelVisCube if present
    if (modelVisCubeOK_p && modelVisCube_p.nelements()>0) {
      blc(0)=trc(0)=1;
      p1.reference(modelVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=2;
      p2.reference(modelVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=3;
      p3.reference(modelVisCube_p(blc,trc).reform(mat));
      tmp=p1;
      p1=p2;
      p2=p3;
      p3=tmp;
    }
    // Do correctedVisCube if present
    if (correctedVisCubeOK_p && correctedVisCube_p.nelements()>0) {
      blc(0)=trc(0)=1;
      p1.reference(correctedVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=2;
      p2.reference(correctedVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=3;
      p3.reference(correctedVisCube_p(blc,trc).reform(mat));
      tmp=p1;
      p1=p2;
      p2=p3;
      p3=tmp;
    }

    // Data is now sorted into canonical order
    corrSorted_p=True;
  }

}

// Unsort correlations: (PP,PQ,QP,QQ) -> (PP,QQ,PQ,QP) 
void VisBuffer::unSortCorr() {
  // This method is for restoring the non-canonical correlation
  //  sort order so that the data matches the order indicated
  //  by corrType()
  // NB: Always unsorts the weightMat()
  // NB: Only works on the visCube-style data
  // NB: It only unsorts the data columns which are already present
  //     (so make sure sortCorr sorted the ones you needed!)
  // NB: This method is a no-op if no sort required, or if
  //     sortCorr hadn't been run since the last unSortCorr

  // If nominal order is non-canonical (only for nCorr=4)
  //   and if data has been sorted
  if (nonCanonCorr() && corrSorted_p) {
    
    // First, do weights
    {
      Vector<Float> wtmp(nRow());
      Vector<Float> w1,w2,w3;
      IPosition wblc(1,0,0), wtrc(3,0,nRow()-1), vec(1,nRow());
      
      wblc(0)=wtrc(0)=1;
      w1.reference(weightMat_p(wblc,wtrc).reform(vec));
      wblc(0)=wtrc(0)=2;
      w2.reference(weightMat_p(wblc,wtrc).reform(vec));
      wblc(0)=wtrc(0)=3;
      w3.reference(weightMat_p(wblc,wtrc).reform(vec));
      wtmp=w3;
      w3=w2;
      w2=w1;
      w1=wtmp;
    }
    // Now do data:

    // Work space, references, coords
    Matrix<Complex> tmp(nChannel(),nRow());
    Matrix<Complex> p1,p2,p3;
    IPosition blc(3,0,0,0), trc(3,0,nChannel()-1,nRow()-1), mat(2,nChannel(),nRow());

    // Do visCube if present
    if (visCubeOK_p && visCube_p.nelements()>0) {
      blc(0)=trc(0)=1;
      p1.reference(visCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=2;
      p2.reference(visCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=3;
      p3.reference(visCube_p(blc,trc).reform(mat));
      tmp=p3;
      p3=p2;
      p2=p1;
      p1=tmp;
    }
    // Do modelVisCube if present
    if (modelVisCubeOK_p && modelVisCube_p.nelements()>0) {
      blc(0)=trc(0)=1;
      p1.reference(modelVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=2;
      p2.reference(modelVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=3;
      p3.reference(modelVisCube_p(blc,trc).reform(mat));
      tmp=p3;
      p3=p2;
      p2=p1;
      p1=tmp;
    }
    // Do correctedVisCube if present
    if (correctedVisCubeOK_p && correctedVisCube_p.nelements()>0) {
      blc(0)=trc(0)=1;
      p1.reference(correctedVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=2;
      p2.reference(correctedVisCube_p(blc,trc).reform(mat));
      blc(0)=trc(0)=3;
      p3.reference(correctedVisCube_p(blc,trc).reform(mat));
      tmp=p3;
      p3=p2;
      p2=p1;
      p1=tmp;
    }

    // Data is now back to corrType order
    corrSorted_p=False;
  }

}

Bool VisBuffer::nonCanonCorr() {
  Vector<Int>& corrs(corrType());
  // Only a meaningful question is all 4 corrs present
  if (corrs.nelements()==4)
    // (assumes corrs(0) is RR or XX)
    return (corrs(1)==Stokes::LL || corrs(1)==Stokes::YY);
  else
    // Assumed OK (fewer than 4 elements, or in canonical order already)
    return False;
}

// Fill weight matrix from sigma matrix
void VisBuffer::resetWeightMat() {
  
  // fill sigmaMat_p, size weightMat_p storage
  sigmaMat();   
  IPosition ip(sigmaMat_p.shape());
  weightMat_p.resize(ip);
  
  Int nPol(ip(0));
  Int nRow(ip(1));

  // Weight is inverse square of sigma (or zero[?])
  Float *w=weightMat_p.data();
  Float *s=sigmaMat_p.data();
  for (Int irow=0;irow<nRow;++irow)
    for (Int ipol=0;ipol<nPol;++ipol,++w,++s)
      if (*s >0.0f)
	*w = 1.0f/square(*s);
      else
	*w = 0.0f;

  // Scale by (unselected!) # of channels 
  //  (to stay aligned with original nominal weights)
  Int nchan=msColumns().spectralWindow().numChan()(spectralWindow());
  weightMat_p*=Float(nchan);

  // weightMat_p now OK
  weightMatOK_p=True;

}

// Rotate visibility phase for phase center offsets
void VisBuffer::phaseCenterShift(Double dx,Double dy) {

  // no-op if no net shift
  if (!(abs(dx)>0 || abs(dy)>0)) return;

  // Offsets in radians (input is arcsec)
  //   sign convention will _correct_ data
  dx*=(-C::pi/180.0/3600.0);
  dy*=(-C::pi/180.0/3600.0);

  // Extra path as fraction of U and V
  Vector<Double> udx;
  udx=uvwMat().row(0);
  Vector<Double> vdy;
  vdy=uvwMat().row(1);
  udx*=dx;    // in m
  vdy*=dy;

  // Combine axes
  udx+=vdy;
  udx/=C::c;  // in light-seconds
  udx*=(2.0*C::pi);  // in radian.seconds/cycle

  Vector<Double> freq(frequency());
  Double ph;
  Complex cph;

  for (Int irow=0;irow<nRow();++irow) 
    for (Int ichn=0;ichn<nChannel();++ichn) {
      // Calculate the Complex factor for this row and channel
      ph=udx(irow)*freq(ichn);
      cph=Complex(cos(ph),sin(ph));
      // Shift each correlation:
      for (Int icor=0;icor<nCorr();++icor) {
	if (visCubeOK_p)          visCube_p(icor,ichn,irow)*=cph;
	if (modelVisCubeOK_p)     modelVisCube_p(icor,ichn,irow)*=cph;
	if (correctedVisCubeOK_p) correctedVisCube_p(icor,ichn,irow)*=cph;
      }
    }
  
}



// Divide visCube by modelVisCube
void VisBuffer::normalize(const Bool& phaseOnly) {

  // NB: phase-only now handled by SolvableVisCal
  //   (we will remove phaseOnly parameter later)


  // NB: Handles pol-dep weights in chan-indep way
  // TBD: optimizations?
  // TBD: Handle channel-dep weights?

  // Only if all relevant columns are present
  if (visCubeOK_p && modelVisCubeOK_p && weightMatOK_p) {

    //    cout << "Normalizing!----------------------------" << endl;

    Int nCor=nCorr();

    // Amplitude data
    Float amp(1.0);
    Vector<Float> ampCorr(nCor);
    Vector<Int> n(nCor);

    Bool* flR=flagRow().data();
    Bool* fl =flag().data();

    for (Int irow=0;irow<nRow();irow++,flR++) {
      if (!*flR) {
	ampCorr=0.0f;
	n=0;
	for (Int ich=0;ich<nChannel();ich++,fl++) {
	  if (!*fl) {
	    for (Int icorr=0;icorr<nCor;icorr++) {
	      amp=abs(modelVisCube_p(icorr,ich,irow));
	      if (amp>0.0f) {
		visCube_p(icorr,ich,irow)=Complex( DComplex(visCube_p(icorr,ich,irow))/
						   DComplex(modelVisCube_p(icorr,ich,irow)) );

		modelVisCube_p(icorr,ich,irow)=Complex(1.0);
		ampCorr(icorr)+=amp;
		n(icorr)++;
	      }
	      else
		// zero data if model is zero
		visCube_p(icorr,ich,irow)=0.0;
	    }
	  }
	}

	for (Int icorr=0;icorr<nCor;icorr++) {
	  if (n(icorr)>0)
	    weightMat_p(icorr,irow)*=square(ampCorr(icorr)/Float(n(icorr)));
	  else
	    weightMat_p(icorr,irow)=0.0f;
	}
	
      }
      else {
	// Zero weight on this flagged row
	weightMat_p.column(irow)=0.0f;
	
	// Advance fl over this row
	fl+=nChannel();
      }
    }
  }
  else
    throw(AipsError("Failed to normalize data by model!"));
}

Vector<Int> VisBuffer::vecIntRange(const MSCalEnums::colDef& calEnum) const
{
  // Return a column range for a generic integer column as
  // identified by the enum specification in class MSCalEnums

  // Prepare the flag column masking
  LogicalArray mask(!flagRow());
  MaskedArray<Int>* maskArray;

  // A dummy vector for columns not yet supported (returns a value of [-1]);
  Vector<Int> nullIndex(antenna1().shape(), -1);

  switch (calEnum) {
    // ANTENNA1
  case MSC::ANTENNA1: {
    maskArray = new MaskedArray<Int>(antenna1(), mask);
    break;
  };
  // ANTENNA2
  case MSC::ANTENNA2: {
    maskArray = new MaskedArray<Int>(antenna2(), mask);
    break;
  };
  // FEED1
  case MSC::FEED1: {
    maskArray = new MaskedArray<Int>(feed1(), mask);
    break;
  };
  // FIELD_ID
  case MSC::FIELD_ID: {
    Vector<Int> fieldIdVec(antenna1().shape(), fieldId());
    maskArray = new MaskedArray<Int>(fieldIdVec, mask);
    break;
  };
  // ARRAY_ID
  case MSC::ARRAY_ID: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // OBSERVATION_ID
  case MSC::OBSERVATION_ID: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // SCAN_NUMBER
  case MSC::SCAN_NUMBER: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // PROCESSOR_ID
  case MSC::PROCESSOR_ID: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // PHASE_ID
  case MSC::PHASE_ID: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // STATE_ID
  case MSC::STATE_ID: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // PULSAR_BIN
  case MSC::PULSAR_BIN: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // PULSAR_GATE_ID
  case MSC::PULSAR_GATE_ID: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // FREQ_GROUP
  case MSC::FREQ_GROUP: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };
  // CALIBRATION_GROUP
  case MSC::CALIBRATION_GROUP: {
    maskArray = new MaskedArray<Int>(nullIndex, mask);
    break;
  };

  default: {
    throw(AipsError("Request for non-existent uv-data column"));
  };
  };

  // Return only unique indices
  Vector<Int> retval = unique(maskArray->getCompressedArray());
  if (maskArray) delete(maskArray);

  return retval;
};

Vector<Int> VisBuffer::antIdRange() const
{
  // Return a column range for ANTENNA_ID, including the
  // union of the ANTENNA1 and ANTENNA2 columns indices

  Vector<Int> ant1 = vecIntRange(MSC::ANTENNA1);
  Vector<Int> ant2 = vecIntRange(MSC::ANTENNA2);
  Vector<Int> ant12 = concatenateArray(ant1, ant2);

  // Return only unique index values
  return unique(ant12);
};

Bool VisBuffer::timeRange(MEpoch& rTime, MVEpoch& rTimeEP, 
			  MVEpoch& rInterval) const
{
  // Return the time range of data in the vis buffer
  // (makes simplistic assumptions in the absence of
  // interval information for now)
  
  // Initialization
  Bool retval = False;

  if (nRow() > 0) {
    retval = True;
    LogicalArray mask(!flagRow());
    MaskedArray<Double> maskTime(time(), mask);
    Double minTime = min(maskTime);
    Double maxTime = max(maskTime);
    // Mean time
    rTime = MEpoch(Quantity((minTime+maxTime)/2, "s"));
    // Extra precision time is always null for now
    rTimeEP = MVEpoch(Quantity(0, "s"));
    // Interval
    rInterval = MVEpoch(Quantity(maxTime-minTime, "s"));
  };
  return retval;
};


Vector<uInt>& VisBuffer::rowIds() {

  if(!rowIdsOK_p) {

    rowIdsOK_p=True;
    visIter_p->rowIds(rowIds_p);
  }
  return rowIds_p;
  


}


void VisBuffer::updateCoordInfo()
{
  antenna1();
  antenna2();
  arrayId();
  fieldId();
  spectralWindow();
  time();
  frequency();
  nRow();
  checkMSId();
  feed1();
  feed2();
  feed1_pa();
  feed2_pa();
  direction1();
  direction2();
}

void VisBuffer::setVisCube(Complex c)
{
  visCube_p.resize(visIter_p->visibilityShape());
  visCube_p.set(c);
  visCubeOK_p=True; 
}
void VisBuffer::setModelVisCube(Complex c)
{
  modelVisCube_p.resize(visIter_p->visibilityShape());
  modelVisCube_p.set(c);
  modelVisCubeOK_p=True;
}
void VisBuffer::setCorrectedVisCube(Complex c)
{
  correctedVisCube_p.resize(visIter_p->visibilityShape());
  correctedVisCube_p.set(c);
  correctedVisCubeOK_p=True;
}
void VisBuffer::setVisCube(const Cube<Complex>& vis)
{
  visCube_p.resize(vis.shape());
  visCube_p=vis;
  visCubeOK_p=True;
}
void VisBuffer::setModelVisCube(const Cube<Complex>& vis)
{
  modelVisCube_p.resize(vis.shape());
  modelVisCube_p=vis;
  modelVisCubeOK_p=True;
}

void VisBuffer::setModelVisCube(const Vector<Float>& stokes)
{

  /*
  cout << "Specified Stokes Parameters: " << stokes << endl;

  cout << "polFrame() = " << polFrame() 
       << " " << MSIter::Circular
       << " " << MSIter::Linear
       << endl;
  */

  // Stokes parameters, nominally unpolarized, unit I
  Float I(1.0),Q(0.0),U(0.0),V(0.0);

  // Only fill as many as are specified, up to 4 (unspecified will be assumed zero)
  for (uInt i=0;i<stokes.nelements();++i)
    switch (i) {
    case 0: { I=stokes(i); break; }
    case 1: { Q=stokes(i); break; }
    case 2: { U=stokes(i); break; }
    case 3: { V=stokes(i); break; }
    default: { break; }
    }

  // Convert to correlations, according to basis
  Vector<Complex> stkvis(4,Complex(0.0));  // initially all zero
  switch (polFrame()) {
  case MSIter::Circular: {
    stkvis(0)=Complex(I+V);
    stkvis(1)=Complex(Q,U);
    stkvis(2)=Complex(Q,-U);
    stkvis(3)=Complex(I-V);
    break;
  }
  case MSIter::Linear: {
    stkvis(0)=Complex(I+Q);
    stkvis(1)=Complex(U,V);
    stkvis(2)=Complex(U,-V);
    stkvis(3)=Complex(I-Q);
    break;
  }
  default:
    throw(AipsError("Model-setting only works for CIRCULAR and LINEAR bases, for now."));
    break;
  }
    
  // A map onto the actual correlations in the VisBuffer
  Vector<Int> corrmap;
  corrmap=corrType();
  corrmap-=corrmap(0);
  // This MUST yield indices < 4
  if (max(corrmap)>3)
    throw(AipsError("HELP! The correlations in the data are not normal!"));
  

  // Set the modelVisCube accordingly
  modelVisCube_p.resize(visIter_p->visibilityShape());
  modelVisCube_p.set(0.0);
  for (Int icorr=0;icorr<nCorr();++icorr) 
    if (abs(stkvis(corrmap(icorr)))>0.0)
      modelVisCube_p(Slice(icorr,1,1),Slice(),Slice()).set(stkvis(corrmap(icorr)));
  modelVisCubeOK_p=True;

  // Lookup flux density calibrator scaling, and apply it per channel...
  //  TBD



}


void VisBuffer::setCorrectedVisCube(const Cube<Complex>& vis)
{
  correctedVisCube_p.resize(vis.shape());
  correctedVisCube_p=vis;
  correctedVisCubeOK_p=True;
}

void VisBuffer::refModelVis(const Matrix<CStokesVector>& mvis)
{
  modelVisibility_p.resize();
  modelVisibility_p.reference(mvis);
  modelVisOK_p=True;
}

void VisBuffer::removeScratchCols() 
{
  // removes scratch data from the vb
  modelVisibility_p.resize();
  modelVisOK_p=False;
  correctedVisibility_p.resize();
  correctedVisOK_p=False;
}


Int & VisBuffer::fillnCorr() 
{ 
  nCorrOK_p=True; 
  nCorr_p=corrType().nelements();
  return nCorr_p;
 }

Int & VisBuffer::fillnChannel() 
{ 
  nChannelOK_p=True; 
  //  nChannel_p=visIter_p->channelGroupSize();
  nChannel_p=channel().nelements();
  return nChannel_p;
 }

Vector<Int> & VisBuffer::fillChannel()
{ channelOK_p=True; return visIter_p->channel(channel_p);}

Int & VisBuffer::fillnRow()
{ 
  nRowOK_p=True; 
  nRow_p=visIter_p->nRow(); 
  return nRow_p; 
}

Vector<Int>& VisBuffer::fillAnt1()
{ ant1OK_p=True; return visIter_p->antenna1(antenna1_p);}
Vector<Int>& VisBuffer::fillAnt2()
{ ant2OK_p=True; return visIter_p->antenna2(antenna2_p);}
Vector<Int>& VisBuffer::fillFeed1()
{ feed1OK_p=True; return visIter_p->feed1(feed1_p);}
Vector<Int>& VisBuffer::fillFeed2()
{ feed2OK_p=True; return visIter_p->feed2(feed2_p);}
Vector<SquareMatrix<Complex,2> >& VisBuffer::fillCjones()
{ cjonesOK_p=True; return visIter_p->CJones(cjones_p); }
Vector<Int>& VisBuffer::fillCorrType()
{ corrTypeOK_p=True; return visIter_p->corrType(corrType_p); }

// calling fillFeed1_pa or fillFeed2_pa will fill antenna, feed
// and time caches automatically
Vector<Float>& VisBuffer::fillFeed1_pa()
{
  // fill feed, antenna and time caches, if not filled before
  feed1(); antenna1(); time();
  feed1_paOK_p=True;
  feed1_pa_p.resize(antenna1_p.nelements()); // could also use nRow()

  // now actual calculations  
  for (uInt row=0; row<feed1_pa_p.nelements(); ++row) {
       const Vector<Float>& ant_pa=feed_pa(time_p(row)); // caching inside
           // ROVisibilityIterator, if the time doesn't change. Otherwise
	   // we should probably fill both buffers for feed1 and feed2
	   // simultaneously to speed up things.
       DebugAssert((uInt(antenna1_p(row))<ant_pa.nelements()),AipsError);
       DebugAssert(antenna1_p(row)>=0,AipsError);	   
       feed1_pa_p(row)=ant_pa(antenna1_p(row));
       // currently feed_pa returns only the first feed position angle
       // we need to add an offset if this row correspods to a
       // different feed
       if (feed1_p(row))  // an if-statement to avoid unnecessary operations
                          // in the single feed case, everything would
			  // work without it.
           feed1_pa_p(row)+=visIter_p->receptorAngles()(0,
	                             antenna1_p(row),feed1_p(row))-
			    visIter_p->receptorAngles()(0,antenna1_p(row),0);
  }
  return feed1_pa_p;
}

Vector<Float>& VisBuffer::fillFeed2_pa()
{
  // fill feed, antenna and time caches, if not filled before
  feed2(); antenna2(); time();
  feed2_paOK_p=True;
  feed2_pa_p.resize(antenna2_p.nelements()); // could also use nRow()

  // now actual calculations  
  for (uInt row=0; row<feed2_pa_p.nelements(); ++row) {
       const Vector<Float>& ant_pa=feed_pa(time_p(row)); // caching inside
           // ROVisibilityIterator, if the time doesn't change. Otherwise
	   // we should probably fill both buffers for feed1 and feed2
	   // simultaneously to speed up things.
       DebugAssert((uInt(antenna2_p(row))<ant_pa.nelements()),AipsError);
       DebugAssert(antenna2_p(row)>=0,AipsError);	   
       feed2_pa_p(row)=ant_pa(antenna2_p(row));
       // currently feed_pa returns only the first feed position angle
       // we need to add an offset if this row correspods to a
       // different feed
       if (feed2_p(row))  // an if-statement to avoid unnecessary operations
                          // in the single feed case, everything would
			  // work without it.
           feed2_pa_p(row)+=visIter_p->receptorAngles()(0,
	                             antenna2_p(row),feed2_p(row))-
			    visIter_p->receptorAngles()(0,antenna2_p(row),0);
  }
  return feed2_pa_p;
}

Vector<MDirection>& VisBuffer::fillDirection1()
{
  // fill feed1_pa cache, antenna, feed and time will be filled automatically 
  feed1_pa();
  direction1OK_p=True;
  direction1_p.resize(antenna1_p.nelements()); // could also use nRow()
  const ROMSPointingColumns& mspc=msColumns().pointing();
  
  if (visIter_p->allBeamOffsetsZero() && mspc.pointingIndex(antenna1()(0),
      time()(0))<0) {
        // if no true pointing information is found
        // just return the phase center from the field table
        direction1_p.set(phaseCenter());
        return direction1_p;
  }
  for (uInt row=0; row<antenna1_p.nelements(); ++row) {
       DebugAssert(antenna1_p(row)>=0,AipsError);	   
       DebugAssert(feed1_p(row)>=0,AipsError);
       Int pointIndex1 = mspc.pointingIndex(antenna1()(row),time()(row));
       // if no true pointing information is found
       // use the phase center from the field table       
       if (pointIndex1>=0)
           direction1_p(row)=mspc.directionMeas(pointIndex1,time()(row));
       else
           direction1_p(row)=phaseCenter();
       if (!visIter_p->allBeamOffsetsZero()) { 
           RigidVector<Double, 2> beamOffset = 
	        visIter_p->getBeamOffsets()(0,antenna1_p(row),feed1_p(row));
           if (visIter_p->antennaMounts()(antenna1_p(row))=="ALT-AZ" ||
               visIter_p->antennaMounts()(antenna1_p(row))=="alt-az") {
               SquareMatrix<Double, 2> xform(SquareMatrix<Double, 2>::General); 
	          // SquareMatrix' default constructor is a bit strange. 
		  // We will probably need to change it in the future
               Double cpa=cos(feed1_pa_p(row));
	       Double spa=sin(feed1_pa_p(row));
               xform(0,0)=cpa;
	       xform(1,1)=cpa;
	       xform(0,1)=-spa;
	       xform(1,0)=spa;
               beamOffset*=xform; // parallactic angle rotation
           }       
           // x direction is flipped to convert az-el type frame to ra-dec
           direction1_p(row).shift(-beamOffset(0),beamOffset(1), True);
      }
  }	 
  return direction1_p; 
}

Vector<MDirection>& VisBuffer::fillDirection2()
{
  // fill feed2_pa cache, antenna, feed and time will be filled automatically 
  feed2_pa();
  direction2OK_p=True;
  direction2_p.resize(antenna2_p.nelements()); // could also use nRow()
  const ROMSPointingColumns& mspc=msColumns().pointing();
  
  if (visIter_p->allBeamOffsetsZero() && mspc.pointingIndex(antenna2()(0),
      time()(0))<0) {
        // if no true pointing information is found
        // just return the phase center from the field table
      direction2_p.set(phaseCenter());
      return direction2_p;
  }
  for (uInt row=0; row<antenna2_p.nelements(); ++row) {
       DebugAssert(antenna2_p(row)>=0,AipsError);	   
       DebugAssert(feed2_p(row)>=0,AipsError);	   
       Int pointIndex2 = mspc.pointingIndex(antenna2()(row),time()(row));
       // if no true pointing information is found
       // use the phase center from the field table       
       if (pointIndex2>=0)
           direction2_p(row)=mspc.directionMeas(pointIndex2,time()(row));
       else
           direction2_p(row)=phaseCenter();
       if (!visIter_p->allBeamOffsetsZero()) { 
           RigidVector<Double, 2> beamOffset = 
	        visIter_p->getBeamOffsets()(0,antenna2_p(row),feed2_p(row));
           if (visIter_p->antennaMounts()(antenna2_p(row))=="ALT-AZ" ||
	   visIter_p->antennaMounts()(antenna2_p(row))=="alt-az") {
               SquareMatrix<Double, 2> xform(SquareMatrix<Double, 2>::General); 
	          // SquareMatrix' default constructor is a bit strange. 
		  // We will probably need to change it in the future
               Double cpa=cos(feed2_pa_p(row));
	       Double spa=sin(feed2_pa_p(row)); 
               xform(0,0)=cpa;
	       xform(1,1)=cpa;
	       xform(0,1)=-spa;
	       xform(1,0)=spa;
               beamOffset*=xform; // parallactic angle rotation
           }          
           // x direction is flipped to convert az-el type frame to ra-dec
           direction2_p(row).shift(-beamOffset(0),beamOffset(1), True);
       }
  }	 
  return direction2_p; 
}

Int& VisBuffer::fillFieldId()
{ 
  fieldIdOK_p=True; 
  fieldId_p=visIter_p->fieldId(); 
  return fieldId_p; 
}

Int& VisBuffer::fillArrayId()
{ 
  arrayIdOK_p=True; 
  arrayId_p=visIter_p->arrayId(); 
  return arrayId_p; 
}

Matrix<Bool>& VisBuffer::fillFlag()
{ flagOK_p=True; return visIter_p->flag(flag_p); }
Cube<Bool>& VisBuffer::fillFlagCube()
{ flagCubeOK_p=True; return visIter_p->flag(flagCube_p); }
Vector<Bool>& VisBuffer::fillFlagRow()
{ flagRowOK_p=True; return visIter_p->flagRow(flagRow_p);}
Vector<Int>& VisBuffer::fillScan()
{ scanOK_p=True; return visIter_p->scan(scan_p);}
Vector<Double>& VisBuffer::fillFreq()
{ freqOK_p=True; return visIter_p->frequency(frequency_p); }
Vector<Double>& VisBuffer::fillLSRFreq()
{ lsrFreqOK_p=True; return visIter_p->lsrFrequency(lsrFrequency_p); }
MDirection& VisBuffer::fillPhaseCenter()
{ phaseCenterOK_p=True; return phaseCenter_p=visIter_p->phaseCenter();}

Int& VisBuffer::fillPolFrame()
{ 
  polFrameOK_p=True; 
  polFrame_p=visIter_p->polFrame();
  return polFrame_p;
}
Vector<Float>& VisBuffer::fillSigma()
{ sigmaOK_p=True; return visIter_p->sigma(sigma_p);}

Matrix<Float>& VisBuffer::fillSigmaMat()
{ sigmaMatOK_p=True; return visIter_p->sigmaMat(sigmaMat_p);}

Int& VisBuffer::fillSpW()
{ 
  spwOK_p=True; 
  spectralWindow_p=visIter_p->spectralWindow();
  return spectralWindow_p;
}

Vector<Double>& VisBuffer::fillTime()
{ timeOK_p=True; return visIter_p->time(time_p);}

Vector<Double>& VisBuffer::fillTimeInterval()
{ timeIntervalOK_p=True; return visIter_p->timeInterval(timeInterval_p);}

Vector<RigidVector<Double,3> >& VisBuffer::filluvw()
{ uvwOK_p=True; return visIter_p->uvw(uvw_p);}

Matrix<Double>& VisBuffer::filluvwMat()
{ uvwMatOK_p=True; return visIter_p->uvwMat(uvwMat_p);}

Matrix<CStokesVector>& 
VisBuffer::fillVis(VisibilityIterator::DataColumn whichOne)
{
  switch (whichOne) {
  case VisibilityIterator::Model:
    modelVisOK_p=True;
    return visIter_p->visibility(modelVisibility_p,whichOne);
    break;
  case VisibilityIterator::Corrected:
    correctedVisOK_p=True;
    return visIter_p->visibility(correctedVisibility_p,whichOne);
    break;    
  case VisibilityIterator::Observed:
  default:
    visOK_p=True; 
    return visIter_p->visibility(visibility_p,whichOne);
    break;
  }
}

Cube<Complex>& VisBuffer::fillVisCube(VisibilityIterator::DataColumn whichOne)
{ 
  switch(whichOne) {
  case VisibilityIterator::Model:
    modelVisCubeOK_p=True; 
    return visIter_p->visibility(modelVisCube_p,whichOne);
    break;
  case VisibilityIterator::Corrected:
    correctedVisCubeOK_p=True; 
    return visIter_p->visibility(correctedVisCube_p,whichOne);
    break;
  case VisibilityIterator::Observed:
  default:
    visCubeOK_p=True; 
    return visIter_p->visibility(visCube_p,whichOne);
    break;
  }
}

Vector<Float>& VisBuffer::fillWeight()
{ weightOK_p=True; return visIter_p->weight(weight_p);}

Matrix<Float>& VisBuffer::fillWeightMat()
{ weightMatOK_p=True; return visIter_p->weightMat(weightMat_p);}

Cube<Float>& VisBuffer::fillWeightSpectrum()
{ weightSpectrumOK_p=True; return visIter_p->weightSpectrum(weightSpectrum_p);}

Matrix<Float>& VisBuffer::fillImagingWeight()
{ imagingWeightOK_p=True; return visIter_p->imagingWeight(imagingWeight_p);}

const Vector<Float>& VisBuffer::feed_pa(Double time) const
{return visIter_p->feed_pa(time);}

const Float& VisBuffer::parang0(Double time) const
{return visIter_p->parang0(time);}

const Vector<Float>& VisBuffer::parang(Double time) const
{return visIter_p->parang(time);}

const MDirection& VisBuffer::azel0(Double time) const
{return visIter_p->azel0(time);}

Vector<Double>& VisBuffer::azel0Vec(Double time, Vector<Double>& azelVec) const {
  MDirection azelMeas=This->azel0(time);
  azelVec.resize(2);
  azelVec=azelMeas.getAngle("deg").getValue();
  return azelVec;

}

const Vector<MDirection>& VisBuffer::azel(Double time) const
{return visIter_p->azel(time);}


Matrix<Double>& VisBuffer::azelMat(Double time, Matrix<Double>& azelMat) const {
  Vector<MDirection> azelMeas=This->azel(time);
  azelMat.resize(2,azelMeas.nelements());
  for (uInt iant=0;iant<azelMeas.nelements();++iant)
    azelMat.column(iant) = (azelMeas(iant).getAngle("deg").getValue());
  return azelMat;

}

Double VisBuffer::hourang(Double time) const {
  return visIter_p->hourang(time);
}


Vector<Int> VisBuffer::unique(const Vector<Int>& indices) const
{
  // Filter integer index arrays for unique values
  //
  uInt n = indices.nelements();
  Vector<Int> uniqIndices(n);
  if (n > 0) {
    // Sort temporary array in place
    Vector<Int> sortedIndices = indices.copy();
    GenSort<Int>::sort(sortedIndices);

    // Extract unique elements
    uniqIndices(0) = sortedIndices(0);
    uInt nUniq = 1;
    for (uInt i=1; i < n; i++) {
      if (sortedIndices(i) != uniqIndices(nUniq-1)) {
	uniqIndices(nUniq++) = sortedIndices(i);
      };
    };
    uniqIndices.resize(nUniq, True);
  };
  return uniqIndices;



};

Bool VisBuffer::checkMSId() {
  //if this is not a new iteration then don't even check;
  //Let the state be
  if(msOK_p)
    return False;
  if( visIter_p!=static_cast<ROVisibilityIterator*>(0)){ 
    if(oldMSId_p != visIter_p->msId()){
      oldMSId_p = visIter_p->msId();
      newMS_p=True;
   
    }   
    else{
      newMS_p=False;
    }
    msOK_p=True;
    return newMS_p;
    
  }
  return False;

}


} //# NAMESPACE CASA - END

