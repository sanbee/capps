//# ClarkCleanModel.h: this defines ClarkCleanModel
//# Copyright (C) 1996,1997,1998,1999,2000,2003
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
//#
//# $Id$

#ifndef SYNTHESIS_CLARKCLEANMODEL_H
#define SYNTHESIS_CLARKCLEANMODEL_H

#include <casa/aips.h>
#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/Vector.h>
#include <casa/Arrays/Array.h>
#include <synthesis/MeasurementEquations/ArrayModel.h>
#include <synthesis/MeasurementEquations/Iterate.h>
//#include <synthesis/MeasurementEquations/ResidualEquation.h>
#include <synthesis/MeasurementEquations/ConvolutionEquation.h>
#include <casa/Logging/LogIO.h>

namespace casa { //# NAMESPACE CASA - BEGIN

class ClarkCleanProgress;

// <summary>
// A Class for performing the Clark Clean Algorithm on Arrays
// </summary>

// <use visibility=export>

// <reviewed reviewer="" date="yyyy/mm/dd" tests="" demos="">
// </reviewed>

// <prerequisite> 
// <li> ResidualEquation/ConvolutionEquation 
// <li> LinearModel/LinearEquation Paradigm 
// </prerequisite>
//
// <etymology>
// This class is called ClarkCleanModel because thats the algorithm it uses
// deconvolve the model. 
// </etymology>
//
// <synopsis>
// This class is used to perform the Clark Clean Algorithm on an
// Array. Only the deconvolved model of the sky are directly stored by this
// class. The point spread function (psf) and convolved (dirty) image are
// stored in a companion class which is must be derived from
// ResidualEquation. 
// 
// The cleaning works like this. The user constructs a ClarkCleanModel by
// specifying an initial model of the sky. This can by be
// one,two,three... dimensional depending on the dimension of the psf (see
// below). The user then constructs a class which implements the forward
// equation between the model and the dirty image. Typically this will be
// the ConvolutionEquation class, although any class which has a
// ResidualEquation interface will be work (but perhaps very slowly, as the
// ConvolutionEquation class has member functions optimised for cleaning)
//
// The user then calls the solve() function (with the appropriate equation
// class as an arguement), and this class will perform the Clark clean.
// The various clean parameters are set (prior to calling solve) using the
// functions derived from the Iterate class, in particular setGain(),
// setNumberIterations() & setThreshold() (to set a flux limit). 
// 
// The solve() function does not return either the deconvolved model or the
// residuals. The solved model can be obtained using the getModel() function
// (derived from ArrayModel()) and the residual can be obtained using the
// residual() member function of the Convolution/Residual Equation Class.
// 
// The size and shape of the model used in this class MUST be the same as
// the convolved data (Dirty Image), stored in the companion
// ResidualEquation Class. However the model (and convolved data) can have
// more dimensions than the psf, as well as a different size (either larger
// or smaller). When the dimensionality is different the cleaning is done
// independendtly in each "plane" of the model. (Note this has not
// been implemented yet but is relatively simple to do if necessary). 
//
// This multi-dimensionalty is exploited when cleaning arrays of
// StokesVectors. Here the Array of StokesVectors is decomposed into a stack
// of 4 Floating point arrays and the cleaning is done on all the the arrays
// simultaneosly. The criterion for choosing the brightest pixel has been
// generalised by using the "length" of the Stokesvector in 4 dimensional
// space. 
//
// A companion class to this one is MaskedClarkCleanModel. This provides
// the same functionality but is used with MaskedArrays which indicate which
// regions of the model to search for clean components. 
//
// </synopsis>
//
// <example>
// <srcblock>
// Matrix<Float> psf(12,12), dirty(10,10), initialModel(10,10);
// ...put appropriate values into psf, dirty, & initialModel....
// ClarkCleanModel<Float> deconvolvedModel(initialModel); 
// ConvolutionEquation convEqn(psf, dirty);
// deconvolvedModel.setGain(0.2); 
// deconvolvedModel.setNumberIterations(1000);
// Bool convWorked = deconvolvedModel.solve(convEqn);
// Array<Float> finalModel, residuals;
// if (convWorked){
//   finalModel = deconvolvedModel.getModel();
//   ConvEqn.residual(deconvolvedModel, finalResidual);
// }
// </srcblock> 
// </example>
//
// <motivation>
// This class is needed to deconvolve images.
// </motivation>
//
// <templating arg=T>
// I have tested this class with Arrays of
//    <li> Float
//    <li> StokesVector
// </templating>
//
// <todo asof="1996/05/02">
//   <li> Make changes so that multidimensions work as advertised
//   <li> compare timing with other clean implementations (ie, Mark's
//   CleanTools, SDE, AIPS & miriad) 
// </todo>

class ClarkCleanModel: 
  public ArrayModel<Float>,
  public Iterate
{
public:
  // The default constructor does nothing more than initialise a zero length
  // array to hold the deconvolved model. If this constructor is used then 
  // the actual model must be set using the setModel() function of the
  // ArrayModel class.
  ClarkCleanModel();
  // Construct the ClarkCleanModel object and initialise the model.
  ClarkCleanModel(Array<Float> & model);
  // Construct the ClarkCleanModel object and initialise the model ans mask
  ClarkCleanModel(Array<Float> & model, Array<Float> & mask);

  void getMask(Array<Float>& mask) const;
  void setMask(const Array<Float>& mask);
  void setMask(Array<Float> & mask);

  void getModel(Array<Float>& model) const;
  void setModel(const Array<Float>& model);
  void setModel(Array<Float> & model);

  // Set/get the progress display 
  // <group>
  virtual void setProgress(ClarkCleanProgress& ccp) { itsProgressPtr = &ccp; }
  virtual ClarkCleanProgress& getProgress() { return *itsProgressPtr; }
  // </group>

  // Using a Clark clean deconvolution proceedure solve for an improved
  // estimate of the deconvolved object. The convolution/residual equation
  // contains the psf and dirty image. When called with a ResidualEquation
  // arguement a quite general interface is used that is slow. The
  // convolution equation contains functions that speed things up. The
  // functions return False if the deconvolution could not be done.
  // <group>
  Bool solve(ConvolutionEquation & eqn);
  Bool singleSolve(ConvolutionEquation & eqn, Array<Float>& residual);
  // </group>

  // These functions set various "knobs" that the user can tweak and are
  // specific to the Clark clean algorithm. The more generic parameters
  // ie. clean gain, and maximum residual fluxlimit, are set using functions
  // in the Iterate base class. The get functions return the value that was
  // actually used after the cleaning was done.
  // <group>
  // set the size of the PSF used in the minor iterations. If not set it
  // defaults to the largest useful Psf (ie. min(modelSize*2, psfSize))
  virtual void setPsfPatchSize(const IPosition & psfPatchSize); 
  virtual IPosition getPsfPatchSize(); 
  // Set the size of the histogram used to determine how many pixels are
  // "active" in a minor iteration. Default value is 1000 is OK for
  // everything except very small cleans.
  virtual void setHistLength(const uInt HistBins ); 
  virtual uInt getHistLength(); 
  // Set the maximum number of minor iterations to perform for each major
  // cycle. 
  virtual void setMaxNumberMinorIterations(const uInt maxNumMinorIterations); 
  virtual uInt getMaxNumberMinorIterations();
  // Set and get the initial number of iterations
  virtual void setInitialNumberIterations(const uInt initialNumberIterations); 
  virtual uInt getInitialNumberIterations();
  // Set the maximum number of major cycles to perform
  virtual void setMaxNumberMajorCycles(const uInt maxNumMajorCycles); 
  virtual uInt getMaxNumberMajorCycles();
  // Set the maximum number of active pixels to use in the minor
  // iterations. The specified number can be exceeded if the topmost bin of
  // the histogram contains more pixels than specified here. The default is
  // 10,000 which is suitable for images of 512by512 pixels. Reduce this for
  // smaller images and increase it for larger ones. 
  virtual void setMaxNumPix(const uInt maxNumPix ); 
  virtual uInt getMaxNumPix(); 
  // Set the maximum exterior psf value. This is used to determine when to
  // stop the minor itartions. This is normally determined from the psf and
  // the number set here is only used if this cannot be determined. The
  // default is zero.
  virtual void setMaxExtPsf(const Float maxExtPsf ); 
  virtual Float getMaxExtPsf(); 
  // An exponent on the F(m,n) factor (see Clark[1980]) which influences how
  // quickly active pixels are treated as unreliable. Larger values mean
  // more major iterations. The default is zero. I have no experience on
  // when to use this factor.
  virtual void setSpeedup(const Float speedup ); 
  virtual Float getSpeedup(); 
  // The structure of various AIPS++ algorithms creates major cycles around
  // the Clark Clean (or other deconvolution algrithms.  The cycleSpeedup
  // parameter causes the threshold to edge up as
  // thresh = thresh_0 * 2^( iter/cycleSpeedup ); 
  // ignored if cycleSpeedup <= 0.
  virtual void setCycleSpeedup(const Float speedup ); 
  virtual Float getCycleSpeedup(); 
  // We are overwriting Iterate's threshold() method to put out speedup in it
  virtual const Float threshold();
  // The user can be asked whether to stop after every minor cycle
  virtual void setChoose(const Bool askForChoice);
  virtual Bool getChoose();
  // </group>

private:
// Do all the minor iterations for one major cycle. Cleaning stops
// when the flux or iteration limit is reached.
  void doMinorIterations(Array<Float> & model, 
			 Matrix<Float> & pixelValue, 
			 const Matrix<Int> & pixelPos, 
			 const Int numPix,
			 Matrix<Float> & psfPatch,
			 Float fluxLimit, 
			 uInt & numberIterations, 
			 Float Fmn, 
			 const uInt totalIterations,
			 Float& totalflux);
// Find all the pixels in the residual that are greater than fluxlimit, and
// store the values in the pixelsValue Matrix, and their positions in the
// pixelPos Matrix. Increases the size of the output matrices as
// necessary, but does not decrease them. So the actual number of "active"
// pixels is returned. This will always be less than (or equal to) the matrix
// size.
  Int cacheActivePixels(Matrix<Float> & pixVal, Matrix<Int> & pixPos, 
			 const Array<Float> & data, const Float fluxLimit);
// make histogram of absolute values in array
  void absHistogram(Vector<Int> & hist, Float & minVal, 
		    Float & maxVal, const Array<Float> & data);
// Determine the flux limit if we only select the maxNumPix biggest
// residuals. Flux limit is not exact due to quantising by the histogram
  Float biggestResiduals(Float & maxRes, const uInt maxNumPix, 
			 const Float fluxLimit, const Array<Float> & residual);
// Work out the size of the Psf patch to use. 
  Float getPsfPatch(Array<Float>& psfPatch, ConvolutionEquation& eqn);
// The maximum residual is the absolute maximum.
  Float maxResidual(const Array<Float> & residual);
  void maxVect(Vector<Float> & maxVal, Float & absVal, Int & offset,
	       const Matrix<Float> & pixVal, const Int numPix);
  void subtractComponent(Matrix<Float> & pixVal, const Matrix<Int> & pixPos,
			 const Int numPix, const Vector<Float> & maxVal,
			 const Vector<Int> & maxPos, const Matrix<Float> & psf);
  Float absMaxBeyondDist(const IPosition &maxDist, const IPosition &centre,
			 const Array<Float> &array);
  Bool stopnow();

  uInt theHistBins;
  Float theMaxExtPsf;
  uInt theMaxNumberMinorIterations;
  uInt theInitialNumberIterations;
  Int theMaxNumberMajorCycles;
  uInt theMaxNumPix;
  IPosition thePsfPatchSize;
  Float theSpeedup;
  Float theCycleSpeedup;
  Bool theChoose;
  Array<Float> theMask;
  LogIO theLog;
  // There are too many iterations counters.
  // This one is required for theCycleSpeedup and threshold(),
  // and just keeps track of the total iterations done by THIS
  // ClarkCleanModel
  Int theIterCounter; 
  ClarkCleanProgress* itsProgressPtr;
  Bool itsJustStarting;
};


} //# NAMESPACE CASA - END

#endif
