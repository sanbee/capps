#ifndef SYNTHESIS_CDATATOGRIDIMPLDEF
#define SYNTHESIS_CDATATOGRIDIMPLDEF
#include <casa/Arrays/Matrix.h>
#include <synthesis/TransformMachines/VBStore.h>

__global__ void kernel_cuBlank(uInt *vbs,Int n);

// template
// __global__ void kernel_cuDataToGridImpl_p(DComplex* gridStore,  Int* gridShape, //4-elements
					  
// 					  const uInt *subGridShape,//[2],
// 					  const uInt *BLCXi, const uInt *BLCYi,
// 					  const uInt *TRCXi, const uInt *TRCYi,
					  
// 					  const Complex *visCube_ptr, const Float* imgWts_ptr,
// 					  const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
// 					  const Double *uvw_ptr,
					  
// 					  const Int nRow, const Int rbeg, const Int rend, 
// 					  const Int nDataChan,const Int nDataPol, 
// 					  const Int startChan, const Int endChan, const Int vbSpw,
// 					  const Double *vbFreq,
					  
// 					  const Complex *cfV[2],
// 					  Int *cfShape,//[4],
// 					  Float *sampling,//[2], 
// 					  const Int *support, //[2]
					  
// 					  Double* sumWt_ptr,
// 					  const Bool dopsf, const Bool accumCFs,
// 					  const Int* polMap_ptr, const Int *chanMap_ptr,
// 					  const Double *uvwScale_ptr, const Double *offset_ptr,
// 					  const Double *dphase_ptr, Int XThGrid, Int YThGrid);

// template
// __global__ void kernel_cuDataToGridImpl_p(Complex* gridStore,  Int* gridShape, //4-elements
					  
// 					  const uInt *subGridShape,//[2],
// 					  const uInt *BLCXi, const uInt *BLCYi,
// 					  const uInt *TRCXi, const uInt *TRCYi,
					  
// 					  const Complex *visCube_ptr, const Float* imgWts_ptr,
// 					  const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
// 					  const Double *uvw_ptr,
					  
// 					  const Int nRow, const Int rbeg, const Int rend, 
// 					  const Int nDataChan,const Int nDataPol, 
// 					  const Int startChan, const Int endChan, const Int vbSpw,
// 					  const Double *vbFreq,
					  
// 					  const Complex *cfV[2],
// 					  Int *cfShape,//[4],
// 					  Float *sampling,//[2], 
// 					  const Int *support, //[2]
					  
// 					  Double* sumWt_ptr,
// 					  const Bool dopsf, const Bool accumCFs,
// 					  const Int* polMap_ptr, const Int *chanMap_ptr,
// 					  const Double *uvwScale_ptr, const Double *offset_ptr,
// 					  const Double *dphase_ptr, Int XThGrid, Int YThGrid
// 					  );
//
//----------------------------------------------------------------------
//
template 
void cuDataToGridImpl_p(Complex* gridStore,  Int* gridShape, //4-elements
			
			const uInt *subGridShape,//[2],
			const uInt *BLCXi, const uInt *BLCYi,
			const uInt *TRCXi, const uInt *TRCYi,
			
			const Complex *visCube_ptr, const Float* imgWts_ptr,
			const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
			const Double *uvw_ptr,
			
			const Int nRow, const Int rbeg, const Int rend, 
			const Int nDataChan,const Int nDataPol, 
			const Int startChan, const Int endChan, const Int vbSpw,
			const Double *vbFreq,
			
			const Complex *cfV[2],
			Int *cfShape,//[4],
			Float *sampling,//[2], 
			const Int *support, //[2]
			
			Double* sumWt_ptr,
			const Bool dopsf, const Bool accumCFs,
			const Int* polMap_ptr, const Int *chanMap_ptr,
			const Double *uvwScale_ptr, const Double *offset_ptr,
			const Double *dphase_ptr, Int XThGrid, Int YThGrid);

template 
void cuDataToGridImpl_p(DComplex* gridStore,  Int* gridShape, //4-elements
			
			const uInt *subGridShape,//[2],
			const uInt *BLCXi, const uInt *BLCYi,
			const uInt *TRCXi, const uInt *TRCYi,
			
			const Complex *visCube_ptr, const Float* imgWts_ptr,
			const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
			const Double *uvw_ptr,
			
			const Int nRow, const Int rbeg, const Int rend, 
			const Int nDataChan,const Int nDataPol, const Int startChan, 
			const Int endChan, const Int vbSpw,
			const Double *vbFreq,
			
			const Complex *cfV[2],
			Int *cfShape,//[4], //[4]
			Float *sampling,//[2], 
			const Int *support, //[2]
			
			Double* sumWt_ptr,
			const Bool dopsf, const Bool accumCFs,
			const Int* polMap_ptr, const Int *chanMap_ptr,
			const Double *uvwScale_ptr, const Double *offset_ptr,
			const Double *dphase_ptr, Int XThGrid, Int YThGrid);
//
//----------------------------------------------------------------------
//
template 
void cDataToGridImpl2_p(Complex* gridStore,  Int* gridShape, //4-elements
			
			const uInt *subGridShape,//[2],
			const uInt *BLCXi, const uInt *BLCYi,
			const uInt *TRCXi, const uInt *TRCYi,
			
			const Complex *visCube_ptr, const Float* imgWts_ptr,
			const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
			const Double *uvw_ptr,
			
			const Int nRow, const Int rbeg, const Int rend, 
			const Int nDataChan,const Int nDataPol, 
			const Int startChan, const Int endChan, const Int vbSpw,
			const Double *vbFreq,
			
			const Complex *cfV[2],
			Int *cfShape,//[4],
			Float *sampling,//[2], 
			const Int *support, //[2]
			
			Double* sumWt_ptr,
			const Bool dopsf, const Bool accumCFs,
			const Int* polMap_ptr, const Int *chanMap_ptr,
			const Double *uvwScale_ptr, const Double *offset_ptr,
			const Double *dphase_ptr, Int XThGrid, Int YThGrid);

template 
void cDataToGridImpl2_p(DComplex* gridStore,  Int* gridShape, //4-elements
			
			const uInt *subGridShape,//[2],
			const uInt *BLCXi, const uInt *BLCYi,
			const uInt *TRCXi, const uInt *TRCYi,
			
			const Complex *visCube_ptr, const Float* imgWts_ptr,
			const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
			const Double *uvw_ptr,
			
			const Int nRow, const Int rbeg, const Int rend, 
			const Int nDataChan,const Int nDataPol, const Int startChan, 
			const Int endChan, const Int vbSpw,
			const Double *vbFreq,
			
			const Complex *cfV[2],
			Int *cfShape,//[4], //[4]
			Float *sampling,//[2], 
			const Int *support, //[2]
			
			Double* sumWt_ptr,
			const Bool dopsf, const Bool accumCFs,
			const Int* polMap_ptr, const Int *chanMap_ptr,
			const Double *uvwScale_ptr, const Double *offset_ptr,
			const Double *dphase_ptr, Int XThGrid, Int YThGrid);

template
void cDataToGridImpl_p(Complex* gridStore, Int* gridShape, VBStore* vbs,
		       Matrix<Double>* sumwt, const Bool dopsf,
		       const Int* polMap_ptr, const Int *chanMap_ptr,
		       const Double *uvwScale_ptr, const Double *offset_ptr,
		       const Double *dphase_ptr,
		       Int XThGrid, Int YThGrid);
template
void cDataToGridImpl_p(DComplex* gridStore, Int* gridShape, VBStore* vbs,
		       Matrix<Double>* sumwt, const Bool dopsf,
		       const Int* polMap_ptr, const Int *chanMap_ptr,
		       const Double *uvwScale_ptr, const Double *offset_ptr,
		       const Double *dphase_ptr,Int XThGrid, Int YThGrid);

template
Complex caccumulateOnGrid(Complex* gridStore, const Int* gridInc_p, const Complex *cached_phaseGrad_p,
			  const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			  const Complex* convFuncV, const Int *cfInc_p,Complex nvalue,
			  Double wVal, Int *supBLC_ptr, Int *supTRC_ptr,
			  Float* scaledSampling_ptr, Double* off_ptr, Int* convOrigin_ptr, 
			  Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr, Bool finitePointingOffset,
			  Bool doPSFOnly, Bool& foundCFPeak);

template
Complex caccumulateOnGrid(DComplex* gridStore, const Int* gridInc_p, const Complex *cached_phaseGrad_p,
			  const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			  const Complex* convFuncV, const Int *cfInc_p,Complex nvalue,
			  Double wVal, Int *supBLC_ptr, Int *supTRC_ptr,
			  Float* scaledSampling_ptr, Double* off_ptr, Int* convOrigin_ptr, 
			  Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr, Bool finitePointingOffset,
			  Bool doPSFOnly, Bool& foundCFPeak);
//
//----------------------------------------------------------------------
//

template
void addTo4DArray(Complex *store, const Int *iPos, const Int* inc, 
		  Complex& nvalue, Complex& wt);
template
void addTo4DArray(DComplex *store, const Int *iPos, const Int* inc, 
		  Complex& nvalue, Complex& wt);

#endif // 
