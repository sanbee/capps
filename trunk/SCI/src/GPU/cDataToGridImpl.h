#ifndef SYNTHESIS_CDATATOGRIDIMPL
#define SYNTHESIS_CDATATOGRIDIMPL

#include <casa/Arrays/Matrix.h>
#include <synthesis/TransformMachines/VBStore.h>
#include <cuComplex.h>

namespace casa { //# NAMESPACE CASA - BEGIN


  typedef void (*ComplexGridder)(Complex* gridStore, Int* gridShape, VBStore* vbs,
				  Matrix<Double>* sumwt, const Bool dopsf,
				  const Int* polMap_ptr, const Int *chanMap_ptr,
				  const Double *uvwScale_ptr, const Double *offset_ptr,
				  const Double *dphase_ptr, Int XThGrid, Int YThGrid);
  typedef void (*DComplexGridder)(DComplex* gridStore, Int* gridShape, VBStore* vbs,
				  Matrix<Double>* sumwt, const Bool dopsf,
				  const Int* polMap_ptr, const Int *chanMap_ptr,
				  const Double *uvwScale_ptr, const Double *offset_ptr,
				  const Double *dphase_ptr, Int XThGrid, Int YThGrid);

  void cuBlank(uInt* t,Int n);

  // template <class T>
  // __global__ void kernel_cuDataToGridImpl_p(T* gridStore,  Int* gridShape, //4-elements
					    
  // 					    const uInt *subGridShape,//[2],
  // 					    const uInt *BLCXi, const uInt *BLCYi,
  // 					    const uInt *TRCXi, const uInt *TRCYi,
					    
  // 					    const Complex *visCube_ptr, const Float* imgWts_ptr,
  // 					    const Bool *flagCube_ptr, const Bool *rowFlag_ptr,
  // 					    const Double *uvw_ptr,
					    
  // 					    const Int nRow, const Int rbeg, const Int rend, 
  // 					    const Int nDataChan,const Int nDataPol, 
  // 					    const Int startChan, const Int endChan, const Int vbSpw,
  // 					    const Double *vbFreq,
					    
  // 					    const Complex *cfV[2],
  // 					    Int *cfShape,//[4], //[4]
  // 					    Float *sampling,//[2], 
  // 					    const Int *support, //[2]
					    
  // 					    Double* sumWt_ptr,
  // 					    const Bool dopsf, const Bool accumCFs,
  // 					    const Int* polMap_ptr, const Int *chanMap_ptr,
  // 					    const Double *uvwScale_ptr, const Double *offset_ptr,
  // 					    const Double *dphase_ptr, Int XThGrid, Int YThGrid);
  template <class T>
  void cuDataToGridImpl_p(T* gridStore,  Int* gridShape, //4-elements
			  
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
			  
			  Complex **cfV,//[2], 
			  Int *cfShape,//[4], 
			  Float *sampling,//[2], 
			  const Int *support, //[2]

  			  Double* sumWt_ptr,
  			  const Bool dopsf, const Bool accumCFs,
  			  const Int* polMap_ptr, const Int *chanMap_ptr,
  			  const Double *uvwScale_ptr, const Double *offset_ptr,
  			  const Double *dphase_ptr, Int XThGrid, Int YThGrid, Int *gridHits);

  template <class T>
  __global__
  void cuDataToGridImpl2_p(T* gridStore,  Int* gridShape, //4-elements
			  
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
			   
			   Complex **cfV,//[2], 
			   Int *cfShape,//[4], 
			   Float *sampling,//[2], 
			   const Int *support, //[2]
			   
			   Double* sumWt_ptr,
			   const Bool dopsf, const Bool accumCFs,
			   const Int* polMap_ptr, const Int *chanMap_ptr,
			   const Double *uvwScale_ptr, const Double *offset_ptr,
			   const Double *dphase_ptr, Int XThGrid, Int YThGrid,
			   Int *gridHits);
  
  template <class T>
  void cDataToGridImpl2_p(T* gridStore,  Int* gridShape, //4-elements
			  
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
			  
			  const Complex **cfV,//[2], 
			  Int *cfShape,//[4], 
			  Float *sampling,//[2], 
			  const Int *support, //[2]
			  
  			  Double* sumWt_ptr,
  			  const Bool dopsf, const Bool accumCFs,
  			  const Int* polMap_ptr, const Int *chanMap_ptr,
  			  const Double *uvwScale_ptr, const Double *offset_ptr,
  			  const Double *dphase_ptr, Int XThGrid, Int YThGrid);
  
  template <class T>
  void cDataToGridImpl_p(T* gridStore, Int* gridShape, VBStore* vbs,
			 Matrix<Double>* sumwt, 
			 const Bool dopsf,
			 const Int* polMap_ptr, const Int *chanMap_ptr,
			 const Double *uvwScale_ptr, const Double *offset_ptr,
			 const Double *dphase_ptr, Int XThGrid=0, Int YThGrid=0);
  
  void csgrid(Double pos[2], Int loc[3], Double off[3], Complex& phasor, 
	      const Int& irow, const Double* uvw, const Double& dphase, 
	      const Double& freq, const Double* scale, const Double* offset,
	      const Float sampling[2]);
  
  Bool ccomputeSupport(const uInt *BLCXi_ptr, const uInt *BLCYi_ptr,
		       const uInt *TRCXi_ptr, const uInt *TRCYi_ptr,
		       const uInt subGridShape[2],
		       const Int& XThGrid, const Int& YThGrid,
		       const Int support[2], const Float sampling[2],
		       const Double pos[2], const Int loc[3],
		       Float iblc[2], Float itrc[2]);
  
  Complex* cgetConvFunc_p(Int cfShape[4], VBStore* vbs, Double& wVal, Int& fndx, Int& wndx, 
			  //Int **mNdx, Int  **conjMNdx,
			  Int mNdx[4][1], Int  conjMNdx[4][1],
			  Int& ipol, uInt& mRow);
  
  
  void ccachePhaseGrad_g(Complex *cached_phaseGrad_p, Int phaseGradNX, Int phaseGradNY,
			 Double* cached_PointingOffset_p, Double* pointingOffset,
			 Int cfShape[4], Int convOrigin[4]);
  
  template <class T>
  Complex caccumulateOnGrid(T* gridStore, const Int* gridInc_p, const Complex *cached_phaseGrad_p,
			    const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			    const Complex* convFuncV, const Int *cfInc_p,Complex nvalue,
			    Double wVal, Int *supBLC_ptr, Int *supTRC_ptr,
			    Float* scaledSampling_ptr, Double* off_ptr, Int* convOrigin_ptr, 
			    Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr, Bool finitePointingOffset,
			    Bool doPSFOnly, Bool& foundCFPeak);
  template <class T>
  __device__
  cuComplex cuaccumulateOnGrid(T* gridStore, const Int* gridInc_p, const cuComplex *cached_phaseGrad_p,
			       const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			       const cuComplex* convFuncV, const Int *cfInc_p,cuComplex nvalue,
			       Double wVal, Int *supBLC_ptr, Int *supTRC_ptr, const Int *support_ptr,
			       Float* scaledSampling_ptr, Double* off_ptr, Int* convOrigin_ptr, 
			       Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr, Bool finitePointingOffset,
			       Bool doPSFOnly, Bool& foundCFPeak, Int& gridHits);
  template <class T>
  __device__
  cuComplex cuaccumulateOnGrid2(T* gridStore, const Int* gridInc_p, const cuComplex *cached_phaseGrad_p,
			       const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			       const cuComplex* convFuncV, const Int *cfInc_p,cuComplex nvalue,
			       Double wVal, Int *supBLC_ptr, Int *supTRC_ptr, const Int *support_ptr,
			       Float* scaledSampling_ptr, Double* off_ptr, Int* convOrigin_ptr, 
			       Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr, Bool finitePointingOffset,
			       Bool doPSFOnly, Bool& foundCFPeak, Int& gridHits);
  //
  //----------------------------------------------------------------------
  //
  inline Complex getFrom4DArray(const Complex * store,
				const Int* iPos, const Int inc[4])
  {return *(store+(iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3]));};
  //
  //----------------------------------------------------------------------
  //
  inline void cacheAxisIncrements(const Int n[4], Int inc[4])
  {inc[0]=1; inc[1]=inc[0]*n[0]; inc[2]=inc[1]*n[1]; inc[3]=inc[2]*n[2];(void)n[3];}
  //
  //----------------------------------------------------------------------
  //
  template <class T>
  void addTo4DArray(T *store, const Int *iPos, const Int* inc, 
		    Complex& nvalue, Complex& wt)
  {store[iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3]] += (nvalue*wt);}
  //
  //----------------------------------------------------------------------
  //----------------------------------------------------------------------
  //----------------------------------------------------------------------
  //----------------------------------------------------------------------
  //----------------------------------------------------------------------
  // CUDA VERSIONS
  //
#define CU_GET_FROM_4DARRAY(store, iPos, inc) (*((store)+((iPos)[0] + (iPos)[1]*(inc)[1] + (iPos)[2]*(inc)[2] +(iPos)[3]*(inc)[3])))
  
#define CU_CACHE_AXIS_INCREMENTS(n,inc)   ({(inc)[0]=1; (inc)[1]=(inc)[0]*(n)[0]; (inc)[2]=(inc)[1]*(n)[1]; (inc)[3]=(inc)[2]*(n)[2];})
  
#define NINT(v)	((Int)std::floor(v+0.5))
  
  // __global__
  // inline cuComplex cugetFrom4DArray(const cuComplex * store,
  // 				const Int* iPos, const Int inc[4])
  // {return *(store+(iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3]));};
  //
  //----------------------------------------------------------------------
  //
  // __global__
  // inline void cucacheAxisIncrements(const Int n[4], Int inc[4])
  // {inc[0]=1; inc[1]=inc[0]*n[0]; inc[2]=inc[1]*n[1]; inc[3]=inc[2]*n[2];(void)n[3];}
  //
  //----------------------------------------------------------------------
  //
  // Since the complex types in CUDA and CASA are different types,
  // following can't be templated (sigh!)
  //
  __device__
  void cuaddTo4DArray(Complex *store, const Int *iPos, const Int* inc, 
		      cuComplex nvalue, cuComplex wt);
  // {
  //   cuComplex tmp=cuCmulf(nvalue,wt);
  //   int n=iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3];
  //   ((cuComplex *)store)[n].x += tmp.x;
  //   ((cuComplex *)store)[n].y += tmp.y;
  // }
  __device__
  void cuaddTo4DArray(DComplex *store, const Int *iPos, const Int* inc, 
		      cuComplex nvalue, cuComplex wt);
  // {
  //   cuComplex tmp=cuCmulf(nvalue,wt);
  //   int n=iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3];
  //   ((cuDoubleComplex *)store)[n].x += tmp.x;
  //   ((cuDoubleComplex *)store)[n].y += tmp.y;
  // }
  //
  //----------------------------------------------------------------------
  //
  __device__
  void cusgrid(Double pos[3], Int loc[3], Double off[3], cuComplex* phasor, 
	       const Int irow, const Double* uvw, const Double dphase, 
	       const Double freq, const Double* scale, const Double* offset,
	       const Float sampling[2]);
  
  __device__
  Bool cucomputeSupport(const uInt *BLCXi_ptr, const uInt *BLCYi_ptr,
			const uInt *TRCXi_ptr, const uInt *TRCYi_ptr,
			const uInt subGridShape[2],
			const Int XThGrid, const Int YThGrid,
			const Int support[2], const Float sampling[2],
			const Double pos[2], const Int loc[3],
			Float iblc[2], Float itrc[2]);
};
#endif // 

