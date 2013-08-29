// -*- C++ -*-
//#include <synthesis/TransformMachines/cDataToGridImpl.h>
#include <casa/Arrays/Matrix.h>
//#include <cufft.h>
#include <cuComplex.h>
#include "cDataToGridImpl.h"
#include <typeinfo>
#include <stdio.h>

namespace casa{

  __global__ void kernel_cuBlank(uInt *vbs,Int n)
  {
    printf("SubGrid: (%d %d) (%d %d) ", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  };

  void cuBlank(uInt* vbs,Int n)
  {
    Int NB=1, NT=1;
    dim3 dimBlock ( NB, NB, 1 ) ;
    dim3 dimThread( NT, NT, 1 ) ;
    
    //    kernel_cuBlank<<<dimBlock, dimThread>>>(vbs,n);
    cudaDeviceSynchronize();
  }
  
  __global__ void kernel_cuDataToGridImpl_p(Complex* gridStore,  Int* gridShape, //4-elements
					    
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
					    Int *cfShape,//[4], //[4]
					    Float *sampling,//[2], 
					    const Int *support, //[2]
					    
					    Double* sumWt_ptr,
					    const Bool dopsf, const Bool accumCFs,
					    const Int* polMap_ptr, const Int *chanMap_ptr,
					    const Double *uvwScale_ptr, const Double *offset_ptr,
					    const Double *dphase_ptr, Int XThGrid, Int YThGrid)
  {
    //printf("SubGrid: %d %d: %d %d %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y,nRow,rbeg,rend,nDataChan,nDataPol,startChan,endChan,vbSpw);
    printf("SubGrid: %d %d: %d %d vis=%f %f\n", blockIdx.x, blockIdx.y, BLCXi[0], BLCXi[20],visCube_ptr[10],visCube_ptr[20]);
  };

  __global__ void kernel_cuDataToGridImpl_p(DComplex* gridStore,  Int* gridShape, //4-elements
					    
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
					    Int *cfShape,//[4], //[4]
					    Float *sampling,//[2], 
					    const Int *support, //[2]
					    
					    Double* sumWt_ptr,
					    const Bool dopsf, const Bool accumCFs,
					    const Int* polMap_ptr, const Int *chanMap_ptr,
					    const Double *uvwScale_ptr, const Double *offset_ptr,
					    const Double *dphase_ptr, Int XThGrid, Int YThGrid)
  {
    //    printf("DSubGrid: %d %d: %d %d %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y,nRow,rbeg,rend,nDataChan,nDataPol,startChan,endChan,vbSpw);
    printf("DSubGrid: %d %d: %d %d vis=%f %f %d %d %d %d %d %d %d\n", blockIdx.x, blockIdx.y, BLCXi[0], BLCXi[20], 
	   ((cuComplex*)visCube_ptr)[10].x,((cuComplex *)visCube_ptr)[20].x,nRow, nDataChan, nDataPol,
	   cfShape[0], cfShape[1], cfShape[2], cfShape[3]);
  };
  
  template <class T>
  void cuDataToGridImpl_p(T* gridStore,  Int* gridShape, //4-elements
			  
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
  			  Int *cfShape,//[4], //[4]
  			  Float *sampling,//[2], 
			  const Int *support, //[2]
			  
  			  Double* sumWt_ptr,
  			  const Bool dopsf, const Bool accumCFs,
  			  const Int* polMap_ptr, const Int *chanMap_ptr,
  			  const Double *uvwScale_ptr, const Double *offset_ptr,
  			  const Double *dphase_ptr, Int XThGrid, Int YThGrid)
  {
    static Int tt=0;
    Int NB=1, NT=1;
    dim3 dimBlock ( NB, NB, 1 ) ;
    dim3 dimThread( NT, NT, 1 ) ;
    
    // kernel_cuDataToGridImpl_p<<<dimBlock,dimThread>>>(gridStore, gridShape, vbs, sumwt, dopsf, polMap_ptr, chanMap_ptr,
    // 						      uvwScale_ptr, offset_ptr, dphase_ptr, XThGrid, YThGrid);
    
    //    cerr << BLCXi << endl;

    // kernel_cuDataToGridImpl_p<<<dimBlock,dimThread>>>(gridStore, gridShape,subGridShape,BLCXi,BLCYi,TRCXi,TRCYi,
    // 						      visCube_ptr,imgWts_ptr,flagCube_ptr,rowFlag_ptr,uvw_ptr,
						      
    // 						      nRow,rbeg,rend,nDataChan,nDataPol,startChan,endChan,vbSpw,
    // 						      vbFreq,
						      
    // 						      cfV,cfShape,sampling,support,
		       
    // 						      sumWt_ptr,dopsf,accumCFs,polMap_ptr,chanMap_ptr,
    // 						      uvwScale_ptr,offset_ptr,dphase_ptr,XThGrid,YThGrid);
    // cudaDeviceSynchronize();
    // if (tt++ > 10) exit(0);

    cDataToGridImpl2_p(gridStore, gridShape,subGridShape,BLCXi,BLCYi,TRCXi,TRCYi,
    		       visCube_ptr,imgWts_ptr,flagCube_ptr,rowFlag_ptr,uvw_ptr,
		       
    		       nRow,rbeg,rend,nDataChan,nDataPol,startChan,endChan,vbSpw,
    		       vbFreq,
		       
    		       cfV,cfShape,sampling,support,
		       
    		       sumWt_ptr,dopsf,accumCFs,polMap_ptr,chanMap_ptr,
    		       uvwScale_ptr,offset_ptr,dphase_ptr,XThGrid,YThGrid);
    
    cudaError_t err=cudaGetLastError();
    if (err != cudaSuccess)
      {
	cerr << "###Cuda error: Failed to run the kernel " << cudaGetErrorString (err) << endl;
	exit(0);
      }
  };
  
  template <class T>
  void cDataToGridImpl2_p(T* gridStore,  Int* gridShape, //4-elements
			  
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
  			  Int *cfShape,//[4], //[4]
  			  Float *sampling,//[2], 
			  const Int *support, //[2]
			  
  			  Double* sumWt_ptr,
  			  const Bool dopsf, const Bool accumCFs,
  			  const Int* polMap_ptr, const Int *chanMap_ptr,
  			  const Double *uvwScale_ptr, const Double *offset_ptr,
  			  const Double *dphase_ptr, Int XThGrid, Int YThGrid)
  {
    //LogIO log_l(LogOrigin("ProtoVR[R&D]","DataToGridImpl_p"));
    
    //    Complex *cfV[2];
    
    Int nw, nCFFreq, nx,ny, nGridPol, nGridChan;
    Int targetIMChan, targetIMPol;
    
    Int loc[3], iloc[4],tiloc[4];
    Int convOrigin[4], gridInc_l[4], cfInc_l[4]; 
    Double pos[2], off[3];
    Int igrdpos[4];
    
    Complex phasor, nvalue, wt;
    Complex norm;
    Bool Dummy;
    // Bool * flagCube_ptr=vbs->flagCube_p.getStorage(Dummy);
    // Bool * rowFlag_ptr = vbs->rowFlag_p.getStorage(Dummy);
    // Float * imgWts_ptr = vbs->imagingWeight_p.getStorage(Dummy);
    // Complex * visCube_ptr = vbs->visCube_p.getStorage(Dummy);
    
    Double *pointingOffset_ptr=NULL; //vbs->cfBSt_p.pointingOffset
    Double *cached_PointingOffset_ptr=NULL;
    
    
    nx=gridShape[0]; ny=gridShape[1];
    nGridPol=gridShape[2]; nGridChan=gridShape[3];
    Bool gDummy;
    
    //    Double *freq=vbs->freq_p.getStorage(Dummy);
    
    cacheAxisIncrements(gridShape, gridInc_l);
    
    // nCFFreq = vbs->cfBSt_p.shape[0]; // shape[0]: nChan, shape[1]: nW, shape[2]: nPol
    // nw = vbs->cfBSt_p.shape[1];
    
    iloc[0]=iloc[1]=iloc[2]=iloc[3]=0;
    
    // if (accumCFs)
    //   {
    // 	startChan = vbs->startChan_p;
    // 	endChan = vbs->endChan_p;
    //   }
    // else 
    //   {
    // 	startChan = 0;
    // 	endChan = vbs->nDataChan_p;
    //   }
    
    Bool finitePointingOffsets = False;
    for(Int irow=rbeg; irow< rend; irow++)
      {   
  	if(!(*(rowFlag_ptr+irow)))
  	  {   
  	    for(Int ichan=startChan; ichan< endChan; ichan++)
  	      {
  		if (*(imgWts_ptr + ichan+irow*nDataChan)!=0.0) 
  		  {  
  		    targetIMChan=chanMap_ptr[ichan];
		    
  		    if((targetIMChan>=0) && (targetIMChan<nGridChan)) 
  		      {
  			// Double dataWVal = vbs->vb_p->uvw()(irow)(2);
  			Double dataWVal = 0;
  			//			if (vbs->uvw_p.nelements() > 0) dataWVal = vbs->uvw_p(irow,2);
  			if (uvw_ptr != NULL) dataWVal = uvw_ptr[irow+nRow*2];
			
  			Int wndx = 0;//(int)(sqrt(vbs->cfBSt_p.wIncr*abs(dataWVal*vbFreq[ichan]/C::c)));
			
  			Int cfFreqNdx=0;
  			// if (vbs->conjBeams_p) cfFreqNdx = vbs->cfBSt_p.conjFreqNdxMap[vbSpw][ichan];
  			// else cfFreqNdx = vbs->cfBSt_p.freqNdxMap[vbSpw][ichan];
			
  			Float s;
  			// s=vbs->cfBSt_p.CFBStorage->sampling;
  			// support[0]=vbs->cfBSt_p.CFBStorage->xSupport;
  			// support[1]=vbs->cfBSt_p.CFBStorage->ySupport;
			
  			// sampling[0] = sampling[1] = SynthesisUtils::nint(s);
			
  			//			const Double *uvw_ptr=vbs->uvw_p.getStorage(Dummy);
			
  			// *uvwScale_ptr=uvwScale_p.getStorage(Dummy),
  			// *offset_ptr=offset_p.getStorage(Dummy);;
			
  			csgrid(pos,loc,off, phasor, irow, uvw_ptr, dphase_ptr[irow], vbFreq[ichan], 
  			       uvwScale_ptr, offset_ptr, sampling);
			
  			Float cfblc[2], cftrc[2];
  			//		    pos[0]=1024.1;pos[1]=1025.6;
  			Bool onMyGrid=
			  ccomputeSupport(BLCXi,BLCYi, TRCXi, TRCYi, subGridShape,
					  XThGrid, YThGrid, support, sampling, pos, loc,cfblc,cftrc);
			//			onMyGrid=ccomputeSupport(vbs, XThGrid, YThGrid, support, sampling, pos, loc,cfblc,cftrc);
  			// 		    if (onGrid(nx, ny, nw, loc, support)) 
  			if (onMyGrid)
  			  {
  			    Int iblc[2], itrc[2];
			    
  			    iblc[0]=SynthesisUtils::nint((cfblc[0]-pos[0]));///sampling[0]);
  			    iblc[1]=SynthesisUtils::nint((cfblc[1]-pos[1]));///sampling[1]);
  			    itrc[0]=SynthesisUtils::nint((cftrc[0]-pos[0]));///sampling[0]);
  			    itrc[1]=SynthesisUtils::nint((cftrc[1]-pos[1]));///sampling[1]); 
			    
  			    Int dx=abs(itrc[0]-iblc[0])+1, dy=abs(itrc[1]-iblc[1])+1;
  			    Float cfFractioanlArea = (dx*dy)/(float)square(abs(support[1]+support[0])+1);
			    
  			    // Loop over all image-plane polarization planes.
  			    for(Int ipol=0; ipol< nDataPol; ipol++) 
  			      { 
  				if((!(*(flagCube_ptr + ipol + ichan*nDataPol + irow*nDataPol*nDataChan))))
  				  {  
  				    targetIMPol=polMap_ptr[ipol];
  				    if ((targetIMPol>=0) && (targetIMPol<nGridPol)) 
  				      {
  					igrdpos[2]=targetIMPol; igrdpos[3]=targetIMChan;
					
  					// if(accumCFs)     allPolNChanDone_l(ipol,ichan,irow)=True;
  					if(dopsf) nvalue=Complex(*(imgWts_ptr + ichan + irow*nDataChan));
  					else      nvalue= *(imgWts_ptr+ichan+irow*nDataChan)*
  						    (*(visCube_ptr+ipol+ichan*nDataPol+irow*nDataChan*nDataPol)*phasor);
					
  					norm = 0.0;
  					Bool foundCFPeak=False;
  					uInt nMueller=1; //vbs->cfBSt_p.nMueller
  					for (uInt mRow=0;mRow<nMueller; mRow++) 
  					  {
  					    const Complex* convFuncV;
  					    Int muellerElementsIndex[4][1] ={{0},{},{},{1}};
  					    Int conjMuellerElementsIndex[4][1] ={{1},{},{},{0}};
					    Int polNdx;
					    if (dataWVal > 0.0) polNdx=muellerElementsIndex[ipol][mRow];
					    else                polNdx=conjMuellerElementsIndex[ipol][mRow];
					    convFuncV = cfV[polNdx];
					    
  					    // convFuncV=cgetConvFunc_p(cfShape, vbs, dataWVal, cfFreqNdx, wndx, 
  					    // 			     // vbs->cfBSt_p.muellerElementsIndex,
  					    // 			     // vbs->cfBSt_p.conjMuellerElementsIndex, 
  					    // 			     muellerElementsIndex, conjMuellerElementsIndex,
  					    // 			     ipol,  mRow);
					    
  					    convOrigin[0]=cfShape[0]/2;
  					    convOrigin[1]=cfShape[1]/2;
  					    convOrigin[2]=cfShape[2]/2;
  					    convOrigin[3]=cfShape[3]/2;
  					    Bool psfOnly=((dopsf==True) && (accumCFs==False));
					    
  					    Int cachedPhaseGradNX=0,cachedPhaseGradNY=0;
  					    Complex *cached_PhaseGrad_ptr=NULL;
					    
  					    if (finitePointingOffsets && !psfOnly)
  					      ccachePhaseGrad_g(cached_PhaseGrad_ptr, cachedPhaseGradNX, cachedPhaseGradNY,	
  								cached_PointingOffset_ptr, pointingOffset_ptr, cfShape, convOrigin);//, cfRefFreq);//, vbs->imRefFreq());
					    
  					    cacheAxisIncrements(cfShape, cfInc_l);
  					    norm += caccumulateOnGrid(gridStore, gridInc_l, cached_PhaseGrad_ptr, 
  								      cachedPhaseGradNX, cachedPhaseGradNY,
  								      convFuncV, cfInc_l, nvalue,dataWVal,
  								      iblc, itrc, sampling, off, 
  								      convOrigin, cfShape, loc, igrdpos,
  								      finitePointingOffsets, psfOnly, foundCFPeak);
  					  }
					
  					*(sumWt_ptr+targetIMPol+targetIMChan*nGridPol)+= *(imgWts_ptr+ichan+irow*nDataChan)*abs(norm);
  				      }
  				  }
  			      } // End poln-loop
  			  }
  		      }
  		  }
  	      } // End chan-loop
  	  }
      } // End row-loop
    //exit(0);
  }
  
  template <class T>
  void cDataToGridImpl_p(T* gridStore,  Int* gridShape /*4-elements*/,
			 VBStore* vbs, Matrix<Double>* sumwt,
			 const Bool dopsf,
			 const Int* polMap_ptr, const Int *chanMap_ptr,
			 const Double *uvwScale_ptr, const Double *offset_ptr,
			 const Double *dphase_ptr, Int XThGrid, Int YThGrid)
  {
    //LogIO log_l(LogOrigin("ProtoVR[R&D]","DataToGridImpl_p"));
    
    // Complex tmp;
    // Bool isGridSinglePrecision=(typeid(gridStore[0]) == typeid(tmp));
    // cerr << "cuisGridSinglePrecision = " << isGridSinglePrecision << endl;
    
    
    Int nGridPol, nGridChan, nx, ny, nw, nCFFreq;
    Int targetIMChan, targetIMPol, rbeg, rend;
    Int startChan, endChan;
    Bool accumCFs;
    
    Float sampling[2],scaledSampling[2];
    Int support[2],loc[3], iloc[4],tiloc[4],scaledSupport[2];
    Int convOrigin[4], gridInc_l[4], cfInc_l[4]; 
    Double pos[2], off[3];
    Int igrdpos[4];
    
    Complex phasor, nvalue, wt;
    Complex norm;
    Int cfShape[4];
    Bool Dummy;
    Bool * flagCube_ptr=vbs->flagCube_p.getStorage(Dummy);
    Bool * rowFlag_ptr = vbs->rowFlag_p.getStorage(Dummy);
    Float * imgWts_ptr = vbs->imagingWeight_p.getStorage(Dummy);
    Complex * visCube_ptr = vbs->visCube_p.getStorage(Dummy);
    Double *sumWt_ptr=sumwt->getStorage(Dummy);
    
    //  Vector<Double> pointingOffset(cfb.getPointingOffset());
    // Double *pointingOffset_ptr=vbs->cfBSt_p.pointingOffset,
    //   *cached_PointingOffset_ptr=cached_PointingOffset_p.getStorage(Dummy);
    Double *pointingOffset_ptr=vbs->cfBSt_p.pointingOffset,
      *cached_PointingOffset_ptr=NULL;
    
    //    cerr << "Data_dptr = " << vbs->visCube_dptr << endl;
    
    
    Int vbSpw=vbs->spwID_p;
    
    
    rbeg = vbs->beginRow_p;
    rend = vbs->endRow_p;
    
    nx=gridShape[0]; ny=gridShape[1];
    nGridPol=gridShape[2]; nGridChan=gridShape[3];
    Bool gDummy;
    
    Double *freq=vbs->freq_p.getStorage(Dummy);
    
    cacheAxisIncrements(gridShape, gridInc_l);
    
    nCFFreq = vbs->cfBSt_p.shape[0]; // shape[0]: nChan, shape[1]: nW, shape[2]: nPol
    nw = vbs->cfBSt_p.shape[1];
    
    iloc[0]=iloc[1]=iloc[2]=iloc[3]=0;
    Int nDataChan=vbs->nDataChan_p,
      nDataPol = vbs->nDataPol_p;
    accumCFs=vbs->accumCFs_p;
    if (accumCFs)
      {
	startChan = vbs->startChan_p;
	endChan = vbs->endChan_p;
      }
    else 
      {
	startChan = 0;
	endChan = vbs->nDataChan_p;
      }
    
    
    //  cerr << "ProtoVR: " << rbeg << " " << rend << " " << startChan << " " << endChan << " " << nDataChan << " " << nDataPol << endl;
    
    // Bool finitePointingOffsets= (
    // 			       (fabs(pointingOffset_ptr[0])>0) ||  
    // 			       (fabs(pointingOffset_ptr[1])>0)
    // 			       );
    Bool finitePointingOffsets = False;
    for(Int irow=rbeg; irow< rend; irow++)
      {   
	if(!(*(rowFlag_ptr+irow)))
	  {   
	    for(Int ichan=startChan; ichan< endChan; ichan++)
	      {
		if (*(imgWts_ptr + ichan+irow*nDataChan)!=0.0) 
		  {  
		    targetIMChan=chanMap_ptr[ichan];
		    
		    if((targetIMChan>=0) && (targetIMChan<nGridChan)) 
		      {
			// Double dataWVal = vbs->vb_p->uvw()(irow)(2);
			Double dataWVal = 0;
			if (vbs->uvw_p.nelements() > 0) dataWVal = vbs->uvw_p(irow,2);
			
			Int wndx = (int)(sqrt(vbs->cfBSt_p.wIncr*abs(dataWVal*freq[ichan]/C::c)));
			
			Int cfFreqNdx;
			if (vbs->conjBeams_p) cfFreqNdx = vbs->cfBSt_p.conjFreqNdxMap[vbSpw][ichan];
			else cfFreqNdx = vbs->cfBSt_p.freqNdxMap[vbSpw][ichan];
			
			Float s;
			s=vbs->cfBSt_p.CFBStorage->sampling;
			support[0]=vbs->cfBSt_p.CFBStorage->xSupport;
			support[1]=vbs->cfBSt_p.CFBStorage->ySupport;
			
			sampling[0] = sampling[1] = SynthesisUtils::nint(s);
			
			const Double *uvw_ptr=NULL;
			if (vbs->uvw_p.nelements() > 0) uvw_ptr=vbs->uvw_p.getStorage(Dummy);
			// *uvwScale_ptr=uvwScale_p.getStorage(Dummy),
			// *offset_ptr=offset_p.getStorage(Dummy);;
			
			csgrid(pos,loc,off, phasor, irow, uvw_ptr, dphase_ptr[irow], freq[ichan], 
			       uvwScale_ptr, offset_ptr, sampling);
			
			Float cfblc[2], cftrc[2];
			//		    pos[0]=1024.1;pos[1]=1025.6;
			
			uInt subGridShape[2]={vbs->BLCXi.shape()(0), vbs->BLCXi.shape()(1)};
			const uInt *BLCXi_ptr=vbs->BLCXi.getStorage(Dummy);
			const uInt *BLCYi_ptr=vbs->BLCYi.getStorage(Dummy);
			const uInt *TRCXi_ptr=vbs->TRCXi.getStorage(Dummy);
			const uInt *TRCYi_ptr=vbs->TRCYi.getStorage(Dummy);
			Bool onMyGrid=
			  ccomputeSupport(BLCXi_ptr,BLCYi_ptr, TRCXi_ptr, TRCYi_ptr, subGridShape,
					  XThGrid, YThGrid, support, sampling, pos, loc,cfblc,cftrc);
			// ccomputeSupport(vbs, XThGrid, YThGrid, support, sampling, pos, loc,cfblc,cftrc);
			// 		    if (onGrid(nx, ny, nw, loc, support)) 
			if (onMyGrid)
			  {
			    
			    Int iblc[2], itrc[2];
			    
			    iblc[0]=SynthesisUtils::nint((cfblc[0]-pos[0]));///sampling[0]);
			    iblc[1]=SynthesisUtils::nint((cfblc[1]-pos[1]));///sampling[1]);
			    itrc[0]=SynthesisUtils::nint((cftrc[0]-pos[0]));///sampling[0]);
			    itrc[1]=SynthesisUtils::nint((cftrc[1]-pos[1]));///sampling[1]); 
			    
			    Int dx=abs(itrc[0]-iblc[0])+1, dy=abs(itrc[1]-iblc[1])+1;
			    Float cfFractioanlArea = (dx*dy)/(float)square(abs(support[1]+support[0])+1);
			    
			    // if ((irow < 3) && (ichan == 2))
			    //   {
			    //     // cerr << irow << "#[" << XThGrid << "," << YThGrid << "] " 
			    //     // 	 << (cfblc[0]) << " " << (cftrc[0]) << " " 
			    //     // 	 << (cfblc[1]) << " " << (cftrc[1]) << " " 
			    //     // 	 << dx << " " << dy << " " << (dx*dy) << " " << cfFractionalArea
			    //     // 	 << endl;
			    //     // cerr << irow << "#[" << XThGrid << "," << YThGrid << "] " 
			    //     // 	 << (pos[0]) << " " << (pos[1]) << endl;
			    
			    //     cerr << irow << " [" << XThGrid << "," << YThGrid << "] " 
			    // 	   << iblc[0] << " " << itrc[0] << " " 
			    // 	   << iblc[1] << " " << itrc[1] << " " << rend
			    // 	   << endl;
			    //   }
			    
			    // Loop over all image-plane polarization planes.
			    for(Int ipol=0; ipol< nDataPol; ipol++) 
			      { 
				if((!(*(flagCube_ptr + ipol + ichan*nDataPol + irow*nDataPol*nDataChan))))
				  {  
				    targetIMPol=polMap_ptr[ipol];
				    if ((targetIMPol>=0) && (targetIMPol<nGridPol)) 
				      {
					igrdpos[2]=targetIMPol; igrdpos[3]=targetIMChan;
					
					// if(accumCFs)     allPolNChanDone_l(ipol,ichan,irow)=True;
					if(dopsf) nvalue=Complex(*(imgWts_ptr + ichan + irow*nDataChan));
					else      nvalue= *(imgWts_ptr+ichan+irow*nDataChan)*
						    (*(visCube_ptr+ipol+ichan*nDataPol+irow*nDataChan*nDataPol)*phasor);
					
					norm = 0.0;
					// for (uInt mRow=0;mRow<conjMNdx[ipol].nelements(); mRow++) 
					// for (uInt mRow=0;mRow<vbs.cfBSt_p.conjMuellerElementsIndex[ipol].nelements(); mRow++) 
					Bool foundCFPeak=False;
					for (uInt mRow=0;mRow<vbs->cfBSt_p.nMueller; mRow++) 
					  {
					    Complex* convFuncV;
					    // CUWORK:  Essentially CFC.getCellPtr(FNDX, WNDX, POLNDX)
					    // CUWORK: CFC wrapper
					    Int muellerElementsIndex[4][1]     = {{0},{},{},{1}};
					    Int conjMuellerElementsIndex[4][1] = {{1},{},{},{0}};
					    
					    convFuncV=cgetConvFunc_p(cfShape, vbs, dataWVal, cfFreqNdx, wndx, 
								     // vbs->cfBSt_p.muellerElementsIndex,
								     // vbs->cfBSt_p.conjMuellerElementsIndex, 
								     muellerElementsIndex, conjMuellerElementsIndex,
								     ipol,  mRow);
					    
					    convOrigin[0]=cfShape[0]/2;
					    convOrigin[1]=cfShape[1]/2;
					    convOrigin[2]=cfShape[2]/2;
					    convOrigin[3]=cfShape[3]/2;
					    Bool psfOnly=((dopsf==True) && (accumCFs==False));
					    // // CUWORK: Convert to a global function with native types
					    
					    // Int cachedPhaseGradNX=cached_phaseGrad_p.shape()[0],
					    //   cachedPhaseGradNY=cached_phaseGrad_p.shape()[1];
					    // Complex *cached_PhaseGrad_ptr=cached_phaseGrad_p.getStorage(Dummy);
					    Int cachedPhaseGradNX=0,cachedPhaseGradNY=0;
					    Complex *cached_PhaseGrad_ptr=NULL;
					    
					    if (finitePointingOffsets && !psfOnly)
					      ccachePhaseGrad_g(cached_PhaseGrad_ptr, cachedPhaseGradNX, cachedPhaseGradNY,	
								cached_PointingOffset_ptr, pointingOffset_ptr, cfShape, convOrigin);//, cfRefFreq);//, vbs->imRefFreq());
					    
					    cacheAxisIncrements(cfShape, cfInc_l);
					    //cerr << gridShape[0] << " " << gridShape[1] << " " << gridInc_p[0] << " " << gridInc_p[0] << endl;
					    norm += caccumulateOnGrid(gridStore, gridInc_l, cached_PhaseGrad_ptr, 
								      cachedPhaseGradNX, cachedPhaseGradNY,
								      convFuncV, cfInc_l, nvalue,dataWVal,
								      iblc, itrc,/*support,*/ sampling, off, 
								      convOrigin, cfShape, loc, igrdpos,
								      finitePointingOffsets,psfOnly,foundCFPeak);
					  }
					
					//sumwt(targetIMPol,targetIMChan) += vbs->imagingWeight_p(ichan, irow);//*abs(norm);
					//cerr << sumwt << " " << targetIMPol << " " << targetIMChan << " " << vbs->imagingWeight_p(ichan, irow) << " " << abs(norm) << endl;
					// Int dx=abs(itrc[0]-iblc[0]+1), dy=abs(itrc[1]-iblc[1]+1);
					// Float cfPixArea = (float)square(abs(support[0]-support[1]+1));
					
					// Accumulate data weight only when the CF peak was used. This can also be done via
					// fractional area of the CF used, and probably should be done that for high accuracy 
					// using the norm of the CF.
					//if (foundCFPeak) 
					*(sumWt_ptr+targetIMPol+targetIMChan*nGridPol)+= *(imgWts_ptr+ichan+irow*nDataChan)*abs(norm);
				      }
				  }
			      } // End poln-loop
			  }
		      }
		  }
	      } // End chan-loop
	  }
      } // End row-loop
    //exit(0);
  }
  //
  //---------------------------------------------------------------------------------
  //
  void csgrid(Double pos[2], Int loc[3], Double off[3], Complex& phasor, 
	      const Int& irow, const Double* uvw_ptr, const Double& dphase, 
	      const Double& freq, const Double* scale, const Double* offset,
	      const Float sampling[2])
  {
    Double phase;
    //Vector<Double> uvw_l(3,0); // This allows gridding of weights
    Double uvw_l[3]={0.0,0.0,0.0}; // This allows gridding of weights
    
    //    const Double *uvw_ptr=uvw.getStorage(dd);
    // centered on the uv-origin
    //  if (uvw.nelements() > 0) for(Int i=0;i<3;i++) uvw_l[i]=uvw(i,irow);
    // if (uvw.nelements() > 0) for(Int i=0;i<3;i++) uvw_l[i]=uvw_ptr[i+irow*3];
    if (uvw_ptr != NULL) for(Int i=0;i<3;i++) uvw_l[i]=uvw_ptr[i+irow*3];
    
    pos[2]=sqrt(abs(scale[2]*uvw_l[2]*freq/C::c))+offset[2];
    loc[2]=SynthesisUtils::nint(pos[2]);
    off[2]=0;
    
    for(Int idim=0;idim<2;idim++)
      {
	pos[idim]=scale[idim]*uvw_l[idim]*freq/C::c+(offset[idim]);
	loc[idim]=SynthesisUtils::nint(pos[idim]);
	//	off[idim]=SynthesisUtils::nint((loc[idim]-pos[idim])*sampling[idim]+1);
	off[idim]=SynthesisUtils::nint((loc[idim]-pos[idim])*sampling[idim]);
      }
    
    if (dphase != 0.0)
      {
	phase=-2.0*C::pi*dphase*freq/C::c;
	Double sp,cp;
	sincos(phase,&sp,&cp);
	//      phasor=Complex(cos(phase), sin(phase));
	phasor=Complex(cp,sp);
      }
    else
      phasor=Complex(1.0);
    // cerr << "### " << pos[0] << " " << offset[0] << " " << loc[0] << " " << off[0] << " " << uvw_l[0] << endl;
    // exit(0);
  }
  //
  //---------------------------------------------------------------------------------
  //
  Bool ccomputeSupport(const uInt *BLCXi_ptr, const uInt *BLCYi_ptr,
		       const uInt *TRCXi_ptr, const uInt *TRCYi_ptr,
		       const uInt subGridShape[2],
		       const Int& XThGrid, const Int& YThGrid,
		       const Int support[2], const Float sampling[2],
		       const Double pos[2], const Int loc[3],
		       Float iblc[2], Float itrc[2])
  {
    //    Int sup[2] = {support[0]*sampling[0], support[1]*sampling[1]};
    Int sup[2] = {support[0], support[1]};
    // Int subGridShape[2]={vbs->BLCXi.shape()(0), vbs->BLCXi.shape()(1)};
    // Bool Dummy;
    // const uInt *BLCXi_ptr=vbs->BLCXi.getStorage(Dummy);
    // const uInt *BLCYi_ptr=vbs->BLCYi.getStorage(Dummy);
    // const uInt *TRCXi_ptr=vbs->TRCXi.getStorage(Dummy);
    // const uInt *TRCYi_ptr=vbs->TRCYi.getStorage(Dummy);
    
    Int blc[2] = {BLCXi_ptr[XThGrid + YThGrid*subGridShape[0]], BLCYi_ptr[XThGrid + YThGrid*subGridShape[0]]};
    Int trc[2] = {TRCXi_ptr[XThGrid + YThGrid*subGridShape[0]], TRCYi_ptr[XThGrid + YThGrid*subGridShape[0]]};
    
    // Int blc[2] = {vbs->BLCXi(XThGrid, YThGrid), vbs->BLCYi(XThGrid, YThGrid)};
    // Int trc[2] = {vbs->TRCXi(XThGrid, YThGrid), vbs->TRCYi(XThGrid, YThGrid)};
    
    Float vblc[2]={pos[0]-sup[0],pos[1]-sup[1]}, vtrc[2]={pos[0]+sup[0],pos[1]+sup[1]};

    if (SynthesisUtils::checkIntersection(blc,trc,vblc,vtrc))
      {
	SynthesisUtils::calcIntersection(blc,trc,vblc,vtrc,iblc,itrc);
	return True;
      }
    return False;
  }
  //
  //---------------------------------------------------------------------------------
  //
  Complex* cgetConvFunc_p(Int cfShape[4], VBStore* vbs,
			  Double& wVal, Int& fndx, Int& wndx,
			  //Int **mNdx, Int  **conjMNdx,
			  Int mNdx[4][1], Int conjMNdx[4][1],
			  Int& ipol, uInt& mRow)
  {
    Bool Dummy;
    Complex *tt;
    CFCStruct *tcfc;
    Int polNdx, shape[3];
    
    if (wVal > 0.0) polNdx=mNdx[ipol][mRow];
    else            polNdx=conjMNdx[ipol][mRow];
    
    tcfc=vbs->cfBSt_p.getCFB(fndx,wndx,polNdx);
    
    tt=tcfc->CFCStorage;
    cfShape[0]=tcfc->shape[0];
    cfShape[1]=tcfc->shape[1];
    
    return tt;
  };
  //
  //---------------------------------------------------------------------------------
  //
  void ccachePhaseGrad_g(Complex *cached_phaseGrad_p, Int phaseGradNX, Int phaseGradNY,
			 Double* cached_PointingOffset_p, Double* pointingOffset,
			 Int cfShape[4], Int convOrigin[4])
  {
    if (
	((fabs(pointingOffset[0]-cached_PointingOffset_p[0])) > 1e-6) ||
	((fabs(pointingOffset[1]-cached_PointingOffset_p[1])) > 1e-6) ||
	(phaseGradNX < cfShape[0]) || (phaseGradNY < cfShape[1])
	)
      {
	cerr << "Computing phase gradiant for pointing offset " 
	     << "[" << pointingOffset[0] << "," << pointingOffset[1] << "] ["
	     << cfShape[0] << "," << cfShape[1] << "]" << endl;
	
	Int nx=cfShape[0], ny=cfShape[1];
	Double grad;
	Complex phx,phy;
	
	cerr << "Resize cached_phaseGrad_p !!!" << endl;
	//	cached_phaseGrad_p.resize(nx,ny);
	cached_PointingOffset_p[0] = pointingOffset[0];
	cached_PointingOffset_p[1] = pointingOffset[1];
	
	for(Int ix=0;ix<nx;ix++)
	  {
	    grad = (ix-convOrigin[0])*pointingOffset[0];
	    Double sx,cx;
	    sincos(grad,&sx,&cx);
	    //	    phx = Complex(cos(grad),sin(grad));
	    phx = Complex(cx,sx);
	    for(Int iy=0;iy<ny;iy++)
	      {
		grad = (iy-convOrigin[1])*pointingOffset[1];
		Double sy,cy;
		sincos(grad,&sy,&cy);
		//		phy = Complex(cos(grad),sin(grad));
		phy = Complex(cy,sy);
		cached_phaseGrad_p[ix+iy*phaseGradNY]=phx*phy;
	      }
	  }
      }
  }
  //
  //---------------------------------------------------------------------------------
  //
  template <class T>
  Complex caccumulateOnGrid(T* gridStore,
			    const Int* gridInc_p,
			    const Complex *cached_phaseGrad_p,
			    const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			    const Complex* convFuncV, 
			    const Int *cfInc_p,
			    Complex nvalue,Double wVal, 
			    Int *supBLC_ptr, Int *supTRC_ptr,//Int* scaledSupport_ptr, 
			    Float* scaledSampling_ptr, 
			    Double* off_ptr, Int* convOrigin_ptr, 
			    Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr,
			    Bool finitePointingOffset,
			    Bool doPSFOnly, Bool& foundCFPeak)
  {
    Int iloc_ptr[4]={0,0,0,0};//   for (int i=0;i<4;i++) iloc_ptr[i]=0;
    
    Complex wt, cfArea=1.0; 
    Complex norm=0.0;
    Int Nth = 1;
    
    Bool finitePointingOffset_l=finitePointingOffset;
    Bool doPSFOnly_l=doPSFOnly;
    Double wVal_l=wVal;
    Complex nvalue_l=nvalue;
    
    Int phaseGradOrigin_l[2]; 
    //    phaseGradOrigin_l = cached_phaseGrad_p.shape()/2;
    phaseGradOrigin_l[0] = cachedPhaseGradNX/2;
    phaseGradOrigin_l[1] = cachedPhaseGradNY/2;
    
    //    for(Int iy=-scaledSupport_ptr[1]; iy <= scaledSupport_ptr[1]; iy++) 
    for(Int iy=supBLC_ptr[1]; iy <= supTRC_ptr[1]; iy++) 
      {
	iloc_ptr[1]=(Int)((scaledSampling_ptr[1]*iy+off_ptr[1])-1)+convOrigin_ptr[1];
	iGrdpos_ptr[1]=loc_ptr[1]+iy;
	
	//	for(Int ix=-scaledSupport_ptr[0]; ix <= scaledSupport_ptr[0]; ix++) 
	for(Int ix=supBLC_ptr[0]; ix <= supTRC_ptr[0]; ix++) 
	  {
	    iloc_ptr[0]=(Int)((scaledSampling_ptr[0]*ix+off_ptr[0])-1)+convOrigin_ptr[0];
	    iGrdpos_ptr[0]=loc_ptr[0]+ix;
	    {
	      if (ix==0 and iy==0) foundCFPeak=True;
	      wt = getFrom4DArray((const Complex * __restrict__ &)convFuncV, 
				  iloc_ptr,cfInc_p)/cfArea;
	      if (wVal > 0.0) {wt = conj(wt);}
	      norm += (wt);
	      if (finitePointingOffset && !doPSFOnly) 
		wt *= cached_phaseGrad_p[iloc_ptr[0]+phaseGradOrigin_l[0]+
					 iloc_ptr[1]+phaseGradOrigin_l[1]*cachedPhaseGradNY];
	      
	      // The following uses raw index on the 4D grid
	      addTo4DArray(gridStore,iGrdpos_ptr,gridInc_p, nvalue,wt);
	    }
	  }
      }
    return norm;
  }
#include "cDataToGridImpl_def.h"
  
};
