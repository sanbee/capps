// -*- C++ -*-
//#include <synthesis/TransformMachines/cDataToGridImpl.h>
#include <casa/Arrays/Matrix.h>
//#include <cufft.h>
#include <cuComplex.h>
#include "cDataToGridImpl.h"
#include <typeinfo>
#include <stdio.h>
#include "./GPUGEOM.h"

extern "C" {
#include <cuUtils.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
};

namespace casa{

#include "cDataToGridImpl_testcode.cu"

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
			  
			  Complex **cfV, //[2]
  			  Int *cfShape,//[4], //[4]
  			  Float *sampling,//[2], 
			  const Int *support, //[2]
			  
  			  Double* sumWt_ptr,
  			  const Bool dopsf, const Bool accumCFs,
  			  const Int* polMap_ptr, const Int *chanMap_ptr,
  			  const Double *uvwScale_ptr, const Double *offset_ptr,
  			  const Double *dphase_ptr, Int XThGrid, Int YThGrid,
			  Int *gridHits)
  {
    // This is a host-side function!!!

    //    printf("DSubGridShape = %d %d\n", subGridShape[0], subGridShape[1]);
    // Int NB=16;
    // Int NT=17;
    Int NB=XBLOCKSIZE;
    Int NT=XTHREADSIZE;

    dim3 dimBlock ( NB, NB, 1 ) ;
    dim3 dimThread( NT, NT, 1 ) ;
    
    //cudaProfilerStart();


    cuDataToGridImpl2_p<<<dimBlock,dimThread>>>(gridStore, gridShape,subGridShape,BLCXi,BLCYi,TRCXi,TRCYi,
						visCube_ptr,imgWts_ptr,flagCube_ptr,rowFlag_ptr,uvw_ptr,
						      
						nRow,rbeg,rend,nDataChan,nDataPol,startChan,endChan,vbSpw,
						vbFreq,
						      
						cfV,cfShape,sampling,support,
		       
						sumWt_ptr,dopsf,accumCFs,polMap_ptr,chanMap_ptr,
						uvwScale_ptr,offset_ptr,dphase_ptr,XThGrid,YThGrid,gridHits);



    //    cudaDeviceSynchronize();

    // cDataToGridImpl2_p(gridStore, gridShape,subGridShape,BLCXi,BLCYi,TRCXi,TRCYi,
    // 		       visCube_ptr,imgWts_ptr,flagCube_ptr,rowFlag_ptr,uvw_ptr,
		       
    // 		       nRow,rbeg,rend,nDataChan,nDataPol,startChan,endChan,vbSpw,
    // 		       vbFreq,
		       
    // 		       cfV,cfShape,sampling,support,
		       
    // 		       sumWt_ptr,dopsf,accumCFs,polMap_ptr,chanMap_ptr,
    // 		       uvwScale_ptr,offset_ptr,dphase_ptr,XThGrid,YThGrid);
    
    cudaError_t err=cudaGetLastError();
    if (err != cudaSuccess)
      {
	cerr << "###Cuda error: Failed to run the kernel " << cudaGetErrorString (err) << endl;
	exit(0);
      }
    //cudaProfilerStop();
  };
  //
  //---------------------------------------------------------------------------------
  // The following function is the CUDA kernel for gridding.
  //
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
			   const Int nDataChan,const Int nDataPol, 
			   const Int startChan, const Int endChan, const Int vbSpw,
			   const Double *vbFreq,
			   
			   Complex **cfV,//[2],
			   Int *cfShape,//[4], //[4]
			   Float *sampling,//[2], 
			   const Int *support, //[2]
			   
			   Double* sumWt_ptr,
			   const Bool dopsf, const Bool accumCFs,
			   const Int* polMap_ptr, const Int *chanMap_ptr,
			   const Double *uvwScale_ptr, const Double *offset_ptr,
			   const Double *dphase_ptr, Int XThGrid, Int YThGrid,
			   Int *gridHits)
  {
    XThGrid = blockIdx.x;
    YThGrid = blockIdx.y;

    Int nw, nCFFreq, nx,ny, nGridPol, nGridChan;
    Int targetIMChan, targetIMPol;
    
    Int loc[3], iloc[4],tiloc[4];
    Int convOrigin[4], gridInc_l[4], cfInc_l[4]; 
    Double pos[3], off[3];
    Int igrdpos[4];
    
    cuComplex phasor, nvalue;
    cuComplex norm;
    Bool Dummy;
    Double *pointingOffset_ptr=NULL;
    Double *cached_PointingOffset_ptr=NULL;
    
    nx=gridShape[0]; ny=gridShape[1];
    nGridPol=gridShape[2]; nGridChan=gridShape[3];
    Bool gDummy;

    CU_CACHE_AXIS_INCREMENTS(gridShape, gridInc_l);
    
    iloc[0]=iloc[1]=iloc[2]=iloc[3]=0;
    
    // Loop over all Rows, and all channels and polarization in each
    // row.  if the data[Row, Chan, Pol] data point is not flagged,
    // use it's UVW co-ordinate to determine if the current Block is
    // touched by the CF for this data point (this decision is in the
    // variable onMyGrid below).  
    //
    // If onMyGrid == True, then determine the pixels within this
    // block that need to do the addition.  With threads-per-block
    // equal to the support size of the CF (currently 8x8 pixels), the
    // CF pixels will map to the threadIdx.  This is done in the
    // cuaccumulateToGrid() function below.  Inside this function, we
    // loop over all the pixels of the CF (8x8 pixels), but do the
    // gridding (accumultion) only if the CF pixel index matches the
    // threadIdx.x and threadIdx.y.
    //

    convOrigin[0]=cfShape[0]/2;	    convOrigin[1]=cfShape[1]/2;
    convOrigin[2]=cfShape[2]/2;	    convOrigin[3]=cfShape[3]/2;
    CU_CACHE_AXIS_INCREMENTS(cfShape, cfInc_l);

    Bool finitePointingOffsets = False;
    for(Int irow=rbeg; irow< rend; irow++)
      {   
  	if(!(*(rowFlag_ptr+irow)))
  	  {   
	    const Float *imgWts_Chan_offset=imgWts_ptr + irow*nDataChan;
  	    for(Int ichan=startChan; ichan< endChan; ichan++)
  	      {
  		//if (*(imgWts_ptr + ichan+irow*nDataChan)!=0.0) 
		const Float imgWts_Chan = *(imgWts_Chan_offset+ichan);
		if ((imgWts_Chan)!=0.0) 
  		  {  
  		    targetIMChan=chanMap_ptr[ichan];
		    
  		    if((targetIMChan>=0) && (targetIMChan<nGridChan)) 
  		      {
  			Double dataWVal = 0;
  			// if (uvw_ptr != NULL) dataWVal = uvw_ptr[irow+nRow*2];
			
  			Int wndx = 0;//(int)(sqrt(vbs->cfBSt_p.wIncr*abs(dataWVal*vbFreq[ichan]/C::c)));
			
  			Int cfFreqNdx=0;
  			Float s;
			
  			cusgrid(pos,loc,off, &phasor, irow, uvw_ptr, dphase_ptr[irow], vbFreq[ichan], 
  			       uvwScale_ptr, offset_ptr, sampling);
			
  			Float cfblc[2], cftrc[2];

  			Bool onMyGrid=
			  cucomputeSupport(BLCXi,BLCYi, TRCXi, TRCYi, subGridShape,
					   XThGrid, YThGrid, support, sampling, pos, 
					   loc,cfblc,cftrc);

  			if (onMyGrid)
  			  {
			    // Gather some stats
			    // gridHits[XThGrid + YThGrid*subGridShape[0]]++;

			    //			    printf("%d %d %d\n",XThGrid, YThGrid,gridHits[XThGrid + YThGrid*subGridShape[0]]);
			    
			    Int iblc[2], itrc[2];
  			    iblc[0]=NINT((cfblc[0]-pos[0]));///sampling[0]);
  			    iblc[1]=NINT((cfblc[1]-pos[1]));///sampling[1]);
  			    itrc[0]=NINT((cftrc[0]-pos[0]));///sampling[0]);
  			    itrc[1]=NINT((cftrc[1]-pos[1]));///sampling[1]); 

  			    // Loop over all image-plane polarization planes.
  			    for(Int ipol=0; ipol< nDataPol; ipol++) 
  			      { 
				const Int iCiP_offset = ipol + ichan*nDataPol + irow*nDataPol*nDataChan;
				const Bool iCiPFlagCube = *(flagCube_ptr + iCiP_offset);
  				//if((!(*(flagCube_ptr + iCiP_offset))))
				if((!(iCiPFlagCube)))
  				  {  
  				    targetIMPol=polMap_ptr[ipol];
  				    if ((targetIMPol>=0) && (targetIMPol<nGridPol)) 
  				      {
  					igrdpos[2]=targetIMPol; igrdpos[3]=targetIMChan;
					
  					//if(dopsf) {nvalue.x=(*(imgWts_ptr + ichan + irow*nDataChan));nvalue.y=0.0;}
					if(dopsf) {nvalue.x=((imgWts_Chan));nvalue.y=0.0;}
  					else      
					  {
					    cuComplex vis;
					    // Float twt;
					    //twt=*(imgWts_ptr+ichan+irow*nDataChan);
					    //twt=(imgWts_Chan);
					    vis=cuCmulf(*((cuComplex *)visCube_ptr+ iCiP_offset),phasor);
					    
					    nvalue.x= imgWts_Chan * vis.x;
					    nvalue.y= imgWts_Chan * vis.y;
					  }
					
  					norm.x = norm.y = 0.0;
  					Bool foundCFPeak=False;
  					uInt nMueller=1; //vbs->cfBSt_p.nMueller
  					for (uInt mRow=0;mRow<nMueller; mRow++) 
  					  {
  					    const cuComplex* convFuncV;
  					    Int muellerElementsIndex[4][1] ={{0},{},{},{1}};
  					    Int conjMuellerElementsIndex[4][1] ={{1},{},{},{0}};
					    Int polNdx;
					    if (dataWVal > 0.0) polNdx=muellerElementsIndex[ipol][mRow];
					    else                polNdx=conjMuellerElementsIndex[ipol][mRow];
					    convFuncV = (cuComplex *)cfV[polNdx];
					    
  					    // convOrigin[0]=cfShape[0]/2;	    convOrigin[1]=cfShape[1]/2;
  					    // convOrigin[2]=cfShape[2]/2;	    convOrigin[3]=cfShape[3]/2;
  					    Bool psfOnly=((dopsf==True) && (accumCFs==False));
					    
  					    Int cachedPhaseGradNX=0,cachedPhaseGradNY=0;
  					    cuComplex *cached_PhaseGrad_ptr=NULL;
					    
					    cuComplex tmpNorm; tmpNorm.x=tmpNorm.y=0.0;

  					    tmpNorm = cuaccumulateOnGrid(gridStore, gridInc_l, cached_PhaseGrad_ptr, 
					    				 cachedPhaseGradNX, cachedPhaseGradNY,
					    				 convFuncV, cfInc_l, nvalue,dataWVal,
					    				 iblc, itrc, support, sampling, off, 
					    				 convOrigin, cfShape, loc, igrdpos,
					    				 finitePointingOffsets, psfOnly, foundCFPeak,
					    				 gridHits[XThGrid + YThGrid*subGridShape[0]]);
					    norm.x += tmpNorm.x; norm.y += tmpNorm.y;
  					  }
					
  					//*(sumWt_ptr+targetIMPol+targetIMChan*nGridPol)+= *(imgWts_ptr+ichan+irow*nDataChan)*cuCabsf(norm);
					*(sumWt_ptr+targetIMPol+targetIMChan*nGridPol)+= (imgWts_Chan)*cuCabsf(norm);
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
  template <class T>
  __device__
  cuComplex cuaccumulateOnGrid(T* gridStore, const Int* gridInc_p, const cuComplex *cached_phaseGrad_p,
			       const Int cachedPhaseGradNX, const Int cachedPhaseGradNY,
			       const cuComplex* convFuncV, const Int *cfInc_p, cuComplex nvalue,
			       Double wVal, 
			       Int *supBLC_ptr, Int *supTRC_ptr,
			       const Int *support_ptr,
			       Float* scaledSampling_ptr, 
			       Double* off_ptr, Int* convOrigin_ptr, 
			       Int* cfShape, Int* loc_ptr, Int* iGrdpos_ptr,
			       Bool finitePointingOffset,
			       Bool doPSFOnly, Bool& foundCFPeak,
			       Int& gridHits)
  {
    Int iloc_ptr[4]={0,0,0,0};
    // !!! Converting sampling and offset to Ints. Check if this still gives correct results.
    Int scaledSampling_l[2]={(Int)scaledSampling_ptr[0], (Int)scaledSampling_ptr[1]};
    Int off_l[2]={(Int)off_ptr[0], off_ptr[1]};
    
    cuComplex wt;
    //cuComplex cfArea;cfArea.x=1.0; 
    cuComplex norm;norm.x=norm.y=0.0;

    //    Bool finitePointingOffset_l=finitePointingOffset;
    //    Bool doPSFOnly_l=doPSFOnly;
    Double wVal_l=wVal;
    cuComplex nvalue_l=nvalue;
    
    // Int phaseGradOrigin_l[2]; 

    // phaseGradOrigin_l[0] = cachedPhaseGradNX/2;
    // phaseGradOrigin_l[1] = cachedPhaseGradNY/2;
    
    Int xOff=off_l[0]-1+convOrigin_ptr[0], 
      yOff = off_l[1]-1+convOrigin_ptr[1];

    for(Int iy=supBLC_ptr[1]; iy <= supTRC_ptr[1]; iy++) 
      {
	//iloc_ptr[1]=((scaledSampling_l[1]*iy+off_l[1])-1)+convOrigin_ptr[1];
	iloc_ptr[1]=scaledSampling_l[1]*iy+yOff;
	iGrdpos_ptr[1]=loc_ptr[1]+iy;

	for(Int ix=supBLC_ptr[0]; ix <= supTRC_ptr[0]; ix++) 
	  {
	    //iloc_ptr[0]=((scaledSampling_l[0]*ix+off_l[0])-1)+convOrigin_ptr[0];
	    iloc_ptr[0]=scaledSampling_l[0]*ix+xOff;
	    iGrdpos_ptr[0]=loc_ptr[0]+ix;
	    {
	      if (ix==0 and iy==0) foundCFPeak=True;

	      if (ix+support_ptr[0]==threadIdx.x and iy+support_ptr[1]==threadIdx.y)
		{
		  // printf("       # : %d %d %d %d \n",ix+support_ptr[0], iy+support_ptr[1], threadIdx.x,threadIdx.y);
		  wt = CU_GET_FROM_4DARRAY(convFuncV, iloc_ptr,cfInc_p);///cfArea;

		  // !!!UNCOMMENT THE FOLLOWING 2 LINES
		  // if (wVal > 0.0) {wt = cuConjf(wt);}
		  norm = cuCaddf(norm,wt);
	      

		  // !!! ENABLE COMPUTING
		  // The following uses raw index on the 4D grid
		  cuaddTo4DArray(gridStore,iGrdpos_ptr,gridInc_p, nvalue,wt);
		  //gridHits++;
		}
	    }
	  }
      }
    return norm;
  }
  //
  //---------------------------------------------------------------------------------
  //
  __device__
  void cusgrid(Double pos[3], Int loc[3], Double off[3], cuComplex* phasor, 
	      const Int irow, const Double* uvw_ptr, const Double dphase, 
	      const Double freq, const Double* scale, const Double* offset,
	      const Float sampling[2])
  {
    Float phase;
    Float uvw_l[3]={0.0,0.0,0.0}; // This allows gridding of weights
    Float LambdaInv=freq/299792458.0;
    Float offset_l[3]={offset[0],offset[1],offset[2]};
    Float freq_l=freq;
    Float scale_l[3]={scale[0],scale[1],scale[2]};
    Float pos_l[3],sampling_l[3]={sampling[0], sampling[1], sampling[2]};
    // centered on the uv-origin
    //  if (uvw.nelements() > 0) for(Int i=0;i<3;i++) uvw_l[i]=uvw(i,irow);
    // if (uvw.nelements() > 0) for(Int i=0;i<3;i++) uvw_l[i]=uvw_ptr[i+irow*3];
    if (uvw_ptr != NULL) 
      {
	for(Int i=0;i<3;i++) 
	  {
	    uvw_l[i]=uvw_ptr[i+irow*3];
	  }
      }
    // else 
    //   printf("cusgrid::UVW == 0\n");
    
    pos_l[2]=0;//__fsqrt_rn(abs(scale_l[2]*uvw_l[2]*LambdaInv))+offset_l[2];
    loc[2]=0;//NINT(pos_l[2]);
    off[2]=0;
    
    for(Int idim=0;idim<2;idim++)
      {
	pos_l[idim]=scale_l[idim]*uvw_l[idim]*LambdaInv+(offset_l[idim]);
	loc[idim]=NINT(pos_l[idim]);
	//	off[idim]=SynthesisUtils::nint((loc[idim]-pos_l[idim])*sampling[idim]+1);
	off[idim]=NINT((loc[idim]-(Float)pos_l[idim])*sampling[idim]);
      }
    
    if (fabs(dphase) >= 1e-8)
      {
	phase=-2.0*M_PI*dphase*LambdaInv;
	Float sp,cp;
	sincos(phase,&sp,&cp);
	(*phasor).x=cp;
	(*phasor).y=sp;
      }
    else
      {
	(*phasor).x=1.0;
	(*phasor).y=0.0;
      }
    pos[0]=pos_l[0]; pos[1]=pos_l[1]; pos[2]=pos_l[2];
  }
  //
  //---------------------------------------------------------------------------------
  //
  __device__
  void cuaddTo4DArray(Complex *store, const Int *iPos, const Int* inc, 
		      cuComplex nvalue, cuComplex wt)
  {
    cuComplex tmp=cuCmulf(nvalue,wt);

    int n=iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3];
    ((cuComplex *)store)[n].x += tmp.x;
    ((cuComplex *)store)[n].y += tmp.y;


  }
  //
  //---------------------------------------------------------------------------------
  //
  __device__
  void cuaddTo4DArray(DComplex *store, const Int *iPos, const Int* inc, 
		      cuComplex nvalue, cuComplex wt)
  {
    cuComplex tmp=cuCmulf(nvalue,wt);
    int n=iPos[0] + iPos[1]*inc[1] + iPos[2]*inc[2] +iPos[3]*inc[3];

    ((cuDoubleComplex *)store)[n].x += tmp.x;
    ((cuDoubleComplex *)store)[n].y += tmp.y;

    // cuComplex tmp;
    // tmp.x=((cuDoubleComplex *)store)[iPos[0]].x;
    // tmp.y=((cuDoubleComplex *)store)[iPos[0]].y;
  }
  //
  //---------------------------------------------------------------------------------
  //
  __device__
  void cucalcIntersection(const Int blc1[2], const Int trc1[2], 
			  const Float blc2[2], const Float trc2[2],
			  Float blc[2], Float trc[2])
  {
    Float dblc, dtrc;
    for (Int i=0;i<2;i++)
      {
        dblc = blc2[i] - blc1[i];
        dtrc = trc2[i] - trc1[i];

        if ((dblc >= 0) and (dtrc >= 0))
	  {
            blc[i] = blc1[i] + dblc;
            trc[i] = trc2[i] - dtrc;
	  }
        else if ((dblc >= 0) and (dtrc < 0))
	  {
            blc[i] = blc1[i] + dblc;
            trc[i] = trc1[i] + dtrc;
	  }
        else if ((dblc < 0) and (dtrc >= 0))
	  {
            blc[i] = blc2[i] - dblc;
            trc[i] = trc2[i] - dtrc;
	  }
        else
	  {
            blc[i] = blc2[i] - dblc;
            trc[i] = trc1[i] + dtrc;
	  }
      }
  }
  //
  // Check if the two rectangles interset (courtesy U.Rau).
  //
  __device__
  Bool cucheckIntersection(const Int blc1[2], const Int trc1[2], const Float blc2[2], const Float trc2[2])
  {
    // blc1[2] = {xmin1, ymin1}; 
    // blc2[2] = {xmin2, ymin2};
    // trc1[2] = {xmax1, ymax1};
    // trc2[2] = {xmax2, ymax2};

    if ((blc1[0] > trc2[0]) || (trc1[0] < blc2[0]) || (blc1[1] > trc2[1]) || (trc1[1] < blc2[1])) 
      return False;
    else
      return True;
  }
  //
  //---------------------------------------------------------------------------------
  //
  __device__
  Bool cucomputeSupport(const uInt *BLCXi_ptr, const uInt *BLCYi_ptr,
		       const uInt *TRCXi_ptr, const uInt *TRCYi_ptr,
		       const uInt subGridShape[2],
		       const Int XThGrid, const Int YThGrid,
		       const Int support[2], const Float sampling[2],
		       const Double pos[2], const Int loc[3],
		       Float iblc[2], Float itrc[2])
  {

    Int sup[2] = {support[0], support[1]};
    
    Int blc[2] = {BLCXi_ptr[XThGrid + YThGrid*subGridShape[0]], BLCYi_ptr[XThGrid + YThGrid*subGridShape[0]]};
    Int trc[2] = {TRCXi_ptr[XThGrid + YThGrid*subGridShape[0]], TRCYi_ptr[XThGrid + YThGrid*subGridShape[0]]};
    
    Float vblc[2]={pos[0]-sup[0],pos[1]-sup[1]}, vtrc[2]={pos[0]+sup[0],pos[1]+sup[1]};
    
    if (cucheckIntersection(blc,trc,vblc,vtrc))
      {
	cucalcIntersection(blc,trc,vblc,vtrc,iblc,itrc);
	return True;
      }
    return False;
  }
#include "cDataToGridImpl_def.h"
  
};
