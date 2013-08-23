// -*- C -*-
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <Utils.h>
#include <math.h>
#include <cuUtils.h>
#include <vector>
/* #include <cuWTerm.h> */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define USE_AUTO 
//#undef USE_AUTO 
#define BLOCKSIZE 128
#define GRIDSIZE (2048*2048/128)

#define OVERSAMPLING 20

namespace casa{
  //
  //--------------------------------------------
  //
  void * allocateDeviceBuffer(int N)
  {
    void *d_buf;
    cudaMalloc((void**)&d_buf, N);
    if (cudaGetLastError() != cudaSuccess)
      {
	fprintf(stderr, "Cuda error: Failed to allocate\n");
	return 0;
      }
    
    return d_buf;
  }
  //
  //--------------------------------------------
  //
  int sendBufferToDevice(void *d_buf, void *h_buf, int N)
  {
    cudaMemcpy(d_buf, h_buf, N, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
      {
	fprintf(stderr, "Cuda error: Failed to send\n");
	return 0;
      }
    return 1;
  }
  //
  //--------------------------------------------
  //
  int getBufferFromDevice(void *h_buf, void *d_buf, int N)
  {
    cudaMemcpy(h_buf, d_buf, N, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess)
      {
	fprintf(stderr, "Cuda error: Failed to get\n");
	return 0;
      }
    return 1;
  }
  //
  //--------------------------------------------
  //
  int makeCUFFTPlan(cufftHandle *plan, int NX, int NY, cufftType type /*CUFFT_C2C*/)
  {
    /* Create a 2D FFT plan. */
    if (cufftPlan2d(plan, NX, NY, type) != CUFFT_SUCCESS)
      {
	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
	return 0;
      }
    return 1;
  }
  //
  //--------------------------------------------
  //
  int setCompatibilityMode(cufftHandle& plan, cufftCompatibility mode /*CUFFT_COMPATIBILITY_NATIVE*/)
  {
    if (cufftSetCompatibilityMode(plan, mode)!= CUFFT_SUCCESS)
      {
	fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
	return 0;
      }
    return 1;
  }
  //
  //--------------------------------------------
  //
  int inPlaceCUFFTC2C(cufftHandle& plan, cufftComplex *d_buf, int dir /*CUFFT_FORWARD */)
  {
    if (cufftExecC2C(plan, d_buf, d_buf, dir) != CUFFT_SUCCESS)
      {
	fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
	return 0;
      }
    cudaThreadSynchronize();
    return 1;
  }
  //
  //--------------------------------------------
  //
  cudaError
    freeHost(void* h_mem, memoryMode memMode)
  {
    if( PINNED == memMode ) {
      return cudaFreeHost(h_mem);
    }
    else {
      free(h_mem);
    }
    return cudaSuccess;
  }
  //
  //--------------------------------------------
  //
  cudaError
    mallocHost(void** h_mem ,uint memSize, memoryMode memMode, bool wc)
  {
    if( PINNED == memMode ) {
#if CUDART_VERSION >= 2020
      return cudaHostAlloc( h_mem, memSize, (wc) ? cudaHostAllocWriteCombined : 0 );
#else
      if (wc) {printf("Write-Combined unavailable on CUDART_VERSION less than 2020, running is: %d", CUDART_VERSION);
        return cudaMallocHost( h_mem, memSize );
#endif
      }
      else { // PAGEABLE memory mode
        *h_mem = malloc( memSize );
      }
      
      return cudaSuccess;
    }
    //
    //--------------------------------------------
    //
    cudaError
      memCpy(void* sink, void* source, uint memSize, cudaMemcpyKind direction, memoryMode memMode)
    {
      /* if( PINNED == memMode ) { */
      /*   return cudaMemcpyAsync( sink, source, memSize, direction, 0); */
      /* } */
      /* else { */
      /*   return cudaMemcpy( sink, source, memSize, direction); */
      /* } */
      return cudaMemcpy( sink, source, memSize, direction);
    }
    //
    //--------------------------------------------
    //
    void cpu_wTermApplySky(cufftComplex* screen, const int nx, const int ny,
			   const int tileWidthX, const int tileWidthY, 
			   const double wPixel,  const float sampling, 
			   const double wScale,  const int inner,      
			   const bool isNoOp)
    {
      double wValue=(wPixel*wPixel)/wScale;
      double twoPiW=2.0*M_PI*double(wValue);
      int convSize = nx;
      
      if (!isNoOp)
      	{
      	  for (int iy=-inner/2;iy<inner/2;iy++)
      	    {
      	      double m=sampling*double(iy);
      	      double msq=m*m;
      	      for (int ix=-inner/2;ix<inner/2;ix++)
      		{
      		  double l=sampling*double(ix);
      		  double rsq=l*l+msq;
      		  if(rsq<1.0)
      		    {
      		      double phase=twoPiW*(sqrt(1.0-rsq)-1.0);
		      cufftComplex w;w.x=cos(phase); w.y=sin(phase);
		      screen[ix+convSize/2 + (iy+convSize/2)*ny]=
			cuCmulf(screen[ix+convSize/2 + (iy+convSize/2)*ny], w); 
      		      /* float wre=cos(phase), wim=sin(phase); */
      		      /* float re=screen[ix+convSize/2 + (iy+convSize/2)*ny].x, */
      		      /* 	im=screen[ix+convSize/2 + (iy+convSize/2)*ny].y; */
      		      /* screen[ix+convSize/2 + (iy+convSize/2)*ny].x=re*wre - im*wim; */
      		      /* screen[ix+convSize/2 + (iy+convSize/2)*ny].y=re*wim + im*wre; */
      		    }
      		}
      	    }
      	}
    }
    //
    //===========================================
    //--------------------------------------------
    //
    __global__ void kernel_wTermApplySky(cufftComplex* screen, 
					 const cufftComplex* aTerm, 
					 const int nx, const int ny,
					 const int tileWidthX, const int tileWidthY, 
					 const double wPixel,
					 const float sampling, const double wScale, 
					 const int inner,      const bool isNoOp)
    {
      unsigned int col = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int row = tileWidthY*blockIdx.y + threadIdx.y ;

      __shared__ float twoPiW;
      twoPiW=__fmul_rn(2.0,M_PI);
      int originx=nx/2, originy=ny/2, tix, tiy;

      /* for (col=blockIdx.x * tileWidthX + threadIdx.x; col < nx; col +=tileWidthX * gridDim.x) */
      /* 	for (row=blockIdx.y * tileWidthY + threadIdx.y; row < ny; row +=tileWidthY * gridDim.y) */
	  {
      int ix=row-inner/2, iy=col-inner/2;
      tix=ix+originx; tiy=iy+originy;

      float m=sampling*float(ix), l=sampling*float(iy);
      float rsq=(l*l+m*m);
      /* float m=__fmul_rn(sampling,float(ix)), l=__fmul_rn(sampling,float(iy)); */
      /* float rsq=__fadd_rn(__fmul_rn(l,l),__fmul_rn(m,m)); */

      if (rsq<1.0)
	{
	  //	  float wValue = wPixel*wPixel/wScale;
	  float wValue=__fdividef((wPixel*wPixel),wScale);
	  twoPiW = __fmul_rn(twoPiW, wValue);
	  //	  float phase = twoPiW*(sqrt(1.0-rsq)-1.0);
	  float phase=__fmul_rn(twoPiW,
				(__fsqrt_rn(1.0-rsq)-1.0)
				);
	  cufftComplex w; __sincosf(phase, &(w.y),&(w.x));
	  screen[tix*ny+tiy] = cuCmulf(w,aTerm[tix*ny+tiy]);
	  //screen[tix*ny+tiy] = w;
	}
      else
	{
	  screen[tix*ny+tiy] = make_cuFloatComplex(0.0,0.0);
	}
	  }
    }
    //
    //--------------------------------------------
    //
    void wTermApplySky(cufftComplex* screen,  
		       const cufftComplex* aTerm,  
		       const int& nx, const int& ny,
		       const int tileWidthX, const int tileWidthY, 
		       const double& wPixel,
		       const float& sampling, const double& wScale, 
		       const int& inner,      const bool& isNoOp)
    {
#ifdef USE_AUTO
      {
	int WIDTH=ny;
	dim3 dimGrid ( WIDTH/tileWidthX , WIDTH/tileWidthY ,1 ) ;
	dim3 dimBlock( tileWidthX, tileWidthY, 1 ) ;
	
	kernel_wTermApplySky <<<dimGrid,dimBlock>>> (screen, aTerm, nx, ny, tileWidthX, tileWidthY,wPixel, sampling,
						     wScale, inner,isNoOp);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE , 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE,1,1);
	kernel_wTermApplySky <<<dimGrid,dimBlock>>> (screen, aTerm, nx, ny, tileWidthX, tileWidthY,wPixel, sampling, 
						     wScale, inner,isNoOp);
      }
#endif
    }
    //
    //===========================================
    //--------------------------------------------
    //
    __global__ void kernel_setBuf(cufftComplex *d_buf, const int nx, const int ny, 
				  const int tileWidthX, const int tileWidthY, 
				  cufftComplex val)
    {
      int WIDTH=ny;
      
      // calculate thread id
      unsigned int col = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int row = tileWidthY*blockIdx.y + threadIdx.y ;
      d_buf[row*WIDTH+col] = val;
    }
    //
    //--------------------------------------------
    //
    void setBuf(cufftComplex *d_buf, const int nx, const int ny, 
		const int tileWidthX, const int tileWidthY, 
		cufftComplex val)
    {
#ifdef USE_AUTO
      {
	int WIDTH=ny;
	dim3 dimGrid ( WIDTH/tileWidthX , WIDTH/tileWidthY ,1 ) ;
	dim3 dimBlock( tileWidthX, tileWidthY, 1 ) ;
	
	kernel_setBuf<<<dimGrid,dimBlock>>> ( d_buf,nx,ny,tileWidthX, tileWidthY,val);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE ,1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	
	kernel_setBuf<<<dimGrid,dimBlock>>> ( d_buf,nx,ny,tileWidthX, tileWidthY,val);
      }
#endif
    }
    //
    //===========================================
    //--------------------------------------------
    //
    __global__ void kernel_mulBuf(cufftComplex *target_d_buf, const cufftComplex* source_d_buf, 
				  const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      int WIDTH=ny;
      
      // calculate thread id
      unsigned int col = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int row = tileWidthY*blockIdx.y + threadIdx.y ;
      target_d_buf[row*WIDTH+col] = cuCmulf(target_d_buf[row*WIDTH+col], source_d_buf[row*WIDTH+col]);
    }
    //
    //--------------------------------------------
    //
    void mulBuf(cufftComplex *target_d_buf, const cufftComplex* source_d_buf, 
		const int& nx, const int& ny, const int tileWidthX, const int tileWidthY)
    {
#ifdef USE_AUTO
      {
	int WIDTH=ny;
	dim3 dimGrid ( WIDTH/tileWidthX , WIDTH/tileWidthY ,1 ) ;
	dim3 dimBlock( tileWidthX, tileWidthY, 1 ) ;
	
	kernel_mulBuf<<<dimGrid,dimBlock>>>(target_d_buf, source_d_buf, nx,ny,tileWidthX, tileWidthY);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE, 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	
	kernel_mulBuf<<<dimGrid,dimBlock>>>(target_d_buf, source_d_buf, nx,ny,tileWidthX, tileWidthY);
      }
#endif
    }
    //
    //--------------------------------------------
    //
    void cpuflip(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      int cx=nx/2, cy=ny/2;
      //      cuComplex sign=make_cuFloatComplex(1.0,0.0);
      
      for (int i=0; i<cx; i++)
	for (int j=0; j< cy; j++)
	  {
	    cufftComplex tmp;
	    // if (((cx+i)+(cy+j))%2 == 0) sign.x = -1.0; else sign.x = 1.0;
	    //tmp=cuCmulf(buf[i+j*ny],sign);
	    tmp=buf[i+j*ny];

	    //if (((i)+(j))%2 == 0) sign.x = -1.0;  else sign.x = 1.0;
	    //buf[i+j*ny] = cuCmulf(buf[cx+i + (cy+j)*ny],sign);
	    buf[i+j*ny] = buf[cx+i + (cy+j)*ny];

	    buf[cx+i + (cy+j)*ny] = tmp;
	  }
      for (int i=cx; i < nx; i++)
	for (int j=0; j < cy; j++)
	  {
	    cufftComplex tmp;

	    /* if (((i-cx)+(j+cy))%2 == 0) sign.x = -1.0; else sign.x = 1.0; */
	    /* tmp=cuCmulf(buf[i-cx +(j+cy)*ny],sign); */
	    tmp=buf[i-cx +(j+cy)*ny];

	    /* if (((i)+(j))%2 == 0) sign.x = -1.0; else sign.x = 1.0; */
	    /* buf[i-cx +(j+cy)*ny] = cuCmulf(buf[i + j*ny],sign); */
	    buf[i-cx +(j+cy)*ny] = buf[i + j*ny];

	    buf[i + j*ny] = tmp;
	  }
    }
    //
    //===========================================
    // Following is the GPU kernel equivalent of the cpuflip function
    //
    //  +--------------------+
    //  |         :          |
    //  |         :          |
    //  |    1    :     2    |
    //  |         :          |
    //  |....................|
    //  |         :          |
    //  |    4    :     3    |
    //  |         :          |
    //  |         :          |
    //  +--------------------+
    //
    // This function copies data from quadrant 1 to 3 (and from 3 to 1) and 4 to 2 (and from 2 to 4).
    // While copying, it also now flips the sign of the pixel values if the target pixel (i,j) satisfies (i+j)%2 != 0.
    // This therefore effectively combines the flipSign() kernel in flip() kernel itself (i.e., flipSign() does not
    // need to be envoked).  This saves ~15% in run-time.
    //
#define FLIPSIGN(i,j,ny,val) ({if (((i)+(j))%2 != 0) {(val[(i)+(j)*(ny)]).x *=-1.0; (val[(i)+(j)*(ny)]).y *= -1.0;}})
    __global__ void kernel_flip(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      // calculate thread id
      unsigned int i = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int j = tileWidthY*blockIdx.y + threadIdx.y ;
      unsigned int cx=nx/2, cy=ny/2;
      cuComplex sign=make_cuFloatComplex(1.0,0.0);

      cufftComplex tmp;

      if (i < cx)
	{
	  if (((cx+i)+(cy+j))%2 == 0) sign.x=1.0; else sign.x = -1.0;
	  tmp=cuCmulf(buf[i+j*ny],sign);
	  
	  if ((i+j)%2 == 0) sign.x=1.0; else sign.x = -1.0;
	  buf[i+j*ny] = cuCmulf(buf[cx+i + (cy+j)*ny],sign);

	  buf[cx+i + (cy+j)*ny] = tmp;

	  /* The commented out code below is cleaner code, but which
	     runs ~10% slower!!!  Don't understand why. */

	  /* tmp=buf[i+j*ny]; */
	  /* buf[i+j*ny] = buf[cx+i + (cy+j)*ny]; */
	  /* FLIPSIGN(i,j,ny,buf); */
	  /* tx=cx+i; ty=cy+j; */
	  /* buf[tx + ty*ny] = tmp; */
	  /* FLIPSIGN(tx, ty,ny, buf); */
	}
      else
	{
	  if ((i+j)%2 == 0) sign.x=1.0; else sign.x = -1.0;
	  tmp=cuCmulf(buf[i-cx +(j+cy)*ny],sign);

	  if (((i-cx)+(j+cy))%2 == 0) sign.x=1.0; else sign.x = -1.0;
	  buf[i-cx +(j+cy)*ny] = cuCmulf(buf[i + j*ny],sign);
	    
	  buf[i + j*ny] = tmp;

	  /* The commented out code below is cleaner code, but which
	     runs ~10% slower!!!  Don't understand why. */

	  /* unsigned int tx,ty; */
	  /* tx=i-cx; ty=j+cy; */
	  /* tmp=buf[tx +ty*ny]; */
	  /* buf[i-cx +(j+cy)*ny] = buf[i + j*ny]; */
	  /* FLIPSIGN(tx, ty,ny, buf); */
	  /* buf[i + j*ny] = tmp; */
	  /* FLIPSIGN(i,j,ny,buf); */
	}
    }
//buf, 4,4,2,2
__global__ void kernel_newflip(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      // calculate thread id
      unsigned int i = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int j = tileWidthY*blockIdx.y + threadIdx.y ;
      
      int cx=nx/2, cy=ny/2;
      cufftComplex tmp;

      if (i < cx  && j <cy) 
        {
          //printf("i=%d, j=%d, tmp=%d, buf[%d]=%d\n", i,j, tmp, (cx+i + (cy+j)*ny), buf[cx+i + (cy+j)*ny]);
          tmp=buf[i+j*ny];
          buf[i+j*ny] = buf[cx+i + (cy+j)*ny];
          buf[cx+i + (cy+j)*ny] = tmp;
        }
      else if (j < cy)
        {
          //printf("i=%d, j=%d, cx=%d cy=%d nx=%d ny=%d\n",i,j,cx,cy,nx,ny);
          //printf("i=%d, j=%d, buf[%d]=%d buf_s[%d]=%d\n", i,j, (i-cx + (cy+j)*ny),buf[i-cx+(cy+j)*ny],(i + j*ny), buf[i + j*ny]);
          tmp=buf[i-cx +(j+cy)*ny];
          buf[i-cx +(j+cy)*ny] = buf[i + j*ny];
          buf[i + j*ny] = tmp;
        }
    }
    //
    //--------------------------------------------
    //
    //
    //--------------------------------------------
    //
    void flip(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      /* dim3 dimGrid ( (nx/tileWidthX) , (ny/tileWidthY) ,1 ) ; */
      /* dim3 dimBlock( tileWidthX, tileWidthY, 1 ) ; */

      /* kernel_newflip<<<dimGrid,dimBlock>>>(buf, nx,ny,tileWidthX, tileWidthY); */

#ifdef USE_AUTO
      {
	dim3 dimGrid ( nx/tileWidthX , ny/(2*tileWidthY) ,1 ) ;
	dim3 dimBlock( tileWidthX, tileWidthY, 1 ) ;
	
	kernel_flip<<<dimGrid,dimBlock>>>(buf, nx,ny,tileWidthX, tileWidthY);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE , 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	
	kernel_flip<<<dimGrid,dimBlock>>>(buf, nx,ny,tileWidthX, tileWidthY);
      }
#endif
    }
    //
    //============================================
    //--------------------------------------------
    //
    __global__ void kernel_flipSign(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      // calculate thread id
      unsigned int i = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int j = tileWidthY*blockIdx.y + threadIdx.y ;
      float sign;
      {
	//	sign=__pow(-1.0,(float)(i+j));
	if ((i+j)%2 == 0) sign=1.0; else sign=-1.0;
	buf[i + j*ny].x = buf[i + j*ny].x*sign;
	buf[i + j*ny].y = buf[i + j*ny].y*sign;
      }
    }
    //
    //--------------------------------------------
    //
    void flipSign(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
#ifdef USE_AUTO
      {
	dim3 dimGrid ( nx/tileWidthX , ny/tileWidthY ,1 ) ;
	dim3 dimBlock( tileWidthY, tileWidthY, 1 ) ;
	kernel_flipSign<<<dimGrid,dimBlock>>>(buf, nx,ny,tileWidthX, tileWidthY);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE , 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	kernel_flipSign<<<dimGrid,dimBlock>>>(buf, nx,ny,tileWidthX, tileWidthY);
      }
#endif
    }
    //
    //============================================
    //--------------------------------------------
    //
    //
    //============================================
    //--------------------------------------------
    //
    //
    //============================================
    //--------------------------------------------
    //
#undef HAS_OMP
#ifdef HAS_OMP
#include <omp.h>
#endif
    
    //
    //----------------------------------------------------------------------
    // A global method for use in OMP'ed findSupport() below
    //
    // void archPeak(const float& threshold, const int& origin, const Block<int>& cfShape, const Complex* funcPtr, 
    // 		const int& nCFS, const int& PixInc,const int& th, const int& R, 
    // 		Block<int>& maxR)
    __device__ void cuArchPeak(const float& threshold, const int& origin, const int nx, const int ny, 
			       const cuComplex* funcPtr, const int& PixInc,const int& th, 
			       const int& R, int* maxR)
    {
      /* thrust::device_vector<cuComplex> vals; */
      /* thrust::device_vector<int> ndx; ndx.assign(2,0); */
      int ndx[2];
      int NSteps;
      //Check every PixInc pixel along a circle of radius R
      NSteps = 90*R/PixInc; 
      int valsNelements=(int)(NSteps+0.5);
      //      vals.resize((int)(NSteps+0.5));
      cuComplex *vals;//[valsNelements];

      cuComplex zero=make_cuFloatComplex(0.0, 0.0);
      //vals.assign(valsNelements,zero);
      for (int ii=0;ii<valsNelements; ii++) vals[ii]=zero;
      //      vals=0;
      
      for(int pix=0;pix<NSteps;pix++)
	{
	  ndx[0]=(int)(origin + R*sin(2.0*M_PI*pix*PixInc/R));
	  ndx[1]=(int)(origin + R*cos(2.0*M_PI*pix*PixInc/R));
	  
	  if ((ndx[0] < nx) && (ndx[1] < ny))
	    //vals[pix]=func(ndx);
	    vals[pix]=funcPtr[ndx[0]+ndx[1]*nx];
	}
      
      maxR[th]=-R;
      for (uint i=0;i<valsNelements;i++)
	if (cuCabsf(vals[i]) > threshold)
	  {
	    maxR[th]=R;
	    break;
	  }
      //		th++;
    }

    __device__ bool cuFindSupport(cuComplex* funcPtr_d, const int nx, const int ny, 
				  float& threshold, int& origin, int& radius)
    {
      int PixInc=1, R0, R1, R, convSize;
      bool found=false;
      uint Nth=1, threadID=0;
      
      
      convSize = nx;
#ifdef HAS_OMP
      Nth = max(omp_get_max_threads()-2,1);
#endif
      
      int *maxR_p;
      /* std::vector<int> maxR(Nth); */
      /* maxR_p=maxR.data(); */
      
      R1 = convSize/2-2;
      radius=R1;
      while (R1 > 1)
	{
	  R0 = R1; R1 -= Nth;
	  
#pragma omp parallel default(none) firstprivate(R0,R1)  private(R,threadID) shared(origin, threshold, PixInc,maxR_p ,nCFS,funcPtr_d) num_threads(Nth)
	  { 
#pragma omp for
	    for(R=R0;R>R1;R--)
	      {
#ifdef HAS_OMP
		threadID=omp_get_thread_num();
#endif
		cuArchPeak(threshold, origin, nx, ny, funcPtr_d, PixInc, threadID, R, maxR_p);
	      }
	    ///#pragma omp barrier
	  }///omp 	    
	  
	  for (uint th=0;th<Nth;th++)
	    {
	      if (maxR_p[th] > 0)
		{
		  found=true; 
		  if (maxR_p[th] < radius) radius=maxR_p[th]; 
		}
	    }
	  if (found) 
	    return found;
	}
      return found;
    }
    
    __device__ bool cuSetUpCFSupport(cuComplex* func_d, const int nx, const int ny, 
			  int& xSupport, int& ySupport,
			  const float& sampling, const cuComplex& peak)
    {
      xSupport = ySupport = -1;
      //      int convFuncOrigin=func.shape()[0]/2, R; 
      int convFuncOrigin=nx/2, R; 
      bool found=false;
      float threshold;

      if (cuCabsf(peak) != 0) threshold = cuCabsf(peak);
      else 
	threshold   = (cuCabsf(func_d[convFuncOrigin+convFuncOrigin*ny]));

      threshold *= 1e-3;

      if (found = cuFindSupport(func_d,nx, ny,threshold,convFuncOrigin,R))
	xSupport=ySupport=int(0.5+float(R)/sampling)+1;

      if (xSupport*sampling > convFuncOrigin)
	  xSupport = ySupport = (int)(convFuncOrigin/sampling);

      return found;
    }
    
    __device__ bool cuResizeCF(cuComplex* func_d, const int nx, const int ny, 
		    int& xSupport, int& ySupport,
		    const float& sampling, const cuComplex& peak)
    {
      bool found = cuSetUpCFSupport(func_d, nx, ny, xSupport, ySupport, sampling,peak);
      
      return true;
      
      /* //int supportBuffer = aTerm_p->getOversampling()*2; */
      /* int ConvFuncOrigin=nx/2;  // Conv. Func. is half that size of convSize */
      
      /* int supportBuffer = OVERSAMPLING*2; */
      /* int bot=(int)(ConvFuncOrigin-sampling*xSupport-supportBuffer),//-convSampling/2,  */
      /* 	top=(int)(ConvFuncOrigin+sampling*xSupport+supportBuffer);//+convSampling/2; */
      /* //    bot *= 2; top *= 2; */
      /* bot = max(0,bot); */
      /* top = min(top, nx-1); */
      
      /* Array<Complex> tmp; */
      /* IPosition blc(4,bot,bot,0,0), trc(4,top,top,0,0); */
      /* // */
      /* // Cut out the conv. func., copy in a temp. array, resize the */
      /* // CFStore.data, and copy the cutout version to CFStore.data. */
      /* // */
      /* tmp = func(blc,trc); */
      /* func.resize(tmp.shape()); */
      /* func = tmp;  */
      /* return found; */
    }
    
  };
