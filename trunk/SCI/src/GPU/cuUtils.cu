// -*- C -*-
#include <cuda_runtime.h>
#include <stdio.h>
#include <Utils.h>
#include <math.h>
#include <cuUtils.h>

#define USE_AUTO 
//#undef USE_AUTO 
#define BLOCKSIZE 128
#define GRIDSIZE (2048*2048/128)

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
					 cufftComplex* aTerm, 
					 const int nx, const int ny,
					 const int tileWidthX, const int tileWidthY, 
					 const double wPixel,
					 const float sampling, const double wScale, 
					 const int inner,      const bool isNoOp)
    {
      unsigned int col = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int row = tileWidthY*blockIdx.y + threadIdx.y ;
      int originx=nx/2, originy=ny/2, tix, tiy;
      int ix=row-inner/2, iy=col-inner/2;
      tix=ix+originx; tiy=iy+originy;
      
      
      double m=sampling*double(ix), l=sampling*double(iy);
      double rsq=(l*l+m*m);

      if (rsq<1.0)
	{
	  double wValue=(wPixel*wPixel)/wScale;
	  double phase=2.0*M_PI*double(wValue)*(sqrt(1.0-rsq)-1.0);
	  cufftComplex w; __sincosf(phase, &(w.y),&(w.x));
	  screen[tix*ny+tiy] = cuCmulf(w,aTerm[tix*ny+tiy]);
	  //screen[tix*ny+tiy] = w;
	}
      else
	{
	  screen[tix*ny+tiy] = make_cuFloatComplex(0.0,0.0);
	}
    }
    //
    //--------------------------------------------
    //
    void wTermApplySky(cufftComplex* screen,  
		       cufftComplex* aTerm,  
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
      
      for (int i=0; i<cx; i++)
	for (int j=0; j< cy; j++)
	  {
	    cufftComplex tmp;
	    tmp=buf[i+j*ny];
	    buf[i+j*ny] = buf[cx+i + (cy+j)*ny];
	    buf[cx+i + (cy+j)*ny] = tmp;
	  }
      for (int i=cx; i < nx; i++)
	for (int j=0; j < cy; j++)
	  {
	    cufftComplex tmp;
	    tmp=buf[i-cx +(j+cy)*ny];
	    buf[i-cx +(j+cy)*ny] = buf[i + j*ny];
	    buf[i + j*ny] = tmp;
	  }
    }
    //
    //===========================================
    // Following is the GPU kernel equivalent of the cpuflip function
    //
    __global__ void kernel_flip(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
      // calculate thread id
      unsigned int i = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int j = tileWidthY*blockIdx.y + threadIdx.y ;
      
      int cx=nx/2, cy=ny/2;
      cufftComplex tmp;
      
      if (i < cx)
	{
	  tmp=buf[i+j*ny];
	  buf[i+j*ny] = buf[cx+i + (cy+j)*ny];
	  buf[cx+i + (cy+j)*ny] = tmp;
	}
      else
	{
	  tmp=buf[i-cx +(j+cy)*ny];
	  buf[i-cx +(j+cy)*ny] = buf[i + j*ny];
	  buf[i + j*ny] = tmp;
	}
    }
    //
    //--------------------------------------------
    //
    void flip(cufftComplex *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
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
	//	sign=pow(-1.0,i+j);
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
    
  };
