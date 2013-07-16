// -*- C -*-
#include <cuda_runtime.h>
#include <stdio.h>
#include <Utils.h>
#include <math.h>
#include <cuUtils.h>

#define USE_AUTO 
#undef USE_AUTO 
#define GRIDSIZE 2048
#define BLOCKSIZE 32

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
			   const int TILE_WIDTH, const double wPixel,
			   const float sampling, const double wScale, 
			   const int inner,      const bool isNoOp)
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
    __global__ void kernel_wTermApplySky(cufftComplex* screen, const int nx, const int ny,
					 const int TILE_WIDTH, const double wPixel,
					 const float sampling, const double wScale, 
					 const int inner,      const bool isNoOp)
    {
      int WIDTH=ny;
      
      unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
      unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
      double wValue=(wPixel*wPixel)/wScale;
      int convSize = nx;
      
      double twoPiW=2.0*M_PI*double(wValue);
      
      int ix=row-inner/2, iy=col-inner/2;
      double m=sampling*double(ix), l=sampling*double(iy);
      double rsq=(l*l+m*m);
      
      if (rsq<1.0)
	{
	  double phase=twoPiW*(sqrt(1.0-rsq)-1.0);
	  int tix=ix+convSize/2, tiy=iy+convSize/2;
	  cufftComplex w;w.x=cos(phase); w.y=sin(phase);
	  
	  /* float wre=cos(phase), wim=sin(phase); */
	  /* float re=screen[row*WIDTH+col].x, */
	  /*   im=screen[row*WIDTH+col].y; */
	  /* screen[tix*WIDTH+tiy].x=re*wre - im*wim; */
	  /* screen[tix*WIDTH+tiy].y=re*wim + im*wre; */
	  
	  screen[tix*WIDTH+tiy] = cuCmulf(screen[tix*WIDTH+tiy], w);
	}
      
      
      /* if (!isNoOp) */
      /* 	{ */
      /* 	  for (int iy=-inner/2;iy<inner/2;iy++)  */
      /* 	    { */
      /* 	      double m=sampling*double(iy); */
      /* 	      double msq=m*m; */
      /* 	      for (int ix=-inner/2;ix<inner/2;ix++)  */
      /* 		{ */
      /* 		  double l=sampling*double(ix); */
      /* 		  double rsq=l*l+msq; */
      /* 		  if(rsq<1.0)  */
      /* 		    { */
      /* 		      double phase=twoPiW*(sqrt(1.0-rsq)-1.0); */
      /* 		      float re=screen[ix+convSize/2 + (iy+convSize/2)*ny].x, */
      /* 			im=screen[ix+convSize/2 + (iy+convSize/2)*ny].y; */
      /* 		      float wre=cos(phase), wim=sin(phase); */
      /* 		      screen[ix+convSize/2 + (iy+convSize/2)*ny].x=re*wre - im*wim; */
      /* 		      screen[ix+convSize/2 + (iy+convSize/2)*ny].y=re*wim + im*wre; */
      /* 		    } */
      /* 		} */
      /* 	    } */
      /* 	} */
    }
    //
    //--------------------------------------------
    //
    void wTermApplySky(cufftComplex* screen,  const int& nx, const int& ny,
		       const int& TILE_WIDTH, const double& wPixel,
		       const float& sampling, const double& wScale, 
		       const int& inner,      const bool& isNoOp)
    {
#ifdef USE_AUTO
      {
	int WIDTH=ny;
	dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
	
	kernel_wTermApplySky <<<dimGrid,dimBlock>>> (screen, nx, ny, TILE_WIDTH,wPixel, sampling,
						     wScale, inner,isNoOp);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE , 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE,1,1);
	kernel_wTermApplySky <<<dimGrid,dimBlock>>> (screen, nx, ny, TILE_WIDTH,wPixel, sampling, 
						     wScale, inner,isNoOp);
      }
#endif
    }
    //
    //===========================================
    //--------------------------------------------
    //
    __global__ void kernel_setBuf(cufftComplex *d_buf, const int nx, const int ny, 
				  const int TILE_WIDTH, cufftComplex val)
    {
      int WIDTH=ny;
      
      // calculate thread id
      unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
      unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
      d_buf[row*WIDTH+col] = val;
    }
    //
    //--------------------------------------------
    //
    void setBuf(cufftComplex *d_buf, const int nx, const int ny, 
		const int TILE_WIDTH, cufftComplex val)
    {
#ifdef USE_AUTO
      {
	int WIDTH=ny;
	dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
	
	kernel_setBuf<<<dimGrid,dimBlock>>> ( d_buf,nx,ny,TILE_WIDTH,val);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE ,1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	
	kernel_setBuf<<<dimGrid,dimBlock>>> ( d_buf,nx,ny,TILE_WIDTH,val);
      }
#endif
    }
    //
    //===========================================
    //--------------------------------------------
    //
    __global__ void kernel_mulBuf(cufftComplex *target_d_buf, const cufftComplex* source_d_buf, 
				  const int nx, const int ny, const int TILE_WIDTH)
    {
      int WIDTH=ny;
      
      // calculate thread id
      unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
      unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
      target_d_buf[row*WIDTH+col] = cuCmulf(target_d_buf[row*WIDTH+col], source_d_buf[row*WIDTH+col]);
    }
    //
    //--------------------------------------------
    //
    void mulBuf(cufftComplex *target_d_buf, const cufftComplex* source_d_buf, 
		const int& nx, const int& ny, const int TILE_WIDTH)
    {
#ifdef USE_AUTO
      {
	int WIDTH=ny;
	dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
	
	kernel_mulBuf<<<dimGrid,dimBlock>>>(target_d_buf, source_d_buf, nx,ny,TILE_WIDTH);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE, 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	
	kernel_mulBuf<<<dimGrid,dimBlock>>>(target_d_buf, source_d_buf, nx,ny,TILE_WIDTH);
      }
#endif
    }
    //
    //--------------------------------------------
    //
    void cpuflip(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH)
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
    __global__ void kernel_flip(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH)
    {
      // calculate thread id
      unsigned int i = TILE_WIDTH*blockIdx.x + threadIdx.x ;
      unsigned int j = TILE_WIDTH*blockIdx.y + threadIdx.y ;
      
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
    void flip(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH)
    {
#ifdef USE_AUTO
      {
	dim3 dimGrid ( nx/TILE_WIDTH , ny/(2*TILE_WIDTH) ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
	
	kernel_flip<<<dimGrid,dimBlock>>>(buf, nx,ny,TILE_WIDTH);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE , 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	
	kernel_flip<<<dimGrid,dimBlock>>>(buf, nx,ny,TILE_WIDTH);
      }
#endif
    }
    //
    //============================================
    //--------------------------------------------
    //
    __global__ void kernel_flipSign(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH)
    {
      // calculate thread id
      unsigned int i = TILE_WIDTH*blockIdx.x + threadIdx.x ;
      unsigned int j = TILE_WIDTH*blockIdx.y + threadIdx.y ;
      
      /* for (int i=0; i<nx; i++) */
      /* 	for (int j=0; j<ny; j++) */
      {
	float sign=powf(-1.0,i+j);
	buf[i + j*ny].x = buf[i + j*ny].x*sign;
	buf[i + j*ny].y = buf[i + j*ny].y*sign;
      }
    }
    //
    //--------------------------------------------
    //
    void flipSign(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH)
    {
#ifdef USE_AUTO
      {
	dim3 dimGrid ( nx/TILE_WIDTH , ny/TILE_WIDTH ,1 ) ;
	dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
	kernel_flipSign<<<dimGrid,dimBlock>>>(buf, nx,ny,TILE_WIDTH);
      }
#else
      {
	dim3 dimGrid ( GRIDSIZE , 1 ,1 ) ;
	dim3 dimBlock( BLOCKSIZE, 1, 1 ) ;
	kernel_flipSign<<<dimGrid,dimBlock>>>(buf, nx,ny,TILE_WIDTH);
      }
#endif
    }
    
  };
