#include <cufft.h>
#ifndef UTILS_H
#define UTILS_H

namespace casa{
  enum memoryMode { PINNED, PAGEABLE };

  void * allocateDeviceBuffer(int N);
  int sendBufferToDevice(void *d_buf, void *h_buf, int N);
  int getBufferFromDevice(void *h_buf, void *d_buf, int N);
  int makeCUFFTPlan(cufftHandle *plan, int NX, int NY, cufftType type /*CUFFT_C2C*/);
  int setCompatibilityMode(cufftHandle& plan, cufftCompatibility mode /*CUFFT_COMPATIBILITY_NATIVE*/);
  int inPlaceCUFFTC2C(cufftHandle& plan, cufftComplex *d_buf, int dir /*CUFFT_FORWARD */);
  cudaError freeHost(void* h_mem, memoryMode memMode);
  cudaError mallocHost(void** h_mem ,uInt memSize, memoryMode memMode, Bool wc);
  cudaError memCpy(void* sink, void* source, uInt memSize, cudaMemcpyKind direction, memoryMode memMode);
  void wTermApplySky(cufftComplex* screen, const int& nx, const int& ny, const int& TILE_WIDTH,
		     const double& wPixel,
		     const float& sampling, 
		     const double& wScale, 
		     const int& inner,
		     const bool& isNoOp);
  void kernel_wTermApplySky(cufftComplex* screen, const int nx, const int ny, const int TILE_WIDTH,
			    const double wPixel,
			    const float sampling, 
			    const double wScale, 
			    const int inner,
			    const bool isNoOp);
  void cpu_wTermApplySky(cufftComplex* screen, const int nx, const int ny, const int TILE_WIDTH,
			 const double wPixel,
			 const float sampling, 
			 const double wScale, 
			 const int inner,
			 const bool isNoOp);

  __global__ void kernel_setBuf(cufftComplex *d_buf, const int& nx, const int& ny, cufftComplex& val);
  void setBuf(cufftComplex *d_buf, const int nx, const int ny, const int TILE_WIDTH, cufftComplex val);


  __global__ void kernel_mulBuf(cufftComplex *target_d_buf, const cufftComplex* source_d_buf, const int nx, const int ny, const int TILE_WIDTH);
  void mulBuf(cufftComplex *target_d_buf, const cufftComplex* source_d_buf, const int& nx, const int& ny, const int TILE_WIDTH);


  void flip(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH);
  __global__ void kernel_flip(cufftComplex *buf, const int nx, const int ny, const int TILE_WIDTH);

  __global__ void  matrixMulCUDA(cufftComplex *A, cufftComplex *B, int wA, int wB);

  void flipSign(cufftComplex *buf, const int nx, const int ny, const int TITLE_WIDTH);


};

#endif
