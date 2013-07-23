#include <cufft.h>
#ifndef UTILS_H
#define UTILS_H

namespace casa{
  enum memoryMode { PINNED, PAGEABLE };

  void * allocateHostBuffer(int N);
  void * allocateDeviceBuffer(int N);
  int sendBufferToDevice(void *d_buf, void *h_buf, int N);
  int getBufferFromDevice(void *h_buf, void *d_buf, int N);
  int makeCUFFTPlan(cufftHandle *plan, int NX, int NY, cufftType type /*CUFFT_C2C*/);
  int setCompatibilityMode(cufftHandle& plan, cufftCompatibility mode /*CUFFT_COMPATIBILITY_NATIVE*/);
  int inPlaceCUFFTC2C(cufftHandle& plan, cufftComplex *d_buf, int dir /*CUFFT_FORWARD */);
  cudaError freeHost(void* h_mem, memoryMode memMode);
  cudaError mallocHost(void** h_mem ,uint memSize, memoryMode memMode, bool wc);
  cudaError memCpy(void* sink, void* source, uint memSize, cudaMemcpyKind direction, memoryMode memMode);
};

#endif
