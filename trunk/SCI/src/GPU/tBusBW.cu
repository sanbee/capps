// -*- C -*-
#include <cuda_runtime.h>
#include <complex.h>
#include <cufft.h>
#include <stdlib.h>
#include <Utils.h>

#include <casa/aips.h>
#include <Utils.h>
#include <casa/OS/Timer.h>
#include <iostream>

using namespace std;
using namespace casa;

__global__ void divide2(cufftComplex *d_buf, int *N)
{
  // Your code was just using blocks, that will be very inefficient.

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  d_buf[i].x = __fdiv_rn(d_buf[i].x, N[0]);

  d_buf[i].y = __fdiv_rn(d_buf[i].y, N[0]);
}

__global__ void divide(cufftComplex *d_buf, int *N)
{
  d_buf[blockIdx.x].x = d_buf[blockIdx.x].x / N[0];
  d_buf[blockIdx.x].y = d_buf[blockIdx.x].y / N[0];
}

int main(int argc, char **argv)
{
  Timer timer;
  int NX=1024*5, NY = 1024*5, N=NX*NY*sizeof(cufftComplex);
  void *h_buf;
  Double t=0;
  int maphostmem=0;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if (prop.canMapHostMemory)
    {
      cerr << "Yes!  Can map." << endl;
      //maphostmem=1;
    }

  timer.mark();
  cudaSetDevice(0);
  cudaDeviceReset();
  t=timer.all();
  cerr << "Device init. time: " << t << endl;
  //
  //-----------------------------------------------
  // Allocated an array on the host and fill it with 
  // some non-zero values
  //
  timer.mark();
  //  h_buf = (void *)malloc(N);
  mallocHost(&h_buf ,N, PINNED, False);

  cerr << "Host allocation time: " << timer.all() << endl;
  /* for (int i=0;i<NX*NY;i++) */
  /*   ((float *)h_buf)[2*i]=100.0; */

  ((float *)h_buf)[NX/2 * NY]=100.0;
  
  /* ((float *)h_buf)[100]=100.0; */
  /* ((float *)h_buf)[2]=200.0; */
  // ((float *)h_buf)[3]=0.0;
  cerr << ((float *)h_buf)[100] << " " 
       << endl;
  //
  //-----------------------------------------------
  // Allocate the buffer on the device
  //
  cufftComplex *d_buf;
  timer.mark();
  d_buf=(cufftComplex *)allocateDeviceBuffer(N);
  t=timer.all();
  cerr << "Device allocation time: " << t << endl;
  //
  //-----------------------------------------------
  // Send the data from host to the device
  //
  if (!maphostmem)
    {
      timer.mark();
      sendBufferToDevice(d_buf, h_buf, N);
      t=timer.all();
      cerr << "Send to device time: " << t << endl;
      cerr << "Send bandwidth for " << N << " bytes: " << N/t << endl;
    }
  //
  //-----------------------------------------------
  // Make a plan for C2C CUFFT
  //
  cufftHandle plan=0;
  timer.mark();
  makeCUFFTPlan(&plan, NX, NY, CUFFT_C2C);
  cerr << "Plan id = " << plan << endl;
  t=timer.all();
  cerr << "Time for plan: " << t << endl;
  //
  //-----------------------------------------------
  // Set the CUFFT compatibility mode
  //
  timer.mark();
  setCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
  t=timer.all();
  cerr << "Time for setting compatibility mode: " << t << endl;
  //
  //-----------------------------------------------
  // Do NFFT C2C FFTs.  Each time, divide the result
  // by NPix (to normalize the result) and re-use the 
  // device buffer for the next FFT
  //
  Double tfft=0;
  Int NFFT=10;
  if (argc > 1) sscanf(argv[1],"%d",&NFFT);
  int NPix=NX*NY;
  int *d_NPix=(int *)allocateDeviceBuffer(sizeof(int));
  sendBufferToDevice(d_NPix,&NPix,sizeof(int));
  cufftComplex *a_map;
  if (maphostmem)
    cudaHostGetDevicePointer(&a_map, h_buf, 0);
  else 
    a_map = d_buf;
  cerr << "Value = " << ((float *)h_buf)[NX/2 * NY] << endl;
  for(int i=0;i<NFFT;i++)
    {
      timer.mark();
      inPlaceCUFFTC2C(plan, (cufftComplex *)a_map, CUFFT_FORWARD);
      cudaThreadSynchronize();
      if (!(i%2)) 
	{
	  int threadsPerBlock = (256); // This we can tune
	  int numBlocks = ceil ( (NX*NY) / threadsPerBlock);
	  divide2<<<numBlocks, threadsPerBlock>>>(a_map,d_NPix);
	  //	  divide<<<NPix,1>>>(a_map,d_NPix);
	}
      tfft+=timer.all();
    }
  cerr << "Time for FFT: Per FFT " << tfft/NFFT << " Total for " << NFFT << " FFTs " << tfft << endl;
  //
  //-----------------------------------------------
  // Copy the values from the device buffer back to
  // the host buffer
  //
  if (!maphostmem)
    {
      timer.mark();
      memCpy(h_buf, d_buf, N, cudaMemcpyDeviceToHost,PINNED);
      //getBufferFromDevice(h_buf, d_buf, N);
      t=timer.all();
      cerr << "Get from device time: " << t << endl;
      cerr << "Get bandwidth for " << N << " bytes: " << N/t << endl;
    }

  cerr << ((float *)h_buf)[0] << " "
       << ((float *)h_buf)[1] << " "
       << ((float *)h_buf)[2] << " "
       << ((float *)h_buf)[3] << " "
       << endl;

}
