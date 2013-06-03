#include<cuda_runtime.h>
#include <complex>
#include "/usr/local/cuda-5.5/include/cufft.h"

namespace casa 
{
    //CUFFT Call replacing the FFT call in AntenaaAterm.cc file

    int call_cufft(cufftComplex *pointer, int  NX, int NY)
    {
        printf("Inside Call_cuda.cu file\n");
        cufftHandle plan;

        printf("sizeof(cufftComplex) = %d NX=%d NY=%d\n", sizeof(cufftComplex), NX, NY);
        cudaMalloc((void**)&pointer, sizeof(cufftComplex)*NX*(NY));
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");
            return 0;
        }

        /* Create a 2D FFT plan. */
        if (cufftPlan2d(&plan, NX, NY, CUFFT_C2C) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            return 0;
        }


        if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
            return 0;
        }

        if (cufftExecC2C(plan, (cufftComplex *)pointer, (cufftComplex *)pointer, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            return 0;
        }
        if (cudaDeviceSynchronize() != cudaSuccess){
  	    fprintf(stderr, "Cuda error: Failed to synchronize\n");
   	    return 0;
        }
        printf("After devicesync\n");
#if 1
        cufftDestroy(plan);
        cudaFree(pointer);
#endif

        return 0;
    }
}
