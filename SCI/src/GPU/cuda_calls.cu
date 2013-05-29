#include<cuda_runtime.h>
#include <complex>
#include "/usr/local/cuda-5.5/include/cufft.h"

namespace casa 
{
    //CUFFT Call replacing the FFT call in AntenaaAterm.cc file

    int call_cufft(cufftDoubleComplex *pointer, int  NX, int NY)
    {
        printf("Inside Call_cuda.cu file\n");
        cufftHandle plan;

        cudaMalloc((void**)&pointer, sizeof(cufftDoubleComplex)*NX*(NY));
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");
            return 0;
        }
        printf("cudamalloc done\n");

        /* Create a 2D FFT plan. */
        if (cufftPlan2d(&plan, NX, NY, CUFFT_C2C) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            return 0;
        }
        printf("2d FFT Plan is done\n");

        if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
            return 0;
        }
        printf("After comtmode\n");

        if (cufftExecC2C(plan, (cufftComplex *)pointer, (cufftComplex *)pointer, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            return 0;
        }
        printf("After cufftexec\n");
        return 0;
    }
}
