#include<cuda_runtime.h>

#include "/usr/local/cuda-5.5/include/cufft.h"
#if 0
//CUFFT Call replacing the FFT call in AntenaaAterm.cc file
int call_cufft(complex *pointer, int  NX, int NY)
{
    cufftHandle plan;
    cufftDoubleComplex *data;
    data = pointer;

    int n[NRANK] = {NX, NY};

    cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*NX*(NY));
    if (cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;
    }

    /* Create a 2D FFT plan. */
    if (cufftPlan2D(&plan, NX, NY, CUFFT_C2C) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to create plan\n");
        return;
    }

    if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
        return;
    }


    if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        return;
    }
    return 0;
}
#endif
