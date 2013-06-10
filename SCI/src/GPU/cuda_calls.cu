#include<cuda_runtime.h>
#include <complex>
#include <complex.h>
#include "/usr/local/cuda-5.5/include/cufft.h"
#include "AntennaATerm.h"

namespace casa 
{
    //CUFFT Call replacing the FFT call in AntenaaAterm.cc file

    int call_cufft(Complex *h_pointer, int  NX, int NY)
    {
        printf("Inside Call_cuda.cu file\n");
        cufftHandle plan;
        cufftComplex *d_pointer;

        printf("sizeof(cufftComplex) = %d NX=%d NY=%d\n", sizeof(cufftComplex), NX, NY);

        #if 0 
        int i=0;

        Complex tmp=0.0;
        for(i=0;i<NX*NY ; i+=100)
        {
             if (h_pointer[i] > tmp) tmp=h_pointer[i];
        }
        cout << "Max in cuda_calls= " << tmp << endl;
        #endif


        cudaMalloc((void**)&d_pointer, sizeof(cufftComplex)*NX*(NY));
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");
            return 0;
        }
       
        cudaMemcpy(d_pointer, h_pointer, sizeof(cufftComplex)*NX*(NY), cudaMemcpyHostToDevice);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");
            return 0;
        }
        

        /* Create a 2D FFT plan. */
        if (cufftPlan2d(&plan, NX, NY, CUFFT_C2C) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            return 0;
        }



        //if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS){
        if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_FFTW_PADDING)!= CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
            return 0;
        }

        if (cufftExecC2C(plan, d_pointer, d_pointer, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
            return 0;
        }
        cudaMemcpy(h_pointer, d_pointer, sizeof(cufftComplex)*NX*(NY), cudaMemcpyDeviceToHost);
        if (cudaGetLastError() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to allocate\n");
            return 0;
        }
       
        printf("After devicesync\n");
#if 1
        cufftDestroy(plan);
        cudaFree(d_pointer);
#endif
#if 0
        int p;
        for (p=0;p<NX*NY;p++)
        cout << "FFTed Data by GPU = " << h_pointer[p] << endl;
        #endif

        return 0;
    }
}
