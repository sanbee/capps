#include <complex>
//#include "/usr/local/cuda-5.5/include/cufft.h"
#include <cufft.h>

namespace casa
{
    //CUFFT Call replacing the FFT call in AntenaaAterm.cc file

  //    int call_cufft(Complex *, int  , int , int);
    int call_cufft(cufftComplex*, int  , int , int);
}
