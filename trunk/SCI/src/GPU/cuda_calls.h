#include <complex>
namespace casa
{
//CUFFT Call replacing the FFT call in AntenaaAterm.cc file
int call_cufft(Complex *, int  , int );
}