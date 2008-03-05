#if !defined(PARAMS_H)
#define PARAMS_H
#include <vector>
#include <FluxonCompList.h>
#include <gsl/gsl_multifit_nlin.h>

//#define IMG vector

template <class T> class Params
{
public:
  Params(void *P)  {ParamsPtr = P;}

  inline IMAGETYPE<T>*              ResImg()    {return (IMAGETYPE<T> *)((int *)            ParamsPtr)[0];}
  inline FluxonCompList<T>*         ModComp()   {return (FluxonCompList<T> *)((int *)       ParamsPtr)[1];}
  inline gsl_multifit_fdfsolver* Minimizer() {return (gsl_multifit_fdfsolver*)((int *)ParamsPtr)[2];}
  inline vector<int>*               CompList()  {return (vector<int> *)((int *)             ParamsPtr)[3];}
  inline IMAGETYPE<T>*              DImg()      {return (IMAGETYPE<T> *)((int *)            ParamsPtr)[4];}
  inline IMAGETYPE<T>*              MImg()      {return (IMAGETYPE<T> *)((int *)            ParamsPtr)[5];}
  inline FluxonCompList<T>*         ModelDI()   {return (FluxonCompList<T> *)((int *)       ParamsPtr)[6];}
  inline FluxonCompList<T>*         PSF()       {return (FluxonCompList<T> *)((int *)       ParamsPtr)[7];}
  
private:
  void *ParamsPtr;
};
#endif
