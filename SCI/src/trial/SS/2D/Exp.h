#if !defined(EXP_H)
#define EXP_H

#include <stdlib.h>
#include <math.h>

template<class T> class Exp
{
private:
  T EStep;
  T *ETable;
  int Size;
public:
  Exp<T>() {EStep=0; ETable=NULL;Size=0;};
  Exp<T>(int n, T Step) {EStep=Size=0;ETable=NULL;Build(n,Step);};
  inline void Build(int n, T Step)
  {
    if (ETable) free(ETable);
    cerr << "Making exp() table..." << endl;
    ETable=(T *)malloc(sizeof(T)*n);
    Size = n;
    EStep = Step;

    for (int i=0;i<n;i++) ETable[i]=exp(-i*Step);
  }
  inline T Exp<T>::operator()(T arg) 
  {
    int N=(int)(-arg/EStep); 

    //    return (fabs(N)>=Size)?0:((ETable[N]-ETable[N+1])*arg + ETable[N]);

    return (abs(N)>=Size)?0:ETable[N];
  }
};
  
#endif
