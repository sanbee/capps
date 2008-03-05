//-*- C++ -*-
#ifndef ARRAY2D_H
#define ARRAY2D_H
#include <iostream>
#include <stdlib.h>

using namespace std;
template <class T> class Array2D {
public:
  Array2D(){Nx=Ny=0; buf=NULL;};
  Array2D(int N, int M);

  ~Array2D();

  inline int mapNdx(int i, int j)         {return i*Ny+j;}
  T operator()(int i, int j)              {return buf[mapNdx(i,j)];};
  T *operator[](int i)                    {return &buf[i*Ny];};
  T *getRow(int i)                        {return &buf[i*Ny];};
  inline void setVal(int i, int j, T val) {buf[mapNdx(i,j)] = val;};

  inline void addVal(int i, int j, T val) {buf[mapNdx(i,j)] += val;};
  inline void subVal(int i, int j, T val) {buf[mapNdx(i,j)] -= val;};
  //  float operator=(int i, int j, float val) {(this->buf)[j*Nx + i]=val;}
  
  void resize(int i, int j);
  void size(int &xsize,int &ysize)        {xsize=Nx;ysize=Ny;}
  int size(int which=0)                   {return (which == 0)?Nx:Ny;}
  void assign(T val)                      {for (int i=0;i<Nx;i++) for (int j=0;j<Ny;j++) setVal(i,j,val);}

  void normalize(T val)                   {for (int i=0;i<Nx;i++) for (int j=0;j<Ny;j++) setVal(i,j,(*this)(i,j)/val);}

  Array2D<T> &operator=(Array2D<T> &val) 
  {
    int inpX,inpY;
    val.size(inpX,inpY);

    resize(inpX,inpY);

    for (int i=0;i<Nx;i++) 
      for (int j=0;j<Ny;j++) 
        setVal(i,j,(T)(val(i,j)));
  }

  T *getStorage() {return buf;};

private:
  unsigned int Nx, Ny;
  T *buf;
};

template <class T> Array2D<T>::Array2D<T>(int N,int M)
{
  //buf = (T *)calloc(sizeof(T)*N*M);
  buf = (T *)calloc(N*M,sizeof(T));
  Ny=N; Nx=M;
}

template <class T> Array2D<T>::~Array2D<T>()
{
  if (buf) free(buf);
  buf = NULL;
}

template <class T> void Array2D<T>::resize(int N, int M)
{
  //
  // Realloc only if the total size has changed.  Important to not
  // blindly realloc - when the array is declared static for use again 
  // and again in a loop.  This will prevent memory fragmentation which
  // can otherwise slow things down in iterative programs.
  //
  if (N*M != Nx*Ny) buf = (T *)realloc(buf,sizeof(T)*N*M);
  Nx=N;Ny=M;
}

//
// Method to write the data to the outputstream
//
template <class T> ostream& operator<<(ostream& os, Array2D<T>& a)
{
  int Nx, Ny;
  a.size(Nx,Ny);
  for (int i=0;i<Nx;i++)
    {
      for (int j=0;j<Ny;j++)
	os << i << " " << j << " " << a(i,j) << endl;
      os << endl;
    }
  return os;
}

#endif
