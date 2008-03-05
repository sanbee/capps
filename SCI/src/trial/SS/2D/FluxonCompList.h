#if !defined(FLUXONCOMPLIST_H)
#define FLUXONCOMPLIST_H
#include <iostream.h>
#include <Fluxon2D.h>
#include <vector.h>

template <class T> class FluxonCompList
{
  //
  // Public face...
  //
public:
  //--------------------------------------------------------------------------------
  // Constructors
  //
  FluxonCompList<T>():CompList() {};
  //--------------------------------------------------------------------------------
  // Get the no. of Fluxon components
  //
  inline void resize(int N) {CompList.resize(N);MaxSize=CompList.size();}
  inline void insert(Fluxon2D<T>& f, int i) {CompList[i] = f;}
  inline int size() {return MaxSize;}
  //--------------------------------------------------------------------------------
  // Copy the component list to another list.
  // 
  inline FluxonCompList<T>& operator=(FluxonCompList<T> f)
  {
    int N;
    CompList.resize(N=f.size());
    for(int i=0;i<N;i++) CompList[i] = f[i];
    MaxSize = CompList.size();
    return *this;
  }
  //--------------------------------------------------------------------------------
  // Add a Fluxon to the FluxonCompList
  //
  inline FluxonCompList<T>& operator+=(Fluxon2D<T>& f)
  {
    int N;
    N = CompList.size()+1;

    CompList.resize(N);
    CompList[N-1] = f;
    MaxSize = CompList.size();
    return *this;
  }
  //--------------------------------------------------------------------------------
  // Get the i th. component
  //
  inline Fluxon2D<T>& operator[](int i) {return CompList[i];}
  //--------------------------------------------------------------------------------
  // Compute the value at x
  //
  inline T operator()(T x,T y) 
  {
    T value=0;
    for (int i=0;i<size();i++) value += CompList[i](x,y);
    return value;
  }
  //--------------------------------------------------------------------------------
  // Convolve the FluxonCompList with f
  //
  inline FluxonCompList<T> convolve(Fluxon2D<T>& f)
  {
    //
    // Make it static - to make sure you don't
    // fragment the memory when called in loops
    //
    static FluxonCompList<T> tmp; 
    Fluxon2D<T> t;
    int N = size();
    tmp.resize(N);
    for (int i=0;i<N;i++) 
      {
	t = CompList[i].convolve(f);
	tmp.insert(t,i);
      }
    return tmp;
  }
  //--------------------------------------------------------------------------------
  // Convolve two FluxonCompLists
  //
  inline FluxonCompList<T> convolve(FluxonCompList<T>& FCL)
  {
    //
    // Make it static so that it does not fragement
    // the memory when called in loops where N*M 
    // does not change
    //
    static FluxonCompList<T> tmp;
    Fluxon2D<T> t;
    int N=FCL.size(),M=size(),C=0;

    tmp.resize(N*M);

    for (int i=0;i<N;i++)
      for(int j=0;j<M;j++)
	{
	  t = CompList[j].convolve(FCL[i]);
	  tmp.insert(t,C++);
	}
    return tmp;
  }  
  //--------------------------------------------------------------------------------
  // Return the auto-correlation function.
  // First N*(N-1)/2 components are 2(P_i * P_j).  The last N elements are P_i*P_i
  inline FluxonCompList<T> acf()
  {
    //
    // Make it static so that it does not fragement
    // the memory when called in loops where N*M 
    // does not change
    //
    static FluxonCompList<T> tmp;
    Fluxon2D<T> F;
    int N=size(),C=0;

    tmp.resize(N*(N-1)/2 + N);

    for (int i=0;i<N;i++) 
      for (int j=i+1;j<N;j++) 
	{
	  F = CompList[i].convolve(CompList[j]);
	  F.setAmp(F.getAmp()*2);
	  tmp.insert(F,C++);
	}

    for (int i=0;i<N;i++) 
      {
	F = CompList[i].convolve(CompList[i]);
	tmp.insert(F,C++);
      }
    return tmp;
  }  
  //
  //
  //
  void setSize(int n) {MaxSize = n<MaxSize?n:MaxSize;}
  void resetSize() {MaxSize = CompList.size();}
  //
  // Private face...
  //
private:
  vector<Fluxon2D<T> > CompList;
  int MaxSize;
};
//
//================================================================================
//

//
// Global scope operaters related to the class...
//
//
//--------------------------------------------------------------------------------
// Method to add a FluxonCompList to an image.  
//
template <class T> IMAGETYPE<T>& operator+=(IMAGETYPE<T>& Img, FluxonCompList<T>& fImg) 
{
  int N;
  N=fImg.size();
  for (int i=0;i<N;i++)  Img += fImg[i];

  return Img;
}
//--------------------------------------------------------------------------------
// Method to subtract a FluxonCompList from an image.  
//
template <class T> IMAGETYPE<T>& operator-=(IMAGETYPE<T>& Img, FluxonCompList<T>& fImg) 
{
  int N;
  N=fImg.size();
  for (int i=0;i<N;i++)  Img -= fImg[i];

  return Img;
}
//--------------------------------------------------------------------------------
// Write the parameters of the component list to an output stream.
//
template <class T> ostream& operator<<(ostream& os, FluxonCompList<T>& Img)
{
  for (int i=0;i<Img.size(); i++)    os << Img[i] << endl;
  return os;
}
//--------------------------------------------------------------------------------
// Read the parameters of the component list from an input stream
//
template <class T> istream& operator>>(istream& is, FluxonCompList<T>& Img)
{
  Fluxon2D<T> f;
  int N;
  is >> N;
  Img.resize(N);
  for(int i=0;i<N;i++)
    {
      is >> f;
      //      f.setPos(f.getPos()+1);
      Img.insert(f,i);
    }

  return is;
}
#endif
