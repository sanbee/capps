#if !defined(FLUXON_H)
#define FLUXON_H

#include <iostream.h>
#include <math.h>
#include <Exp.h>
#include <vector.h>
#include "Common.h"

// #include <vector.h>
// #define IMAGETYPE Array2D
#define ZERO_SCALE 2.5
#define FITPOS 0
#define NPARAMS 2

using namespace std;

typedef double FTYPE;
//
// Name space for global parameters which may be used by the objects
// and can be set easily by the user.
//
namespace GlobalParams {
  FTYPE NSigma = 4.0;
  FTYPE XCenter = 512;
  FTYPE YCenter = 512;
  Exp<FTYPE> TableOfExponential;
};

  //
  // Restricted scale-space
  //
#define MAX_SCALE 2.7
#define ATAN_MAX 1.57079563
#define ATAN_MAX2 3.1415926
  inline float scale (float s,float maxS=MAX_SCALE)              {return (maxS/2)*(1+atan(s)/ATAN_MAX);}
  inline float parameterizedScale(float s, float maxS=MAX_SCALE) {return tan((2*s/maxS-1)*ATAN_MAX);}
  inline float dScale(float s,float maxS=MAX_SCALE)              {return (maxS/ATAN_MAX2)*(1.0/(1+s*s));}


template <class T> class Fluxon2D
{
public:
  //
  // Constructors
  //
  //  Fluxon2D<T>()
  //  {
  //    itsA=1.0; itsPos=0; itsScale=1.0;itsShift=0;
  //    itsExpTable = GlobalParams::TableOfExponential;
  //  };
  enum PARAMS {NOOFPARAMS=NPARAMS};
  enum Derivatives {DA=0, DXP=1, DYP=2, DS=3};
  Fluxon2D<T>(Exp<T>& ExpTable=GlobalParams::TableOfExponential) 
  {
    itsA=1.0; itsXPos=itsYPos=0; itsScale=scale(1000.0);
    itsXShift=GlobalParams::XCenter;
    itsYShift=GlobalParams::YCenter;
    itsExpTable = ExpTable;
    itsDerivatives.resize(NOOFPARAMS);
    pitsScale=parameterizedScale(10000.0);
  }

  Fluxon2D<T>(Exp<T>& ExpTable, T A, T XPos, T YPos, T Scale, 
	      T XShift=GlobalParams::XCenter,T YShift=GlobalParams::YCenter)  
  {
    itsA=A; itsXPos=XPos; itsYPos=YPos; itsScale=scale(Scale);
    itsXShift=XShift;itsYShift=YShift;
    itsExpTable = ExpTable;
    itsDerivatives.resize(NOOFPARAMS);
    pitsScale=parameterizedScale(itsScale);
  };
  //
  // Get the parameters...
  //
  inline void setAmp(T Amp)                {itsA = Amp;}
  inline void setPos(T XPos, T YPos)       {itsXPos = (XPos); itsYPos=(YPos);}
  inline void setXPos(T XPos)              {itsXPos = (XPos);}
  inline void setYPos(T YPos)              {itsYPos = (YPos);}
  inline void setScale(T Scale)            {itsScale=scale(Scale);pitsScale=parameterizedScale(itsScale);}
  inline void setPScale(T Scale)           {itsScale=(Scale);pitsScale=parameterizedScale(Scale);}
  inline void setShift(T XShift, T YShift) {itsXShift = XShift; itsYShift=YShift;}
  inline void setParams(T Amp, T XPos, T YPos, T Scale, T XShift, T YShift)
  {setAmp(Amp); setPos(XPos,YPos); setScale(Scale); setShift(XShift,YShift);}
  //
  // ...and set the parameters.
  //
  inline T getAmp()               {return (itsA);    }
  inline T getXPos()              {return (itsXPos); }
  inline T getYPos()              {return (itsYPos); }
  inline T getScale()             {return itsScale;}
  inline T getXShift()            {return itsXShift; }
  inline T getYShift()            {return itsYShift; }
  inline int getNoOfParams()      {return NOOFPARAMS;}
  inline void getRange(int MaxSize, int &nx0, int &nx1, int &ny0, int &ny1)
  {
    T R=GlobalParams::NSigma/getScale();
    nx0 = (nx0=(int)(getXPos()-R+1))<0?0:nx0;
    nx1 = (nx1=(int)(getXPos()+R+1))>MaxSize?MaxSize:nx1;

    ny0 = (ny0=(int)(getYPos()-R+1))<0?0:ny0;
    ny1 = (ny1=(int)(getYPos()+R+1))>MaxSize?MaxSize:ny1;
    /*
    nx0=0;nx1=MaxSize;
    ny0=0;ny1=MaxSize;
    */
  }
  inline int inRange(T x, T y, int nx0, int nx1, int ny0, int ny1) {return ((x >= nx0) && (x <nx1) && (y >= ny0) && (y < ny1));}
  //
  // Assignment operator
  //
  /*
  Fluxon2D<T>& operator=(Fluxon2D<T> f) 
  {
    itsA     = f.getAmp();   itsPos   = f.getPos(); 
    itsScale = f.getScale(); itsShift = f.getShift(); 
    return *this;
  }
  */
  //
  // Convolution operator
  //
  inline Fluxon2D<T> convolve(Fluxon2D<T>& f) 
  {
    T d = sqrt(itsScale*itsScale + f.getScale()*f.getScale());
    Fluxon2D<T> tmp(f);

    tmp.setParams(itsA*f.getAmp() * sqrt(2*M_PI)/d,
		  itsXPos + f.getXPos() - f.getXShift(),
		  itsYPos + f.getYPos() - f.getYShift(),
		  (itsScale*f.getScale())/d,
		  f.getXShift(),f.getYShift());
    return tmp;
  }
  //
  // Area under the Fluxon2D
  //
  inline T narea() {return itsA*sqrt(2*M_PI)/itsScale;};
  //
  // Value of the Fluxon2D at x
  //
  inline T operator()(T x,T y) 
  {
    T arg; 
    //
    //    if (abs(x-getPos()) > GlobalParams::NSigma/getScale()) return 0;
    arg = ((x-getXPos())*(x-getXPos())+(y-getYPos())*(y-getYPos()))*itsScale*itsScale;
    return itsA*itsExpTable(-arg/2);
    //    return itsA*exp(-arg/2);
  }

  //
  // Evaluate partial derivatives at (x,y)
  //
  inline void evalDerivatives(T x, T y) 
  {
    T tmp = 2*operator()(x,y), x2=(x-itsXPos), y2=(y-itsYPos);

    itsDerivatives[DA]  =  -tmp/(itsA);                     // -2*P_k/A_k
#ifdef FITPOS
    itsDerivatives[DXP] =  -x2*itsScale*tmp;       // -2*[x-x_k] * Scale^2 * P_k
    itsDerivatives[DYP] =  -y2*itsScale*tmp;       // -2 * [y-y_k] * Scale^2 * P_k
#endif
    itsDerivatives[DS]  =   (x2*x2 + y2*y2)*itsScale*dScale(pitsScale)*tmp;   // 2 * ([x-x_k]^2 + [y-y_k]^2) * Scale * P_k
  }
  inline T dF(Derivatives D) {return itsDerivatives[D];}
  
private:
  T itsA, itsXPos, itsYPos, itsScale, pitsScale, itsXShift,itsYShift;;

  Exp<T> itsExpTable;
  vector<T> itsDerivatives;
};
//
// Method to add a Fluxon2D to an image.  
//
template <class T> IMAGETYPE<T>& operator+=(IMAGETYPE<T>& Img, Fluxon2D<T>& f) 
{
  int nx0,nx1,ny0,ny1,MAXSIZE;

  Img.size(MAXSIZE,MAXSIZE);
  
  f.getRange(MAXSIZE, nx0, nx1, ny0, ny1);

  for(int i=nx0;i<nx1;i++) 
      {
	for(int j=ny0;j<ny1;j++) 
	  {
	    Img.addVal(i,j,f((T)i,(T)j));
	  }
      }

  return Img;
}
//
// Method to subtract a Fluxon2D from an image.  
//
template <class T> IMAGETYPE<T>& operator-=(IMAGETYPE<T>& Img, Fluxon2D<T>& f) 
{
  int nx0,nx1,ny0,ny1,MAXSIZE;
  Img.size(MAXSIZE,MAXSIZE);
  f.getRange(MAXSIZE, nx0, nx1, ny0, ny1);

  for(int i=nx0;i<nx1;i++) 
    for(int j=ny0;j<ny1;j++) 
      Img.subVal(i,j,f((T)i,(T)j));

  return Img;
}
//
// Method to write the parameters of the Fluxon2D to an output stream
//
template <class T> ostream& operator<<(ostream& os, Fluxon2D<T>& f)
{
  os << f.getAmp() << " " << f.getXPos() << " " << f.getYPos() << " " << f.getScale();
  return os;
}
//
// Method to read the parameters of the Fluxon2D from an input stream
//
template <class T> istream& operator>>(istream& is, Fluxon2D<T>& f)
{
  T Amp, PosX, PosY, Scale;
  is >> Amp >> PosX >> PosY >> Scale;
  f.setAmp(Amp);
  f.setXPos(PosX);
  f.setYPos(PosY);
  f.setScale(Scale);
  return is;
}
#endif
