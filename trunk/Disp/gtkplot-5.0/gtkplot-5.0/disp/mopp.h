#if !defined(MOPP_GLOBALS)
#define MOPP_GLOBALS
#include <values.h>

gdouble *XYPanel::X=NULL;
gint XYPanel::Id=0;
gfloat XYPanel::XDataMax=MAXFLOAT, XYPanel::XDataMin=MINFLOAT;

extern "C" {
  void GridPacker(int n, int np, gfloat *cw, gfloat *ch, 
		  gfloat *pw, gfloat *ph, gfloat *px, gfloat *py)
  {
    int k1=3,k2;
    k2=(int)(np/k1+1);
    *cw = *pw*k1;  
    *ch = *ph*k2;
    *px = (n%k1)* *pw; *py=int((n/k1)* *ph);
  };
}

#endif
