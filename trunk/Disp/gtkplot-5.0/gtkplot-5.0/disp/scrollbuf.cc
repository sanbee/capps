#include <scrollbuf.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <namespace.h>
void ScrollBuf::reset(int N, int L)
{
  Clear();
  NBufs = N;
  Lengths = L;
  init();
}

void ScrollBuf::Clear()
{
  int i;
  if (Allocated)
    {
      for (i=0;i<NBufs;i++)
	delete [] buffers[i];
      delete [] buffers;
      Allocated=0;
    }
  NBufs=0; Lengths=0;
}

ScrollBuf::ScrollBuf(int N, int L)
{
  NBufs = (N);
  Lengths = (L);
  Allocated = 0;
  init ();
}

void ScrollBuf::init()
{ 
  if (Lengths>0)
    {
      if (!Allocated)
	{
	  buffers = new float * [NBufs];
	  for (int i=0;i<NBufs;i++)
	    buffers[i] = new float [Lengths];
	  Allocated = 1;
	}
    }
}
 
ScrollBuf::~ScrollBuf()
{
  Clear();
}

void ScrollBuf::add(float data, int Id)
{
  scroll(Id,1);
  buffers[Id][Lengths-1]=data;
}

void ScrollBuf::scroll(int Id, int n=1)
{
  if (!(n>Lengths-1))
    {
      if (Id <0 || Id > NBufs)
	cerr << "Wronge ID for ScrollBuf::scroll()" << endl;
      {
	float *d=buffers[Id];
	memcpy(d,&d[n],sizeof(float)*(Lengths-1));
      }
    }
}    
