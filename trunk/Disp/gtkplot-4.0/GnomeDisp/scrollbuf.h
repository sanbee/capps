// $Id: scrollbuf.h,v 1.2 1998/02/06 09:24:54 sanjay Exp $
// -*-C++-*-
#if !defined(SCROLLBUF_H)
#define SCROLLBUF_H

#include <stdio.h>

class ScrollBuf {
 private:
  float **buffers;
  int NBufs,Lengths,Allocated;
  void init();
 public:
  ScrollBuf(){Allocated=NBufs=Lengths=0;buffers=NULL;};
  ScrollBuf(int, int);
  ~ScrollBuf();

  void Clear();
  void reset(int,int);
  void add(float data,int Id=0);
  void scroll(int Id, int n);
  float* buf(int Id=0){return buffers[Id];};
  int capacity(int Id=0) {return Lengths;};
};

#endif
