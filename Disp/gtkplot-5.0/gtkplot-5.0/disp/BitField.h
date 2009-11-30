//-*-C++-*-
// $Id: BitField.h,v 1.4 1999/02/06 11:13:27 sanjay Exp $
#ifndef BITFIELD_H
#define BITFIELD_H

#define BYTELEN 8*sizeof(char)
#include <stdio.h>
#include <stdlib.h>

class BitField{
public:
  BitField(){Size=0;Bits=0;CurPos=0;};
  BitField(int NBits);
  ~BitField() {Size=0;if(Bits) free(Bits);Bits=0;}

  int resize(int N);
  void clear() {if (Bits) free(Bits);Bits=NULL;Size=0;}
  void toggle(int N);
  void set(int N);
  int length() {return Size;};
  int count();
  void setIterator(int N=0){CurPos=N;};
  int next();
  int operator()(int N);

  char *getBuf() {return Bits;}
  void fprtBits(FILE *);
private:
  char *Bits;
  int Size;
  unsigned int CurPos;
};

#endif
