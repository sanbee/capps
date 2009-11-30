// $Id: BitField.cc,v 1.5 1999/02/06 11:13:45 sanjay Exp $
#include <BitField.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>
//
//-----------------------------------------------------------------
//
BitField::BitField(int N)
{
  CurPos = 0;
  if (Size*BYTELEN < (unsigned int)N)  resize(N);
}
//
//-----------------------------------------------------------------
//
int BitField::resize(int N)
{
  int S=length();
  Size = (int)ceil(double(N+1)/BYTELEN);
  if ((Bits=(char *)realloc(Bits,sizeof(char)*length()))==NULL)
    {
      perror("###Fatal error in BitField::resize: ");
      exit(-1);
    }
  for (int i=S;i<length();i++) Bits[i]=(char)NULL;

  return length();
}
//
//-----------------------------------------------------------------
//
void BitField::toggle(int N)
{
  if ((N==0) && (Size==0)) resize(1);

  if ((unsigned int)N >= Size*BYTELEN) resize(N);
  
  Bits[N/BYTELEN] ^= 1<<(N%BYTELEN);
}
//
//-----------------------------------------------------------------
//
void BitField::set(int N)
{
  if ((N==0) && (Size==0)) resize(1);

  if ((unsigned int)N >= length()*BYTELEN) resize(N);
  
  Bits[N/BYTELEN] |= 1<<(N%BYTELEN);
}
//
//-----------------------------------------------------------------
//
int BitField::operator()(int N)
{
  if ((unsigned int)N > length()*BYTELEN) return 0;
  return (Bits[(N/BYTELEN)] & (1<<(N%BYTELEN)))>0;
}
//
//-----------------------------------------------------------------
//
int BitField::count()
{
  unsigned int i,N=0;
  for (i=0;i<length()*BYTELEN;i++)
    if (operator()(i)) 
      N++;
  return (int)N;
}
//
//-----------------------------------------------------------------
//
int BitField::next()
{
  int N=length()*BYTELEN,R=-1;
  
  if (CurPos < (unsigned int)N)
    for (unsigned int i=CurPos;i<(unsigned int)N;i++)
      {
	CurPos++;
	if (operator()(i)) 
	  {R=CurPos-1;break;}
      }
  else  CurPos = 0;

  return R;
}
//
//-----------------------------------------------------------------
//
void BitField::fprtBits(FILE *fd)
{
  int i,N=length()*BYTELEN;

  for(i=0;i<N;i++)
    if (operator()(i)) fprintf(fd,"%d ",i);
}
