//-*-C++-*-
//
// $Id$
//
#if !defined(CLERROROBJ_H)
#define CLERROROBJ_H
#ifdef __cplusplus
#include <ErrorObj.h>
class clError: public ErrorObj 
{
public:
  clError() {};
  clError(const char *m, const char *i, int l=0): ErrorObj(m,i,l) {};
};

extern "C" {
inline int clThrowUp(const char *m, const char *i, int l){throw(clError(m,i,l));}
	   }
#else
#include <stdio.h>
inline int clThrowUp(const char *m, const char *i, int l){fprintf(stderr,"%s: %s\n",m,i);return l;}
#endif

#endif
