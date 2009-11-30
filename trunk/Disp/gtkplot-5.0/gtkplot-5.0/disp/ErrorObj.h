//-*-C++-*-
#if !defined(ERROROBJ_H)
#define      ERROROBJ_H
#include <string>
#include <iostream>
#include <namespace.h>

#undef HANDLE_EXCEPTIONS
#define HANDLE_EXCEPTIONS(str)    try {str} catch(...) {throw;}

class ErrorObj{
 public:
  enum {Informational=100,Recoverable,Severe,Fatal};
  ErrorObj():Id(),Msg(),Src(),Message(){/*Id=Msg=Src=Message=NULL;*/};
  ErrorObj(const char *m,const char *i,int l=0);
  ~ErrorObj()
    {
//      if (Msg) delete Msg; if (Id) delete Id; 
//      if (Src) delete Src; if (Message) delete Message;
    };
  
  void SetSource(const char *m=0);
  const char *Source() {return Src.c_str();}
  int Severity() {return Level;}
  const char *what();

  ostream &operator<<(const char *m) {return cerr << m;}
  ostream &operator<<(ErrorObj &E) {return cerr << E;}
  friend ostream &operator<<(ostream& o,const ErrorObj&);

 private:
  string Id,Msg,Src,Message;
  int Level;
};

ostream &operator<<(ostream& o, const ErrorObj &E);

#endif
