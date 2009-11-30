//-*-C++-*-
#if !defined(ERROROBJ_H)
#define      ERROROBJ_H
#include <string.h>
#include <iostream.h>

#define HANDLE_EXCEPTIONS(str)    try {str} catch(...) {throw;}

class ErrorObj{
 public:
  enum {Informational=100,Recoverable,Severe,Fatal};
  ErrorObj(){Id=Msg=Src=Message=NULL;};
  ErrorObj(const char *m,const char *i,int l=0);
  ~ErrorObj()
    {
//      if (Msg) delete Msg; if (Id) delete Id; 
//      if (Src) delete Src; if (Message) delete Message;
    };
  
  void SetSource(const char *m=0);
  char *Source() {return Src;}
  int Severity() {return Level;}
  const char *what();

  ostream &operator<<(const char *m) {return cerr << m;}
  ostream &operator<<(ErrorObj &E) {return cerr << E;}
  friend ostream &operator<<(ostream& o,const ErrorObj&);

 private:
  char *Id,*Msg,*Src,*Message;
  int Level;
};

ostream &operator<<(ostream& o, const ErrorObj &E);

#endif
