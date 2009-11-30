#include <ErrorObj.h>

ErrorObj::ErrorObj(const char *m, const char *i, int l)
  //:Id(i,strlen(i)),Msg(m,strlen(m))
{
  if (i)  {Id = new char [strlen(i)+1];  strcpy(Id,i);}
  if (m)  {Msg = new char [strlen(m)+1]; strcpy(Msg,m);}
  Level=l;
}

void ErrorObj::SetSource(const char *s)
{
  if (s) {Src = new char [strlen(s)+1]; strcpy(Src,s);}
}
const char* ErrorObj::what()
{
  int N=0;
  if (Id) N += strlen(Id);
  if (Msg) N += strlen(Msg) + 2 + 1;

  if (N) 
    {
      Message = new char [N];
      if (Id) {strcpy(Message,Id); strcat(Message,": ");}
      if (Msg) strcat(Message,Msg);
    }
  return Message;
}

ostream &operator<<(ostream& o, const ErrorObj &E)
{
  if (E.Id && (strlen(E.Id) > 0))    o << E.Id;
  if (E.Msg && (strlen(E.Msg) > 0))  o <<": "<< E.Msg;
  return o;
}
