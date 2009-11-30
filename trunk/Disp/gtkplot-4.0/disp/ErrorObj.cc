#include <ErrorObj.h>

ErrorObj::ErrorObj(const char *m, const char *i, int l)
  :Id(i),Msg(m)
{
  //  if (i)  {if (Id) delete Id;   Id = new char [strlen(i)+1];  strcpy(Id,i);}
  //  if (m)  {if (Msg) delete Msg; Msg = new char [strlen(m)+1]; strcpy(Msg,m);}
  //  Msg=m; Id=i;
  Level=l;
}

void ErrorObj::SetSource(const char *s)
{
  /*
  if (s) {if (Src) delete Src; 
  Src = new char [strlen(s)+1]; strcpy(Src,s);}
  */
  Src=s;
}
const char* ErrorObj::what()
{
  int N=0;
  //  if (Id) N += strlen(Id);
  //  if (Msg) N += strlen(Msg) + 2 + 1;
  N = Id.size() + Msg.size();

  /*
  if (N) 
    {
      if (Message) delete Message; Message = new char [N];
      if (Id) {strcpy(Message,Id); strcat(Message,": ");}
      if (Msg) strcat(Message,Msg);
    }
  return Message;
  */
  return (Id + ": " + Msg).c_str();
}

ostream &operator<<(ostream& o, const ErrorObj &E)
{
  /*
  if (E.Id && (strlen(E.Id) > 0))    o << E.Id;
  if (E.Msg && (strlen(E.Msg) > 0))  o << ": " << E.Msg;
  return o;
  */
  o << E.Id << ": " << E.Msg;
  return o;
}
