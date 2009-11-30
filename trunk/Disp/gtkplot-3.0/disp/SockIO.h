// -*- C++ -*-
// $Id$
#if !defined(SOCKIO_H)
#define      SOCKIO_H
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <Protocol.h>
#include <strings.h>

class SockIO
{
public:
  SockIO()  {SockID=-1;memset((void *)&sin,0,sizeof(sin));AckNeeded=1;};
  ~SockIO() {shut();};
  //---------------------------------------------------------------------------
  void reset(int fd,struct sockaddr_in *sin_in, int AckRequired=1)
  {shut(); SockID=fd; memcpy((void*)&sin,(void*)sin_in,sizeof(sin));
   AckNeeded = AckRequired;}
  //---------------------------------------------------------------------------
  int init(u_short Port,int Passive=1,char *Addr=NULL,
	   int domain=AF_INET, 
	   int type=SOCK_STREAM, 
	   int protocol=IPPROTO_TCP);
  //---------------------------------------------------------------------------
  void over() {if (SockID != -1) close(SockID);};
  //---------------------------------------------------------------------------
  int konnekt();
  //---------------------------------------------------------------------------
  int shutup(int how) {int i=shutdown(SockID,how);SockID=-1;return i;}
  //---------------------------------------------------------------------------
  int shut() 
  {
    if (SockID > -1) close(SockID);SockID=-1;  
    memset((void *)&sin,0,sizeof(sin));
    return SockID;
  };
  //---------------------------------------------------------------------------
  int Send(char* Msg, int len);
  //---------------------------------------------------------------------------
  int Receive(char* Msg, int len);
  //---------------------------------------------------------------------------
  void xcept(SockIO &P)
  {
    struct sockaddr_in mysin; unsigned int len;  
    int fd= accept(SockID,(struct sockaddr*)&mysin,&len);
    P.reset(fd,&mysin);
  }
  //---------------------------------------------------------------------------
  int descriptor() {return SockID;}
  //---------------------------------------------------------------------------
  FILE* fd(char* mode) {return fdopen(SockID,mode);}
  //---------------------------------------------------------------------------
  u_short port() {return ntohs(sin.sin_port);}
  //---------------------------------------------------------------------------
  struct sockaddr_in* addr() {return &sin;};
  //---------------------------------------------------------------------------

private:
  struct sockaddr_in sin;
  int SockID, AckNeeded;
};

#endif      
