#include <SockIO.h>
#include <ErrorObj.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>

int SockIO::init(u_short Port,int Passive,char *Addr, 
		 int domain,
		 int type,
		 int protocol)
{
  //
  // Shut if a connection was active.
  //
  struct hostent *phe;

  HANDLE_EXCEPTIONS(
  shut();
  memset((void *)&sin,0,sizeof(sin));

  if (Addr==NULL) sin.sin_addr.s_addr = INADDR_ANY;
  else if ( (phe=gethostbyname(Addr)))
      bcopy(phe->h_addr, (char *)&sin.sin_addr, phe->h_length);
    else   
      sin.sin_addr.s_addr = INADDR_ANY;

  sin.sin_family      = domain;
  sin.sin_port        = htons(Port);

  if ((SockID = socket(domain,type,protocol)) == -1)
    {
      perror("###Error");
      throw(ErrorObj("Error in opening socket","###Error",
		     ErrorObj::Recoverable));
    }
  if (Passive)
    {
      int Reuse=1;
      if (setsockopt(SockID, SOL_SOCKET, SO_REUSEADDR, 
		     (void *)&Reuse, sizeof(Reuse)) == -1)
	{
	  perror("###Error");
	  shut();
	  throw(ErrorObj("Error in setsockopt","###Error",
			 ErrorObj::Recoverable));
	}
      
      if (bind(SockID, (struct sockaddr *)&sin, sizeof(sin))<0)
	{
	  perror("###Error");
	  shut();
	  throw(ErrorObj("Error in binding socket","###Error",
			 ErrorObj::Recoverable));
	}

      if (listen(SockID,2) < 0)
	{
	  perror("###Error");
	  shut();
	  throw(ErrorObj("Error in listening on socket","###Error",
			 ErrorObj::Recoverable));
	}
    }
  return SockID;
)
};
//
//---------------------------------------------------------------------
//
int SockIO::konnekt()
{
  if (connect(SockID, (struct sockaddr *)&sin, sizeof(sin))<0)
    {
      perror("###Error");
      throw(ErrorObj("Error in connecting to socket","###Error",
		     ErrorObj::Recoverable));
    }
  return 1;
}
//
//---------------------------------------------------------------------
//
int SockIO::Send(string& Msg)
{
  unsigned long i;

  i=htonl(Msg.size());
  send(SockID, (char *)&i,sizeof(unsigned long),0);
  i=send(SockID, &Msg[0], Msg.size(), 0);

  if (AckNeeded) 
    {
      char Ack[32];
      recv(SockID,(char *)&i,sizeof(unsigned long),0);
      i=htonl(i);
      i=recv(SockID,Ack,i,0);
    }

  return i;
};
//
//---------------------------------------------------------------------
//
int SockIO::Send(char *Msg, int len)
{
  unsigned long i;
  //  unsigned long n=strlen(Msg);

  //  write(SockID,(char *)&n,sizeof(int));
  //  write(SockID,Msg,n);
  i=htonl(len);
  send(SockID, (char *)&i,sizeof(unsigned long),0);
  i=send(SockID, Msg, len, 0);

  if (AckNeeded) 
    {
      char Ack[32];
      recv(SockID,(char *)&i,sizeof(unsigned long),0);
      i=htonl(i);
      i=recv(SockID,Ack,i,0);
    }

  return i;
};
//
//---------------------------------------------------------------------
//
int SockIO::Receive(string &Msg)
{
  unsigned long n;
  int m;
  if (!(m=recv(SockID,(char *)&n,sizeof(unsigned long),0))) return m;
  n=ntohl(n);
  Msg.resize(n);
  n=recv(SockID,&Msg[0],n,0);
  if (AckNeeded) 
    {
      n=strlen(DONE);
      n=htonl(n);
      send(SockID, (char *)&n,sizeof(unsigned long),0);
      n=send(SockID,DONE,strlen(DONE),0);
    }
  return n;
};
//
//---------------------------------------------------------------------
//
int SockIO::Receive(char *Msg, int len)
{
  unsigned long n;
//    read(SockID,&n,sizeof(int));
//    n=n>len?len:n;
//    read(SockID,Msg,n);
  recv(SockID,(char *)&n,sizeof(unsigned long),0);
  n=ntohl(n);
  n=n>(unsigned long)len?(unsigned long)len:n;
  n=recv(SockID,Msg,n,0);
  if (AckNeeded) 
    {
      n=strlen(DONE);
      n=htonl(n);
      send(SockID, (char *)&n,sizeof(unsigned long),0);
      n=send(SockID,DONE,strlen(DONE),0);
    }
  return n;
};
