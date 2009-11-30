#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

main()
{
  char Msg[128];
  struct sockaddr_in sin;
  struct hostent *phe;
  struct sockaddr_in mysin; int len;  
  int Reuse=1,fd;
  unsigned long n;
  int SockID;

  memset((void *)&sin,NULL,sizeof(sin));

  sin.sin_addr.s_addr = INADDR_ANY;
  sin.sin_family      = AF_INET;
  sin.sin_port        = htons(1857);

  if ((SockID = socket(AF_INET,SOCK_STREAM,0)) == -1)
    perror("###Error");

  if (setsockopt(SockID, SOL_SOCKET, SO_REUSEADDR, 
		 (void *)&Reuse, sizeof(Reuse)) == -1)
    perror("###Error");
      
  if (bind(SockID, (struct sockaddr *)&sin, sizeof(sin))<0)
    perror("###Error");

  if (listen(SockID,2) < 0)
    perror("###Error");

  scanf("%s",Msg);

  fd= accept(SockID,(struct sockaddr *)&mysin,&len);

  if (recv(fd,(char *)&n,sizeof(unsigned long),0)<0)
    perror("###Error:");
  n = ntohl(n);
  printf("N=%d\n",n);
  recv(fd,Msg,n,0);

  fprintf(stderr,"Msg=%s\n",Msg);
}  
