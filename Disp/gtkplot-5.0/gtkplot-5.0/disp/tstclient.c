#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string.h>
#include <Protocol.h>

main(int argc, char **argv)
{
  int fd,cmdfd,datafd,n;
  u_short port,cmdport,dataport;
  struct sockaddr_in sin;
  struct hostent *phe;
  char Msg[128],tmp[128];

  memset((void *)&sin,NULL,sizeof(sin));
  sin.sin_family      = AF_INET;
  sin.sin_port        = htons(1857);
  if ( phe=gethostbyname(argv[1]))
    bcopy(phe->h_addr, (char *)&sin.sin_addr, phe->h_length);

  if ((fd = socket(AF_INET,SOCK_STREAM,0)) == -1)
    {perror("###Error");close(fd);exit(-1);}

  if (connect(fd, (struct sockaddr *)&sin, sizeof(sin))<0)
    {perror("###Error");close(fd);exit(-1);}

  n=write(fd,GREETINGS,strlen(GREETINGS));
  fprintf(stderr,"Wrote %d bytes\n",n);

  read(fd,Msg,128);
  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&cmdport);
  fprintf(stderr,"Got command port no. %d\n",cmdport);
  write(fd,SURE,strlen(SURE));
  /*  
  read(fd,Msg,128);
  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&dataport);
  fprintf(stderr,"Got data port no. %d\n",dataport);
  write(fd,SURE,strlen(SURE));
  */
  /*----------------------------------------------------------------*/
  memset((void *)&sin,NULL,sizeof(sin));
  sin.sin_family      = AF_INET;
  sin.sin_port        = htons(cmdport);
  if ( phe=gethostbyname(argv[1]))
    bcopy(phe->h_addr, (char *)&sin.sin_addr, phe->h_length);

  if ((cmdfd = socket(AF_INET,SOCK_STREAM,0)) == -1)
    {perror("###Error");close(cmdfd);exit(-1);}

  if (connect(cmdfd, (struct sockaddr *)&sin, sizeof(sin))<0)
    {
      perror("###Error");fprintf(stderr,"on port %d.\n",cmdport);
      close(cmdfd);exit(-1);
    }

  n=write(cmdfd,GREETINGS,strlen(GREETINGS));
  while(scanf("%s",Msg) != 0)
    n=write(cmdfd,Msg,strlen(Msg));

  /*----------------------------------------------------------------*/
  close(fd);
}
  

