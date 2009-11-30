#include <stdio.h>
#include <mp.h>

main(int argc, char **argv)
{
  int n;
  char Msg[128];
  SockIO CommandPort, DataPort;

  try {  Dock(argv[1],CommandPort,DataPort); }
  catch (ErrorObj x) {x << x.what() << endl; exit(0);}

  while(fgets(Msg,128,stdin) != NULL)
    {
      Msg[strlen(Msg)-1]='\0';
      if (!strncmp(Msg,"sleep",5)) sleep(5);
      if (!strncmp(Msg,"start",5)) 
	{
	  n=0;
	  while(1)
	    {
	      sprintf(Msg,"%f %f %f\0",(float)n++,
		      (float)sin(2*3.14*n/20),(float)cos(2*3.14*n/20));
	      DataPort.Send(Msg,strlen(Msg));
	      CommandPort.Send("plot",4);
	      sleep(1);
	    }
	}

      else if (!strncmp(Msg,"data",4))
	n=DataPort.Send(&Msg[4],strlen(Msg)-4);
      else
      {
	n=CommandPort.Send(Msg,strlen(Msg));
        if(!strncmp(Msg,"init",strlen("init")))
	  CommandPort.Receive(Msg, strlen(DONE));
      }
      if (!strncmp(Msg,"shutdown",5)) 
       {CommandPort.Receive(Msg,strlen(DONE));break;}
    }
  CommandPort.shutup(2); DataPort.shutup(2);
}


