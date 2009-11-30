#include <stdio.h>
#include <mp.h>

main(int argc, char **argv)
{
  int n;
  float D[5];int i=0;
  char Msg[128],*r;
  float Data[3];
  SockIO CommandPort, DataPort;

  D[0]=0;
  if (argc < 2) 
    {
      cerr << "Usage: " << argv[0] << " DisplayName [FileName]" << endl;
      exit(0);
    }
  try {  Dock(argv[1],CommandPort,DataPort); }
  catch (ErrorObj x) {x << x.what() << endl; exit(0);}

  if (argc > 2)
    {
      FILE *fd=fopen(argv[2],"r");

      while(fgets(Msg,128,fd) != NULL)
	{
	  Msg[strlen(Msg)-1]='\0';
	  if (!strncmp(Msg,"data",4))
	    {
	      // char *v=strtok(&Msg[4]," ");
	      for (i=0;i<100;i++) 
		{
		  D[1]=(float)sin(2*3.1415*D[0]/100.0);
		  D[2]=(float)cos(2*3.1415*D[0]/100.0);
		  // while (v) {sscanf(v,"%f",&D[i++]);v=strtok(NULL," ");}
		  // n=DataPort.Send(&Msg[4],strlen(Msg)-4);
		  n=DataPort.Send((char *)D,sizeof(float)*3);
		  D[0]++;
		}
cerr << "Done" << endl;
	    }
	  else
	    {
	      n=CommandPort.Send(Msg,strlen(Msg));
	      if(!strncmp(Msg,"init",strlen("init")))
		CommandPort.Receive(Msg, strlen(DONE));
	    }
	  if (!strncmp(Msg,"wait",4)) 
	    CommandPort.Receive(Msg,strlen(SURE));
	  else if (!strncmp(Msg,"shutdown",8)) 
	    {CommandPort.Receive(Msg,strlen(DONE));break;}
	}
      fclose(fd);
    }
	
  while((r=fgets(Msg,128,stdin)) != NULL)
    {
      Msg[strlen(Msg)-1]='\0';
      if (!strncmp(Msg,"data",4))
	n=DataPort.Send(&Msg[4],strlen(Msg)-4);
      else
      {
	n=CommandPort.Send(Msg,strlen(Msg));
        if(!strncmp(Msg,"init",strlen("init")))
	  CommandPort.Receive(Msg, strlen(DONE));
      }
      if (!strncmp(Msg,"wait",4)) 
	CommandPort.Receive(Msg,strlen(SURE));
      else if (!strncmp(Msg,"shutdown",8)) 
       {
	 CommandPort.shutup(2); DataPort.shutup(2);
	 CommandPort.Receive(Msg,strlen(DONE));
	 exit(0);
       }
    }
  if (r==NULL) CommandPort.Send("shutdown",8);
}
