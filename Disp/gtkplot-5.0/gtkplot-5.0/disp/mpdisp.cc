#include <stdio.h>
#include <mp.h>
#include <vector>

main(int argc, char **argv)
{
  float D[5];int i=0;
  char Msg[128],*r;
  vector<float> Data;
  SockIO CommandPort, DataPort;
  int Commands=0,Cols=0,Rows=0;

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

      while((fgets(Msg,128,fd) != NULL) && (!feof(fd)))
	{
	  Msg[strlen(Msg)-1]='\0';

	  i=0;
	  while ((Msg[i] == ' ') && (Msg[i] != '\0')) i++; r=&Msg[i];
	  if (!strncmp(r,"sleep",5))
	    sleep(5);
	  if (!strncmp(r,"begin",5))
	    Commands=1;
	  else if (!strncmp(r,"end",3))
	    Commands=0;
	  else if (!strncmp(r,"data",4) && !Commands)
	    {
	      if (!Rows || !Cols)
		cerr << "No Rows or Cols given" << endl;
	      else
		{
		  cerr << Cols << " " << Rows << endl;
		  Data.resize(Cols);
		  for (int j=0;j<Rows;j++)
		    {
		      for (int i=0;i<Cols;i++)
			if (!fscanf(fd,"%f",&Data[i])) break;
		      //		      for (int i=0;i<Cols;i++)
		      //			fprintf(stderr,"%f ",Data[i]);
		      //		      fprintf(stderr,"\n");
		      DataPort.Send((char *)&Data[0],sizeof(float)*Cols);
		    }
		}
	      //	      break;
	    }
	  else if(!strncmp(r,"init",strlen("init")))
	    {
	      CommandPort.Send(r,strlen(r));
	      CommandPort.Receive(Msg, strlen(DONE));
	    }
	  else if(!strncmp(r,"columns",7))
	    {
	      CommandPort.Send(r,strlen(r));
	      sscanf(&r[8],"%d",&Cols);
	    }
	  else if(!strncmp(r,"npoints",7))
	    {
	      CommandPort.Send(r,strlen(r));
	      sscanf(&r[8],"%d",&Rows);
	    }
	  else if (!strncmp(r,"wait",4)) 
	    CommandPort.Receive(Msg,strlen(SURE));
	  else if (!strncmp(r,"shutdown",8)) 
	    {CommandPort.Receive(Msg,strlen(DONE));break;}
	  else if (!strncmp(r,"sleep",5)) 
	    {int n;sscanf(&r[5],"%d",&n);sleep(n);}
	  else 
	    if (r && strlen(r)) CommandPort.Send(r,strlen(r));
	}
      fclose(fd);
    }
	
  while((r=fgets(Msg,128,stdin)) != NULL)
    {
      Msg[strlen(Msg)-1]='\0';
      if(!strncmp(Msg,"init",strlen("init")))
	  CommandPort.Receive(Msg, strlen(DONE));
      if (!strncmp(Msg,"wait",4)) 
	CommandPort.Receive(Msg,strlen(SURE));
      else if (!strncmp(Msg,"shutdown",8)) 
       {
	 CommandPort.shutup(2); DataPort.shutup(2);
	 CommandPort.Receive(Msg,strlen(DONE));
	 exit(0);
       }
      else CommandPort.Send(Msg,strlen(Msg));
    }
  if (r==NULL) CommandPort.Send("shutdown",8);
}
