#include <stdio.h>
#include <mp.h>
#include <string>
#include <vector>
#include <CommH.h>

main(int argc, char **argv)
{
  int n,NCols;
  string Msg;
  vector<float> FloatData;

  char tmp[512];
  int SleepTime=5;
  SockIO CommandPort, DataPort;

  try {  nDock(argv[1],CommandPort,DataPort); }
  catch (ErrorObj x) {x << x.what() << endl; exit(0);}

  while(fgets(tmp,512,stdin)!=NULL)
    {
      Msg = tmp; Msg[Msg.size()-1]='\0';
      Msg.resize(Msg.size()+1);Msg[Msg.size()]='\0';

      if (!strncmp(&Msg[0],"columns",7))
	sscanf(&Msg[8],"%d",&NCols);
      if (!strncmp(&Msg[0],"sleep",5)) 
	sscanf(&Msg[6],"%d",&SleepTime);
      else if (!strncmp(&Msg[0],"wait",5))
	CommandPort.Receive(&Msg[0],strlen(SURE));
      else if (!strncmp(&Msg[0],"start",5)) 
	{
	  n=200;
	  while(1)
	    {
	      /*
	      Msg.resize(0);
	      sprintf(tmp,"%f ",(float)n++);
	      Msg+= tmp;
	      for (int i=0;i<60;i++)
		{
		  sprintf(tmp,"%f ",(float)sin(2*3.14*n/20));
		  Msg +=" "; Msg += tmp;
		}
	      DataPort.Send(Msg);
	      CommandPort.Send("plot",4);
	      CommandPort.Receive(Msg);
	      Msg.resize(Msg.size()+1);Msg[Msg.size()]='\0';
	      */
	      FloatData.resize(NCols);
	      FloatData[0]=n++;
	      for (int i=1;i<NCols;i++)
		FloatData[i]=(i+1)*sin(2*3.14*n/20);
	      DataPort.Send((char *)&FloatData[0],sizeof(float)*NCols);
	      CommandPort.Send("plot",4);
	      //	      CommandPort.Receive(Msg);

	      sleep(SleepTime);
	    }
	}
      else if (!strncmp(&Msg[0],"data",4))
	n=DataPort.Send(&Msg[4],strlen(&Msg[0])-4);
      else
      {
	n=CommandPort.Send(Msg);
        if(!strncmp(&Msg[0],"init",strlen("init")))
	  CommandPort.Receive(Msg);
      }
      if (!strncmp(&Msg[0],"shutdown",5)) 
       {CommandPort.Receive(Msg);break;}
    }
  CommandPort.shutup(2); DataPort.shutup(2);
}


