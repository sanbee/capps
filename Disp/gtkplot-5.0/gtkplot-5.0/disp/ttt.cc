#include <stdio.h>
#include <mp.h>
#include <string>
#include <vector>

main(int argc, char **argv)
{
  int n;
  string Msg;
  vector<float> FloatData;
  FILE *fd;

  char tmp[512];
  int SleepTime=5;
  SockIO CommandPort, DataPort;


  try {  Dock(argv[1],CommandPort,DataPort); }
  catch (ErrorObj x) {x << x.what() << endl; exit(0);}

  fd = fopen(argv[2],"r");

  while(fgets(tmp,512,fd)!=NULL)
    {
      Msg = tmp; Msg[Msg.size()-1]='\0';
      Msg.resize(Msg.size()+1);Msg[Msg.size()]='\0';
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

	      FloatData.resize(61);
	      FloatData[0]=n++;
	      for (int i=0;i<61;i++)
		{
		  fscanf(stdin, "%f",&FloatData[i]);
		  cerr << FloatData[i] << " ";
		}
	      cerr << endl;
	      //		FloatData[i]=(i+1)*sin(2*3.14*n/20);
	      DataPort.Send((char *)&FloatData[0],sizeof(float)*61);
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

