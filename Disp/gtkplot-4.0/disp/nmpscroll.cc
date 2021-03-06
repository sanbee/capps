/*-------------------------------------------------------------------------
    Attempt (a shoddy attempt though) at writing a generalized translator
    from input data/command sequences to that required by mpsrvr.

    History record:

      ??????          First version.  Dark ages.
                                          S.Bhatnagar

      May,2000        Always supplies binary data to mpsrvr.  By default
                      it also expects binary data on the input.  If
                      "#ASCII 1" is found in the header (and it must be
                       BEFORE "#COLS" commands in the header), it can
                      take ASCII data at the input.
                                          S.Bhatnagar
--------------------------------------------------------------------------*/

#include <stdio.h>
#include <mp.h>
#include <vector>
#include <CommH.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <math.h>
#define EOH      "#End"
#define NCOLS    "#NCOLS"
#define NROWS    "#NROWS"
#define LABELS   "#LABEL"
#define ASCII    "#ASCII"
#define SCROLL   "#SCROLL"
#define RESET    "#RESET"

main(int argc, char **argv)
{
  float D[5];int i=0;
  int PX=800, PY=70,DOASCII=0,DOSCROLL=1,DORESET=-1;
  char Msg[256*3];
  char *r;
  //  vector<float> Data;
  float *Data;
  SockIO CommandPort, DataPort;
  int Commands=0,Cols=0,Rows=0, NPoints=1000,Finish=0;
  float f;
  D[0]=0;
  if (argc < 5) 
    {
      cerr << "Usage: " << argv[0] << " DisplayName NPoints PX PY [FileName]" << endl;
      exit(0);
    }

  try {  nDock(argv[1],CommandPort,DataPort); }
  catch (ErrorObj x) {x << x.what() << endl; exit(0);}

  sscanf(argv[2],"%d",&NPoints);
  if (sscanf(argv[3],"%d",&PX)!=1) PX=800;
  if (sscanf(argv[4],"%d",&PY)!=1) PY=70;

  if (argc > 2)
    {
      FILE *fd=stdin;
      if (strcmp(argv[5],"-")) fd=fopen(argv[5],"r");

      sprintf(Msg,"npoints=%d%c",NPoints,'\0');
      CommandPort.Send(Msg,strlen(Msg));

      sprintf(Msg,"psize=%d,%d%c",PX,PY,'\0');
      CommandPort.Send(Msg,strlen(Msg));
	  
      sprintf(Msg,"setscroll=%d%c",DOSCROLL,'\0');
      CommandPort.Send(Msg,strlen(Msg));

      while((fgets(Msg,128,fd) != NULL) && (!(Finish=feof(fd))))
	{
	  Msg[strlen(Msg)-1]='\0';

	  i=0;
	  while ((Msg[i] == ' ') && (Msg[i] != '\0')) i++; r=&Msg[i];
	  if (!strncmp(r,"end",3))
	    Commands=0;
	  else if (!strncmp(r,ASCII,strlen(ASCII)))
	    {
	      sscanf(&r[6],"%d",&DOASCII);
	    }
	  else if (!strncmp(r,SCROLL,strlen(SCROLL)))
	    {
	      sscanf(&r[7],"%d",&DOSCROLL);
	      sprintf(Msg,"setscroll=%d%c",DOSCROLL,'\0');
	      CommandPort.Send(Msg,strlen(Msg));
	    }
	  else if (!strncmp(r,NROWS,strlen(NROWS)))
	    {
	      int NN;
	      sscanf(&r[6],"%d",&NN);
	      if (NN < pow(2,31)) NPoints=NN;
	      sprintf(Msg,"npoints=%d%c",NPoints,'\0');
	      CommandPort.Send(Msg,strlen(Msg));
	    }
	  else if (!strncmp(r,RESET,strlen(RESET)))
	    {
	      sscanf(&r[6],"%d",&DORESET);
	    }
	  else if (!strncmp(r,EOH,strlen(EOH)))
	    {
	      int N=0;
	      //	      Data.resize(Cols);
	      Data = (float *)malloc(sizeof(float)*Cols);
	      Rows=0;
	      while(1)
		{
		  int i;
		  for (i=0;i<Cols;i++)
		    if (DOASCII) fscanf(fd,"%f",&Data[i]);
		    else if (fread(&f,sizeof(float),1,fd)!=1) 
		      {
			cerr << "###Error: Something wrong!  Found unexpected EOF" 
			     << endl;break;
			sprintf(Msg,"bye%c",'\0');
			CommandPort.Send(Msg,strlen(Msg));
			sprintf(Msg,"plot%c",'\0');
			CommandPort.Send(Msg,strlen(Msg));
			exit(0);
		      }
		    else Data[i] = f;
		  //		  Data[0]=Rows++;
		  if (feof(fd)) 
		    {
		      sprintf(Msg,"plot%c",'\0');
		      CommandPort.Send(Msg,strlen(Msg));
		      cerr << "###Informational: I found EOF and don't know what to do!"
			   << endl;
		      cerr << "###Informational: Exiting now." << endl;
		      exit(0);
		    }

		  DataPort.Send((char *)&Data[0],sizeof(float)*Cols);
		  if (DOSCROLL)
		    {
		      sprintf(Msg,"plot%c",'\0');
		      CommandPort.Send(Msg,strlen(Msg));
		    }
		  N++;

		  if ((DORESET>0) && N/DORESET>0)
		    {
		      N=0;
		      sprintf(Msg,"plot%c",'\0');
		      CommandPort.Send(Msg,strlen(Msg));
		      sprintf(Msg,"reset=%d%c",0,'\0');
		      CommandPort.Send(Msg,strlen(Msg));
		    }
		}
	    }
	  else if(!strncmp(r,LABELS,strlen(LABELS)))
	    {
	      int n;
	      sscanf(&r[6],"%d",&n);
	      r=strtok(Msg,"  ");
	      r=strtok(NULL,"  ");
	      if (n)
		sprintf(Msg,"legend=%s,%d%c",r,n-1,'\0');
	      else
		sprintf(Msg,"xlabel=%s,%d%c",r,Cols-2,'\0');
	      CommandPort.Send(Msg,strlen(Msg));
	    }
	  else if(!strncmp(r,NCOLS,strlen(NCOLS)))
	    {
	      r=strtok(Msg," ");
	      r=strtok(NULL," ");
	      sscanf(r,"%d",&Cols);
	      cerr << "NCols = " << Cols << endl;
	      if (!Cols)
		cerr << "No Rows or Cols given" << endl;
	      else
		{
		  cerr << Cols << " " << Rows << endl;
		  sprintf(Msg,"columns=%d%c",Cols,'\0');
		  CommandPort.Send(Msg,strlen(Msg));

		  sprintf(Msg,"map=-1,0%c",'\0');
		  for (int i=0;i<Cols-1;i++)
		    {
		      char tmp[24];
		      sprintf(tmp,",%d,%d%c",i,0,'\0');
		      strcat(Msg,tmp);
		    }
		  CommandPort.Send(Msg,strlen(Msg));

		  //		  sprintf(Msg,"ascii=%d%c",DOASCII,'\0');
		  sprintf(Msg,"ascii=0%c",'\0');
		  CommandPort.Send(Msg,strlen(Msg));
		  
		  CommandPort.Send("init",4);
		  CommandPort.Receive(Msg,strlen(DONE));
		}
	    }
	  else if(!strncmp(r,"npoints",7))
	    sscanf(&r[8],"%d",&Rows);
	  else if (!strncmp(r,"wait",4)) 
	    CommandPort.Receive(Msg,strlen(SURE));
	  else if (!strncmp(r,"shutdown",8)) 
	    {CommandPort.Receive(Msg,strlen(DONE));break;}
	  else if (!strncmp(r,"sleep",5)) 
	    {int n;sscanf(&r[5],"%d",&n);sleep(n);}
	  /*
	  else 
	    if (r && strlen(r)) CommandPort.Send(r,strlen(r));
	    */
	}
      fclose(fd);
    }

  if (Finish)
    {
      sprintf(Msg,"shutdown%c",'\0');
      CommandPort.Send(Msg,strlen(Msg));
      exit(0);
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
