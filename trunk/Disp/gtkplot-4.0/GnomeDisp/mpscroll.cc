#include <stdio.h>
#include <mp.h>
#include <vector>

#define EOH      "#End"
#define NCOLS    "#NCOLS"
#define NROWS    "#NROWS"
#define LABELS   "#LABEL"

main(int argc, char **argv)
{
  float D[5];int i=0;
  int PX=800, PY=70;
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

  try {  Dock(argv[1],CommandPort,DataPort); }
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
	  
      sprintf(Msg,"setscroll=%d%c",1,'\0');
      CommandPort.Send(Msg,strlen(Msg));

      while((fgets(Msg,128,fd) != NULL) && (!(Finish=feof(fd))))
	{
	  Msg[strlen(Msg)-1]='\0';

	  i=0;
	  while ((Msg[i] == ' ') && (Msg[i] != '\0')) i++; r=&Msg[i];
	  if (!strncmp(r,"end",3))
	    Commands=0;
	  else if (!strncmp(r,EOH,strlen(EOH)))
	    {
	      //	      Data.resize(Cols);
	      Data = (float *)malloc(sizeof(float)*Cols);
	      Rows=0;
	      while(1)
		{
		  int i;
		  for (i=0;i<Cols;i++)
		    if (fread(&f,sizeof(float),1,fd)!=1) 
		      {
			cerr << "Something wrong!" << endl;break;
		      }
		    else Data[i] = f;
		  Data[0]=Rows++;
		  DataPort.Send((char *)&Data[0],sizeof(float)*Cols);
		  sprintf(Msg,"plot%c",'\0');
		  CommandPort.Send(Msg,strlen(Msg));
		  if (feof(fd)) exit(0);
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
