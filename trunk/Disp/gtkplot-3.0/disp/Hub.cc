//$Id$
#include <Hub.h>
#include "./WidgetLib.h"
#include <ErrorObj.h>
#include <clstring.h>
#include <cl.h>
#include <mach.h>
//
//---------------------------------------------------------------------
//
int Hub::CmdProcessor(int fd, void *data) 
{
  char Msg[MSG_BUFLEN]={'\0'};;
  char *KeyWord=NULL,*Val=NULL;
  static int PW=970,PH=150;

  /*
  CmdIO.Receive(Msg,MSG_BUFLEN);
  cerr << "Cmd: " << Msg << endl;

  BreakStr(Msg,&KeyWord,&Val);

  if (KeyWord)
    {
      if (!strcmp(KeyWord,                   "print"))
	Doer->prtdata();
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "xscale"))
	{
	  double base=10; int scale;
	  char *v=NULL;
	  if (Val) v=strtok(Val,",");
	  if (v) sscanf(v,"%d",&scale);
	  if ((v=strtok(NULL,","))) sscanf(v,"%lf",&base);
	  Doer->SetAttribute(XYPanel::XSCALE,scale,base);
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "yscale"))
	{
	  double base=10; int scale;
	  char *v=NULL;
	  if (Val) v=strtok(Val,",");
	  if (v) sscanf(v,"%d",&scale);
	  if ((v=strtok(NULL,","))) sscanf(v,"%lf",&base);
	  Doer->SetAttribute(XYPanel::YSCALE,scale,base);
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "xtics"))
	{
	  int x0,x1;

	  if (Val) 
	    {
	      sscanf(strtok(Val,","),"%d",&x0);
	      sscanf(strtok(NULL,","),"%d",&x1);
	      cerr << x0 << " " << x1 << endl;
	      Doer->SetAttribute(XYPanel::XTICS0,x0);
	      Doer->SetAttribute(XYPanel::XTICS1,x1);
	    }
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "wait"))
	{
	  CmdIO.Send(SURE,strlen(SURE));
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "clear"))
	{
	  int w=MultiPanel::ALLPANELS,overlay=XYPanel::ALLOVERLAYS;
	  char *v=NULL; if (Val) v=strtok(Val,",");
	  if (v) sscanf(v,"%d",&w);
	  v=strtok(NULL,",");
	  if (v) sscanf(v,"%d",&overlay);
	  try {Doer->Clear(w,overlay);}
	  catch (ErrorObj x) {x << x.what() << endl;}
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "lstyle"))
	{
	  char *v=NULL;
	  if(Val) strtok(Val,",");
	  int p=MultiPanel::ALLPANELS,o=XYPanel::ALLOVERLAYS,
	    ls=FL_NORMAL_XYPLOT;

	  if (v) sscanf(v,"%d",&ls);
	  v=strtok(NULL,",");	  if (v) sscanf(v,"%d",&p);
	  v=strtok(NULL,",");	  if (v) sscanf(v,"%d",&o);
	  try {Doer->SetAttribute(XYPanel::LINESTYLE,ls,p,o);}
	  catch (ErrorObj x) {x << x.what() << endl;}
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "bye"))
	{
	  cleanup(1);
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "shutdown"))
	{
	  cleanup();
	  Doer->Init(0,0);
          CmdIO.Send(DONE,strlen(DONE));
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "reset"))
	{
	  int n=0;
	  if (Val) sscanf(Val,"%d",&n);
	  Counter=n;
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "text"))
        {
	  char *t,*t0; int Which=MultiPanel::ALLPANELS,Align=FL_ALIGN_RIGHT;
	  float x=0,y=0;FL_COLOR Col=FL_YELLOW;
	  if (Val)
	    {
	      t=clstrtok(Val,",",'\\');
	      if ((t0=clstrtok(NULL,",",'\\'))!=NULL) 
		{sscanf(t0,"%d",&Which);cerr << t0 << endl;}
	      if ((t0=clstrtok(NULL,",",'\\'))!=NULL) 
		{sscanf(t0,"%f",&x);cerr << x << endl;}
	      if ((t0=clstrtok(NULL,",",'\\'))!=NULL) 
		{sscanf(t0,"%f",&y);cerr << y << endl;}
	      if ((t0=clstrtok(NULL,",",'\\'))!=NULL) 
		{sscanf(t0,"%d",&Align);cerr << Align << endl;}
	      if ((t0=clstrtok(NULL,",",'\\'))!=NULL) 
		{sscanf(t0,"%d",&Col);cerr << Col << endl;}
	      Doer->PutText(t,(double)x,(double)y,Col,Align,Which);
	    }
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "title"))
	{
	  int n=MultiPanel::ALLPANELS;
	  char *l=NULL, *v=NULL;
	  if (Val)
	    {
	      if (Val[0]!=',') l=strtok(Val,",");
	      else v=strtok(NULL,",");
	      if ((v=strtok(NULL,",")))  sscanf(v,"%d",&n);
	      try {Doer->SetAttribute(XYPanel::TTITLE,l,NULL,n);}
	      catch (ErrorObj x) {x<<x.what()<<endl;}
	    }
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "xlabel"))
	{
	  int n=MultiPanel::ALLPANELS;
	  char *l=NULL, *v=NULL;
	  if (Val)
	    {
	      if (Val[0]!=',') l=strtok(Val,",");
	      else v=strtok(NULL,",");
	      if ((v=strtok(NULL,",")))  sscanf(v,"%d",&n);
	      try {Doer->SetAttribute(XYPanel::XLABEL,l,NULL,n);}
	      catch (ErrorObj x) {x<<x.what()<<endl;}
	    }
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "ylabel"))
	{
	  int n=MultiPanel::ALLPANELS;
	  char *l=NULL, *v=NULL;
	  if (Val)
	    {
	      if (Val[0]!=',') l=strtok(Val,",");
	      else v=strtok(NULL,",");
	      if ((v=strtok(NULL,",")))  sscanf(v,"%d",&n);
	      try {Doer->SetAttribute(XYPanel::YLABEL,l,NULL,n);}
	      catch (ErrorObj x) {x<<x.what()<<endl;}
	    }
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,                   "columns"))
      {
	if (Val) sscanf(Val,"%d",&NoOfCols);
        NoOfOverlays.resize(NoOfCols-1);
	A = (int **)calloc(sizeof(int *), NoOfCols);
	//
	// Allocate the A array and set it's default value
	// Default ==> 1st COL is X and the rest are Yi's
	// Each Yi will be the 1st overlay of individual
	// panels.
	//
	for (int i=0; i < NoOfCols; i++)
	  A[i] = (int *)calloc(sizeof(int), 2);
	A[0][0] = -1;
	for (int i=1; i < NoOfCols; i++)
	  {
	    A[i][0] = i-1;
	    NoOfOverlays[i-1] = 1;
	  }
	//
	// Also determine the no. of panels required.
	//
	NoOfPanels = NoOfCols-1;
      }
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "ascii"))
	{
	  int ASCII;
	  if (Val) sscanf(Val,"%d",&ASCII);
	  if (!ASCII)
	    {
	      Data = (float *)realloc(Data,sizeof(float)*NoOfCols);
	      //	      LoadData=&(Hub::GulpBinData);
	    }
	  else
	    {if (Data) free(Data);Data=NULL;}//	    LoadData=&(Hub::GulpASCIIData);
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "grid"))
	{
	  int n=1;
	  if(Val) sscanf(Val,"%d",&n);
	  if (n) 
	    {
	      Doer->SetAttribute(XYPanel::XGRID,n);//FL_GRID_MAJOR
	      Doer->SetAttribute(XYPanel::YGRID,n);//FL_GRID_MAJOR
	    }
	  else
	    {
	      Doer->SetAttribute(XYPanel::XGRID,FL_GRID_NONE);
	      Doer->SetAttribute(XYPanel::YGRID,FL_GRID_NONE);
	    }
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "npoints"))
	{if (Val) sscanf(Val,"%d",&NPoints);}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "xaxis"))
	{if (Val) sscanf(Val,"%d",&ColAsX);}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "npanels"))
	{if (Val) sscanf(Val,"%d",&NoOfPanels);}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "plot"))
        {
	  Doer->Redraw();
	  CmdIO.Send(SURE,strlen(SURE));
        }
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "psize"))
	{
	  if (Val) 
	    {
	      sscanf(strtok(Val,","),"%d",&PW);
	      sscanf(strtok(NULL,","),"%d",&PH);
	    }
	}	  
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "map"))
        {
	  NoOfPanels = 0;
	  if (Val)
	    {
	      sscanf(strtok(Val,","),"%d",&A[0][0]);
	      sscanf(strtok(NULL,","),"%d",&A[0][1]);
	      for (int i = 1; i < NoOfCols; i++)
		for (int j = 0; j < 2; j++)
		  sscanf(strtok(NULL,","),"%d",&A[i][j]);

	      for (int i = 0; i < NoOfCols; i++)
		{
		  if (A[i][0] != -1)
		    NoOfPanels = A[i][0]>NoOfPanels?A[i][0]:NoOfPanels;
		}
	      NoOfPanels++;
	      NoOfOverlays.resize(NoOfPanels);
	      for (int i = 0; i < NoOfPanels; i++)
		{
		  NoOfOverlays[i] = 0;
		  for (int j = 0; j < NoOfCols; j++)
		    {
		      if (A[j][0] == i)
			NoOfOverlays[i]++;
		    }
		}
	    }
        }
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "init"))
	{
	  cerr << "No of panels = " << NoOfPanels << endl;
	  cerr << "No of npoints = " << NPoints << endl;
	  Doer->Init(NoOfPanels,NPoints);
	  for (int i=0;i<NoOfPanels;i++)
	    {
	      Doer->operator[](i).SetNoOfOverlays(NoOfOverlays[i]);
	      Doer->operator[](i).SetNoOfDataPts(NPoints);
	    }
	  Doer->MakeWindow(MultiPanel::ALLPANELS,0,0,PW,PH);
	  Counter=0;
          CmdIO.Send(DONE,strlen(DONE));
	}
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "realloc"))
	for (int i=0;i<NoOfPanels;i++)
	  Doer->operator[](i).SetNoOfDataPts(NPoints);
//---------------------------------------------------------------------
      else if (!strcmp(KeyWord,              "scroll"))
	{if (Val)        sscanf(Val,"%d",&Scroll);}
//---------------------------------------------------------------------
    }
  if (KeyWord) free(KeyWord);
  if (Val) free(Val);
  */
  return 1;
};
//
//---------------------------------------------------------------------
//
int Hub::DataProcessor(int fd, void *data) 
{
  if (Data) GulpBinData();
  else GulpASCIIData();
  //  this->LoadData();
  return 1;
}
//
//---------------------------------------------------------------------
//
Hub::Hub(MultiPanel *Do)
{
  MachineSpecType Mac;
  Scroll=Counter=NoOfCols=ColAsX=NPoints=NoOfPanels=0;
  set_terminate(MPDefaultErrorHandler);
  Doer=Do;
  Data=NULL;
  A = NULL;
  GetMachineSpec(&Mac);
  Big_Endian = Mac.big_endian;
  //  LoadData=&(Hub::GulpASCIIData);
}
//
//---------------------------------------------------------------------
//
int Hub::init(u_short Port)
{
  //
  // Login port
  //
  try {  CtrlPort.init(Port);}
  catch(ErrorObj x) { x << x.what() << endl;exit(-1);}

  cerr << "Listening on port " << CtrlPort.port() 
       << " for login requests." <<  endl;

  /*
  fl_add_io_callback(CtrlPort.descriptor(),
		     FL_READ,
		     hub_ctrl_io_callback,
		     (void *)this);
  */
  return 1;
}
//
//---------------------------------------------------------------------
//
int Hub::CtrlProcessor(int fd, void *data)
{
  int n;
  char Msg[MSG_BUFLEN]={'\0'};
  SockIO NewPort,Com1,Com2;

  CtrlPort.xcept(NewPort);

  if (((n=NewPort.Receive(Msg, MSG_BUFLEN)) == -1) ||
      (strncmp(Msg,GREETINGS,strlen(GREETINGS))))
    {
      perror("###Error");
      NewPort.over();
      throw(ErrorObj("In initial greetings itself!","###Error",
		     ErrorObj::Recoverable));
    }
  //
  // Having finished the protocol formalities, get down to business.
  //
  // Get the Endian type of the client.
  //

  NewPort.Receive(Msg,MSG_BUFLEN);
  strtok(Msg," ");sscanf(strtok(NULL," "),"%d",&Big_Endian);
  cerr<<"Receiving from Big_Endian = "<<Big_Endian<<" machine"<<endl;

  //-----------------------------------------------------------------------
  // Start negotiation for the command port
  //
  // Setup the command port first.
  //
  try {  Com1.init(CtrlPort.port()+1);}
  catch(ErrorObj x) 
    { 
      x << x.what() << " Port no. " 
	<< Com1.port() << endl;exit(-1);
    }
  //
  // Announce the success
  //
  cerr << "Listening on port " << Com1.port() << " for commands." << endl;
  //
  // Convey the port no. to dock on for commands and receive an ACK.
  // The client must attempt a connection on the said port immediately
  // after Sending the ACK.
  //
  sprintf(Msg,"%d%c",Com1.port(),'/0');
  //  cerr << "Sending..." << Msg << endl;

  NewPort.Send(Msg,strlen(Msg));
  while (NewPort.Receive(Msg,strlen(SURE))!=strlen(SURE));
  //
  // Accept the connection request and attach a 
  // callback function for further conversation.
  //
  Com1.xcept(CmdIO);
  /*
  fl_add_io_callback(CmdIO.descriptor(),
		     FL_READ,
		     hub_cmd_io_callback,
		     (void *)this);
  */

  //-----------------------------------------------------------------------
  // Start negotiation for the data port
  //
  // Setup the data port first.
  //
  try {  Com2.init(CtrlPort.port()+2);}
  catch(ErrorObj x) 
    { 
      x << x.what() << " Port no. " 
	<< Com2.port() << endl;exit(-1);
    }
  //
  // Announce the success
  //
  cerr << "Listening on port " << Com2.port() << " for data." << endl;
  //
  // Convey the port no. to dock on for commands and receive an ACK.
  // The client must attempt a connection on the said port immediately
  // after sending the ACK.
  //
  sprintf(Msg,"%d%c",Com2.port(),'\0');
  //  cerr << "Sending..." << Msg << endl;

  NewPort.Send(Msg,strlen(Msg));
  while (NewPort.Receive(Msg,strlen(SURE))!=strlen(SURE));
  //
  // Accept the connection request and attach a 
  // callback function for further conversation.
  //
  Com2.xcept(DataIO);
  /*
  fl_add_io_callback(DataIO.descriptor(),
		     FL_READ,
		     hub_data_io_callback,
		     (void *)this);

  */
  Com1.over();Com2.over();
  NewPort.over();
  return 1;
}
//
//---------------------------------------------------------------------
//
int Hub::cleanup(int ShutIOOnly)
{
  /*
  if (DataIO.descriptor()!=-1)
    fl_remove_io_callback(DataIO.descriptor(),FL_READ,hub_data_io_callback);
  if (CmdIO.descriptor()!=-1)
    fl_remove_io_callback(CmdIO.descriptor(),FL_READ,hub_cmd_io_callback);

  if (!ShutIOOnly)
    {
      //  CtrlPort.shutup(2); 
      CmdIO.shutup(2); DataIO.shutup(2);
      if (Data) free(Data); if (A) free(A);
    }
  */
  return 1;
}
//
//---------------------------------------------------------------------
//
int Hub::PlotData(int Col, float Data, int C)
{
  if (A[Col][0] == -1)
    for (int j=0; j < NoOfPanels; j++)
      if (Scroll) Doer->operator[](j).AddXData(&Data,1);
      else        Doer->operator[](j).PutXData(&Data,C);
  else
    //    if (Scroll) Doer->operator[](A[Col][0]).AddYData(&Data,1,A[Col][1]);
    //    else      Doer->operator[](A[Col][0]).PutYData(&Data,C,A[Col][1]);

  return 1;
}
//
//---------------------------------------------------------------------
//
int Hub::GulpBinData()
{
  DataIO.Receive((char *)Data,NoOfCols*sizeof(float));
  if (Big_Endian) swap_long((void *)Data,NoOfCols);

  for (int i=0;i<NoOfCols;i++) PlotData(i,Data[i],Counter);
 
  Counter++;
  return Counter;
}
//
//---------------------------------------------------------------------
//
int Hub::GulpASCIIData()
{
  register char Msg[MSG_BUFLEN]={'\0'};
  double D;

  DataIO.Receive(Msg,MSG_BUFLEN);

  for (int i=0;i<NoOfCols;i++)
    {
      if (i == 0) sscanf(strtok(Msg," "),"%lf",&D);
      else        sscanf(strtok(NULL," "),"%lf",&D);
      PlotData(i,D,Counter);
    }
  Counter++;
  return Counter;
}
//
//---------------------------------------------------------------------
// C callback to bounce the call back to C++ obj.
//
extern "C" 
{
  void hub_cmd_io_callback(int fd, void *data)
    {((Hub *)data)->CmdProcessor(fd,data);}
  void hub_data_io_callback(int fd, void *data)
    {((Hub *)data)->DataProcessor(fd,data);}
  void hub_ctrl_io_callback(int fd, void *data)
    {((Hub *)data)->CtrlProcessor(fd,data);}
};
