#include <stdlib.h>
#include <SockIO.h>
#include <Protocol.h>
#include <MultiPanel.h>
#include <vector>
#include <exception>
#include <ErrorObj.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mach.h>

int CommH(int CmdPort, int DataPort,SockIO& CmdSock, SockIO& DataSock)
{
  string Msg;
  SockIO Com1,Com2;
  int Big_Endian;

  try {  Com1.init(CmdPort);}
  catch(ErrorObj x) 
    { 
      x << x.what() << " Port no. " 
	<< Com1.port() << endl;
      exit(-1);
    }
  //
  // Announce the success
  //
  cerr << "schild: Listening on port " << Com1.port() << " for commands." 
       << endl;
  //  n=wait_for_activity(Com1.descriptor());
  //  if (n<0) cerr << "wait failed"<< endl;
  //
  // Accept the connection request and attach a 
  // callback function for further conversation.
  //
  Com1.xcept(CmdSock);
  cerr << CmdSock.descriptor() << endl;

  CmdSock.Receive(Msg);
  strtok(&Msg[0]," ");sscanf((char *)strtok(NULL," "),"%d",&Big_Endian);
  cerr<<"Receiving from Big_Endian = "<<Big_Endian<<" machine"<<endl;

  //  gdk_input_add(CmdIO.descriptor(),
  //		GDK_INPUT_READ,
  //		hub_cmd_io_callback,
  //		(gpointer)this);

  //-----------------------------------------------------------------------
  // Start negotiation for the data port
  //
  // Setup the data port first.
  //
  try {  Com2.init(DataPort);}
  catch(ErrorObj x) 
    { 
      x << x.what() << " Port no. " 
	<< Com2.port() << endl;exit(-1);
    }
  //
  // Announce the success
  //
  cerr << "schild: Listening on port " << Com2.port() << " for data." << endl;
  //
  // Accept the connection request and attach a 
  // callback function for further conversation.
  //
  Com2.xcept(DataSock);

  //  gdk_input_add(DataIO.descriptor(),
  //		GDK_INPUT_READ,
  //		hub_data_io_callback,
  //		(gpointer)this);
  Com1.over();Com2.over();

  //  signal(SIGPIPE,handler);
  //  n=CmdSock.descriptor();  fcntl(n,F_SETOWN);
  //  n=DataSock.descriptor(); fcntl(n,F_SETOWN);
  //  atexit(on_exit_handler);
  return Big_Endian;
}

int nDock(char *Host, SockIO &CmdS, SockIO &DataS)
{
  SockIO S0;
  int cmdport,dataport;
  char Msg[128],*Disp;
  int n;
  short int Big_Endian;

  MachineSpecType Mac;

  GetMachineSpec(&Mac);
  Big_Endian = Mac.big_endian;

  //
  // Open the "well known" port.
  //
  S0.init(1858,0,Host);
  S0.konnekt();

  n=S0.Send(GREETINGS,strlen(GREETINGS));
  Disp=getenv("DISPLAY");
  n=S0.Send(Disp,strlen(Disp));
  //
  // Get the command port and dock on it
  //
  n=S0.Receive(Msg,128); Msg[n]='\0';
  //  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&cmdport);
  fprintf(stderr,"\nnclient: Got command port no. %d\n",cmdport);

  CmdS.init(cmdport,0,Host);
  CmdS.konnekt();

  //
  // Send the first command....announce the endian type of your 
  // client
  //
  sprintf(Msg,"Big_Endian %d%c",Big_Endian,'\0');
  n=CmdS.Send(Msg,strlen(Msg));

  cerr << "nclient: Connected for commands on port " << cmdport << endl;
  sleep(1);
  /*----------------------------------------------------------------*/
  //
  // Get the data port and dock on it
  //
  n=S0.Receive(Msg,128); Msg[n]='\0';
  //  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&dataport);
  fprintf(stderr,"\nnclient: Got data port no. %d\n",dataport);

  DataS.init(dataport,0,Host);
  DataS.konnekt();

  sleep(1);
  cerr << "nclient: Connected for data on port " << dataport << endl;

  S0.shutup(2);
  return 1;
}  

