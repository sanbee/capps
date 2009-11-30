#include <stdlib.h>
#include <SockIO.h>
#include <Protocol.h>
#include <MultiPanel.h>
#include <vector.h>
#include <exception>
#include <ErrorObj.h>
#include <fcntl.h>
#include <sys/stat.h>

int wait_for_activity(vector<int>&);
SockIO CmdIO, DataIO;
int Big_Endian;
//
//----------------------------------------------------------------
//
void on_exit_handler(void)
{
  int PID=getpid();
  cerr << "Got exit signal " << PID << endl;
  exit(0);
}
//
//----------------------------------------------------------------
//
void ShutDown()
{
  cerr << "Socket shutdown" << endl;

  CmdIO.shut(); 
  DataIO.shut(); 
  exit(-1);
} 
//
//----------------------------------------------------------------
//
void handler(int sig)
{
  switch (sig)
    {
    case SIGPIPE:
      ShutDown();
      break;
    }
}

void CommH(int CmdPort, int DataPort,SockIO& CmdSock, SockIO& DataSock)
{
  string Msg;
  SockIO Com1,Com2;
  int n;

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
  if (n<0) cerr << "wait failed"<< endl;
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

  signal(SIGPIPE,handler);
  n=CmdSock.descriptor();  fcntl(n,F_SETOWN);
  n=DataSock.descriptor(); fcntl(n,F_SETOWN);
  atexit(on_exit_handler);
}

main(int argc, char *argv[])
{
  int n, CmdPort, DataPort;
  string Msg;

  sscanf(argv[1],"%d",&CmdPort);
  sscanf(argv[2],"%d",&DataPort);
  
  CommH(CmdPort,DataPort,CmdIO,DataIO);
  CmdIO.Receive(Msg);
}

  
