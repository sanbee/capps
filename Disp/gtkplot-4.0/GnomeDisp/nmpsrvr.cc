/*
    History record:
 
      ????,1999       Dark ages vesion using XForms library.
                                            S.Bhantagar
      ????,2000       Switched to Gtk+.
                                            S.Bhatnagar
      Feb, 2000       Made is server which forkes the plotting 
                      server.  Hence this can handle any number of 
                      in-comming requests and the actual plotting
                      services are provided by the the plotting
                      server (the CLIENTNAME below).

                      With this design, this server has no dependence
                      on the plotting library/widget library being
                      used.
                                            S.Bhatnagar
*/
#include <SockIO.h>
#include <Protocol.h>
#include <MultiPanel.h>
#include <vector.h>
#include <exception>
#include <ErrorObj.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <strstream>
#include <fstream>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

#define CLIENTNAME "mppsrvr"

fd_set fdset;
int CtrlPort=1858;
SockIO Com1,Com2;
SockIO CtrlIO, ChildIO;

typedef struct ServerInfo {
  int CmdPort, DataPort, Key;
} ServerInfo;
//
//----------------------------------------------------------------
// A warpper for select() system call.  The EINTR return status 
// (system call interrupted) is trapped and select simply called
// again.  This can happen since if a child proccess exits while
// the server is waiting on select() (which is where this server
// spends most of it's time).
//
int  Select(int  n,  fd_set  *readfds,  fd_set  *writefds,
       fd_set *exceptfds, struct timeval *timeout)
{
  int m;
  do
    {
      m=select(n, readfds, writefds, exceptfds, timeout);
    }
  while((m<0) && (errno==EINTR));
  return m;
}
//
//----------------------------------------------------------------
// Hunt for free ports which the forked process can open and 
// setup the command and data communication channels with the
// client.
//
int GetFreePorts(int& P0, int& P1)
{
  int Found=0;
  SockIO tmp1;
  P0=6001; P1=P0+1;
  while (!Found && P0<=10000)
    {
      cerr << "Trying port " << P0 << "..." << endl;
      tmp1.shut();
      try
	{
	  tmp1.init(P0);
	  Found=1;
	}
      catch (ErrorObj& x)
	{
	  cerr << x.what() << endl;
	  cerr << P0 << endl;
	  Found=0;
	  P0++;
	}
    }
  cerr << "done.  P0="<<P0<<endl;
  tmp1.shut();
  if (!Found) P0=-1;
  Found=0;
  P1=P0+1;
  while (!Found && P1<=10000)
    {
      cerr << "Trying port " << P1 << "..." << endl;
      try
	{
	  tmp1.init(P1);
	  Found=1;
	  
	}
      catch (ErrorObj x)
	{
	  cerr << x.what() << endl;
	  cerr << P1 << endl;
	  Found=0;
	  P1++;
	}
    }
  if (!Found) P1=-1;
  cerr << "done. P1="<<P1<<endl;
  return ((P0!=-1) && (P1!=-1));
}
//
//----------------------------------------------------------------
// Signal handler
//
void handler(int sig)
{
  int Status,PID;
  switch (sig)
    {
    case SIGPIPE:
      cerr << "Socket shutdown" << endl;
      Com1.shut(); 
      break;
    case SIGCHLD:
      cerr << "Child exited" << endl;
      PID=wait(&Status);
      cerr << "PID=" << PID << " Status=" << Status << endl;
      break;
    }
}
//
// The hub for starting the server and setting up the link between
// the server and the client.
//
// This program always listens on a socket for incomming requests to
// start a copy of the server.
//
int main(int argc, char *argv[])
{
  int n;
  int ChildPid;
  string Msg,ServerChild,Display;
  //  map<int, ServerInfo> SI;
  char CmdPortString[16],DataPortString[16];
  int mygetdtablehi();

  if (argc > 1) sscanf(argv[1],"%d",&CtrlPort);
  if (argc == 3) ServerChild=argv[2]; else ServerChild=CLIENTNAME;
  try {  CtrlIO.init(CtrlPort);}
  catch(ErrorObj x) 
    { 
      x << x.what() << "(port no. " << CtrlPort << ")." 
	<< endl
	<< "###Error: " << "Possibly server already running."
	<< endl;
      exit(-1);
    }
  //
  // Announce the success
  //
  cerr << "Listening on port " << CtrlIO.port() << " for docking requests." 
       << endl;
  //
  // Accept the connection request and attach a 
  // callback function for further conversation.
  //
  
  
  while(1)
    {
      int P0,P1;
      FD_ZERO(&fdset);
      FD_SET(CtrlIO.descriptor(),&fdset);

      n=Select(mygetdtablehi(), &fdset, NULL, NULL, NULL);
      if (n<0) {perror("select");exit(-1);}
      if ((n>0) && FD_ISSET(CtrlIO.descriptor(),&fdset))
	{
	  //
	  // Hunt for 2 free ports.  If any of them turn out to be
	  // -1, a free port could not be found.
	  //
	  // The client will be asked to connect on a port no. -1
	  // which (I hope) will fail.  The forked() process will also
	  // be asked to open and listen on port no. -1, it wil immediately
	  // exit.
	  //
	  cerr << "Hunting for free ports..." << endl;
	  GetFreePorts(P0, P1);
	  cerr << endl << "mpsrvr: Accepting socket connection..." <<endl;
	  CtrlIO.xcept(Com1);
	  sleep(1);
	  if (((n=Com1.Receive(Msg)) == -1) ||
	      (strncmp(&Msg[0],GREETINGS,strlen(GREETINGS))))
	    {
	      perror("###Error");
	      Com1.over();
	      throw(ErrorObj("In initial greetings itself!","###Error",
			     ErrorObj::Recoverable));
	    }
	  signal(SIGPIPE,handler);
	  signal(SIGCHLD,handler);
	  fcntl(Com1.descriptor(),F_SETOWN);
	  Com1.Receive(Display);
	  cerr << "DISPLAY="<<Display << endl;
	  //
	  // Fork a process and replace it with the actual server
	  // program.
	  //	  
	  if ((ChildPid=fork()))
	    {
	      //
	      // Parent process.
	      // Assign two port numbers, fill in the table along with
	      // the PID, and write this info /tmp/mp<PID>.  This file
	      // is then read by the child process, and later removed.
	      //
	      ofstream os;
	      char fname[128];
	      cerr << endl << "forked() child pid=" << ChildPid << endl;
	      //	      SI[ChildPid].CmdPort=P0;//7009;
	      //	      SI[ChildPid].DataPort=P1;//SI[ChildPid].CmdPort+1;
	      sprintf(fname,"/tmp/mp%d%c",ChildPid,'\0');
	      os.open(fname);
	      /*
	      os << ServerChild << " " 
		 << SI[ChildPid].CmdPort << " "<<SI[ChildPid].DataPort << endl;
	      */
	      os << ServerChild << " " 
		 << P0 << " "<< P1 << endl;
	      //	      sprintf(CmdPortString,"%d",SI[ChildPid].CmdPort);
	      //	      sprintf(DataPortString,"%d",SI[ChildPid].DataPort);
	      sprintf(CmdPortString,"%d",P0);
	      sprintf(DataPortString,"%d",P1);
	      cerr << CmdPortString << " " << DataPortString << endl;
	    }
	  else
	    {
	      //
	      // Child process.
	      // Open the file /tmp/mp<PID> and read two port numbers
	      // and use them to open the command and data sockets.  The
	      // client will connect to these ports to send commands
	      // and data.  Remove the /tmp/mp<PID> file.
	      //
	      int PID=getpid();
	      char fname[128];
	      ifstream inp;

	      sleep(1);
	      sprintf(fname,"/tmp/mp%d%c",PID,'\0');
	      inp.open(fname);
	      inp >> fname >> CmdPortString >> DataPortString;
	      inp.close();
	      sprintf(fname,"/tmp/mp%d%c",PID,'\0');
	      unlink(fname);
	      {
		string env="DISPLAY=";
		env += Display;
		if (putenv(env.c_str())<0)
		  cerr << "###Error: " << ServerChild 
		       << ": Could not set the DISPLAY env. var." << endl;
	      }
	      execl(ServerChild.c_str(), ServerChild.c_str(), 
		    CmdPortString, DataPortString, Display.c_str(),NULL);
	    }
	  //
	  // Sleep for a while to give the SERVER CHILD to set
	  // itself up...
	  //
	  sleep(2);
	  //
	  // Send the command and data port numbers to the client
	  //
	  Msg=CmdPortString;
	  cerr << endl << "mpsrvr: Sending command port no. to client..." 
	       << Msg << endl;
	  Com1.Send(&Msg[0],Msg.size());
	  cerr << "done" << endl;

	  Msg=DataPortString;
	  cerr << endl << "mpsrvr: Sending data port no. to client..." 
	       << Msg << endl;
	  Com1.Send(&Msg[0],Msg.size());
	  cerr << "done" << endl;
	  Com1.shut();
	}
    }
}
