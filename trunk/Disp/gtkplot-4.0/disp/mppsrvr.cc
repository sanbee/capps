/*
    History record:

       ????1999       Dark ages version using XFORMS
                                                S. Bhatnagar

       ????2000       The XForms team did not release the new
                      version for IRIX which had the facility for
                      scrolling window.  

                      Finally I hunted for a Free widget library
                      and concluded that I must use the Gtk+ library.  
                      It turned out to be a wise decision.
                                                S. Bhatnagar

       Feb,2000       Changed the design a little bit.  This
                      program is now started by nmpsrvr.  nmpsrvr
                      itself is now a light weight server of inetd
                      kind and hence any number of users can use
                      it.  Earlier, it could be used by only one user
                      and making it handle multiple requests would
                      make it very complex.  Now, each request results
                      into a new mppsrvr (this program) being initiated
                      which provides the plotting services.
                                                S.Bhatnagar

      April,2000      Added colors for individual panels.  Gave up the
                      idea of automatically detemining the colours for
                      the panels which would give good contrast with the
                      background as well as with respect to the neighboring
                      panels.  

                      Colours are now detemined by a simpler system.  File
                      given by the env. variable MPCOLOURS is opened and a
                      list of X recognized colour names is read.  This list
                      of colours is then used for the panels.  The list is 
                      circulated if the number of colours defined is less
                      than the number of panels.  
                                                S.Bhatnagar
 */
#include <iostream.h>
#include <stdlib.h>
#include <MultiPanel.h>
#include <ExitObj.h>
#include <SockIO.h>
#include <Hub.h>
#include <mopp.h>
#include <fstream>
#include <signal.h>
//
//----------------------------------------------------------------
// The actual line plot server.  This is started by the mpserver
// and handles the in-comming commands and data on two sockets.
//

Hub CommHQ;

//
//----------------------------------------------------------------
// Signal handler
//
void handler(int sig)
{
  switch (sig)
    {
    case SIGPIPE:// Broken pipe: write to pipe with no readers
      cerr << "Socket shutdown on SIGPIPE" << endl;
      CommHQ.cleanup();
      break;
    case SIGHUP:// Hangup detected on controlling terminal
                // or death of controlling process
      cerr << "Socket shutdown on SIGHUP" << endl;
      CommHQ.cleanup();
      break;
    case SIGINT:// Interrupt from keyboard
      cerr << "Socket shutdown on SIGINT" << endl;
      CommHQ.cleanup();
      break;
    case SIGQUIT:// Quit from keyboard
      cerr << "Socket shutdown on SIGQUIT" << endl;
      CommHQ.cleanup();
      break;
    case SIGILL://  Illegal Instruction
      cerr << "Socket shutdown on SIGILL" << endl;
      CommHQ.cleanup();
      break;
    case SIGABRT:// Abort signal from abort(3)
      cerr << "Socket shutdown on SIGABRT" << endl;
      CommHQ.cleanup();
      break;
    case SIGFPE://  Floating point exception
      cerr << "Socket shutdown on SIGFPE" << endl;
      CommHQ.cleanup();
      break;
    case SIGSEGV:// Invalid memory reference
      cerr << "Socket shutdown on SIGSEGV" << endl;
      CommHQ.cleanup();
      break;
    case SIGALRM:// Timer signal from alarm(2)
      cerr << "Socket shutdown on SIGALRM" << endl;
      CommHQ.cleanup();
      break;
    case SIGTERM:// Termination signal
      cerr << "Socket shutdown on SIGTERM" << endl;
      CommHQ.cleanup();
      break;
    case SIGTSTP:// Stop typed at tty
      cerr << "Socket shutdown on SIGTSTP" << endl;
      CommHQ.cleanup();
      break;
    }
}
//
//----------------------------------------------------------------
//
int main(int argc, char **argv)
{

  int CmdPort, DataPort;
  string WindowLabel="Plot Tool";

  if (argc > 2)
    {
      sscanf(argv[1],"%d",&CmdPort);
      sscanf(argv[2],"%d",&DataPort);
      cerr << "mppsrvr: DISPLAY=" << argv[3] << endl;
      if (argc > 4) {cerr << argv[4] << endl;WindowLabel=argv[4];}
    }
  else
    {
      int PID=getpid();
      char fname[128];
      ifstream inp;
      sprintf(fname,"/tmp/mp%d%c",PID,'\0');
      inp.open(fname);
      inp >> fname >> CmdPort >> DataPort;
      cout << "mpps: " << fname << " " << CmdPort << DataPort << endl;
    }

  try
    {
      MultiPanel OnDisp(argc, argv,(char *)WindowLabel.c_str());


      if (CmdPort<0 || DataPort<0) 
	{
	  exit(-1);
	}

      gtk_rc_add_default_file("tst.rc");

      CommHQ.reset(&OnDisp);
      //
      // Setup the communication HQ!  Well...basically open a socket, listen
      // on it for incomming requests for docking, and set up the callback
      // function for the same.
      //
      CommHQ.init(CmdPort, DataPort);

      signal(SIGPIPE,handler);
      signal(SIGHUP,handler);
      signal(SIGINT,handler);
      signal(SIGQUIT,handler);
      signal(SIGILL,handler);
      signal(SIGABRT,handler);
      signal(SIGFPE,handler);
      signal(SIGSEGV,handler);
      signal(SIGALRM,handler);
      signal(SIGTSTP,handler);

      OnDisp.Init(0,0/*,GridPacker*/);
      //      OnDisp.MakeWindow(MultiPanel::ALLPANELS,0,0,1,1,0);
      
      CommHQ.Start();
      //      gtk_main();

    }
  catch(ExitObj x) {};
}


