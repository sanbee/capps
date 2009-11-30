#include <iostream.h>
#include <stdlib.h>
#include <MultiPanel.h>
#include <ExitObj.h>
#include <SockIO.h>
#include <Hub.h>
#include <mopp.h>

extern "C" {
void CleanUp(int status, void *data)
{
  ((Hub *)data)->cleanup();
}

};

main(int argc, char **argv)
{
  try
    {
      MultiPanel OnDisp(argc, argv,"Plot Tool");
      Hub CommHQ;


      gtk_rc_add_default_file("tst.rc");

      CommHQ.reset(&OnDisp);


      //
      // Setup the communication HQ!  Well...basically open a socket,
      // listen on it for incomming requests for docking, and set up
      // the callback function for the same.
      //
      CommHQ.init();

      OnDisp.Init(0,0/*,GridPacker*/);
      //      OnDisp.MakeWindow(MultiPanel::ALLPANELS,0,0,1,1,0);
      
      gtk_main();

    }
  catch(ExitObj x) {};
}
