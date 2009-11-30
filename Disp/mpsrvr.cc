#include <iostream.h>
#include <stdlib.h>
#include <MultiPanel.h>
#include <ExitObj.h>
#include <SockIO.h>
#include <Hub.h>
#include <forms.h>

extern "C" {
  void GridPacker(int n, int np, int *cw, int *ch, 
		  int *pw, int *ph, int *px, int *py)
  {
    int k1=3,k2;
    k2=(int)(np/k1+1);
    *cw = *pw*k1;  
    *ch = *ph*k2;
    *px = (n%k1)* *pw; *py=int((n/k1)* *ph);
  };

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

      CommHQ.reset(&OnDisp);

      on_exit(CleanUp,(void *)&CommHQ);

      //
      // Setup the communication HQ!  Well...basically open a socket, listen
      // on it for incomming requests for docking, and set up the callback
      // function for the same.
      //
      CommHQ.init();

      OnDisp.Init(0,0,GridPacker);
      OnDisp.MakeWindow(MultiPanel::ALLPANELS,0,0,1,1,0);
      
      fl_do_forms();
    }
  catch(ExitObj x) {};
}
