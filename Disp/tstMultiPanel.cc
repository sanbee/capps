#include <iostream.h>
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
}

#define NPoints  100

main(int argc, char **argv)
{
  try
    {
      float X[NPoints],Y[NPoints], Range[2];
      MultiPanel OnDisp(argc, argv,"Plot Tool");
      Hub CommHQ(&OnDisp);
      FL_FORM *Form;
      char str[10];
      int N,C;

      //
      // Setup the communication HQ!  Well...basically open a socket, listen
      // on it for incomming requests for docking, and set up the callback
      // function for the same.
      //
      CommHQ.init();
      //
      // Test data
      //
      for (int i=0;i<NPoints;i++)
	{
	  X[i] = i;
	  Y[i] = sin(2*3.1414*sin(2*3.1415*i/10));
	}
      Range[0] = Range[1] = 0.0;
      //
      // Read no. of panels to make from the keyboard
      //
      sscanf(argv[1], "%d",&N);
      //
      // Initialize the the MultiPanel object to have 
      // N panels with NPoints
      //
      OnDisp.Init(N,NPoints,GridPacker);
      //
      // Set the XTICS attributes for all panels
      //
      for (int i=0;i<N;i++)
	{
	  OnDisp[i].SetAttribute(XYPanel::XTICS0,10);
	  OnDisp[i].SetAttribute(XYPanel::XTICS1,10);
	  //	  OnDisp[i].SetStdLabels("X-Axis",XYPanel::XLABEL);
	  //	  OnDisp[i].SetStdLabels("Y|Axis",XYPanel::YLABEL);
	  OnDisp[i].SetNoOfOverlays(2);
	}
      //
      // Make the form and display it on the screen
      //  
      OnDisp.MakeWindow(MultiPanel::ALLPANELS,0,0,970,150);
      //
      // Send the x and y co-oridnates to individual panels and display
      // the graph
      //  
      for (int i=0;i<N;i++)
	{
	  OnDisp[i].AddXData(X,NPoints);
	  OnDisp[i].AddYData(Y,NPoints,0);
	}
      
      for (int i=0;i<NPoints;i++)
	{
	  X[i] = i;
	  Y[i] = cos(2*3.1415*i/10);
	}
      
      for (int i=0;i<N;i++)
	OnDisp[i].AddYData(Y,NPoints,1);
      
      OnDisp.Redraw(0);
      OnDisp.Redraw(1);
      C=OnDisp[0].EraseOverlay(1);
      OnDisp.Redraw();
      Range[0]=0;Range[1]=0; // Autoscale
      OnDisp[0].SetRange(Range,1);
      //
      // Start the interactive loop
      //
      fl_do_forms();
    }
  catch (ErrorObj x) {};
}
