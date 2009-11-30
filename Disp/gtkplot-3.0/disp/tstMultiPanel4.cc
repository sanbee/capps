#include <iostream.h>
#include <MultiPanel.h>
#include <ExitObj.h>
#include <SockIO.h>
#include <Hub.h>
#include "./WidgetLib.h"

extern "C" {
  void GridPacker(int n, int np, gfloat *cw, gfloat *ch, 
		  gfloat *pw, gfloat *ph, gfloat *px, gfloat *py)
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
      gdouble X[NPoints], Y[NPoints];
      //      GtkPlotPoint Data[NPoints];
      float Range[2];
      MultiPanel OnDisp(argc, argv,"Plot Tool");
      Hub CommHQ(&OnDisp);
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
      OnDisp.Init(N,NPoints);//GridPacker);
      for (int i=0;i<N;i++)	  OnDisp[i].SetNoOfOverlays(1);
      //
      // Make the form and display it on the screen
      //  
      //      OnDisp.MakeWindow(MultiPanel::ALLPANELS,0,0,970,150);
      OnDisp.MakeWindow(MultiPanel::ALLPANELS,0.5,0.5,700,100);
      //
      // Send the x and y co-oridnates to individual panels and display
      // the graph
      //  
      for (int i=0;i<NPoints;i++)
	{
	  X[i] = i;
	  Y[i] = cos(2*3.1415*i/10);
	}
      
      for (int i=0;i<N;i++)
	OnDisp[i].AddYData(Data,NPoints,0);
      
      gtk_plot_axis_labels_set_numbers(GTK_PLOT(OnDisp[0].GetObj()),
				       GTK_PLOT_AXIS_LEFT,
				       GTK_PLOT_LABEL_EXP,
				       0);
      //
      // Set the XTICS attributes for all panels
      //
      for (int i=0;i<N;i++)
	{
	  OnDisp[i].SetAttribute(XYPanel::XTICS0,20.0);   // Major ticks
	  OnDisp[i].SetAttribute(XYPanel::XTICS1,10.0);  // Minor ticks
	  OnDisp[i].SetAttribute(XYPanel::YTICS0,0.2); // Major Ticks
	  OnDisp[i].SetAttribute(XYPanel::YTICS1,0.1); // Minor ticks
	  OnDisp[i].SetAttribute(XYPanel::XTITLE,0);
	  OnDisp[i].SetAttribute(XYPanel::XLABEL,0);
	}

      OnDisp.GetRange(Range,0,0);
      OnDisp.SetRange(Range,0);

      OnDisp.GetRange(Range,1,0);
      OnDisp.SetRange(Range,1);
      cerr << Range[0] << " " << Range[1] << endl;

      OnDisp.Redraw(0);
      //      OnDisp.Redraw(1);
      C=OnDisp[0].EraseOverlay(1);
      OnDisp.Redraw();

      //
      // Start the interactive loop
      //
      gtk_main();
      
    }
  catch (ErrorObj x) {};
}
