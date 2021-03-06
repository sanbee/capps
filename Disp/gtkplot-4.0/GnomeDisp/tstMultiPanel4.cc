#include <iostream.h>
#include <MultiPanel.h>
#include <ExitObj.h>
#include <SockIO.h>
#include <Hub.h>
#include "./WidgetLib.h"
#include "./mopp.h"
#include <math.h>
#include <mp.h>
#include "./interface.h"
#define NPoints  100
void
gtk_color_selection_hsv_to_rgb (double  h, double  s, double  v,
				double *r, double *g, double *b);
void
gtk_color_selection_rgb_to_hsv (double  r, double  g, double  b,
				double *h, double *s, double *v);
main(int argc, char **argv)
{
  try
    {
      gdouble X[NPoints], Y[NPoints];
      double rgb[3]={0.4,0.4,0.5},hsv[3];
      //      GtkPlotPoint Data[NPoints];
      float Range[2];
      MultiPanel OnDisp(argc, argv,"Plot Tool");
      Hub CommHQ(&OnDisp);
      int N;

      //
      // Setup the communication HQ!  Well...basically open a socket, listen
      // on it for incomming requests for docking, and set up the callback
      // function for the same.
      //
      //      CommHQ.init(1858,1859);
      //
      // Read no. of panels to make from the commandline
      //
      sscanf(argv[1], "%d",&N);

      //
      // Initialize the the MultiPanel object to have 
      // N panels with NPoints
      //
      OnDisp.Init(N,NPoints);//GridPacker);
      for (int i=0;i<N;i++)	  OnDisp[i].SetNoOfOverlays(2);
      //
      // Make the form and display it on the screen
      //  
      OnDisp.MakeWindow(MultiPanel::ALLPANELS,35.0,10.0,700,100);
      
      //
      // Test data
      //
      for (int j=0;j<N;j++)
	{
	for (int i=0;i<NPoints;i++)
	  {
	    X[0]=i;
	    Y[0]=(j+1)*sin(2*3.1414*X[0]*10/NPoints);
	    OnDisp[0].PutXData(X[0],i);
	    OnDisp[j].PutYData(Y[0],i);
	    OnDisp[j].PutYData(5*Y[0],i,1);
	  }
	//	OnDisp[j].prtdata();
	}
      Range[0] = Range[1] = 0.0;
      /*
      for (int i=0;i<N;i++)
      	{
      	  OnDisp[i].AddYData(Y,NPoints,0);
	  OnDisp[i].AddYData(Y,57,0);
      	  OnDisp[i].AddXData(X,NPoints,0);
      	}
      */
      gtk_plot_axis_labels_set_numbers(GTK_PLOT(OnDisp[0].GetObj()),
				       GTK_PLOT_AXIS_LEFT,
				       GTK_PLOT_LABEL_EXP,
				       0);
      //
      // Set the XTICS attributes for all panels
      //
      for (int i=0;i<N;i++)
	{
	  vector<char *> ColorList;
	  int n;
	  double bg_rgb[3]={0.4,0.4,0.5};
	  GdkColor color;

	  n=MakeColorList(ColorList);

	  OnDisp[i].SetAttribute(XYPanel::XTICS0,20.0);   // Major ticks
	  OnDisp[i].SetAttribute(XYPanel::XTICS1,10.0);  // Minor ticks
	  OnDisp[i].SetAttribute(XYPanel::YTICS0,0.2); // Major Ticks
	  OnDisp[i].SetAttribute(XYPanel::YTICS1,0.1); // Minor ticks
	  OnDisp[i].SetAttribute(XYPanel::XTITLE,0);
	  OnDisp[i].SetAttribute(XYPanel::XLABEL,0);
	  //	  OnDisp[i].SetAttribute(XYPanel::GRAPH_BG_COLOUR,rgb);
	  cerr << "RGB_BG[" << i << "]: " << rgb[0] << " " << rgb[1] << " " << rgb[2]
	       << endl;
	  gtk_color_selection_rgb_to_hsv(rgb[0],rgb[1],rgb[2],
					 &hsv[0],&hsv[1],&hsv[2]);

	  //	  cerr <<"HVS_BG: " << hsv[0] << " " << hsv[1] << " " << hsv[2] << endl;

	  //
	  // Find the HSV with highest contrast.
	  //
	  hsv[0] = ((int)hsv[0] + 180)%360;
	  hsv[1] = hsv[1] + 0.5;
	  hsv[2] = hsv[2] + 0.5;
	  if (hsv[1] > 1.0) hsv[1] = hsv[1] - 1.0;
	  if (hsv[2] > 1.0) hsv[2] = hsv[2] - 1.0;
	  //	  cerr <<"HSV_FG: " << hsv[0] << " " << hsv[1] << " " << hsv[2] << endl;
	  gtk_color_selection_hsv_to_rgb(hsv[0],hsv[1],hsv[2],
					 &rgb[0],&rgb[1],&rgb[2]);

	  cerr << "RGB_FG[" << i << "]: " << rgb[0] << " " << rgb[1] << " " << rgb[2]
	       << endl;
	  cerr << ColorList[i] << endl;
	  gdk_color_parse(ColorList[i], &color);
	  //	  OnDisp[i].SetAttribute(XYPanel::GRAPH_FG_COLOUR,rgb,0);
	  rgb[0]=color.red;
	  rgb[1]=color.green;
	  rgb[2]=color.blue;
	  OnDisp[i].SetAttribute(XYPanel::GRAPH_FG_COLOUR,ColorList[i],NULL,-1);

	  //
	  // Change the HSV to slightly different.  This will be
	  // the BG for the next panel.
	  //
	  gtk_color_selection_rgb_to_hsv(rgb[0],rgb[1],rgb[2],
					 &hsv[0],&hsv[1],&hsv[2]);
	  hsv[0] = ((int)hsv[0] + (180/N))%360;
	  hsv[1] = hsv[1] + (0.5);
	  hsv[2] = hsv[2] + (0.5);
	  if (hsv[1] > 1.0) hsv[1] = hsv[1] - 1.0;
	  if (hsv[2] > 1.0) hsv[2] = hsv[2] - 1.0;
	  gtk_color_selection_hsv_to_rgb(hsv[0],hsv[1],hsv[2],
					 &rgb[0],&rgb[1],&rgb[2]);
 	}

      OnDisp.GetRange(Range,0,0);
      OnDisp.SetRange(Range,0);
      cerr << Range[0] << " " << Range[1] << endl;

      OnDisp.GetRange(Range,1,0);
      OnDisp.SetRange(Range,1);
      cerr << Range[0] << " " << Range[1] << endl;

      //      OnDisp.Redraw(0);
      //      OnDisp.Redraw(1);
      //      C=OnDisp[0].EraseOverlay(1);
      for (int i=0;i<N;i++)
	{
	  //	  OnDisp[i].PutText("Test String",0.5,0.5,0,0);
	  OnDisp[i].SetAttribute(XYPanel::DATALEGEND,1,0);
	  OnDisp[i].SetAttribute(XYPanel::DATALEGEND,"Sin",NULL,0);

	}
      //      OnDisp.Redraw();
      for (int i=0;i<N;i++)
	{
	  //	  OnDisp[i].DeleteText("Test String");
	}
      OnDisp.Redraw();
      //
      // Start the interactive loop
      //
      gtk_main();
      
    }
  catch (ErrorObj x) {};
}
