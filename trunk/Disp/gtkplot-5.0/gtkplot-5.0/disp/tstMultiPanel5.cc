#include <iostream>
#include <MultiPanel.h>
#include <ExitObj.h>
#include <SockIO.h>
#include <Hub.h>
#include "./WidgetLib.h"
#include "./mopp.h"
#include <math.h>
#include <mp.h>
#include "./interface.h"
#include <cl.h>
#include <clinteract.h>
#include <readline/readline.h>
using namespace std;
int NPoints=100;
void
gtk_color_selection_hsv_to_rgb (double  h, double  s, double  v,
				double *r, double *g, double *b);
void
gtk_color_selection_rgb_to_hsv (double  r, double  g, double  b,
				double *h, double *s, double *v);


MultiPanel OnDisp;
void clreader(char *buf)
{
  if (buf && strlen(buf))
    cerr << "This is what I got: " << buf << endl;
  if (buf==string("redraw"))
    {
      cerr << "Redrawing.." << endl;
      OnDisp.Redraw();
    }
  if (buf==string("print"))
    OnDisp.print(string("tst.ps"));
  if (buf==string("quit"))
    exit(0);
}
void reader(gpointer data, gint source, GdkInputCondition cond)
{
    rl_callback_read_char() ;
}
void UI(bool restart,int &argc, char **argv, int &npanels, int &npoints)
{
  if (!restart)
    {
      BeginCL(argc,argv);
      char TBuf[FILENAME_MAX];
      clgetConfigFile(TBuf,argv[0]);strcat(TBuf,".config");
      clloadConfig(TBuf);
      clInteractive(0);
    }
  else
   clRetry();
  try
    {
      int i;
      i=1;clgetIValp("npanels",npanels,i);
      i=1;clgetIValp("npoints",npoints,i);
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}

int main(int argc, char **argv)
{
  int N=5;
  UI(false, argc, argv, N, NPoints);

  try
    {
      gdouble X[NPoints], Y[NPoints];
      double rgb[3]={0.4,0.4,0.5},hsv[3];
      float Range[2];
      OnDisp.SetUp(argc, argv, "Plot Tool");
      Hub CommHQ(&OnDisp);

      //
      // Set the number of panels and the number of points in each
      // panel.  A predefined packer is used to put all panels
      // vertically.  A different packer can be set if required.
      //
      // Number of points per panel is also defined.  This might need
      // to be different from each panel.
      //
      //
      OnDisp.Init(N,NPoints);
      //
      // Set the number of overlays per panel.
      // This and NPoints together allocats internal buffers per panel.
      //
      for (int i=0;i<N;i++)	  OnDisp[i].SetNoOfOverlays(2);
      //
      // Make the form and display it on the screen.  This is the top
      // level window.
      //  
      OnDisp.MakeWindow(MultiPanel::ALLPANELS,35.0,10.0,1000,100);
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
      //
      // Such calls must migerate into the MultiPanel object to keep
      // the top-level interface Widegt library independent.
      //
      //
      gtk_plot_axis_labels_set_numbers(GTK_PLOT(OnDisp[0].GetObj()),
				       GTK_PLOT_AXIS_LEFT,
				       GTK_PLOT_LABEL_EXP,
				       0);
      //
      // Set the XTICS attributes for all panels
      //
      for (int i=0;i<N;i++)
	{
	  int n;
	  vector<char *> ColorList;
	  //
	  // Get the use-defined list of overlay colours.  
	  //
	  n=MakeColorList(ColorList);

	  OnDisp[i].SetAttribute(XYPanel::XTICS0,20.0);  // Major ticks
	  OnDisp[i].SetAttribute(XYPanel::XTICS1,10.0);  // Minor ticks
	  OnDisp[i].SetAttribute(XYPanel::YTICS0,0.2);   // Major Ticks
	  OnDisp[i].SetAttribute(XYPanel::YTICS1,0.1);   // Minor ticks
	  //
	  // No x-labels except for the last panel.
	  //
	  OnDisp[i].SetAttribute(XYPanel::XTITLE,0);
	  OnDisp[N-1].SetAttribute(XYPanel::XTITLE,1);
	  if (i<N-1) OnDisp[i].SetAttribute(XYPanel::XLABEL,0);
	  //
	  // Cycle through the list of colours
	  //
	  if (n>0)
	    {
	      int k=i;
	      k= (i+1)%ColorList.size();
	      OnDisp[i].SetAttribute(XYPanel::GRAPH_FG_COLOUR,ColorList[k],NULL,-1);
	    }
 	}

      OnDisp.GetRange(Range,0,0);      OnDisp.SetRange(Range,0);
      OnDisp.GetRange(Range,1,0);      OnDisp.SetRange(Range,1);

      int C=OnDisp[0].EraseOverlay(1);
      OnDisp.Redraw(0);      OnDisp.Redraw(1);
      for (int i=0;i<N;i++)
	{
	  int angle=0;
	  //	  OnDisp[i].PutText("Test String",0.5,0.5,0,angle); // This works
	  OnDisp[i].SetAttribute(XYPanel::DATALEGEND,1,0);
	  OnDisp[i].SetAttribute(XYPanel::DATALEGEND,"Sin",NULL,0);

	}
      // for (int i=0;i<N;i++)
      // 	{ 
      // 	  OnDisp[i].DeleteText("Test String"); // This does not work with GTKPlot 5.0
      // 	}
      OnDisp.Redraw();
      //
      // Start the interactive loop.  This keeps the window active as
      // well as give interactive control.  
      //
      // The stdin input stream is added as one of the input stream to
      // the GTK loop (gtk_loop).  The callback for this stream is
      // reader().  Which in-turn calls clreader() (via a bit more
      // complicated callback meachims of ReadLine lib. so that even
      // commandline editing remains available).
      //
      char *prompt="tst>";
      rl_callback_handler_install (prompt,clreader);
      gtk_input_add_full(fileno(stdin), GDK_INPUT_READ, reader,
			 NULL, NULL, NULL);
      gtk_main();
    }
  catch (ErrorObj x) {};
}
