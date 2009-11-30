// -*- C++ -*-
// $Id$
#if !defined(XYPANEL_H)
#define XYPANEL_H

#include "./WidgetLib.h"
#include "../gtkplot.h"
#include "../gtkplotlayout.h"
#include "../gtkplotcanvas.h"
#include "../gtkplotps.h"
#include <scrollbuf.h>
#include <float.Vec.h>
#include <vector>
#include <namespace.h>
#include <iostream>

extern "C"{
  void gtk_plot_layout_paint (GtkWidget *widget);
  void xp_ymax_callback(GtkAdjustment* Ob, GtkWidget* data);
  void xp_ymin_callback(GtkAdjustment* Ob, GtkWidget* data);
  int PanelPreHandler(GtkWidget *ob, GdkEvent* Event,
		      gpointer data);
};

class XYPanel {
public:
  XYPanel();
  ~XYPanel();
  GtkWidget *Make(GtkWidget *Kanwas, 
		  gint KanwasWidth, gint KanwasHeight,
		  gint X0, gint Y0, gint Width, gint Height,
		  gint makeYScrollBars);
  
  void prtdata()
    {
      int i;
      cerr << NoOfDataPts << " " << NoOfOverlays << endl;
      for (i=0;i<NoOfDataPts;i++)
	{
	  cerr << Yi[0]->x[i] << " "<< Yi[0]->y[i] << endl;
	}
    };
  
  void MakeStorage(int WhichOverlay);
  void Init(int NoOfPoints, gdouble *X, int NoOfOverlays=1)
    {SetNoOfOverlays(NoOfOverlays);SetNoOfDataPts(NoOfPoints);}
  
  void Clear(int WhichOverlay=ALLOVERLAYS);
  void Plot(int WhichOverlay=ALLOVERLAYS);

  void freeze()   {/*GTK_PLOT_SET_FLAGS(CHART,GTK_PLOT_FREEZE);*/}
  void unfreeze() {/*GTK_PLOT_UNSET_FLAGS(CHART,GTK_PLOT_FREEZE);*/}
  int EraseOverlay(int N)                {return 0;}
  int UnEraseOverlay(int N,gint Col)     {return 0;}

  void GetRange(float* XRange, int Axis, int Mode=RANGEDATA);
  int GetNoOfOverlays()                  {return NoOfOverlays;};
  int GetAttribute(int WhichAttr, short int WhichOverlay=ALLOVERLAYS);
  
  void SetXRangingMode(unsigned short Mode) {XAutoRanging=Mode;}
  void SetYRangingMode(unsigned short Mode) {YAutoRanging=Mode;}
  int SetRange(float Range[2], int Axis);
  int SetNoOfOverlays(int N);
  int SetNoOfDataPts(int NData);
  int SetAttribute(int WhichAttr, gint * Val, int WhichOverlay=ALLOVERLAYS){return 1;};
  int SetAttribute(int WhichAttr, gdouble * Val, int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, gdouble Val, int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, gint Val, int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, GdkColor &Colour, int WhichOverlay=ALLOVERLAYS)
    {return 1;};
  int SetAttribute(int WhichAttr, char *Val0, char *Val1=NULL,
		   int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, gdouble& val0, gdouble& val1);
  float GetMax(int n=ALLOVERLAYS);
  float GetMin(int n=ALLOVERLAYS);
  
  
  void PutText(char* Text, double x, double y, 
	       char *Col=NULL,
	       int Angle=0);
  void DeleteText(const char* Text);
  
  int PreHandler(GtkWidget *ob, GdkEvent* Event,
		 gpointer data);
  
  void Redraw(short int WhichOverlay=-1){};
  void Show(){if (CHART) gtk_widget_show(CHART);}
  GtkWidget *GetObj(int Enum=XYCHART);
  GtkAdjustment* GetSliderAdj(int Enum);
  void ScrollData(gdouble *DataTarget, gdouble *DataSource, 
		  int TargetSize,int SourceSize);
  void AddXData(gdouble *X, int Size, int OverLay = 0);
  void AddYData(gdouble *Y, int Size, int OverLay=0);
  void PutXData(gdouble Xp, int Where);
  void PutYData(gdouble Yp, int Where,int WhichOverlay=0);
  
  void YMinCallback(GtkAdjustment *Ob=NULL, GtkWidget* data=0);
  void YMaxCallback(GtkAdjustment *Ob=NULL, GtkWidget* data=0);
  void AutoSetTicks(gfloat& Min, gfloat& Max,GtkOrientation Axis);
  int MinSliderMove(GtkWidget *ob, float fac);
  int MaxSliderMove(GtkWidget *ob, float fac);
  void print(string& filename, gint paper=GTK_PLOT_LETTER, gint orientation=GTK_PLOT_PORTRAIT,
	     gint epsflags=0)
  {
    gtk_plot_export_ps(GTK_PLOT(CHART), (char *)filename.c_str(),
		       orientation, epsflags, paper);

  }
  //
  // The attributes names
  //
  enum {ALLOVERLAYS=-1};
  
  enum 
  {
    FRAME_BOX_TYPE, XYPLOT_TYPE, XYPLOT_FONT,
    LINESTYLE, PLOTSTYLE, AXISTYPE, 
    GRAPH_BG_COLOUR, GRAPH_FG_COLOUR, OVERLAY_COLOR, LABEL_COLOR, LABEL_FONT,
    XTICS0,YTICS0,XTICS1,YTICS1,
    XLABEL,YLABEL,
    XTITLE,YTITLE,TTITLE,
    DATALEGEND,
    XGRID,YGRID,
    XSCALE,YSCALE,
    XRANGE,YRANGE,

    XYCHART,YMINSLIDER, YMAXSLIDER,
    YMINSLIDERADJ,YMAXSLIDERADJ,
    
    RANGEDATA, RANGESLIDER  /* Read the range from data or the slider*/

  };
  
private:
  int NoOfDataPts, NoOfOverlays;
  unsigned short int ScrollStep, AxisType,Chosen, XAutoRanging, YAutoRanging;
  gdouble XTics0, YTics0, XTics1,YTics1;
  gint XGridType,YGridType;
  gfloat XMin, XMax, YMin, YMax;
  gfloat YDataMin, YDataMax;
  GtkWidget *CHART, *YMax_Slider, *YMin_Slider, *Canvas;
  GtkAdjustment *YMin_Adj, *YMax_Adj;
  
  vector< GtkPlotData * > Yi;

  static gdouble *X;
  static gint Id;
  static gfloat XDataMin, XDataMax;
  // gdouble *X;
  // gint Id;
  // gfloat XDataMin, XDataMax;
};

#endif
