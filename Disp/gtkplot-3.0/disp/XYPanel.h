// -*- C++ -*-
// $Id$
#if !defined(XYPANEL_H)
#define XYPANEL_H

#include "./WidgetLib.h"
#include <scrollbuf.h>
#include <float.Vec.h>
#include <vector>


extern "C"{
  void xp_ymax_callback(GtkWidget *Ob, long data);
  void xp_ymin_callback(GtkWidget *Ob, long data);
  int PanelPreHandler(GtkWidget *ob, int Event,
		      gint Mx, gint My,
		      int Key, void *RawEvent);
  int xp_SliderPreHandlerDown(GtkWidget *ob, int Event,
			      gint Mx, gint My,
			      int Key, void *RawEvent);
  int xp_SliderPreHandlerUp(GtkWidget *ob, int Event,
			    gint Mx, gint My,
			    int Key, void *RawEvent);
};

class XYPanel {
public:
  XYPanel(GtkWidget *Form=NULL);
  ~XYPanel();
  GtkWidget *Make(GtkWidget *Form, gfloat X0, gfloat Y0, gfloat Width, gfloat Height);
  
  void prtdata()
    {
      int i;
      cerr << NoOfDataPts << " " << NoOfOverlays << endl;
      for (i=0;i<NoOfDataPts;i++)
	{
	  cerr << Yi[0][i].points->x << " "<< Yi[0][i].points->y << endl;
	}
    };
  
  void FreeChart();
  void Init(int NoOfPoints, int NoOfOverlays=1)
    {SetNoOfOverlays(NoOfOverlays);SetNoOfDataPts(NoOfPoints);}
  
  void Clear(int WhichOverlay=ALLOVERLAYS);
  void Plot(int WhichOverlay=ALLOVERLAYS);
  
  void GetRange(float XRange[2], int Axis);
  void SetRange(float Range[2], int Axis);
  
  int GetNoOfOverlays() {return NoOfOverlays;};
  int SetNoOfOverlays(int N);
  int EraseOverlay(int N) 
    {int C=Colour[N];SetAttribute(OVERLAY_COLOR,GraphBGColour,N);
    return C;}
  int UnEraseOverlay(int N,gint Col) 
    {int C=Colour[N];SetAttribute(OVERLAY_COLOR,Col,N);return C;}
  
  int SetNoOfDataPts(int NData);
  int SetAttribute(int WhichAttr, gdouble Val, int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, gint Val, int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, GdkColor &Colour, int WhichOverlay=ALLOVERLAYS)
    {return 1;};
  int SetAttribute(int WhichAttr, char *Val0, char *Val1=NULL,
		   int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, int scale, double base);
  int GetAttribute(int WhichAttr, short int WhichOverlay=ALLOVERLAYS);
  
  void PutText(char* Text, double x, double y, 
	       gint Col,
	       int Align=0);
  void DeleteText(const char* Text);
  
  float GetMax(int n=ALLOVERLAYS);
  float GetMin(int n=ALLOVERLAYS);
  
  int PreHandler(GtkWidget *ob, int Event,
		 gint Mx, gint My,
		 int Key, void *RawEvent);
  
  GtkWidget *XYPanel::AddText(int X0, int Y0, 
			      int Width, int Height, 
			      char *Str,
			      gint Colour,
			      int TextType=0);
  
  void Redraw(short int WhichOverlay=-1){};
  void Show(){if (CHART) gtk_widget_show(CHART);}
  GtkWidget *GetObj(int Enum=0){return CHART;};
  
  void AddXData(float *X, int Size);
  void AddYData(gdouble *Y, int Size, int OverLay=0);
  void PutXData(float *Xp, int Where)                  {Yi[0][Where].points->x=*Xp;}
  void PutYData(float *Yp, int Where,int WhichOverlay=0) 
    {Yi[WhichOverlay][Where].points->y=*Yp;}
  
  void YMinCallback(GtkWidget *Ob=NULL, long data=0);
  void YMaxCallback(GtkWidget *Ob=NULL, long data=0);
  int SliderPreHandlerUp(GtkWidget *, int event, 
			 gint , gint,
			 int , void *);
  int SliderPreHandlerDown(GtkWidget *, int event, 
			   gint , gint,
			   int , void *);
  int MinSliderMove(GtkWidget *ob, float fac);
  int MaxSliderMove(GtkWidget *ob, float fac);
  //
  // The attributes names
  //
  enum {ALLOVERLAYS=-1};
  
  enum {
    FRAME_BOX_TYPE, XYPLOT_TYPE, XYPLOT_FONT,
    LINESTYLE, PLOTSTYLE, AXISTYPE, 
    GRAPH_BG_COLOUR, OVERLAY_COLOR, LABEL_COLOR, LABEL_FONT,
    XTICS0,YTICS0,XTICS1,YTICS1,
    X_FIXED_AXIS0, X_FIXED_AXIS1,
    Y_FIXED_AXIS0, Y_FIXED_AXIS1,
    XLABEL,YLABEL,
    XTITLE,YTITLE,TTITLE,
    ALPHAXTICS,ALPHAYTICS,XGRID,YGRID,
    XFIXEDAXIS,YFIXEDAXIS,
    XSCALE,YSCALE
  };
  
  enum {COPY, ADD};
  
private:
  int NoOfDataPts, NoOfOverlays;
  unsigned short int ScrollStep, AxisType,Chosen;
  gdouble XTics0, YTics0, XTics1,YTics1;
  gint XGridType,YGridType;
  char *MajorAlphaXTics, *MinorAlphaXTics;
  char *MajorAlphaYTics, *MinorAlphaYTics;
  gfloat XMin, XMax, YMin, YMax;
  GtkWidget *CHART, *YMax_Slider, *YMin_Slider;
  GdkColor  GraphBGColour,GraphFGColour;
  GtkWidget *Form;
  char *XLabel, *YLabel,*Title;
  
  vector<int> TmpSize;
  vector<GtkWidget *> Label;
  
  vector<unsigned short int> Colour, LineStyle, PlotStyle;
  
  vector< GtkPlotData * > Yi;
  //  float *X;
};


#endif
