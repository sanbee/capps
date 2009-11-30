// -*- C++ -*-
// $Id$
#if !defined(XYPANEL_H)
#define XYPANEL_H

extern "C" {
#include <forms.h>
}
#include <scrollbuf.h>
#include <float.Vec.h>
#include <vector>


extern "C"{
  void xp_ymax_callback(FL_OBJECT *Ob, long data);
  void xp_ymin_callback(FL_OBJECT *Ob, long data);
  int PanelPreHandler(FL_OBJECT *ob, int Event,
		      FL_Coord Mx, FL_Coord My,
		      int Key, void *RawEvent);
  int xp_SliderPreHandlerDown(FL_OBJECT *ob, int Event,
			      FL_Coord Mx, FL_Coord My,
			      int Key, void *RawEvent);
  int xp_SliderPreHandlerUp(FL_OBJECT *ob, int Event,
			    FL_Coord Mx, FL_Coord My,
			    int Key, void *RawEvent);
};

class XYPanel {
public:
  XYPanel(FL_FORM *Form=NULL);
  ~XYPanel();
  FL_OBJECT *Make(FL_FORM *Form,int X0,int Y0,int Width, int Height);
  
  void prtdata()
    {
      int i,j;
      cerr << NoOfDataPts << " " << NoOfOverlays << endl;
      for (i=0;i<NoOfDataPts;i++)
	{
	  cerr << X[i] << " "<< Yi[0][i] << endl;
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
  int UnEraseOverlay(int N,FL_COLOR Col) 
    {int C=Colour[N];SetAttribute(OVERLAY_COLOR,Col,N);return C;}
  
  int SetNoOfDataPts(int NData);
  int SetAttribute(int WhichAttr, int Val, int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, char *Val0, char *Val1=NULL,
		   int WhichOverlay=ALLOVERLAYS);
  int SetAttribute(int WhichAttr, int scale, double base);
  int GetAttribute(int WhichAttr, short int WhichOverlay=ALLOVERLAYS);
  
  void PutText(const char* Text, double x, double y, 
	       FL_COLOR Col,
	       int Align=FL_ALIGN_RIGHT);
  void DeleteText(const char* Text);
  
  float GetMax(int n=ALLOVERLAYS);
  float GetMin(int n=ALLOVERLAYS);
  
  //  void SetObjCallback(int ObjEnum, FL_CALLBACKPTR CallBack){};
  int PreHandler(FL_OBJECT *ob, int Event,
		 FL_Coord Mx, FL_Coord My,
		 int Key, void *RawEvent);
  
  FL_OBJECT *XYPanel::AddText(int X0, int Y0, 
			      int Width, int Height, 
			      char *Str,
			      FL_COLOR Colour,
			      int TextType=FL_NORMAL_TEXT);
  
  void Redraw(short int WhichOverlay=-1){};
  
  FL_OBJECT *GetObj(int Enum){};
  
  void AddXData(float *X, int Size);
  void AddYData(float *Y, int Size, int OverLay=0);
  void PutXData(float *Xp, int Where)                  {X[Where]=*Xp;}
  void PutYData(float *Yp, int Where,int WhichOverlay=0) 
    {Yi[WhichOverlay][Where]=*Yp;}
  
  void YMinCallback(FL_OBJECT *Ob=NULL, long data=0);
  void YMaxCallback(FL_OBJECT *Ob=NULL, long data=0);
  int SliderPreHandlerUp(FL_OBJECT *, int event, 
			 FL_Coord , FL_Coord,
			 int , void *);
  int SliderPreHandlerDown(FL_OBJECT *, int event, 
			   FL_Coord , FL_Coord,
			   int , void *);
  int MinSliderMove(FL_OBJECT *ob, float fac);
  int MaxSliderMove(FL_OBJECT *ob, float fac);
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
    XLABEL,YLABEL,TITLE,
    ALPHAXTICS,ALPHAYTICS,XGRID,YGRID,
    XFIXEDAXIS,YFIXEDAXIS,
    XSCALE,YSCALE
  };
  
  enum {COPY, ADD};
  
private:
  int NoOfDataPts, NoOfOverlays;
  unsigned short int ScrollStep, AxisType,Chosen;
  int XTics0, YTics0, XTics1,YTics1,XGridType,YGridType;
  char *MajorAlphaXTics, *MinorAlphaXTics;
  char *MajorAlphaYTics, *MinorAlphaYTics;
  FL_OBJECT *CHART, *YMax_Slider, *YMin_Slider;
  FL_COLOR GraphBGColour,GraphFGColour;
  FL_FORM *Form;
  char *XLabel, *YLabel,*Title;
  
  vector<int> TmpSize;
  vector<FL_OBJECT *> Label;
  
  vector<unsigned short int> Colour, LineStyle, PlotStyle;
  
  vector< float * > Yi;
  float *X;
};


#endif
