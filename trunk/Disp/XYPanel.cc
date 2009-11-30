// $Id$
#include <XYPanel.h>
#include <vector>
#include <iostream.h>
#include <ErrorObj.h>
#define EPSILON  1E-3
//
//-------------------------------------------------------------------
//
XYPanel::XYPanel(FL_FORM *F)
{
  CHART = (FL_OBJECT *)NULL;
  NoOfDataPts = 0; ScrollStep = 1; NoOfOverlays=1;
  SetNoOfOverlays(1);
  //  Label.resize(0);
  X=NULL;
  Yi.resize(1); Yi[0]=NULL; TmpSize.resize(1); TmpSize[0]=0;
  XLabel=NULL,YLabel=NULL; Title=NULL;
  XTics0=YTics0=XTics1=YTics1=3;
  GraphBGColour=FL_BLACK;//FL_PALEGREEN;//FL_COL1;//FL_INACTIVE;
  GraphFGColour=FL_WHEAT;
  XGridType=YGridType=FL_GRID_NONE;
  MajorAlphaXTics=MinorAlphaXTics=NULL;
  MajorAlphaYTics=MinorAlphaYTics=NULL;
  Chosen=0;
}
//
//-------------------------------------------------------------------
//
XYPanel::~XYPanel()
{
  if (XLabel) free(XLabel);
  if (YLabel) free(YLabel);
  if (Title) free(Title);
}
//
//-------------------------------------------------------------------
//
#define OBJECT_BORDER_WIDTH 1
FL_OBJECT* XYPanel::Make(FL_FORM *F, int X0, int Y0,int Width, int Height)
{
  Form=F;
  CHART =  fl_add_xyplot(FL_NORMAL_XYPLOT,X0,Y0,Width-20,Height,"");

  CHART->u_ldata=(long)this;
  CHART->active = 1;
  CHART->input = 1;
  CHART->wantkey=FL_KEY_SPECIAL;
  fl_set_object_prehandler(CHART,PanelPreHandler);

  fl_set_object_boxtype(CHART,FL_FRAME_BOX);
  fl_set_object_color(CHART,GraphBGColour,GraphFGColour);
  fl_set_object_lcol(CHART,FL_SLATEBLUE);
  fl_set_object_lsize(CHART,FL_NORMAL_SIZE);
  fl_set_object_lstyle(CHART,FL_TIMES_STYLE);

  fl_set_xyplot_ytics(CHART,YTics0,YTics1);
  fl_set_xyplot_xtics(CHART,XTics0,XTics1);

  fl_set_object_bw(CHART,1);

  YMax_Slider = fl_add_slider(FL_VERT_SLIDER,X0+Width-20,Y0,10,Height,"");
    fl_set_slider_return(YMax_Slider, FL_RETURN_CHANGED);
    fl_set_object_callback(YMax_Slider,xp_ymax_callback,(long)this);

    YMax_Slider->u_ldata=(long)this;
    YMax_Slider->input = 1;
    YMax_Slider->wantkey=FL_KEY_SPECIAL;
    fl_set_object_prehandler(YMax_Slider, xp_SliderPreHandlerUp);

  YMin_Slider = fl_add_slider(FL_VERT_SLIDER,X0+Width-10,Y0,10,Height,"");
    fl_set_slider_return(YMin_Slider, FL_RETURN_CHANGED);
    fl_set_object_callback(YMin_Slider, xp_ymin_callback, (long)this);

    YMin_Slider->u_ldata=(long)this;
    YMin_Slider->input = 1;
    YMin_Slider->wantkey=FL_KEY_SPECIAL;
    fl_set_object_prehandler(YMin_Slider, xp_SliderPreHandlerDown);

  return CHART;
}
//
//-------------------------------------------------------------------
//
FL_OBJECT *XYPanel::AddText(int X0, int Y0, int Width, int Height, 
			    char *Str, FL_COLOR Colour,int TextType)
{
  /*
  int N=Label.size();
  FL_OBJECT *Obj;
  Label.resize(N+1);

  Label[N] = Obj = fl_add_xyplot_text(CHART,TextType, 
				      X0, Y0, Width, Height, 
				      Str);
  fl_set_object_lcol(Obj,Colour);
  return Label[N];
  */
}
//
//-------------------------------------------------------------------
//
int XYPanel::SetNoOfDataPts(int N) 
{
  NoOfDataPts=N;
  X=(float *)realloc(X,sizeof(float)*NoOfDataPts);
  for (int i=0;i<Yi.size();i++)
    {
      Yi[i] = (float *)realloc(Yi[i],sizeof(float)*NoOfDataPts);
      TmpSize[i]=NoOfDataPts;
    }
}
//
//-------------------------------------------------------------------
//
int XYPanel::SetNoOfOverlays(int N)
{
  int M;

  M=NoOfOverlays;
  NoOfOverlays = N;

  Colour.   resize(NoOfOverlays); 
  //  Label.    resize(NoOfOverlays);
  PlotStyle.resize(NoOfOverlays);
  LineStyle.resize(NoOfOverlays);
  Yi.       resize(NoOfOverlays);
  TmpSize.  resize(NoOfOverlays);

  for (int i=M;i<NoOfOverlays;i++)
    {
      Yi[i]        = NULL;
      TmpSize[i]   = 0;
      Colour[i]    = i%10;
      LineStyle[i] = i%10;
      PlotStyle[i] = FL_NORMAL_XYPLOT;
    }
}
//
//-------------------------------------------------------------------
//
void XYPanel::GetRange(float Range[2], int Axis)
{
  if (Axis==0) fl_get_xyplot_xbounds(CHART,&Range[0],&Range[1]);
  else         fl_get_xyplot_ybounds(CHART,&Range[0],&Range[1]);
}
//
//-------------------------------------------------------------------
//
void XYPanel::SetRange(float Range[2], int Axis)
{
  if (Axis==0) fl_set_xyplot_xbounds(CHART,(double)Range[0],(double)Range[1]);
  else         fl_set_xyplot_ybounds(CHART,(double)Range[0],(double)Range[1]);
}
//
//-------------------------------------------------------------------
//
void XYPanel::Clear(int WhichOverlay)
{
  if (WhichOverlay < 0)
    fl_clear_xyplot(CHART);
  else
    {
      EraseOverlay(WhichOverlay);
      Plot();
    }
}
//
//-------------------------------------------------------------------
//
void XYPanel::Plot(int WhichOverlay)
{
  float *Y=Yi[WhichOverlay];
  
  if (WhichOverlay >= NoOfOverlays)
    throw(ErrorObj("Request overlay for plot does not exits",
		   "###Error",
		   ErrorObj::Recoverable));

  if (WhichOverlay==-1)
    {
      fl_set_xyplot_data(CHART,X,Yi[0],TmpSize[0],
			 Title,XLabel,YLabel);
      for (int i=1;i<NoOfOverlays;i++)
	{
	  fl_add_xyplot_overlay(CHART, i, X, Yi[i], TmpSize[i],
				Colour[i]);
	  fl_set_xyplot_overlay_type(CHART, i,PlotStyle[i]);
	}
    }
	  
  else if (WhichOverlay==0)
    fl_set_xyplot_data(CHART,X,Yi[0],TmpSize[WhichOverlay],
		       Title,XLabel,YLabel);
  else
    {
      fl_add_xyplot_overlay(CHART, WhichOverlay, X, Yi[WhichOverlay],
			    TmpSize[WhichOverlay],
			    Colour[WhichOverlay]);
      fl_set_xyplot_overlay_type(CHART, WhichOverlay,
				 PlotStyle[WhichOverlay]);
    }
}
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, int Val, int WhichItem) 
{
  switch (WhichAttr)
    {
    case XTICS0:           
      XTics0 = Val;
      if (CHART) fl_set_xyplot_xtics(CHART,XTics0,XTics1);
      break;
    case XTICS1:           
      XTics1 = Val;
      if (CHART) fl_set_xyplot_xtics(CHART,XTics0,XTics1);
      break;
    case YTICS0:           
      YTics0 = Val;
      if (CHART) fl_set_xyplot_ytics(CHART,YTics0,YTics1);
      break;
    case YTICS1:           
      YTics1 = Val;
      if (CHART) fl_set_xyplot_ytics(CHART,YTics0,YTics1);
      break;
    case XGRID:             
      XGridType = Val;
      if (CHART) fl_set_xyplot_xgrid(CHART,XGridType);
      break;
    case YGRID:             
      YGridType = Val;
      if (CHART) fl_set_xyplot_ygrid(CHART,YGridType);
      break;
    case GRAPH_BG_COLOUR:   
      GraphBGColour=Val;
      if (CHART) fl_set_object_color(CHART,GraphBGColour,FL_BLUE);
      break;
    case OVERLAY_COLOR:    
      if (WhichItem <= 0)
	for (int i=0;i<Colour.size();i++)
	  Colour[i]=Val;
      else if (WhichItem < Colour.size())
	Colour[WhichItem]=Val;
      else 
	throw(ErrorObj("SetAttribute: Requested item not present.",
		       "###Error: ",
		       ErrorObj::Recoverable));
      break;
    case LINESTYLE:
      if (WhichItem <= 0)
	for (int i=0;i<LineStyle.size();i++)
	  LineStyle[i]=Val;
      else if (WhichItem < LineStyle.size())
	LineStyle[WhichItem]=Val;
      else 
	throw(ErrorObj("SetAttribute: Requested item not present.",
		       "###Error: ",
		       ErrorObj::Recoverable));
      break;
    case PLOTSTYLE: 
      if (WhichItem <= 0)
	for (int i=0;i<PlotStyle.size();i++)
	  PlotStyle[i]=Val;
      else if (WhichItem < PlotStyle.size())
	PlotStyle[WhichItem]=Val;
      else 
	throw(ErrorObj("SetAttribute: Requested item not present.",
		       "###Error",
		       ErrorObj::Recoverable));
      break;
    default: throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			    "###Error",
			    ErrorObj::Recoverable));
    }
};
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, char *Val0, char *Val1,
			  int WhichItem) 
{
  switch (WhichAttr)
    {
    case XFIXEDAXIS:
      fl_set_xyplot_fixed_xaxis(CHART,Val0,Val1);break;
    case YFIXEDAXIS:
      fl_set_xyplot_fixed_yaxis(CHART,Val0,Val1);break;
    case ALPHAXTICS:
      MajorAlphaXTics=(char *)realloc(MajorAlphaXTics,strlen(Val0));
      MinorAlphaXTics=(char *)realloc(MinorAlphaXTics,strlen(Val1));
      if (CHART) fl_set_xyplot_alphaxtics(CHART,Val0,Val1);
      break;
    case ALPHAYTICS:
      MajorAlphaYTics=(char *)realloc(MajorAlphaYTics,strlen(Val0));
      MinorAlphaYTics=(char *)realloc(MinorAlphaYTics,strlen(Val1));
      if (CHART) fl_set_xyplot_alphaytics(CHART,Val0,Val1);
      break;
    case XLABEL:
      if (Val0!=NULL)
	{
	  XLabel=(char *)realloc(XLabel,strlen(Val0));
	  strcpy(XLabel,Val0);
	}
      else {free(XLabel);XLabel=NULL;}
      break;
    case YLABEL:
      if (Val0!=NULL)
	{
	  YLabel=(char *)realloc(YLabel,strlen(Val0));
	  strcpy(YLabel,Val0);
	}
      else {free(YLabel);YLabel=NULL;}
      break;
    case TITLE:
      if (Val0!=NULL)
	{
	  Title=(char *)realloc(Title,strlen(Val0));
	  strcpy(Title,Val0);
	}
      else {free(Title);Title=NULL;}
      break;
    default: throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			    "###Error",
			    ErrorObj::Recoverable));
    }
};
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, int scale, double base)
{
  switch(WhichAttr)
    {
    case XSCALE:
      fl_set_xyplot_xscale(CHART,scale,base);break;
    case YSCALE:
      fl_set_xyplot_yscale(CHART,scale,base);break;
    default: throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			    "###Error",
			    ErrorObj::Recoverable));
    };
}
//
//-------------------------------------------------------------------
//
int XYPanel::GetAttribute(int WhichAttr, short int WhichItem=-1) 
{
  switch (WhichAttr)
    {
    case XTICS0: return XTics0;
    case XTICS1: return XTics1;
    case YTICS0: return YTics0;
    case YTICS1: return YTics1;
    case GRAPH_BG_COLOUR: return GraphBGColour;
    default: throw(ErrorObj("Looking for unknown attribute?!",
			    "###Error",
			    ErrorObj::Recoverable));
    }
};
//
//-------------------------------------------------------------------
//
void XYPanel::PutText(const char* Text, double x, double y, 
		      FL_COLOR Col,
		      int Align=FL_ALIGN_RIGHT)
{
  fl_add_xyplot_text(CHART,x,y,Text,Align,Col);
  //  fl_set_xyplot_key(CHART,0,Text);
}
//
//-------------------------------------------------------------------
//
void XYPanel::DeleteText(const char* Text)
{
    fl_delete_xyplot_text(CHART,Text);
}
//
//-------------------------------------------------------------------
//
void XYPanel::AddXData(float *Xp, int Size)
{
  int From;

  if ((From = Size>=NoOfDataPts?NoOfDataPts:Size) < NoOfDataPts)
    memcpy(X,&X[From],sizeof(float)*(NoOfDataPts-From));

  memcpy(&X[NoOfDataPts-From],Xp,sizeof(float)*(From));
};
//
//-------------------------------------------------------------------
//
void XYPanel::AddYData(float *Yp, int Size,int OverLay)
{
  float *d;
  int From;
  d=Yi[OverLay];

  if ((From = Size>=NoOfDataPts?NoOfDataPts:Size) < NoOfDataPts)
    memcpy(d,&d[From],sizeof(float)*(NoOfDataPts-From));
  
  memcpy(&d[NoOfDataPts-From],Yp,sizeof(float)*(From));
};
//
//-------------------------------------------------------------------
//
float XYPanel::GetMax(int n)
{
  float Max=0;
  if (n==ALLOVERLAYS)
    for (int i=0;i<NoOfOverlays;i++)
      for (int j=0;j<NoOfDataPts;i++)
	if ((Yi[i])[j] > Max) Max=Yi[i][j];
  return Max;
}
//
//-------------------------------------------------------------------
//
float XYPanel::GetMin(int n)
{
  float Min=0;
  if (n==ALLOVERLAYS)
    for (int i=0;i<NoOfOverlays;i++)
      for (int j=0;j<NoOfDataPts;i++)
	if ((Yi[i])[j] < Min) Min=Yi[i][j];
  return Min;
}
//
//-------------------------------------------------------------------
//
void XYPanel::FreeChart()
{
  fl_delete_object(CHART);
  fl_free_object(CHART); 
  CHART=NULL;
}  
//
//----------------------------------------------------------------
//
void XYPanel::YMinCallback(FL_OBJECT *Ob, long data)
{
  float Range[2]={0,0};
  int i;

  GetRange(Range,1);
  if (Ob==NULL) 
    {
      fl_set_slider_value(YMin_Slider,(double)Range[0]);
      fl_set_slider_bounds(YMin_Slider,(double)Range[1],(double)Range[0]);
    }
  else
    {
      Range[0] = (float)fl_get_slider_value(Ob);
      SetRange(Range,1);
    }
}
//
//----------------------------------------------------------------
//
void XYPanel::YMaxCallback(FL_OBJECT *Ob, long data)
{
  float Range[2]={0,0};
  int i;
  
  GetRange(Range,1);
  if (Ob==NULL) 
    {
      fl_set_slider_value(YMax_Slider,(double)Range[1]);
      fl_set_slider_bounds(YMax_Slider,(double)Range[1],(double)Range[0]);
    }
  else
    {
      Range[1] = (float)fl_get_slider_value(Ob);
      SetRange(Range,1);
    }
}
//
//-------------------------------------------------------------------
//
// Event pre-handlers to push the upper limit up or down for Max
// sliders.
//
int XYPanel::SliderPreHandlerUp(FL_OBJECT *ob, int Event,
				FL_Coord Mx, FL_Coord My,
				int Key, void *RawEvent)
{
  char Name[8];
  float fac = 0.1;
  KeySym L_KeySym;
  static short int Shift=0;

  if (RawEvent != NULL && Event == FL_KEYBOARD)
      {
	XLookupString((XKeyEvent *)RawEvent,Name,7,&L_KeySym,NULL);
	if (Shift) fac *= 0.1;
	switch (L_KeySym)
	  {
	  case XK_Up:      MaxSliderMove(ob,fac);  break;
	  case XK_Down:    MaxSliderMove(ob,-fac); break;
	  case XK_Shift_L: Shift = !Shift;break;
	  case XK_Shift_R: Shift = !Shift;break;
	  case XK_F2:      
	    {
	      //
	      // Set the YMax slider upper bound to the max. of
	      // of the plotted data.
	      //
	      float Range[2];
	      GetRange(Range,1);
	      SetRange(Range,1);
	      YMaxCallback();
	      break;
	    }
	  case XK_F1:
	    {
	      //
	      // Set the upper bound of the YMax slider to the
	      // current value of the slider.
	      //
	      double Range[2];
	      fl_get_slider_bounds(ob,&Range[1],&Range[0]);
	      Range[1] = fl_get_slider_value(ob);
	      fl_set_slider_bounds(ob,(double)Range[1],(double)Range[0]);
	      fl_set_slider_value(ob,(double)Range[1]);
	      break;
	    }
	  }

      }
  return !FL_PREEMPT;
}
//
// Event pre-hanlder to push the lower limit down or up for Min sliders.
//
int XYPanel::SliderPreHandlerDown(FL_OBJECT *ob, int Event,
				  FL_Coord Mx, FL_Coord My,
				  int Key, void *RawEvent)
{
  char Name[32];
  float fac = 0.1;
  KeySym L_KeySym;
  static short int Shift=0;

  if (RawEvent != NULL && Event == FL_KEYBOARD)
      {
	XLookupString((XKeyEvent *)RawEvent,Name,10,&L_KeySym,NULL);
	if (Shift) fac *= 0.1;

	switch (L_KeySym)
	  {
	  case XK_Down:    MinSliderMove(ob,fac);  break;
	  case XK_Up:      MinSliderMove(ob,-fac); break;
	  case XK_Shift_L: Shift = !Shift;         break;
	  case XK_Shift_R: Shift = !Shift;         break;
	  case XK_F2:      
	    {
	      //
	      // Set the YMin slider to the lower bound of the
	      // plotted data
	      //
	      float Range[2]={0,0};
	      SetRange(Range,1);
	      YMinCallback();
	      break;
	    }
	  case XK_F1:
	    {
	      //
	      // Set the YMin slider lower bound to the current
	      // value of the slider
	      //
	      double Range[2];
	      fl_get_slider_bounds(ob,&Range[1],&Range[0]);
	      Range[0] = fl_get_slider_value(ob);
	      fl_set_slider_bounds(ob,Range[1],Range[0]);
	      fl_set_slider_value(ob,Range[0]);
	      break;
	    }
	  }

      }
  return !FL_PREEMPT;
}
//-------------------------------------------------------------------
// Reads the focused slider (ob), which is expected to be the Max
// slider, jacks it up by a factor (fac) and sets the value and the
// bounds of all the selected Max sliders to this value.
//
int XYPanel::MaxSliderMove(FL_OBJECT *ob, float fac)
{
  float Range[2]={0,0},V;

  //
  // Find the max value from the focused slider, modify it, and set it
  // as the slider value of all the selected charts.
  //
  GetRange(Range,1);                         // Get Y range
  Range[1]  = (float)fl_get_slider_value(ob);// Read the slider value
  V=Range[1];
  if (fabs(V) <= EPSILON) V=10*EPSILON;
  Range[1] += (V*fac + EPSILON);             // Modify the value 

  fl_set_slider_bounds(YMax_Slider, (double)Range[1],(double)Range[0]);
  fl_set_slider_value (YMax_Slider, (double)Range[1]);

  SetRange(Range,1);

  return FL_PREEMPT;
}
//---------------------------------------------------------------------
// Reads the focused slider (ob), which is expected to be the Min
// slider, reduces it by a factor (fac) and sets the value and the
// bounds of all the selected Min sliders to this value.
//
int XYPanel::MinSliderMove(FL_OBJECT *ob, float fac)
{
  float Range[2]={0,0},V;

  GetRange(Range,1);

  Range[0]  = (float)fl_get_slider_value(ob);
  V=Range[0];
  if (fabs(V) <= EPSILON) V=10*EPSILON;
  Range[0] -= (V*fac + EPSILON);

  fl_set_slider_bounds(YMin_Slider, (double)Range[1],(double)Range[0]);
  fl_set_slider_value (YMin_Slider, (double)Range[0]);

  SetRange(Range,1);

  return FL_PREEMPT;
}
//
//-------------------------------------------------------------------
//
int XYPanel::PreHandler(FL_OBJECT *ob, int Event,
			FL_Coord Mx, FL_Coord My,
			int Key, void *RawEvent)
{
  char Name[8];
  KeySym L_KeySym;

  if (RawEvent != NULL && Event == FL_KEYBOARD)
      {
	XLookupString((XKeyEvent *)RawEvent,Name,7,&L_KeySym,NULL);

	switch (L_KeySym)
	  {
	  case XK_Up:      
	    //
	    // Visually show that the this panel is selected/de-selected
	    //
	    Chosen = !Chosen;
	    if (Chosen) fl_set_object_color(CHART,FL_RED,GraphFGColour);
	    else fl_set_object_color(CHART,GraphBGColour,GraphFGColour);
	    break;
	  case XK_F1:
	    //
	    // Set the max and min sliders bounds to their
	    // respective values.
	    //
	    YMinCallback(NULL,0);
	    YMaxCallback(NULL,0);
	    break;
	  case XK_F2:      
	    {
	      //
	      // Reset the max and min slider bounds to the
	      // range allowed by the data plotted in this
	      // panel.
	      //
	      float Range[2]={0,0};
	      SetRange(Range,1);
	      YMinCallback();
	      YMaxCallback();
	      break;
	    }
	  }

      }
  return !FL_PREEMPT;
}
//
//-------------------------------------------------------------------
//
extern "C" {
  int PanelPreHandler(FL_OBJECT *ob, int Event,
		      FL_Coord Mx, FL_Coord My,
		      int Key, void *RawEvent)
  {
    return ((XYPanel *)ob->u_ldata)->PreHandler(ob,Event,Mx,My,
						Key,RawEvent);
  }
  
  void xp_ymax_callback(FL_OBJECT *Ob, long data)
  {
    ((XYPanel *)data)->YMaxCallback(Ob,data);
  }

  void xp_ymin_callback(FL_OBJECT *Ob, long data)
  {
    ((XYPanel *)data)->YMinCallback(Ob,data);
  }

  int xp_SliderPreHandlerDown(FL_OBJECT *ob, int Event,
			      FL_Coord Mx, FL_Coord My,
			      int Key, void *RawEvent)
  {
    return ((XYPanel *)ob->u_ldata)->SliderPreHandlerDown(ob,Event,Mx,My,
							  Key,RawEvent);
  }

  int xp_SliderPreHandlerUp(FL_OBJECT *ob, int Event,
			    FL_Coord Mx, FL_Coord My,
			    int Key, void *RawEvent)
  {
    return ((XYPanel *)ob->u_ldata)->SliderPreHandlerUp(ob,Event,Mx,My,
							Key,RawEvent);
  }

};
