// $Id$
#include <XYPanel.h>
#include <vector>
#include <iostream.h>
#include <ErrorObj.h>
#define EPSILON  1E-3
//
//-------------------------------------------------------------------
//
XYPanel::XYPanel(GtkWidget *F)
{
  CHART = (GtkWidget *)NULL;
  NoOfDataPts = 0; ScrollStep = 1; NoOfOverlays=1;
  SetNoOfOverlays(1);
  //  Label.resize(0);
  Yi.resize(1); Yi[0]=NULL; TmpSize.resize(1); TmpSize[0]=0;
  XLabel=NULL,YLabel=NULL; Title=NULL;
  XTics0=YTics0=XTics1=YTics1=3;
  //  GraphBGColour=FL_BLACK;//FL_PALEGREEN;//FL_COL1;//FL_INACTIVE;
  //  GraphFGColour=FL_WHEAT;
  //  XGridType=YGridType=FL_GRID_NONE;
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
GtkWidget* XYPanel::Make(GtkWidget *F, gfloat X0, gfloat Y0, gfloat Width, gfloat Height)
{
  Form=F;
  GtkWidget *ScaleSB;

  CHART = gtk_plot_new_with_size(NULL, Width, Height);

  //
  //
  //
  //  ScaledSB = gtk_vscale_new( GTK_PLOT(CHART)->vadj);
  



  //  gtk_plot_set_range(GTK_PLOT(CHART), -1. ,1., -1., 1.4);
  //  gtk_plot_legends_move(GTK_PLOT(CHART), 0.15,0.05);
  gtk_plot_hide_legends(GTK_PLOT(CHART));
  //  gtk_plot_hide_legends_border(GTK_PLOT(CHART));
  /*
  gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 0, XTics0, XTics1); //Left
  gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 1, YTics0, YTics1); //Right
  gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 2, YTics0, YTics1); //Top
  gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 3, YTics0, YTics1); //Bottom
  */

  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_LEFT_AXIS);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_BOTTOM_AXIS);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_RIGHT_AXIS);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_TOP_AXIS);
  gtk_plot_axis_set_ticks(GTK_PLOT(CHART), GTK_PLOT_AXIS_TOP, 0,0);
  gtk_plot_axis_set_ticks(GTK_PLOT(CHART), GTK_PLOT_AXIS_RIGHT, 0,0);
  
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_X0);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_Y0);
  gtk_plot_canvas_add_plot(GTK_PLOT_CANVAS(F), GTK_PLOT(CHART), X0, Y0);
  gtk_plot_canvas_set_active_plot(GTK_PLOT_CANVAS(F),
				  GTK_PLOT(CHART));
  gtk_plot_axis_hide_title(GTK_PLOT(CHART), GTK_PLOT_AXIS_TOP);
  gtk_plot_axis_hide_title(GTK_PLOT(CHART), GTK_PLOT_AXIS_RIGHT);
  /*
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
  */
  return CHART;
}
//
//-------------------------------------------------------------------
//
GtkWidget *XYPanel::AddText(int X0, int Y0, int Width, int Height, 
			    char *Str, gint Colour,int TextType)
{
  /*
  int N=Label.size();
  GtkWidget *Obj;
  Label.resize(N+1);

  Label[N] = Obj = fl_add_xyplot_text(CHART,TextType, 
				      X0, Y0, Width, Height, 
				      Str);
  fl_set_object_lcol(Obj,Colour);
  return Label[N];
  */
return NULL;
}
//
//-------------------------------------------------------------------
//
int XYPanel::SetNoOfDataPts(int N) 
{
  NoOfDataPts=N;
  //  X=(float *)g_realloc(X,sizeof(GtkDataPlot)*NoOfDataPts);
  for (int i=0;i<(int)Yi.size();i++)
    {
      /*
      Yi[i]->points = (GtkPlotPoint *)g_realloc(Yi[i]->points,
						sizeof(GtkPlotPoint)*NoOfDataPts);
      assert(Yi[i]->points != NULL);
      */
      TmpSize[i]=NoOfDataPts;
    }
  return 1;
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
      //      PlotStyle[i] = FL_NORMAL_XYPLOT;
    }
  return 1;
}
//
//-------------------------------------------------------------------
//
void XYPanel::GetRange(float Range[2], int Axis)
{
  Range[0]=Range[1]=0.0;

  if (Axis==0)
    for (int i=0;i<NoOfOverlays;i++)
      for (int j=0;j<NoOfDataPts;j++)
	{
	  if (Yi[i]->points[j].x > Range[1]) Range[1] = Yi[i]->points[j].x;
	  if (Yi[i]->points[j].x < Range[0]) Range[0] = Yi[i]->points[j].x;
	}
  else
    for (int i=0;i<NoOfOverlays;i++)
      for (int j=0;j<NoOfDataPts;j++)
	{
	  if (Yi[i]->points[j].y > Range[1]) Range[1] = Yi[i]->points[j].y;
	  if (Yi[i]->points[j].y < Range[0]) Range[0] = Yi[i]->points[j].y;
	}
}
//
//-------------------------------------------------------------------
//
void XYPanel::SetRange(float Range[2], int Axis)
{
  if (Axis==0) 
    {
      gtk_plot_set_range(GTK_PLOT(CHART),
			 (gdouble)Range[0],(gdouble)Range[1],
			 YMin, YMax);
      XMin = Range[0]; XMax = Range[1];
    }
  else 
    {
      gtk_plot_set_range(GTK_PLOT(CHART),
			 XMin, XMax,
			 (double)Range[0],(double)Range[1]);
      YMin = Range[0]; YMax = Range[1];
    }
}
//
//-------------------------------------------------------------------
//
void XYPanel::Clear(int WhichOverlay)
{
  if (WhichOverlay < 0)
    {
      /*
      gtk_plot_dataset_set_line_attributes(Yi[WhichOverlay],
					   NULL, //GtkPlotLineStyle style,
					   NULL, //gint width,
					   NULL);//GdkColor color);
      */
    }
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
  if (WhichOverlay >= NoOfOverlays)
    throw(ErrorObj("Request overlay for plot does not exits",
		   "###Error",
		   ErrorObj::Recoverable));

  for (int i=0;i<(int)Yi.size();i++)
    if (Yi[i]==NULL) Yi[i] = gtk_plot_dataset_new(GTK_PLOT(CHART));

  if (WhichOverlay==-1)
    {
      for (int i=0;i<(int)Yi.size();i++)
	  gtk_plot_add_dataset(GTK_PLOT(CHART), Yi[i]);
    }
  else
    gtk_plot_add_dataset(GTK_PLOT(CHART), Yi[WhichOverlay]);

  gtk_widget_show(GTK_WIDGET(CHART));
  gtk_widget_queue_draw(GTK_WIDGET(CHART));
}
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, gint Val, int WhichItem) 
{
  switch(WhichAttr)
    {
    case XLABEL:
      gtk_plot_axis_show_labels(GTK_PLOT(CHART),
			     GTK_PLOT_AXIS_BOTTOM,
			     Val);

      break;
    case YLABEL:
      gtk_plot_axis_show_labels(GTK_PLOT(CHART),
			     GTK_PLOT_AXIS_LEFT,
			     Val);

      break;
    case XTITLE:
      if (Val)
	gtk_plot_axis_show_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_BOTTOM);
      else
	gtk_plot_axis_hide_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_BOTTOM);

      break;
    case YTITLE:
      if (Val)
	gtk_plot_axis_show_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_LEFT);
      else
	gtk_plot_axis_hide_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_LEFT);

      break;
    case TTITLE:
      if (Val)
	gtk_plot_axis_show_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_TOP);
      else
	gtk_plot_axis_hide_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_TOP);

      break;
    default:throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			   "###Error", ErrorObj::Recoverable));
    }
  return 1;
}
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, gdouble Val, int WhichItem) 
{
  switch (WhichAttr)
    {
    case XTICS0:           
      XTics0 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), GTK_PLOT_AXIS_BOTTOM, 
					 XTics0, XTics1);
      break;
    case XTICS1:           
      XTics1 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), GTK_PLOT_AXIS_BOTTOM, 
					 XTics0, XTics1);
      break;
    case YTICS0:           
      YTics0 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), GTK_PLOT_AXIS_LEFT, 
					 YTics0, YTics1);
      break;
    case YTICS1:           
      YTics1 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), GTK_PLOT_AXIS_LEFT, 
					 YTics0, YTics1);
      break;
      /*
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
      */
    default: throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			    "###Error",
			    ErrorObj::Recoverable));
    }
  return 1;
};
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, char *Val0, char *Val1,
			  int WhichItem) 
{
  switch (WhichAttr)
    {
      /*
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
      */
    case XTITLE:
      if (Val0!=NULL)
	gtk_plot_axis_set_title(GTK_PLOT(CHART),
				GTK_PLOT_AXIS_BOTTOM,
				Val0);
      break;
    case YTITLE:
      if (Val0!=NULL)
	gtk_plot_axis_set_title(GTK_PLOT(CHART),
				GTK_PLOT_AXIS_LEFT,
				Val0);
      break;
    case TTITLE:
      if (Val0!=NULL)
	gtk_plot_axis_set_title(GTK_PLOT(CHART),
				GTK_PLOT_AXIS_TOP,
				Val0);
      break;
    default: throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			    "###Error",
			    ErrorObj::Recoverable));
    }
  return 1;
};
//
//-------------------------------------------------------------------
//
int XYPanel::SetAttribute(int WhichAttr, int scale, double base)
{
  /*
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
  */
  return 1;
}
//
//-------------------------------------------------------------------
//
int XYPanel::GetAttribute(int WhichAttr, short int WhichItem=-1) 
{
  /*
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
  */
  return 1;
};
//
//-------------------------------------------------------------------
//
void XYPanel::PutText(char* Text, double x, double y, 
		      gint Col,
		      int Align=0)
{
  gtk_plot_put_text(GTK_PLOT(CHART),
		    x, y, 
		    Align, //gint angle,
		    NULL,//gchar *font,	
		    1,//gint height,
		    &GraphFGColour, //GdkColor *foreground,
		    &GraphBGColour, //GdkColor *background,
		    Text); 
}
//
//-------------------------------------------------------------------
//
void XYPanel::DeleteText(const char* Text)
{
  /*
    fl_delete_xyplot_text(CHART,Text);
  */
}
//
//-------------------------------------------------------------------
//
void XYPanel::AddXData(float *Xp, int Size)
{
  /*
  int From;
  if ((From = Size>=NoOfDataPts?NoOfDataPts:Size) < NoOfDataPts)
    memcpy(X,&X[From],sizeof(float)*(NoOfDataPts-From));

  memcpy(&X[NoOfDataPts-From],Xp,sizeof(float)*(From));
  */
};
//
//-------------------------------------------------------------------
//
void XYPanel::AddYData(gdouble *Yp, int Size,int OverLay)
{
  //  GtkPlotPoint *d;
  //  int From;
  //  d=Yi[OverLay]->points;

  if (Yi[OverLay]==NULL) 
    gtk_plot_add_dataset(GTK_PLOT(CHART),
			 (Yi[OverLay] = gtk_plot_dataset_new(GTK_PLOT(CHART))));

  NoOfDataPts = Size;
  gtk_plot_dataset_set_x(Yi[OverLay], Yp);
  gtk_plot_dataset_set_numpoints(Yi[OverLay], Size);


  /*
  if ((From = Size>=NoOfDataPts?NoOfDataPts:Size) < NoOfDataPts)
    memcpy(d,&d[From],sizeof(GtkPlotData)*(NoOfDataPts-From));
  
  memcpy(&d[NoOfDataPts-From],Yp,sizeof(GtkPlotData)*(From));
  */
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
	if (Yi[i]->points[j].y > Max) Max=Yi[i]->points[j].y;
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
	if (Yi[i]->points[j].y < Min) Min=Yi[i]->points[j].y;
  return Min;
}
//
//-------------------------------------------------------------------
//
void XYPanel::FreeChart()
{
  /*
  fl_delete_object(CHART);
  fl_free_object(CHART); 
  CHART=NULL;
  */
}  
//
//----------------------------------------------------------------
//
void XYPanel::YMinCallback(GtkWidget *Ob, long data)
{
  /*
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
  */
}
//
//----------------------------------------------------------------
//
void XYPanel::YMaxCallback(GtkWidget *Ob, long data)
{
  /*  
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
  */
}
//
//-------------------------------------------------------------------
//
// Event pre-handlers to push the upper limit up or down for Max
// sliders.
//
int XYPanel::SliderPreHandlerUp(GtkWidget *ob, int Event,
				gint Mx, gint My,
				int Key, void *RawEvent)
{
  /*
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
  */
  return 0;
}
//
// Event pre-hanlder to push the lower limit down or up for Min sliders.
//
int XYPanel::SliderPreHandlerDown(GtkWidget *ob, int Event,
				  gint Mx, gint My,
				  int Key, void *RawEvent)
{
  /*
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
  */
  return 0;
}
//-------------------------------------------------------------------
// Reads the focused slider (ob), which is expected to be the Max
// slider, jacks it up by a factor (fac) and sets the value and the
// bounds of all the selected Max sliders to this value.
//
int XYPanel::MaxSliderMove(GtkWidget *ob, float fac)
{
  /*
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
  */
  return 0;
}
//---------------------------------------------------------------------
// Reads the focused slider (ob), which is expected to be the Min
// slider, reduces it by a factor (fac) and sets the value and the
// bounds of all the selected Min sliders to this value.
//
int XYPanel::MinSliderMove(GtkWidget *ob, float fac)
{
  /*
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
  */
  return 0;
}
//
//-------------------------------------------------------------------
//
int XYPanel::PreHandler(GtkWidget *ob, int Event,
			gint Mx, gint My,
			int Key, void *RawEvent)
{
  /*
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
  */
  return 0;
}
//
//-------------------------------------------------------------------
//
extern "C" {
  int PanelPreHandler(GtkWidget *ob, int Event,
		      gint Mx, gint My,
		      int Key, void *RawEvent)
  {
    /*
    return ((XYPanel *)ob->u_ldata)->PreHandler(ob,Event,Mx,My,
						Key,RawEvent);
    */
    return 0;
  }
  
  void xp_ymax_callback(GtkWidget *Ob, long data)
  {
    ((XYPanel *)data)->YMaxCallback(Ob,data);
  }

  void xp_ymin_callback(GtkWidget *Ob, long data)
  {
    ((XYPanel *)data)->YMinCallback(Ob,data);
  }

  int xp_SliderPreHandlerDown(GtkWidget *ob, int Event,
			      gint Mx, gint My,
			      int Key, void *RawEvent)
  {
    /*
    return ((XYPanel *)ob->u_ldata)->SliderPreHandlerDown(ob,Event,Mx,My,
							  Key,RawEvent);
    */
    return 0;
  }

  int xp_SliderPreHandlerUp(GtkWidget *ob, int Event,
			    gint Mx, gint My,
			    int Key, void *RawEvent)
  {
    /*
    return ((XYPanel *)ob->u_ldata)->SliderPreHandlerUp(ob,Event,Mx,My,
							Key,RawEvent);
    */
    return 0;
  }

};
