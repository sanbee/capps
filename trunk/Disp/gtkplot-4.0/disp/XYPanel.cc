// $Id$
#include <XYPanel.h>
#include <vector>
#include <iostream>
#include <ErrorObj.h>
#include "./WidgetLib.h"
#include <values.h>
#include "../gtkplot.h"
#define EPSILON  0
//
//-------------------------------------------------------------------
// Initialize the internal data members.  The data x and y ranges are 
// initialize to the extreme values and filled in with the actual values
// by the methods which load the data for the panel, namely Add{X,Y}Data
// or Put(X,y}Data.
//
XYPanel::XYPanel()
{
  CHART = (GtkWidget *)NULL;
  NoOfDataPts = 0; ScrollStep = 1; NoOfOverlays=1;
  SetNoOfOverlays(1);

  Yi.resize(1); Yi[0]=NULL; 

  XTics0=YTics0=XTics1=YTics1=3;
  YMin_Adj = YMax_Adj = NULL;
  Chosen=0;
  XAutoRanging=YAutoRanging=1;
  XDataMin = YDataMin = MINFLOAT;
  XDataMax = YDataMax = MAXFLOAT;
}
//
//-------------------------------------------------------------------
// Release the allocated data.  Since ultimately this XYPanel object 
// can be used where panles would be destroyed and remade, cleaning
// up is important to avoid mem. leaks.  So check this part carefully!
//
XYPanel::~XYPanel()
{
  int NoOfPanels=Yi.size();
  //  for (int i=0;i<NoOfDataPts;i++)
  for (int i=0;i<NoOfPanels;i++)
    {
      if ((Yi[i]) && (Yi[i]->y))
	{
	  g_free(Yi[i]->y); 
	  Yi[i]->y=NULL;
	}
      if ((Yi[i]) && (Yi[i]->x))
	if (Id == 0)
	  {
	    g_free(Yi[i]->x);
	    Yi[i]->x=NULL;
	  }
    }
  if (Id) Id--;
}
//
//-------------------------------------------------------------------
// The Widget lib. dependant method which create the widget and the 
// assocated objects like the y-scale scroll bars.
//
#define OBJECT_BORDER_WIDTH 1
GtkWidget* XYPanel::Make(GtkWidget *Kanwas, gint CW, gint CH,
			 gint X0, gint Y0, 
			 gint Width, gint Height)
{
  //  float ModifiedCW=CW+5, ModifiedCH=CH+50;
  float ModifiedCW=CW, ModifiedCH=CH;
  // int M_X0=X0+15;
 int M_X0=X0;
 char name[16];

 Chosen = (short int)Id++;
 YAutoRanging=XAutoRanging=1;
 XDataMin = YDataMin = MINFLOAT;
 XDataMax = YDataMax = MAXFLOAT;
 

 Canvas = Kanwas;
 CHART = gtk_plot_new_with_size(NULL, Width/ModifiedCW, Height/ModifiedCH);
 
 cerr << "Add at " << Width << " " << ModifiedCW << " " << Height << " " << ModifiedCH << endl;
 cerr << GTK_LAYOUT(Canvas)->width << " " << GTK_LAYOUT(Canvas)->height << endl;

 sprintf(name,"MPChart%3.3d",(int)Chosen);

 gtk_widget_set_name (CHART, name);


  gtk_plot_hide_legends(GTK_PLOT(CHART));

  
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_RIGHT_AXIS);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_TOP_AXIS);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_BOTTOM_AXIS);
  //
  // Hide ticks and labeling of the top and right axis
  //
  gtk_plot_axis_show_ticks (GTK_PLOT(CHART),
			    GTK_PLOT_AXIS_TOP,
			    GTK_PLOT_TICKS_NONE);
  gtk_plot_axis_show_ticks (GTK_PLOT(CHART),
			    GTK_PLOT_AXIS_RIGHT,
			    GTK_PLOT_TICKS_NONE);
  gtk_plot_axis_show_labels (GTK_PLOT(CHART),
			     GTK_PLOT_AXIS_TOP,
			     GTK_PLOT_LABEL_NONE);
  gtk_plot_axis_show_labels (GTK_PLOT(CHART),
			     GTK_PLOT_AXIS_RIGHT,
			     GTK_PLOT_LABEL_NONE);
  
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_X0);
  GTK_PLOT_SET_FLAGS(GTK_PLOT(CHART), GTK_PLOT_SHOW_Y0);
  //
  // Add the plot at the fractional locations wrt the canvas
  // width and height which is units of pixels
  //
  //  gtk_plot_canvas_add_plot(GTK_PLOT_CANVAS(Kanwas), GTK_PLOT(CHART), 
  //			   M_X0/ModifiedCW, (Y0+5)/ModifiedCH);

  gtk_plot_canvas_add_plot(GTK_PLOT_CANVAS(Kanwas), GTK_PLOT(CHART), 
			   M_X0/ModifiedCW, (Y0)/ModifiedCH);

  gtk_plot_canvas_set_active_plot(GTK_PLOT_CANVAS(Kanwas),
				  GTK_PLOT(CHART));
  gtk_plot_axis_hide_title(GTK_PLOT(CHART), GTK_PLOT_AXIS_TOP);
  gtk_plot_axis_hide_title(GTK_PLOT(CHART), GTK_PLOT_AXIS_RIGHT);

  //
  // Create the yrange adjustments and scroll bars
  //
  YMax_Slider = gtk_vscrollbar_new(NULL);
  YMin_Slider = gtk_vscrollbar_new(NULL);


  YMax_Adj = GTK_SCROLLBAR(YMax_Slider)->range.adjustment;
  YMin_Adj = GTK_SCROLLBAR(YMin_Slider)->range.adjustment;
  YMax_Adj->page_size = 0.01;
  YMin_Adj->page_size = 0.01;


  gtk_signal_connect (GTK_OBJECT (YMax_Adj), "value_changed",
		      GTK_SIGNAL_FUNC (xp_ymax_callback), 
		      (GtkWidget *)this);
  gtk_signal_connect (GTK_OBJECT (YMin_Adj), "value_changed",
		      GTK_SIGNAL_FUNC (xp_ymin_callback), 
		      (GtkWidget *)this);

  gtk_widget_set_usize(YMin_Slider, 20, Height);
  gtk_layout_put(GTK_LAYOUT(Kanwas), GTK_WIDGET(YMin_Slider), 
		 Width+M_X0+5, Y0);
  gtk_widget_show(YMin_Slider);

  gtk_widget_set_usize(YMax_Slider, 20, Height);
  gtk_layout_put(GTK_LAYOUT(Kanwas), GTK_WIDGET(YMax_Slider), 
		 Width+M_X0+20, Y0);

  gtk_widget_show(YMax_Slider);

  for (int i=0;i<(int)Yi.size();i++)
    if (Yi[i]==NULL) MakeStorage(i);

  gtk_widget_show(GTK_WIDGET(CHART));

  return CHART;
};
//
//-------------------------------------------------------------------
// Allocate memory for storing the y-data for overlayed plots and 
// X data.  The X data is global pointer shared by all XYPanel objects.
//
void XYPanel::MakeStorage(int WhichOverlay)
{
  if (WhichOverlay >= NoOfOverlays)
    throw(ErrorObj("Requested overlay in MakeStorage does not exist",
		   "###Error",
		   ErrorObj::Recoverable));
  
  if (Yi[WhichOverlay] == NULL)
    {
      //GdkColor color;
      //gint style, width;
      Yi[WhichOverlay] = gtk_plot_dataset_new(GTK_PLOT(CHART));
      gtk_plot_dataset_set_numpoints(Yi[WhichOverlay], NoOfDataPts);
      gtk_plot_add_dataset(GTK_PLOT(CHART), Yi[WhichOverlay]);

      /*
        if (WhichOverlay==0) gdk_color_parse("red", &color);
        else gdk_color_parse("green", &color);
      	gdk_color_alloc(gdk_colormap_get_system(), &color);


      	gtk_plot_dataset_set_line_attributes(Yi[WhichOverlay],
      					     (GtkPlotLineStyle)style,
      					     width,
      					     color);
      */
    }

  if (Yi[WhichOverlay]->y == NULL)
    Yi[WhichOverlay]->y=(gdouble *)g_realloc(Yi[WhichOverlay]->y,
					     sizeof(gdouble)*NoOfDataPts);
  if (X==NULL) X = (gdouble *)g_malloc(sizeof(gdouble)*NoOfDataPts);

  Yi[WhichOverlay]->x=X;

  for (int i=0;i<NoOfDataPts;i++) 
    {
      X[i]=0.0;
      Yi[WhichOverlay]->y[i]=0.0;
    }
}
//
//-------------------------------------------------------------------
// Set the total num. of data points held by the panel.  This is same
// for all overlayed plots.
//
int XYPanel::SetNoOfDataPts(int N)
{
  NoOfDataPts=N;
  return 1;
}
//
//-------------------------------------------------------------------
// Set the no. of overlayed plots this panel will display.
// If this is being set to a new number, resize the Yi array.  
// The way it is written currently, this method should be called before Make
// method which actually allocates the arrays to hold the numbers. 
// May be, right here, one call MakeStorage and write MakeStorage such that
// multiple calls to it will do the right things.
//
int XYPanel::SetNoOfOverlays(int N)
{
  int M;

  M=NoOfOverlays;
  NoOfOverlays = N;

  Yi.       resize(NoOfOverlays);

  for (int i=M;i<NoOfOverlays;i++)
    {
      Yi[i]        = NULL;
    }
  return 1;
}
//
//-------------------------------------------------------------------
// Get the pointers to the various GtkWidgets associated with the panel.
//
GtkWidget* XYPanel::GetObj(int Enum)
{
  switch(Enum)
    {
    case XYCHART:      return CHART;  
    case YMINSLIDER: return YMin_Slider;
    case YMAXSLIDER: return YMax_Slider;
    default:         return NULL;
    }
}
//
//-------------------------------------------------------------------
// Get the GtkAdjustments associated with the Y-scale sliders.
//
GtkAdjustment* XYPanel::GetSliderAdj(int Enum)
{
  switch(Enum)
    {
    case YMINSLIDERADJ: return YMin_Adj;
    case YMAXSLIDERADJ: return YMax_Adj;
    default:            return NULL;
    }
}
//
//-------------------------------------------------------------------
// Get range of the data on Axis (==0 ==> X-axis, ==1 ==> Y-axis).
// It can also provide the ranges of the Y-range sliders.
//
void XYPanel::GetRange(float* Range, int Axis, int Mode)
{
  if (Axis==0)      {Range[0] = XDataMin; Range[1] = XDataMax;}
  else
    if (Mode==RANGESLIDER)
      {
	Range[0] = YMin_Adj->lower;
	Range[1] = YMax_Adj->upper;
      }
    else            {Range[0] = YDataMin; Range[1] = YDataMax;}
}
//
//-------------------------------------------------------------------
// Set the range of the Axis.  For Y-axis, this will also set the 
// sliders range and value.
//
// If y-range is set, it will also emit "changed" signal for the
// y-axis scroll bars with the side effect that the YAutoRanging 
// will be turned off.  Hence, save the current ranging mode and
// set it back after the signal is emitted.
//
int XYPanel::SetRange(float Range[2], int Axis)
{
  int RangingMode;
  switch (Axis)
    {
    case 0:      RangingMode=XAutoRanging; if (!XAutoRanging) return 0;break;
    case 1:      RangingMode=YAutoRanging; if (!YAutoRanging) return 0;break;
    };

  if (Range[0] == Range[1])    GetRange(Range,Axis);

  if (Axis==0) 
    {
      gtk_plot_set_range(GTK_PLOT(CHART),
			 (gdouble)Range[0],(gdouble)Range[1],
			 YMin, YMax);
      XMin = Range[0]; XMax = Range[1];
      AutoSetTicks(XMin,XMax,GTK_ORIENTATION_HORIZONTAL);
    }
  else 
    {
      gtk_plot_set_range(GTK_PLOT(CHART), XMin, XMax,
			 (double)Range[0],(double)Range[1]);
	  
      YMin = Range[0]; YMax = Range[1];
      AutoSetTicks(YMin,YMax,GTK_ORIENTATION_VERTICAL);
      
      YMin_Adj->upper = YMax;
      YMin_Adj->lower = YMin;
      YMax_Adj->upper = YMax;
      YMax_Adj->lower = YMin;
	  

      YMin_Adj->step_increment =  (YMax-YMin)*0.001;
      YMax_Adj->step_increment =  (YMax-YMin)*0.001;
      YMin_Adj->page_increment =  (YMax-YMin)*0.001;
      YMax_Adj->page_increment =  (YMax-YMin)*0.001;
      YMin_Adj->page_size      =  (YMax-YMin)*0.005;
      YMax_Adj->page_size      =  (YMax-YMin)*0.001;

      gtk_adjustment_set_value(YMax_Adj,YMax);
      gtk_adjustment_set_value(YMin_Adj,YMin);

      //      gtk_signal_emit_by_name (GTK_OBJECT (YMax_Adj), "changed");
      //      gtk_signal_emit_by_name (GTK_OBJECT (YMin_Adj), "changed");
      YAutoRanging=RangingMode;
    }
  return 1;
}
//
//-------------------------------------------------------------------
// Supposed to just clear a given overlayed plot.  Not yet operational
// with GtkPlot widget.
//
void XYPanel::Clear(int WhichOverlay)
{
  if (WhichOverlay < 0)
    {
    }
  else
    {
      EraseOverlay(WhichOverlay);
      Plot();
    }
}
//
//-------------------------------------------------------------------
// Does the drawing of the current data(s) on the screen.
// gkt_plot_draw() now draws on the pixmap, and gtk_plot_refresh()
// copies the appropriate part of the pixmap on the screen.
// gtk_plot_refresh() which was inside gtk_plot_draw() earlier (in
// gtkplot.c) has been removed this that causes un-necessary calls to
// gtk_draw_pixmap whenever gtk_widget_queue_draw() is called.
//
void XYPanel::Plot(int WhichOverlay)
{
  if (!(GTK_PLOT_FLAGS(CHART) & GTK_PLOT_FREEZE))
    {
      GdkRectangle area;

      if (WhichOverlay >= NoOfOverlays)
	throw(ErrorObj("Requested overlay for plot does not exist",
		       "###Error",
		       ErrorObj::Recoverable));

      area.x = GTK_WIDGET(CHART)->allocation.x+GTK_LAYOUT(Canvas)->xoffset;
      area.y = GTK_WIDGET(CHART)->allocation.y+GTK_LAYOUT(Canvas)->yoffset;
      area.width = GTK_WIDGET(CHART)->allocation.width;	
      area.height = GTK_WIDGET(CHART)->allocation.height;	

      gtk_plot_draw(GTK_WIDGET(CHART),&area);
      gtk_plot_refresh(GTK_PLOT(CHART),&area);
    }
}
//
//-------------------------------------------------------------------
// Set various attributes.
int XYPanel::SetAttribute(int WhichAttr, gdouble *Val, int WhichItem) 
{
  switch(WhichAttr)
    {
    case GRAPH_BG_COLOUR:   
      {
      GdkColor color={0,0,0,0};
      color.red=(guint16)(Val[0]*65535.0);
      color.green=(guint16)(Val[1]*65535.0);
      color.blue=(guint16)(Val[2]*65535.0);
      gdk_color_alloc(gtk_widget_get_colormap(CHART), &color);

      //      gdk_color_parse("light yellow", &color);
      //      gdk_color_alloc(gtk_widget_get_colormap(active_plot), &color);
      //      gtk_plot_set_background(GTK_PLOT(active_plot), color);
      if (CHART) gtk_plot_set_background(GTK_PLOT(CHART),color);
      /*
      if (CHART) gtk_plot_axis_set_attributes(GTK_PLOT(CHART),
					     GTK_PLOT_AXIS_LEFT,
					     2,
					     color);
					     */
      break;
      }
    case GRAPH_FG_COLOUR:
      {
	GtkPlotData *data;
	GdkColor color={0,0,0,0};
	gint style, width;
	gtk_plot_dataset_get_line_attributes(Yi[WhichItem+1],
					     &style,
					     &width,
					     &color);
	color.red=(guint16)(Val[0]*65535.0);
	color.green=(guint16)(Val[1]*65535.0);
	color.blue=(guint16)(Val[2]*65535.0);

	gdk_color_alloc(gdk_colormap_get_system()/*gtk_widget_get_colormap(CHART)*/, 
			&color);

	data=Yi[WhichItem];
	gtk_plot_dataset_set_line_attributes(Yi[WhichItem+1],
					     (GtkPlotLineStyle)style,
					     width,
					     color);
	break;
      }
    default:throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			   "###Error", ErrorObj::Recoverable));
    }
  return 1;
}
//
//-------------------------------------------------------------------
// Set various attributes.
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
      if (Val)	gtk_plot_axis_show_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_BOTTOM);
      else	gtk_plot_axis_hide_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_BOTTOM);
      break;
    case YTITLE:
      if (Val)	gtk_plot_axis_show_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_LEFT);
      else	gtk_plot_axis_hide_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_LEFT);
      break;
    case TTITLE:
      if (Val)	gtk_plot_axis_show_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_TOP);
      else	gtk_plot_axis_hide_title(GTK_PLOT(CHART),
				 GTK_PLOT_AXIS_TOP);
      break;
    case DATALEGEND:
      if (Val)	gtk_plot_show_legends(GTK_PLOT(CHART));
      else 	gtk_plot_hide_legends(GTK_PLOT(CHART));
      break;
    }
  return 1;
}
//
//-------------------------------------------------------------------
// Set various attributes.
int XYPanel::SetAttribute(int WhichAttr, gdouble Val, int WhichItem) 
{
  switch (WhichAttr)
    {
    case XTICS0:           
      XTics0 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 
					 GTK_ORIENTATION_HORIZONTAL,
					 //GTK_PLOT_AXIS_BOTTOM, 
					 XTics0, XTics1);
      break;
    case XTICS1:           
      XTics1 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 
					 GTK_ORIENTATION_HORIZONTAL,
					 //GTK_PLOT_AXIS_BOTTOM, 
					 XTics0, XTics1);
      break;
    case YTICS0:           
      YTics0 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 
					 GTK_ORIENTATION_VERTICAL,
					 //GTK_PLOT_AXIS_LEFT, 
					 YTics0, YTics1);
      break;
    case YTICS1:           
      YTics1 = Val;
      if (CHART) gtk_plot_axis_set_ticks(GTK_PLOT(CHART), 
					 GTK_ORIENTATION_VERTICAL,
					 //GTK_PLOT_AXIS_LEFT, 
					 YTics0, YTics1);
      break;
    case OVERLAY_COLOR:    
      //      if (WhichItem <= 0)
      //	for (int i=0;i<Colour.size();i++)
      //	  Colour[i]=Val;
      //      else if (WhichItem < Colour.size())
      //	Colour[WhichItem]=Val;
      //      else 
	throw(ErrorObj("SetAttribute: Requested item not present.",
		       "###Error: ",
		       ErrorObj::Recoverable));
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
// Set various attributes.
int XYPanel::SetAttribute(int WhichAttr, char *Val0, char *Val1,
			  int WhichItem) 
{
  switch (WhichAttr)
    {
    case GRAPH_BG_COLOUR:   
      {
      GdkColor color={0,0,0,0};
      gdk_color_parse(Val0, &color);

      gdk_color_alloc(gtk_widget_get_colormap(CHART), &color);

      if (CHART) gtk_plot_set_background(GTK_PLOT(CHART),color);
      break;
      }
    case GRAPH_FG_COLOUR:
      {
	GtkPlotData *data;
	GdkColor color={0,0,0,0};
	gint style, width;
	gtk_plot_dataset_get_line_attributes(Yi[WhichItem+1],
					     &style,
					     &width,
					     &color);
	gdk_color_parse(Val0, &color);
	gdk_color_alloc(gdk_colormap_get_system()/*gtk_widget_get_colormap(CHART)*/, 
			&color);

	data=Yi[WhichItem];
	gtk_plot_dataset_set_line_attributes(Yi[WhichItem+1],
					     (GtkPlotLineStyle)style,
					     width,
					     color);
	break;
      }
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
    case DATALEGEND:
      if (Val0!=NULL)   gtk_plot_dataset_set_legend(Yi[WhichItem],Val0);
      break;
    case XTITLE:
      if (Val0!=NULL)   gtk_plot_axis_set_title(GTK_PLOT(CHART),
						GTK_PLOT_AXIS_BOTTOM,
						Val0);
      break;
    case YTITLE:
      if (Val0!=NULL)   gtk_plot_axis_set_title(GTK_PLOT(CHART),
						GTK_PLOT_AXIS_LEFT,
						Val0);
      break;
    case TTITLE:
      if (Val0!=NULL)	gtk_plot_axis_set_title(GTK_PLOT(CHART),
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
// Set various attributes.
int XYPanel::SetAttribute(int WhichAttr, gdouble& val0, gdouble& val1)
{
  switch(WhichAttr)
    {
      /*
    case XSCALE:
      fl_set_xyplot_xscale(CHART,scale,base);break;
    case YSCALE:
      fl_set_xyplot_yscale(CHART,scale,base);break;
      */
    case XRANGE:
      {
	gfloat Range[2] = {val0,val1};
	SetRange(Range,0);
	break;
      }
    case YRANGE:
      {
	gfloat Range[2] = {val0,val1};
	SetRange(Range,1);
	break;
      }
    default: throw(ErrorObj("Attempt to set invalid attribute igrnored.",
			    "###Error:XYPanel::SetAttribute(int WhichAttr, gdouble& val0, gdouble& val1)",
			    ErrorObj::Recoverable));
    };
  return 1;
}
//
//-------------------------------------------------------------------
// Get various attributes.
int XYPanel::GetAttribute(int WhichAttr, short int WhichItem) 
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
// Put text at locatin x,y (in pixel co-ordinates).
//
void XYPanel::PutText(char* Text, double x, double y, 
		      char *Col,
		      int Angle)
{
  GdkColor *fg;
  GtkPlot *p=GTK_PLOT(CHART);

  if (Col==NULL) fg = &(Yi[0]->line.color);
  else cerr << "FG colour not yet implemented in XYPanel::PutText" << endl;
  gtk_plot_put_text(GTK_PLOT(CHART),
		    p->x+x*p->width, p->y+y*p->height,
		    Angle,
		    p->bottom.label_attr.font,//font
		    p->bottom.label_attr.height,
		    fg, //foreground,
		    &(p->background), //background,
		    0,//GTK_JUSTIFY_CENTER,//Justification
		    Text); 
}
//
//-------------------------------------------------------------------
// Remove text.
//
void XYPanel::DeleteText(const char* Text)
{
  gtk_plot_remove_text(GTK_PLOT(CHART),(char *)Text);
}
//
//-------------------------------------------------------------------
// Scrolls the DataTarget array by TargetSize elements and copies 
// SourceSize elements from DataSource array into the vacated elements
// of DataTarget.  The size of the of the DataTarget array is TargetSize.
//
void XYPanel::ScrollData(gdouble *DataTarget, gdouble *DataSource, 
			 int TargetSize, int SourceSize)
{
  int From;
  if ((From = SourceSize>=TargetSize?TargetSize:SourceSize) < TargetSize)
    memcpy(DataTarget,&DataTarget[From],sizeof(gdouble)*(TargetSize-From));

  memcpy(&DataTarget[TargetSize-From],DataSource,sizeof(gdouble)*(From));
}
//
//-------------------------------------------------------------------
// Copy Xp in X[Where] array and update the XDataMin, XDataMax values.
//
void XYPanel::PutXData(gdouble Xp, int Where) 
{
  Yi[0]->x[Where]=Xp;
  if (XDataMin == MINFLOAT) XDataMin = XDataMax = Xp;
  else 
    {
      if (Xp < XDataMin) XDataMin = Xp;
      if (Xp > XDataMax) XDataMax = Xp;
    }
}
//
//-------------------------------------------------------------------
// Copy Yp in the Where location of the WhichOverlay y-data.  Also
// update the YDataMin and YDataMax values.
//
void XYPanel::PutYData(gdouble Yp, int Where,int WhichOverlay) 
{
  Yi[WhichOverlay]->y[Where]=Yp;
  if (YDataMin == MINFLOAT) YDataMin = YDataMax = Yp;
  else 
    {
      if (Yp < YDataMin) YDataMin = Yp;
      if (Yp > YDataMax) YDataMax = Yp;
    }
}
//
//-------------------------------------------------------------------
// Scroll the X data by Size elements and copy the new values from Xp
// for overlay Overlay.
//
void XYPanel::AddXData(gdouble *Xp, int Size, int Overlay)
{
  if (Size)
    {
      if (Overlay >= NoOfOverlays)
	throw(ErrorObj("Request overlay in AddXData does not exist",
		       "###Error",
		       ErrorObj::Recoverable));

      ScrollData(Yi[Overlay]->x, Xp,
		 gtk_plot_dataset_get_numpoints(Yi[Overlay]),Size);

      XDataMin = XDataMax = (float)Yi[0]->x[0];
      for (int i=0;i<NoOfDataPts;i++)
	{
	  if ((float)Yi[0]->x[i] < XDataMin) XDataMin=(float)Yi[0]->x[i];
	  if ((float)Yi[0]->x[i] > XDataMax) XDataMax=(float)Yi[0]->x[i];
	}
      //      cerr << "XR: " << XDataMin << " " << XDataMax << endl;
    }
};
//
//-------------------------------------------------------------------
// Scroll the X data by Size elements and copy the new values from Xp
// for overlay Overlay.
// 
void XYPanel::AddYData(gdouble *Yp, int Size,int Overlay)
{
  if (Size)
    {
      if (Overlay >= NoOfOverlays)
	throw(ErrorObj("Request overlay in AddYData does not exist",
		       "###Error",
		       ErrorObj::Recoverable));

      if (Yi[Overlay]==NULL) MakeStorage(Overlay);

      ScrollData(Yi[Overlay]->y, Yp,
		 gtk_plot_dataset_get_numpoints(Yi[Overlay]),Size);
      
      if (YDataMin == MINFLOAT) {YDataMin=YDataMax=Yi[0]->y[0];}
      for (int i=0;i<Size;i++)
	{
	  if (Yp[i] < YDataMin) YDataMin=Yp[i];
	  if (Yp[i] > YDataMax) YDataMax=Yp[i];
	}
    }
}
//
//-------------------------------------------------------------------
// Get the Max y value from all overlays
//
float XYPanel::GetMax(int n)
{
  float Max=0;
  if (n==ALLOVERLAYS)
    for (int i=0;i<NoOfOverlays;i++)
      for (int j=0;j<NoOfDataPts;i++)
	if (Yi[i]->y[j] > Max) Max=Yi[i]->y[j];
  return Max;
}
//
//-------------------------------------------------------------------
// Get the min. y-value from all overlays.
//
float XYPanel::GetMin(int n)
{
  float Min=0;
  if (n==ALLOVERLAYS)
    for (int i=0;i<NoOfOverlays;i++)
      for (int j=0;j<NoOfDataPts;i++)
	if (Yi[i]->y[j] < Min) Min=Yi[i]->y[j];
  return Min;
}
//
//----------------------------------------------------------------
// Set the number of major and minor ticks on the axis Axis.  This
// algorithm is bare minimum and needs to be made smarter to get prettier
// display of plots.
//
void XYPanel::AutoSetTicks(gfloat& Min, gfloat& Max,GtkOrientation Axis)
{
  gdouble delta,P;
  int Precision, Style=GTK_PLOT_LABEL_FLOAT;

  delta = Max-Min;

  if (delta)
    {
      gtk_plot_axis_set_ticks (GTK_PLOT(CHART), Axis, delta*0.2, delta*0.1);

      P = log10(fabs(delta*0.2));
      Precision = (int)fabs(P-3);

      if ((P > 3) || (P <= -3)) {Precision=fabs(fabs(P)-2); Style = GTK_PLOT_LABEL_EXP;}
      if (Axis==GTK_ORIENTATION_VERTICAL)
	gtk_plot_axis_labels_set_numbers (GTK_PLOT(CHART),
					  GTK_PLOT_AXIS_LEFT,
					  Style,
					  Precision);
      else
	gtk_plot_axis_labels_set_numbers (GTK_PLOT(CHART),
					  GTK_PLOT_AXIS_BOTTOM,
					  Style,
					  Precision);
    }
}
//
//----------------------------------------------------------------
// Call back called when the YMin scroll bars is fiddled with.
//
void XYPanel::YMinCallback(GtkAdjustment *Ob, GtkWidget* data)
{
  if (YMax_Adj->value > YMin_Adj->value)
    {
      YAutoRanging=0;
      YMin = YMin_Adj->value;
      AutoSetTicks(YMin, YMax, GTK_ORIENTATION_VERTICAL);
      gtk_plot_set_range(GTK_PLOT(CHART), XMin, XMax, YMin, YMax);
      Plot();
    }
  else
    gtk_adjustment_set_value(YMin_Adj,YMax);
}
//
//----------------------------------------------------------------
// Call back for YMax scroll bar.
//
void XYPanel::YMaxCallback(GtkAdjustment *Ob, GtkWidget* data)
{
  if (YMax_Adj->value > YMin_Adj->value)
    {
      YAutoRanging=0;
      YMax = YMax_Adj->value;
      AutoSetTicks(YMin, YMax, GTK_ORIENTATION_VERTICAL);
      gtk_plot_set_range(GTK_PLOT(CHART), XMin, XMax, YMin, YMax);
      Plot();
    }
  else 
    gtk_adjustment_set_value(YMax_Adj,YMin_Adj->value);
}
//
//-------------------------------------------------------------------
// Call back for "click_on_data" event.
//
int XYPanel::PreHandler(GtkWidget *ob,
			GdkEvent *Event,
			gpointer func_data )
{
  cerr << "Data doctoring option disabled (;-)" << endl;
  return 0;
}
//
//-------------------------------------------------------------------
//
extern "C" {
  int PanelPreHandler(GtkWidget *ob, 
		      GdkEvent *Event,
		      gpointer data)
  {
    return ((XYPanel *)data)->PreHandler(ob,Event,data);
  }
  
  void xp_ymax_callback(GtkAdjustment *Ob, GtkWidget* data)
  {
    ((XYPanel *)data)->YMaxCallback(Ob,data);
  }

  void xp_ymin_callback(GtkAdjustment *Ob, GtkWidget* data)
  {
    ((XYPanel *)data)->YMinCallback(Ob,data);
  }
};








