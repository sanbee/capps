// $Id$
#include <MultiPanel.h>
#include <ErrorObj.h>
#include "./WidgetLib.h"
#include <XYPanel.h>
#include <iostream.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ExitObj.h>

extern "C" 
{
#include "./mpd_interface.h"
  GtkWidget*  lookup_widget              (GtkWidget       *widget,
					  const gchar     *widget_name);
  int gnome_init(char *, float, int, char **);

  void gtk_plot_layout_paint(GtkWidget *);
  void DefaultPacker(int n, int np, gfloat *cw, gfloat *ch, 
		     gfloat *w, gfloat *h, gfloat *x, gfloat *y) 
  {
    *w = 0.87;
    *h = 0.9/np;
    *y = *h * n;
    *x = 0.08;

    *w = *cw*0.87;
    *h = *ch/np;
    *y = *h * n;
    *x = *cw*0.08;
  };
};
//
//---------------------------------------------------------------------
//
MultiPanel::MultiPanel(int argc, char **argv, char *Title,
		       int N,int NPoints) 
{  
  HANDLE_EXCEPTIONS(
		    gtk_rc_add_default_file("tst.rc");

		    gtk_init(&argc, &argv);

		    gnome_init ("project1", 0, argc, argv);
		    app1 = create_app1();
		    //		    gtk_widget_show (app1);
		    
		    Canvas=NULL;
		    CtrlPanel=NULL;
		    Init(N,NPoints);
		    );
}
//
//---------------------------------------------------------------------
//
GtkWidget *MultiPanel::MakeWindow(unsigned int Which,
				  gfloat X0, float Y0,
				  gfloat W0, gfloat H0,
				  int Disp, char *Title)
{
  int Height, Width;
  gfloat X,Y,W,H;
  gfloat CW,CH;
  int NP;
  GtkWidget *TopLevelMenuBox;

  NP=NPanels();

  Width = (int)W0;
  Height= (int)H0*NP;
  //  fl_set_border_width(1);
  //
  // Make the graph form
  //
  X=X0; Y=Y0;
  W=W0; H=H0;
  CW = Width;
  CH = Height;
  //
  // Get the size of canvas and the panels from the packer.
  //
  Packer(1,NP,&CW,&CH,&W,&H,&X,&Y);
  //
  // Make the toplevel window.  This is the window which will be
  // the visibile window on the screen.  Inside this will be everything
  // else.
  //
  gtk_widget_set_usize(app1,Width,Height>500?500:Height);
  gtk_widget_set_name (app1, "MP RootWin");
  //
  // Make the canvas
  //
  Canvas = gtk_plot_canvas_new(Width, Height);
  gtk_widget_set_name (Canvas, "Canvas");
  //
  // Capture the signal for DnD on data points (basically take 
  // control of data doctoring!).
  //
  gtk_signal_connect(GTK_OBJECT(Canvas),
		     "click_on_point",
		     GTK_SIGNAL_FUNC(ClickOnPlot_handler),
		     (gpointer)this);

  gtk_signal_connect(GTK_OBJECT(app1),
		     "key_press_event",
		     GTK_SIGNAL_FUNC(mp_key_press_handler),
		     (gpointer)this);

  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), 
			    GTK_PLOT_CANVAS_DND_FLAGS);
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), 
			    GTK_PLOT_CANVAS_ALLOCATE_TITLES);
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), 
			    GTK_PLOT_CANVAS_RESIZE_PLOT);
  //
  // Put the GtkPlotCanvas in the scolled window (which has to be
  // got from the toplevel window).
  //
  Layout = Canvas;
  Scroll1=lookup_widget(app1,"scrolledwindow1");
  gtk_container_add(GTK_CONTAINER(Scroll1),Layout);

  gtk_layout_set_size(GTK_LAYOUT(Layout), Width, Height);
  GTK_LAYOUT(Layout)->hadjustment->step_increment = 5;
  GTK_LAYOUT(Layout)->vadjustment->step_increment = 5;
  Show(Title,app1);
  Show(Title,Layout);
  //
  // Make the multipanel form
  //
  if (Which == (unsigned int)ALLPANELS)
    {
      int i;
      for (i=0;i<NP;i++)
	{
	  float CW_M=CW-15, CH_M=CH-50;
	  X=X0; Y=Y0;
	  W=W0; H=H0;
	  Packer(i,NP,&CW_M, &CH_M, &W, &H, &X, &Y);
	  Panels[i].Make(Layout,CW,CH,X+10,Y+5,W,H);
	}
    }
  else
    {
      //assert(Which < (unsigned int)NPanels());
      Panels[Which].Make(Layout,CW-5,CH-50,X0,Y0,Width,Height);
    }
  return Canvas;
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Init(int N,int NPoints, PACKER_CALLBACK_PTR DefaultP)
{
  /*
  if (Canvas) 
    {
      Hide(Canvas);
      for (int i=0;i<NPanels();i++) 
	Panels[i].FreeChart();
      //      fl_delete_formbrowser(BrowserObj,Canvas);
      //      fl_free_form(Canvas);
      //      Canvas=NULL;
    }
  if (TopLevelWin) 
    {
      //      Hide(TopLevelWin);
      //      fl_delete_object(BrowserObj);
      //      fl_free_object(BrowserObj);
      //      fl_free_form(FormBrowser);
      //      FormBrowser=NULL;
    }
  */
  CtrlPanel=NULL;
  NewPlot=1;
  Panels.resize(N);
  X.resize(NPoints);

  for (int i=0;i<N;i++)  Panels[i].Init(NPoints,&X[0]);

  Packer=DefaultP;
}
//
//---------------------------------------------------------------------
//
void MultiPanel::SetCallBack(unsigned int Which,int ObjEnum, GtkSignalFunc CB)
{
  GtkWidget *Obj;

  if (Which == (unsigned int)ALLPANELS)
    for (int i=0;i<NPanels();i++)
      {
	Obj=Panels[i].GetObj(ObjEnum);
	gtk_signal_connect(GTK_OBJECT(Obj), "clicked",
			   CB, NULL);
      }
  else
    {
      //assert(Which < (unsigned int)NPanels());
      Obj=Panels[Which].GetObj(ObjEnum);
      gtk_signal_connect(GTK_OBJECT(Obj), "pressed",
			 CB, NULL);
    }

}
//
//---------------------------------------------------------------------
//
void MultiPanel::Clear(int Which,int WhichOverlay)
{
  if (Which >= NPanels()) 
    throw(ErrorObj("Requested panel does not exits for clearing",
		   "###Error",
		   ErrorObj::Recoverable));
     if (Which == ALLPANELS)
       for(int i=0;i<NPanels();i++) Panels[i].Clear(WhichOverlay);
     else
       Panels[Which].Clear(WhichOverlay);
}
//
//---------------------------------------------------------------------
//
int MultiPanel::SetAttribute(int WhichAttr, int* Val, 
			     int WhichPanel,
			     int WhichOverLay)
{
  if (WhichPanel >= NPanels()) 
    throw(ErrorObj("Requested panel does not exits for SetAttribute",
		   "###Error",
		   ErrorObj::Recoverable));

  HANDLE_EXCEPTIONS
    (
  if (WhichPanel==ALLPANELS)		    
    for (int i=0;i<NPanels();i++)
      Panels[i].SetAttribute(WhichAttr,Val,WhichOverLay);
  else
    Panels[WhichPanel].SetAttribute(WhichAttr,Val,WhichOverLay);
  );
  return 1;
}
//
//---------------------------------------------------------------------
//
int MultiPanel::SetAttribute(int WhichAttr, int Val, 
			     int WhichPanel,
			     int WhichOverLay)
{
  if (WhichPanel >= NPanels()) 
    throw(ErrorObj("Requested panel does not exits for SetAttribute",
		   "###Error",
		   ErrorObj::Recoverable));

  HANDLE_EXCEPTIONS
    (
  if (WhichPanel==ALLPANELS)		    
    for (int i=0;i<NPanels();i++)
      Panels[i].SetAttribute(WhichAttr,Val,WhichOverLay);
  else
    Panels[WhichPanel].SetAttribute(WhichAttr,Val,WhichOverLay);
  );
  return 1;
}
//
//---------------------------------------------------------------------
//
int MultiPanel::SetAttribute(int WhichAttr, gdouble& val0, gdouble& val1,
			     int WhichPanel)
{
  if (WhichPanel >= NPanels()) 
    throw(ErrorObj("Requested panel does not exits for SetAttribute",
		   "###Error",
		   ErrorObj::Recoverable));
  HANDLE_EXCEPTIONS(
  if (WhichPanel==ALLPANELS)		    
    for (int i=0;i<NPanels();i++)
      Panels[i].SetAttribute(WhichAttr,val0,val1);
  else
    Panels[WhichPanel].SetAttribute(WhichAttr,val0,val1);
);
  return 1;
}
//
//---------------------------------------------------------------------
//
int MultiPanel::SetAttribute(int WhichAttr, char *Val0, char *Val1, 
			     int WhichPanel,
			     int WhichOverLay)
{
  if (WhichPanel >= NPanels()) 
    throw(ErrorObj("Requested panel does not exits for SetAttribute",
		   "###Error",
		   ErrorObj::Recoverable));
  HANDLE_EXCEPTIONS(
  if (WhichPanel==ALLPANELS)
    for (int i=0;i<NPanels();i++)
      Panels[i].SetAttribute(WhichAttr,Val0,Val1,WhichOverLay);
  else
      Panels[WhichPanel].SetAttribute(WhichAttr,Val0,Val1,WhichOverLay);
);
  return 1;
}
//
//---------------------------------------------------------------------
//
void MultiPanel::PutText(char* Text, double x, double y, 
			 char *Col,int Angle,int WhichPanel)
{
    if (WhichPanel==ALLPANELS)
	for (int i=0;i<NPanels();i++)
	    Panels[i].PutText(Text,x,y,Col,Angle);
    else
	Panels[WhichPanel].PutText(Text,x,y,Col,Angle);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::DeleteText(char* Text, int WhichPanel)
{
    if (WhichPanel==ALLPANELS)
	for (int i=0;i<NPanels();i++)
	    Panels[i].DeleteText(Text);
    else
	Panels[WhichPanel].DeleteText(Text);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Show(char *Title,GtkWidget *F,int BorderType)
{
  gtk_widget_show(F);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Hide(GtkWidget *Obj)
{
  gtk_widget_hide(Obj);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Redraw(int AutoScale, int WhichPanel,int WhichOverlay)
{
  float Range[2];

  for (int i=0;i<NPanels();i++) Panels[i].freeze();
  if (WhichPanel==ALLPANELS) 
    for(int i=0;i<NPanels();i++)
      {
	//
	// Set the ranges
	//
	if (AutoScale&AUTOSCALEX)
	  {
	    Panels[i].GetRange(Range,0);
	    Panels[i].SetRange(Range,0);
	  }
	if (AutoScale&AUTOSCALEY)
	  {
	    Panels[i].GetRange(Range,1);
	    Panels[i].SetRange(Range,1);
	  }
	//	Panels[i].Plot(WhichOverlay);
      }
  else
    Panels[WhichPanel].Plot(WhichOverlay);
  for (int i=0;i<NPanels();i++) Panels[i].unfreeze();

  //
  // Drawing on the pixmap gets done, in very subtle manner, by the
  // call to SetRange.  draw_pixmap, which copies the pixmap on to 
  // the screen gets done by the following call.
  //
  gtk_plot_layout_paint(GTK_WIDGET(Layout));
}
//
//---------------------------------------------------------------------
//
GtkWidget *MultiPanel::MakeCtrlPanel(gint X0, gint Y0, 
				     gint W0, gint H0, 
				     char *T)
{
  if (CtrlPanel==NULL)
    {
      GtkWidget *obj, *TopLevelMenuBox;
      float Range[2];

      CtrlPanel = gtk_window_new(GTK_WINDOW_TOPLEVEL);
      TopLevelMenuBox = gtk_hbox_new(FALSE,0);

      gtk_signal_connect(GTK_OBJECT(CtrlPanel), "destroy",
			 GTK_SIGNAL_FUNC(cp_done_callback),
			 this);

      gtk_container_add(GTK_CONTAINER(CtrlPanel),TopLevelMenuBox);
      
      gtk_window_set_title(GTK_WINDOW(CtrlPanel),T);
      gtk_widget_set_usize(CtrlPanel,W0,H0);
      gtk_container_border_width(GTK_CONTAINER(CtrlPanel),0);

      Panels[0].GetRange(Range,0);

      Show(T,TopLevelMenuBox);
      Show(T,CtrlPanel);

      obj = gtk_button_new_with_label("Done");
      gtk_widget_set_usize(obj, 40, 20);
      gtk_signal_connect(GTK_OBJECT(obj), "pressed",
			 (GtkSignalFunc) cp_done_callback, 
			 this);
      gtk_box_pack_end(GTK_BOX(TopLevelMenuBox),obj, FALSE, FALSE,0);
      Show(NULL,obj);
      /*
      //
      // XMin slider
      //
      obj = fl_add_valslider(FL_HOR_NICE_SLIDER,
			     10,H0-50,W0-20,20,"XMin");
      fl_set_object_color(obj,FL_INDIANRED, FL_RED);
      fl_set_slider_return(obj, FL_RETURN_CHANGED);
      fl_set_object_callback(obj,mp_xmin_callback,(gpointer)this);

      fl_set_slider_bounds(obj,(double)Range[0],(double)Range[1]);
      fl_set_slider_value(obj,(double)Range[0]);
      //
      // XMax slider
      //
      obj = fl_add_valslider(FL_HOR_NICE_SLIDER,
			     10,H0-80,W0-20,20,"XMax");
      fl_set_object_color(obj,FL_INDIANRED, FL_RED);
      fl_set_slider_return(obj, FL_RETURN_CHANGED);
      fl_set_object_callback(obj,mp_xmax_callback,(gpointer)this);

      fl_set_slider_bounds(obj,(double)Range[0],(double)Range[1]);
      fl_set_slider_value(obj,(double)Range[1]);

      Panels[0].GetRange(Range,1);
      */
      /*
      //
      // YMin slider
      //
      obj = fl_add_valslider(FL_VERT_NICE_SLIDER,
			     10,10,25,H0-110,"YMin");
      fl_set_object_color(obj,FL_INDIANRED, FL_RED);
      fl_set_slider_return(obj, FL_RETURN_CHANGED);
      fl_set_object_callback(obj,mp_ymin_callback,(gpointer)this);

      fl_set_slider_bounds(obj,(double)Range[1],(double)Range[0]);
      fl_set_slider_value(obj,(double)Range[0]);

      obj->u_ldata=(gpointer)this;
      obj->input = 1;
      obj->wantkey=FL_KEY_SPECIAL;
      fl_set_object_prehandler(obj,mp_SliderPreHandlerDown);
      //
      // YMax slider
      //
      obj = fl_add_valslider(FL_VERT_NICE_SLIDER,
			     40,10,25,H0-110,"YMax");
      fl_set_object_color(obj,FL_INDIANRED, FL_RED);
      fl_set_slider_return(obj, FL_RETURN_CHANGED);
      fl_set_object_callback(obj,mp_ymax_callback,(gpointer)this);

      fl_set_slider_bounds(obj,(double)Range[1],(double)Range[0]);
      fl_set_slider_value(obj,(double)Range[1]);
      
      obj->u_ldata=(gpointer)this;
      obj->input = 1;
      obj->wantkey=FL_KEY_SPECIAL;
      fl_set_object_prehandler(obj,mp_SliderPreHandlerUp);
      */
      //
      // "Done" button
      //
    }

  //  Show(T,TopLevelMenuBox);
  Show(T,CtrlPanel);

  return CtrlPanel;
}
//
//----------------------------------------------------------------
// This routine should talk to the Packer to do correct mapping.
// Currently it assumes that the Packer used is a simple vertical
// stack packer.
int MultiPanel::MapPointerToPanel(int& X, int& Y)
{
  double fy;
  gint CanOff=26;
  /*
  gint CanOff=(CtrlButton->allocation.height)+
    Panels[0].GetObj(XYPanel::YMAXSLIDER)->allocation.y;
  */                                                         
  //
  // CanOff is the y-offset from where the Plots start. This begins
  // at the end of the Control Button + the offset at which the
  // first plot is put (which unfortunately has to be gotten from the 
  // x-co-ord. of the slider of the first plot since the gkt_plot_new
  // does not fill in the GtkWidget strutcture's co-ordinates properly).
  //

  fy = (Y-CanOff+
	 gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(Scroll1))
	 ->value);

  /*
  cerr << Y << " " << Y-26+val << " "
       << CtrlButton->allocation.height << " "
       << Panels[0].GetObj(XYPanel::YMAXSLIDER)->allocation.y << " "
       << val << " " 
       << (GTK_LAYOUT(Layout)->height*GTK_PLOT(Panels[0].GetObj())->height)
       << endl;
  */

  fy /= (GTK_LAYOUT(Layout)->height*GTK_PLOT(Panels[0].GetObj())->height);


  return  (int)(fy<Panels.size()?fy:Panels.size()-1);
}
//
//----------------------------------------------------------------
//
int MultiPanel::KeyPressHandler(GtkWidget *Ob, GdkEventKey *event, 
				gpointer data)
{
  int x,y;
  float Range[2];

  gtk_widget_get_pointer(Ob, &x, &y);
  y= x = MapPointerToPanel(x,y);
  Panels[x].GetRange(Range,1,XYPanel::RANGESLIDER);
  //
  // x now has to slider no. in focus
  //  
  switch(event->keyval)
    {
    case GDK_F1: 
      // Reset the range to that of the data
      Panels[y].SetYRangingMode(1);
      Panels[y].GetRange(Range,1);
      Panels[y].SetRange(Range,1);
      break;
    case GDK_F2:
      // Reset the range to that of the value of the sliders
      Range[0] = Panels[x].GetSliderAdj(XYPanel::YMINSLIDERADJ)->value;
      Range[1] = Panels[x].GetSliderAdj(XYPanel::YMAXSLIDERADJ)->value;
      Panels[y].SetYRangingMode(1);
      Panels[y].SetRange(Range,1);
      Panels[y].SetYRangingMode(0);
      break;
    case GDK_F3:
      // Reset the range to huristic values
      cerr << "Huristic ranging not yet possible " << endl;
      break;
    case GDK_Down: 
      // Push the lower limit up by 10%
      Panels[x].SetYRangingMode(1);
      if ((event->state & GDK_SHIFT_MASK) == GDK_SHIFT_MASK)
	{
	  Range[0] = (0.9)*
	    Panels[x].GetSliderAdj(XYPanel::YMINSLIDERADJ)->value;
	  Range[0] = (Range[0]==0.0)?0.1:Range[0];
	  Panels[x].SetRange(Range,1);
	}
      // Push the lower limit down by 10%
      else
	{
	  Range[0] = (1.1)*
	    Panels[x].GetSliderAdj(XYPanel::YMINSLIDERADJ)->value;
	  Range[0] = (Range[0]==0.0)?-0.1:Range[0];
	  Panels[x].SetRange(Range,1);
	}
      Panels[x].SetYRangingMode(0);
      break;
    case GDK_Up:  
      // Push the upper limit down by 10%
      Panels[x].SetYRangingMode(1);
      if ((event->state & GDK_SHIFT_MASK) == GDK_SHIFT_MASK)
	{
	  Range[1] = (0.9)*Panels[x].GetSliderAdj(XYPanel::YMAXSLIDERADJ)->value;
	  Range[1] = (Range[1]==0.0)?-0.1:Range[1];
	  Panels[x].SetRange(Range,1);
	}
      // Push the upper limit up 10%
      else
	{
	  Range[1] = (1.1)*Panels[x].GetSliderAdj(XYPanel::YMAXSLIDERADJ)->value;
	  Range[1] = (Range[1]==0.0)?0.1:Range[1];
	  Panels[x].SetRange(Range,1);
	}
      Panels[x].SetYRangingMode(0);
      break;
    }

  return TRUE;
}
//
//----------------------------------------------------------------
//
void MultiPanel::DoneCallback(GtkWidget *Ob, gpointer data)
{
  //  throw ExitObj();
  exit(0);
}
//----------------------------------------------------------------
int MultiPanel::CPDoneCallback(GtkWidget *Ob, gpointer data)
{
  Hide(CtrlPanel);
  return (TRUE);
}
//
//----------------------------------------------------------------
//
int MultiPanel::CtrlPanelCallback(GtkWidget *Ob, gpointer data)
{
  MakeCtrlPanel(1,1,500,250,"Control Panel");
  return TRUE;
}
//
//----------------------------------------------------------------
//
int MultiPanel::RescaleAllPanelCallback(GtkWidget *Ob, gpointer data)
{
  for (int i=0;i<NPanels();i++)
    {
      gfloat R[2]={0,0};
      Panels[i].SetRange(R,1);
    }

  return TRUE;
}
//
//----------------------------------------------------------------
//
void MultiPanel::SetRange(float R[2], int Axis, int Which)
{
  if (Which==ALLPANELS)
    for(int i=0;i<NPanels();i++) operator[](i).SetRange(R,Axis);
  else                           operator[](Which).SetRange(R,Axis);
}
//
//----------------------------------------------------------------
//
void MultiPanel::GetRange(float R[2], int Axis, int Which)
{
  if (Which==ALLPANELS)
    for(int i=0;i<NPanels();i++) operator[](i).GetRange(R,Axis);
  else                           operator[](Which).GetRange(R,Axis);
}
//
//---------------------------------------------------------------------
// C callbacks which in turn call the C++ object callback.  The
// callback action is ultimately in control of the C++ object.
//
extern "C" 
{
  void mp_done_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->DoneCallback(Ob,data);
    }

  int cp_done_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->CPDoneCallback(Ob,data);
      return  TRUE;
    }

  int mp_ctrlpanel_callback(GtkWidget *Ob, gpointer data)
    {
      return ((MultiPanel *)data)->CtrlPanelCallback(Ob,data);
    }

  int mp_rescale_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->RescaleAllPanelCallback(Ob,data);
      return TRUE;
    }

  int ClickOnPlot_handler(GtkWidget *Ob, gpointer data)
    {
      cerr << "Data doctoring disabled (;-)!" << endl;
      //      return TRUE;
      return 0;
      /*
      int x,y;
      gtk_widget_get_pointer(Ob, &x, &y);
      cerr << x << " " << y << endl;
      y= x = ((MultiPanel *)data)->MapPointerToPanel(x,y);
      cerr << x << " " << y << endl;
      return ((MultiPanel *)data)->Panels->operator[](y).PreHandler(Ob,NULL,data);
      */
    }

  int mp_key_press_handler(GtkWidget *Ob, GdkEventKey *event, gpointer data)
    {
      return ((MultiPanel *)data)->KeyPressHandler(Ob,event,data);
    }

  /*
  void mp_xmax_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->XMaxCallback(Ob,data);
    }

  void mp_xmin_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->XMinCallback(Ob,data);
    }

  void mp_ymax_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->YMaxCallback(Ob,data);
    }

  void mp_ymin_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->YMinCallback(Ob,data);
    }
  */  
};
//
//---------------------------------------------------------------------
//
/*
//----------------------------------------------------------------
void MultiPanel::XMaxCallback(GtkWidget *Ob, gpointer data)
{
  //  float Range[2];
  //  int i;

  operator[](0).GetRange(Range,0);
  
  Range[1] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,0);
}
//----------------------------------------------------------------
void MultiPanel::XMinCallback(GtkWidget *Ob, gpointer data)
{
  float Range[2];
  //  int i;
  
  operator[](0).GetRange(Range,0);
  
}
//----------------------------------------------------------------
void MultiPanel::YMaxCallback(GtkWidget *Ob, gpointer data)
{
  //  float Range[2];
  //  int i;

  operator[](0).GetRange(Range,1);
  
  Range[1] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,1);
}
//----------------------------------------------------------------
void MultiPanel::YMinCallback(GtkWidget *Ob, gpointer data)
{
  //  float Range[2];
  //  int i;

  operator[](0).GetRange(Range,1);
  
  Range[0] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,1);

}
//
//----------------------------------------------------------------
//
#define EPSILON  1E-3
int MultiPanel::MaxSliderMove(GtkWidget *ob, float fac)
{

  double Range[2];
  float R[2];
    
  GetRange(R,1,0);
  Range[1] = (float)fl_get_slider_value(ob);
    
  Range[1] += (Range[1]*fac+EPSILON); 
    
  fl_set_slider_bounds(ob,(double)Range[1],(double)Range[0]);
  fl_set_slider_value(ob,(double)Range[1]);
    
  R[1]=Range[1];

  ((MultiPanel *)(ob->u_ldata))->SetRange(R,1);
    
  return FL_PREEMPT;

  return 0;
}
//
//----------------------------------------------------------------
//
// Reads the focused slider (ob), which is expected to be the Min
// slider, reduces it by a factor (fac) and sets the value and the
// bounds of all the selected Min sliders to this value.
//
int MultiPanel::MinSliderMove(GtkWidget *ob, float fac)
{

  double Range[2];
  float R[2];
    
  GetRange(R,1,0);
  Range[0] = (float)fl_get_slider_value(ob);
    
  Range[0] -= (Range[0]*fac + EPSILON);
    
  fl_set_slider_bounds(ob,(double)Range[1],(double)Range[0]);
  fl_set_slider_value(ob,(double)Range[0]);
    
  R[0]=Range[0];

  ((MultiPanel *)(ob->u_ldata))->SetRange(R,1);
    
  return FL_PREEMPT;

  return 0;
}
*/
