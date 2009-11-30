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
  void DefaultPacker(int n, int np, gfloat *cw, gfloat *ch, 
		     gfloat *w, gfloat *h, gfloat *x, gfloat *y) 
  {
    *ch=*h*np;
    *cw=*w;
    *x=0;
    *y=*h*n;
  };
};
//
//---------------------------------------------------------------------
//
MultiPanel::MultiPanel(int argc, char **argv, char *Title,
		       int N,int NPoints) 
{  
  HANDLE_EXCEPTIONS(
      gtk_init(&argc, &argv);
      TopLevelWin = gtk_window_new(GTK_WINDOW_TOPLEVEL);
      gtk_window_set_title(GTK_WINDOW(TopLevelWin), Title);
      gtk_widget_set_usize(TopLevelWin,550,100*N);
      gtk_container_border_width(GTK_CONTAINER(TopLevelWin),0);
      // Make a connection to destory signal

      Canvas=NULL;
      Init(N,NPoints);

      Show(NULL,TopLevelWin,0);
);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Init(int N,int NPoints, PACKER_CALLBACK_PTR DefaultP)
{
  if (Canvas) 
    {
      /*
      Hide(Canvas);
      for (int i=0;i<NPanels();i++) 
	Panels[i].FreeChart();
      //      fl_delete_formbrowser(BrowserObj,Canvas);
      //      fl_free_form(Canvas);
      //      Canvas=NULL;
      */
    }
  if (TopLevelWin) 
    {
      Hide(TopLevelWin);
      //      fl_delete_object(BrowserObj);
      //      fl_free_object(BrowserObj);
      //      fl_free_form(FormBrowser);
      //      FormBrowser=NULL;
    }
  NewPlot=1;
  Panels.resize(N);

  for (int i=0;i<N;i++)  Panels[i].Init(NPoints);

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
      assert(Which < (unsigned int)NPanels());
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
int MultiPanel::SetAttribute(int WhichAttr, int scale, double base,
			     int WhichPanel)
{
  if (WhichPanel >= NPanels()) 
    throw(ErrorObj("Requested panel does not exits for SetAttribute",
		   "###Error",
		   ErrorObj::Recoverable));
  HANDLE_EXCEPTIONS(
  if (WhichPanel==ALLPANELS)		    
    for (int i=0;i<NPanels();i++)
      Panels[i].SetAttribute(WhichAttr,scale,base);
  else
    Panels[WhichPanel].SetAttribute(WhichAttr,scale,base);
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
			 gint Col,int Align,int WhichPanel)
{
    if (WhichPanel==ALLPANELS)
	for (int i=0;i<NPanels();i++)
	    Panels[i].PutText(Text,x,y,Col,Align);
    else
	Panels[WhichPanel].PutText(Text,x,y,Col,Align);
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
void MultiPanel::Redraw(int WhichPanel,int WhichOverlay)
{
  //  fl_freeze_form(Canvas);

  if (WhichPanel==ALLPANELS) 
    for(int i=0;i<NPanels();i++)
      {
	Panels[i].Plot(WhichOverlay);
	Panels[i].YMaxCallback();
	Panels[i].YMinCallback();
      }
  else
    Panels[WhichPanel].Plot(WhichOverlay);

  //  fl_unfreeze_form(Canvas);
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
  //
  // Get the size of canvas and the panels from the packer.
  //
  Packer(1,NP,&CW,&CH,&W,&H,&X,&Y);

  //
  // Make the browser form
  //
  // Restrict the size of the top window to fit on a 1024x768 pixel
  // screen.  Ultimately this limit one should get from the X server.
  //
  //  //  Width = 700;
  //  //  Height = 100*NP;
  //  Width  = GTK_PLOT_LETTER_W ;
  //  Height = GTK_PLOT_LETTER_H ;

  TopLevelVBox = gtk_vbox_new(FALSE,0);
  gtk_container_add(GTK_CONTAINER(TopLevelWin),TopLevelVBox);

  TopLevelMenuBox = gtk_hbox_new(FALSE,0);
  gtk_box_pack_start(GTK_BOX(TopLevelVBox),TopLevelMenuBox,FALSE,TRUE,0);
  gtk_widget_show(TopLevelMenuBox);
  //
  // Make control buttons
  //
  QuitButton = gtk_toggle_button_new_with_label("Done");
  gtk_widget_set_usize(QuitButton, 40,20);
  gtk_signal_connect(GTK_OBJECT(QuitButton), "toggled",
	     (GtkSignalFunc) mp_done_callback, Canvas);
  gtk_box_pack_end(GTK_BOX(TopLevelMenuBox),QuitButton, FALSE, TRUE,0);

  CtrlButton = gtk_toggle_button_new_with_label("CtrlPanel");
  gtk_widget_set_usize(CtrlButton, 60,20);
  gtk_signal_connect(GTK_OBJECT(CtrlButton), "pressed",
		     (GtkSignalFunc) mp_ctrlpanel_callback, Canvas);
  gtk_box_pack_start(GTK_BOX(TopLevelMenuBox),CtrlButton, FALSE, FALSE,0);

  //
  // Make the scroll bars
  //
  Scroll1=gtk_scrolled_window_new(NULL,NULL);
  gtk_container_border_width(GTK_CONTAINER(Scroll1),0);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(Scroll1),
				 GTK_POLICY_ALWAYS,GTK_POLICY_ALWAYS);
  gtk_box_pack_start(GTK_BOX(TopLevelVBox),Scroll1, TRUE, TRUE,0);
  //
  // Make the canvas
  //
  Canvas = gtk_plot_canvas_new(Width, Height);
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), 
			    GTK_PLOT_CANVAS_DND_FLAGS);
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), 
			    GTK_PLOT_CANVAS_ALLOCATE_TITLES);
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), 
			    GTK_PLOT_CANVAS_RESIZE_PLOT);
  Layout = Canvas;
  gtk_container_add(GTK_CONTAINER(Scroll1),Layout);
  gtk_layout_set_size(GTK_LAYOUT(Layout), Width, Height);
  GTK_LAYOUT(Layout)->hadjustment->step_increment = 5;
  GTK_LAYOUT(Layout)->vadjustment->step_increment = 5;

  //
  // Make the multipanel form
  //
  if (Which == (unsigned int)ALLPANELS)
    {
      int i;
      for (i=0;i<NP;i++)
	{
	  X=X0; Y=Y0;
	  W=W0; H=H0;
	  Packer(i,NP,&CW, &CH, &W, &H, &X, &Y);
  cerr << X0 << " " << Y0 << " " << X << " " << Y << endl;
  W=0.9;
  H = 0.9/NP;
  Y=H*i;
  X=0.05;
  cerr << X << " " << Y << " " << W << " " << H << endl;
	  Panels[i].Make(Canvas,X,Y,W,H);
	  //	  Panels[i].Make(Canvas,0.15,0.05,0.15,0.05);
	}
    }
  else
    {
      assert(Which < (unsigned int)NPanels());
      Panels[Which].Make(Canvas,X0,Y0,W0,H0);
      //      Panels[Which].Make(Canvas,X0,Y0,0.15,0.5);
    }
  //
  // Display the browser form (which carries with it the
  // graph form)
  //
  if (Disp) 
    {
      Show(Title,TopLevelWin);
      Show(Title,Scroll1);
      Show(Title,Layout);
      //      Show(Title,Canvas);
      Show(NULL,QuitButton);
      Show(NULL,CtrlButton);
      //      for (int i=0;i<Panels.size();i++) Panels[i].Show();
      Show(Title,TopLevelVBox);
    }
  return Canvas;
}
//
//---------------------------------------------------------------------
//
GtkWidget *MultiPanel::MakeCtrlPanel(gint X0, gint Y0, 
				     gint W0, gint H0, 
				     char *T)
{
  //  GtkWidget *obj;
  //  float Range[2];

  
      /*
  if (CtrlPanel==NULL)
    {
      CtrlPanel = fl_bgn_form(FL_FRAME_BOX, W0, H0);

      obj = fl_add_box(FL_FRAME_BOX,0,0,W0,H0,"");


      Panels[0].GetRange(Range,0);
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
      //
      // "Done" button
      //
      obj = fl_add_button(FL_NORMAL_BUTTON,W0-50,H0-25,40,20,
			  "Button");
      fl_set_object_boxtype(obj,FL_FRAME_BOX);

      fl_set_object_color(obj,FL_TOMATO, FL_RED);
      fl_set_object_callback(obj,cp_done_callback,(gpointer)this);
      fl_set_object_label(obj,"Done");
      fl_set_object_resize(obj,FL_RESIZE_NONE);
      fl_end_form();
    }
      */

  /*
  Show(T,CtrlPanel);//FL_TRANSIENT);
  */
  return (CtrlPanel=NULL);
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
void MultiPanel::CPDoneCallback(GtkWidget *Ob, gpointer data)
{
  Hide(CtrlPanel);
}
//----------------------------------------------------------------
void MultiPanel::CtrlPanelCallback(GtkWidget *Ob, gpointer data)
{
  MakeCtrlPanel(1,1,500,250,"Control Panel");
}
//----------------------------------------------------------------
void MultiPanel::XMaxCallback(GtkWidget *Ob, gpointer data)
{
  //  float Range[2];
  //  int i;
  /*
  
  operator[](0).GetRange(Range,0);
  
  Range[1] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,0);
  */
}
//----------------------------------------------------------------
void MultiPanel::XMinCallback(GtkWidget *Ob, gpointer data)
{
  float Range[2];
  //  int i;
  
  operator[](0).GetRange(Range,0);
  
  /*
  Range[0] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,0);
  */
}
//----------------------------------------------------------------
void MultiPanel::YMaxCallback(GtkWidget *Ob, gpointer data)
{
  //  float Range[2];
  //  int i;
  /*
  operator[](0).GetRange(Range,1);
  
  Range[1] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,1);
  */
}
//----------------------------------------------------------------
void MultiPanel::YMinCallback(GtkWidget *Ob, gpointer data)
{
  //  float Range[2];
  //  int i;
  /*
  operator[](0).GetRange(Range,1);
  
  Range[0] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,1);
  */
}
//----------------------------------------------------------------
//   Event pre-handler to push the upper limit up or down for Max sliders.
//----------------------------------------------------------------
int MultiPanel::SliderPreHandlerUp(GtkWidget *ob, int Event,
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
	  }

      }
  return !FL_PREEMPT;
  */
  return 0;
}
//----------------------------------------------------------------
// Event pre-hanlder to push the lower limit down or up for Min sliders.
//----------------------------------------------------------------
int MultiPanel::SliderPreHandlerDown(GtkWidget *ob, int Event,
				     gint  Mx, gint My,
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
	  case XK_Shift_L: Shift = !Shift;break;
	  case XK_Shift_R: Shift = !Shift;break;
	  }

      }
  return !FL_PREEMPT;
  */
  return 0;
}
//
//----------------------------------------------------------------
//
#define EPSILON  1E-3
int MultiPanel::MaxSliderMove(GtkWidget *ob, float fac)
{
  /*
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
  */
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
  /*
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
  */
  return 0;
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

  void cp_done_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->CPDoneCallback(Ob,data);
    }

  void mp_ctrlpanel_callback(GtkWidget *Ob, gpointer data)
    {
      ((MultiPanel *)data)->CtrlPanelCallback(Ob,data);
    }

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

  int mp_SliderPreHandlerDown(GtkWidget *ob, int Event,
			       gint Mx, gint My,
			       int Key, void *RawEvent)
  {
    //    return ((MultiPanel *)ob->u_ldata)->SliderPreHandlerDown(ob,Event,Mx,My,
    //							     Key,RawEvent);
    return 0;
  }

  int mp_SliderPreHandlerUp(GtkWidget *ob, int Event,
			     gint Mx, gint My,
			     int Key, void *RawEvent)
  {
    //    return ((MultiPanel *)ob->u_ldata)->SliderPreHandlerUp(ob,Event,Mx,My,
    //							   Key,RawEvent);
    return 0;
  }
};
//
//---------------------------------------------------------------------
//
