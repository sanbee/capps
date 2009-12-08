// $Id$
#include <MultiPanel.h>
#include <ErrorObj.h>
#include "./WidgetLib.h"
#include <XYPanel.h>
#include <iostream>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ExitObj.h>
#include "./interface.h"
#include <gdk/gdk.h>
#include <gtk/gtk.h>
#include <glib.h>
#include <namespace.h>

char *ThisPointer;
static GtkItemFactoryEntry menu_items[] = {
  { "/_File",         NULL,         NULL, 0, "<Branch>" },
  { "/File/Rescale all",  "<control>R",  (GtkSignalFunc) mp_rescale_callback, 
    0, NULL },
  { "/File/sep1",     NULL,         NULL, 0, "<Separator>" },
  { "/File/Quit",     "<control>Q",  (GtkSignalFunc) mp_done_callback, 
    0, NULL },
  { "/_Help",         NULL,         NULL, 0, "<Branch>" },
  { "/_Help/About",   NULL,         NULL, 0, "<LastBranch>"}
  //  ,{ "/X",             NULL,         NULL, 0, "<LastBranch>" },
};

void gtk_widget_get_size (GtkWidget *widget, int *pcx, int *pcy)
  {
  GtkArg arg ;
  arg.name = "width" ;
  gtk_widget_get (widget, &arg) ;
  *pcx = GTK_VALUE_INT (arg) ;
  arg.name = "height" ;
  gtk_widget_get (widget, &arg) ;
  *pcy = GTK_VALUE_INT (arg) ;
  }


void get_main_menu( GtkWidget  *window,
                    GtkWidget **menubar )
{
  GtkItemFactory *item_factory;
  GtkAccelGroup *accel_group;

  gint nmenu_items = sizeof (menu_items) / sizeof (menu_items[0]);

  accel_group = gtk_accel_group_new ();

  /* This function initializes the item factory.
     Param 1: The type of menu - can be GTK_TYPE_MENU_BAR, GTK_TYPE_MENU,
              or GTK_TYPE_OPTION_MENU.
     Param 2: The path of the menu.
     Param 3: A pointer to a gtk_accel_group.  The item factory sets up
              the accelerator table while generating menus.
  */

  item_factory = gtk_item_factory_new (GTK_TYPE_MENU_BAR, "<main>", 
				       accel_group);

  /* This function generates the menu items. Pass the item factory,
     the number of items in the array, the array itself, and any
     callback data for the the menu items. */
  gtk_item_factory_create_items (item_factory, nmenu_items, menu_items, NULL);

  /* Attach the new accelerator group to the window. */
  gtk_window_add_accel_group (GTK_WINDOW (window), accel_group);

  if (menubar)
    /* Finally, return the actual menu bar created by the item factory. */ 
    *menubar = gtk_item_factory_get_widget (item_factory, "<main>");
}

extern "C" 
{
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
    *h = *ch*0.98/np;
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
  SetUp(argc, argv, Title, N, NPoints);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::SetUp(int argc, char **argv, char *Title,
		       int N, int NPoints)
{
  HANDLE_EXCEPTIONS(
		    gtk_rc_add_default_file("tst.rc");

		    gtk_init(&argc, &argv);

		    //		    gnome_init ("project1", 0, argc, argv);
		    //		    app1 = create_app1();
		    //		    gtk_widget_show (app1);
		    
		    Canvas=NULL;
		    CtrlPanel=NULL;
		    Init(N,NPoints);
		    );
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Init(int N,int NPoints, PACKER_CALLBACK_PTR DefaultP)
{
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
//      assert(Which < (unsigned int)NPanels());
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
  int Width, Height;
  gtk_widget_get_size(GTK_WIDGET(Layout),&Width, &Height);
  //  cout << "Window size = " << Width << " " << Height << endl;

  for (int i=0;i<NPanels();i++) Panels[i].freeze();
  if (WhichPanel==ALLPANELS) 
    for(int i=0;i<NPanels();i++)
      {
	IterMainLoop();
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
  gtk_layout_set_size(GTK_LAYOUT(Canvas),Width,Height);
  //  gtk_plot_layout_paint(GTK_WIDGET(Canvas));
  gtk_plot_paint(GTK_WIDGET(Panels[0].GetObj()),NULL);
  //  gtk_plot_window_paint((TopLevelWin));
}
//
//---------------------------------------------------------------------
//
GtkWidget* MultiPanel::MakePanels(unsigned int NP, 
				  gfloat X0, gfloat Y0,
				  gfloat W0, gfloat H0,
				  int makeYScrollBars)
{
  gint Height, Width;
  gfloat X,Y,W,H;
  gfloat CW,CH;

  Width = (int)W0;
  Height= (int)H0*NP;
  //
  // Make the graph form
  //
  X=X0; Y=Y0;
  W=W0; H=H0;
  CW = Width;
  CH = Height;

  // Show("Progress", ProgressBar);
  // Timer = gtk_timeout_add (100, progress_timeout, ProgressBar);

  gtk_widget_set_usize(TopLevelWin,Width,Height>500?500:Height);
  Scroll1=gtk_scrolled_window_new(NULL,NULL);
  gtk_widget_set_name (Scroll1, "Surface");
  
  gtk_container_border_width(GTK_CONTAINER(Scroll1),0);
  gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(Scroll1),
				 GTK_POLICY_AUTOMATIC,GTK_POLICY_AUTOMATIC);
  gtk_box_pack_start(GTK_BOX(TopLevelVBox),Scroll1, TRUE, TRUE,0);
  gtk_widget_set_name (TopLevelVBox, "VBox");
  //----------------------Non GTKPLOT CALLS (PURE GTK CALLS)-------------------
  //
  // Make the canvas
  //
  Canvas = gtk_plot_canvas_new(Width, Height);
  gtk_widget_set_name (Canvas, "Canvas");
  //
  // Capture the signal for DnD on data points (basically take 
  // control of data doctoring!).
  //
  // gtk_signal_connect(GTK_OBJECT(Canvas),
  // 		     "click_on_point",
  // 		     GTK_SIGNAL_FUNC(ClickOnPlot_handler),
  // 		     (gpointer)this);

  gtk_signal_connect(GTK_OBJECT(TopLevelWin), "key_press_event",
		     GTK_SIGNAL_FUNC(mp_key_press_handler),
		     (gpointer)this);
  gtk_signal_connect(GTK_OBJECT(Canvas), "select_region_pixel",
                     GTK_SIGNAL_FUNC(SelectRegion_handler), 
		     (gpointer)this);
  
  gtk_widget_add_events(GTK_WIDGET(TopLevelWin), GDK_CONFIGURE);
  //
  // DND on points and plots is fun, but irritating.  Disable those.
  // Retain DND of text, legends and labels.
  //
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), GTK_PLOT_CANVAS_DND_FLAGS);
  GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(Canvas), GTK_PLOT_CANVAS_ALLOCATE_TITLES);
  GTK_PLOT_CANVAS_UNSET_FLAGS(GTK_PLOT_CANVAS(Canvas), GTK_PLOT_CANVAS_CAN_DND_POINT);
  GTK_PLOT_CANVAS_UNSET_FLAGS(GTK_PLOT_CANVAS(Canvas), GTK_PLOT_CANVAS_CAN_MOVE_PLOT);
  GTK_PLOT_CANVAS_UNSET_FLAGS(GTK_PLOT_CANVAS(Canvas), GTK_PLOT_CANVAS_RESIZE_PLOT);

  Layout = Canvas;
  gtk_container_add(GTK_CONTAINER(Scroll1),Layout);

  gtk_layout_set_size(GTK_LAYOUT(Layout), Width, Height);

  // gtk_signal_connect(GTK_OBJECT(TopLevelWin), "configure-event",
  // 		     GTK_SIGNAL_FUNC(configure_handler), (gpointer)(GTK_WIDGET(Canvas)));
  gtk_widget_set_usize(GTK_WIDGET(Canvas),Width,Height);
  
  // gtk_signal_connect(GTK_OBJECT(TopLevelWin), "configure-event",
  // 		     GTK_SIGNAL_FUNC(configure_handler), (gpointer)(GTK_LAYOUT(Layout)));


  
  GTK_LAYOUT(Layout)->hadjustment->step_increment = 5;
  GTK_LAYOUT(Layout)->vadjustment->step_increment = 5;
  Show(NULL,Layout);
  Show(NULL,Canvas);

  //
  // Make the multipanel form
  //
  //  if (Which == (unsigned int)ALLPANELS)
  {
    unsigned int i;
    for (i=0;i<NP;i++)
      {
	//	while (g_main_context_iteration(NULL, FALSE));
	//	while (g_main_iteration(TRUE));
	IterMainLoop();
	float CW_M=CW-15, CH_M=CH-50;
	X=X0; Y=Y0;
	W=W0; H=H0;
	Packer(i,NP,&CW_M, &CH_M, &W, &H, &X, &Y);
	Panels[i].Make(Layout,(gint)CW,(gint)CH,(gint)(X+10),(gint)(Y+5),(gint)(W),(gint)H,
		       makeYScrollBars);
      }
  }
  Show(NULL,Scroll1);
  // else
  //   Panels[Which].Make(Layout,(gint)(CW-5),(gint)(CH-50),(gint)X0,(gint)Y0,Width,Height,
  // 		       makeYScrollBars);
  return Canvas;
}
//
//---------------------------------------------------------------------
//
GtkWidget *MultiPanel::MakeWindow(unsigned int Which,
				  gfloat X0, float Y0,
				  gfloat W0, gfloat H0,
				  int Disp, char *Title,
				  int makeYScrollBars)
{
  gint Height, Width;
  gfloat X,Y,W,H;
  gfloat CW,CH;
  int NP;

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
  TopLevelWin = gtk_window_new(GTK_WINDOW_TOPLEVEL);

  gtk_widget_set_name (TopLevelWin, "MP RootWin");

  gtk_window_set_title(GTK_WINDOW(TopLevelWin), Title);
  //  gtk_widget_set_usize(TopLevelWin,Width,Height>500?500:Height);
  //  gtk_widget_set_usize(TopLevelWin,Width,500);

  gtk_container_border_width(GTK_CONTAINER(TopLevelWin),0);
  Show(NULL,TopLevelWin,0);

  //----------------------Non GTKPLOT CALLS (PURE GTK CALLS)-------------------
  // Make the scrolled window
  //
  TopLevelVBox = gtk_vbox_new(FALSE,0);
  ProgressBar = gtk_progress_bar_new();
  gtk_progress_set_activity_mode(GTK_PROGRESS(ProgressBar),TRUE);
  gtk_progress_bar_set_bar_style(GTK_PROGRESS_BAR(ProgressBar),GTK_PROGRESS_CONTINUOUS);
  gtk_progress_set_format_string(GTK_PROGRESS(ProgressBar),"Drawing");
  gtk_progress_set_show_text(GTK_PROGRESS(ProgressBar),TRUE);

  HBox = gtk_hbox_new(FALSE,0);
  gtk_container_add(GTK_CONTAINER(TopLevelWin),TopLevelVBox);

  gtk_box_pack_start(GTK_BOX(TopLevelVBox),HBox,FALSE,TRUE,0);
  //
  //------------------------------------------------------------------
  // Make menu using the ItemFactory.
  //
  ThisPointer=(char *)this;
  get_main_menu (TopLevelWin, &menubar);

  gtk_box_pack_start (GTK_BOX (HBox), menubar, FALSE, TRUE, 0);
  //------------------------------------------------------------------
  gtk_box_pack_end(GTK_BOX(HBox),ProgressBar,FALSE,TRUE,0);

  Show(Title,TopLevelVBox);
  Show(Title, HBox);
  Show(Title,menubar);
  //  Show(Title, ProgressBar);
  Show(Title,TopLevelWin);
  //  Show(Title,TopLevelMenuBox);
  //
  if (Disp) 
    {
      //      Show(Title,Scroll1);
      //      Show(NULL,QuitButton);
      //      Show(NULL,CtrlButton);
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
int MultiPanel::MapPointerToPanel(int X, int Y, bool isWindowPixelLocation)
{
  double fy;
  //  gint CanOff=26;

  /*
  CanOff= (menubar->allocation.height)+ 
    Panels[0].GetObj(XYPanel::YMAXSLIDER)->allocation.y;
  */
  //
  // CanOff is the y-offset from where the Plots start. This begins
  // at the end of the Control Button + the offset at which the
  // first plot is put (which unfortunately has to be gotten from the 
  // x-co-ord. of the slider of the first plot since the gkt_plot_new
  // does not fill in the GtkWidget strutcture's co-ordinates properly).
  //

  /*
  fy = (Y-CanOff+
	 gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(Scroll1))
	 ->value);
  */

  
  //  fy = (Y-CanOff)/(GTK_LAYOUT(Layout)->height*GTK_PLOT(Panels[0].GetObj())->height);
  if (!isWindowPixelLocation) Y+=menubar->allocation.height;
  fy=((Y-((menubar->allocation.height)+ 
       Panels[0].GetObj(XYPanel::YMAXSLIDER)->allocation.y)) /
      (GTK_LAYOUT(Layout)->height*GTK_PLOT(Panels[0].GetObj())->height));

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
      Panels[y].SetYRangingMode(0);
      break;
    case GDK_F2:
      // Reset the range to that of the value of the sliders
      Range[0] = Panels[x].GetSliderAdj(XYPanel::YMINSLIDERADJ)->value;
      Range[1] = Panels[x].GetSliderAdj(XYPanel::YMAXSLIDERADJ)->value;
      Panels[y].SetYRangingMode(1);
      Panels[y].SetRange(Range,1);
      gtk_signal_emit_by_name (GTK_OBJECT(Panels[x].
					  GetSliderAdj(XYPanel::YMAXSLIDERADJ)),
			       "changed");
      gtk_signal_emit_by_name (GTK_OBJECT(Panels[x].
					  GetSliderAdj(XYPanel::YMINSLIDERADJ)),
			       "changed");
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
  gtk_main_quit();
  //  exit(0);
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
//----------------------------------------------------------------------
//
int MultiPanel::RescaleAllPanelCallback(GtkWidget *Ob, gpointer data)
{
  gfloat R[2]={0,0};
  //  EnableProgressMeter();
  FreezeDisplay();
  for (int i=0;i<NPanels();i++)
    {
      //      IterMainLoop();
      Panels[i].SetYRangingMode(1); Panels[i].SetXRangingMode(1);
      Panels[i].GetRange(R,0);      Panels[i].SetRange(R,0);
      Panels[i].GetRange(R,1);      Panels[i].SetRange(R,1);
      Panels[i].SetYRangingMode(0); Panels[i].SetXRangingMode(0);
    }
  UnFreezeDisplay();
  //  DisableProgressMeter();

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
//----------------------------------------------------------------
//
void MultiPanel::EnableProgressMeter() 
{
  Timer = gtk_timeout_add (100, progress_timeout, ProgressBar);
  Show("Progress",GTK_WIDGET(ProgressBar));
}
//
//----------------------------------------------------------------
//
void MultiPanel::DisableProgressMeter() 
{ 
  gtk_timeout_remove(Timer);
  GtkAdjustment *adj;
  adj = GTK_PROGRESS (ProgressBar)->adjustment;
  gtk_progress_set_value (GTK_PROGRESS (ProgressBar), 0.0);
  Hide(GTK_WIDGET(ProgressBar));
};
//
//---------------------------------------------------------------------
// C callbacks which in turn call the C++ object callback.  The
// callback action is ultimately in control of the C++ object.
//
extern "C" 
{
  void mp_done_callback(GtkWidget *Ob, gpointer data)
    {
      //      ((MultiPanel *)data)->DoneCallback(Ob,data);
      ((MultiPanel *)ThisPointer)->DoneCallback(Ob,data);
    }

  int cp_done_callback(GtkWidget *Ob, gpointer data)
    {
      //      ((MultiPanel *)data)->CPDoneCallback(Ob,data);
      ((MultiPanel *)ThisPointer)->CPDoneCallback(Ob,data);
      return  TRUE;
    }

  int mp_ctrlpanel_callback(GtkWidget *Ob, gpointer data)
    {
      //      return ((MultiPanel *)data)->CtrlPanelCallback(Ob,data);
      return ((MultiPanel *)ThisPointer)->CtrlPanelCallback(Ob,data);
    }

  int mp_rescale_callback(GtkWidget *Ob, gpointer data)
    {
      //      ((MultiPanel *)data)->RescaleAllPanelCallback(Ob,data);
      ((MultiPanel *)ThisPointer)->RescaleAllPanelCallback(Ob,data);
      return TRUE;
    }

  int ClickOnPlot_handler(GtkWidget *Ob, gpointer data)
    {
      //      return TRUE;
      /*
      int x,y;
      gtk_widget_get_pointer(Ob, &x, &y);
      cerr << x << " " << y << endl;
      y= x = ((MultiPanel *)data)->MapPointerToPanel(x,y);
      cerr << x << " " << y << endl;
      //      return ((MultiPanel *)data)->Panels->operator[](y).PreHandler(Ob,NULL,data);
      */
      cerr << "Data doctoring disabled (;-)!" << endl;
      return 0;
    }

  int mp_key_press_handler(GtkWidget *Ob, GdkEventKey *event, gpointer data)
    {
      //      return ((MultiPanel *)data)->KeyPressHandler(Ob,event,data);
      return ((MultiPanel *)ThisPointer)->KeyPressHandler(Ob,event,data);
    }

  int configure_handler(GtkWidget *Ob, GdkEvent *event, gpointer data)
  {
    int x, y, w, h;
    x = event->configure.x;
    y = event->configure.y;
    w = event->configure.width;
    h = event->configure.height;
    //    gtk_layout_set_size(GTK_LAYOUT((GtkLayout *)(data)), w, h);
    gtk_widget_set_usize(GTK_WIDGET((GtkWidget *)data),w,h);
    //    gtk_plot_resize(gtk_plot_canvas_get_active_plot(GTK_PLOT_CANVAS((GtkWidget *)(data))),w,h);
    ((MultiPanel *)ThisPointer)->Redraw();
    cerr << "X , Y = " << x << " " << y << " " << w << " " << h << endl;
    return TRUE;
  }

  int SelectRegion_handler(GtkPlotCanvas *canvas,
			   gint cxoff, gint cyoff,
			   gint panelx0, gint panely0,
			   gint panelx1, gint panely1,
			   gdouble x1, gdouble x2,
			   gdouble y2, gdouble y1,
			   gpointer data) 
  {
    int cx0=panelx0+cxoff, cy0=panely0+cyoff,cx1=panelx1+cxoff,cy1=panely1+cyoff;
    int PanelNumber0 = ((MultiPanel*)data)->MapPointerToPanel(cx0,cy0,FALSE),
      PanelNumber1 = ((MultiPanel*)data)->MapPointerToPanel(cx1,cy1,FALSE);
    cout << "Region = " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
    // cout << "Panel pixel = " << panelx0 << " " << panely0 << endl;
    // cout << "Panel pixel = " << panelx1 << " " << panely1 << endl;
    cout << "Panels = " << PanelNumber0 << " " << PanelNumber1 << endl;
    gfloat Range[2];
    ((MultiPanel*)data)->operator[](PanelNumber0).SetXRangingMode(1); 
    ((MultiPanel*)data)->operator[](PanelNumber0).SetYRangingMode(1);
    Range[0]=x1;Range[1]=x2;    ((MultiPanel *)data)->operator[](PanelNumber0).SetRange(Range,0);
    Range[0]=y1;Range[1]=y2;    ((MultiPanel *)data)->operator[](PanelNumber0).SetRange(Range,1);
    ((MultiPanel*)data)->operator[](PanelNumber0).SetXRangingMode(0); 
    ((MultiPanel*)data)->operator[](PanelNumber0).SetYRangingMode(0);
    //cout << "TLC Panel =" <<  ((MultiPanel *)data)->MapPointerToPanel(x1,y1) << endl;
    //cout << "BRC Panel =" <<  ((MultiPanel *)data)->MapPointerToPanel(x2,y2) << endl;
    return TRUE;
  }
  // Update the value of the progress bar so that we get
  // some movement 

  gint progress_timeout( gpointer data )
  {
    gfloat new_val;
    GtkAdjustment *adj;

    /* Calculate the value of the progress bar using the
     * value range set in the adjustment object */

    new_val = gtk_progress_get_value( GTK_PROGRESS(data) ) + 1;

    adj = GTK_PROGRESS (data)->adjustment;
    if (new_val > adj->upper)
      new_val = adj->lower;

    /* Set the new value */
    gtk_progress_set_value (GTK_PROGRESS (data), new_val);

    /* As this is a timeout function, return TRUE so that it
     * continues to get called */
    return(TRUE);
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

