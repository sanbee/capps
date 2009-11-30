// $Id$
#include <MultiPanel.h>
#include <ErrorObj.h>
#include <forms.h>
#include <XYPanel.h>
#include <iostream.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ExitObj.h>
extern "C" 
{
  void DefaultPacker(int n, int np, int *cw, int *ch, 
		     int *w, int *h, int *x, int *y) 
  {
    *ch=*h*np;
    //   *ch=*ch>700?700:*ch;
    //    *cw =*w>1000?1000:*w;
    *cw=*w;
    *x=0;*y=*h*n;
  };
};
//
//---------------------------------------------------------------------
//
MultiPanel::MultiPanel(int argc, char **argv, char *Title,
		       int N,int NPoints) 
{  
  HANDLE_EXCEPTIONS(
      if ((TheDisplay = fl_initialize(&argc, argv, Title, 0,0))==NULL)
	throw(ErrorObj("###Error","Could not open display",
		       ErrorObj::Recoverable));
      GraphForm=NULL;
      FormBrowser=NULL;
      CtrlPanel=NULL;
      Init(N,NPoints);
);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Init(int N,int NPoints, PACKER_CALLBACK_PTR DefaultP)
{
  if (GraphForm) 
    {
      Hide(GraphForm);
      for (int i=0;i<NPanels();i++) 
	Panels[i].FreeChart();
      fl_delete_formbrowser(BrowserObj,GraphForm);
      fl_free_form(GraphForm);
      GraphForm=NULL;
    }
  if (FormBrowser) 
    {
      Hide(FormBrowser);
      fl_delete_object(BrowserObj);
      fl_free_object(BrowserObj);
      fl_free_form(FormBrowser);
      FormBrowser=NULL;
    }
  NewPlot=1;
  Panels.resize(N);

  for (int i=0;i<N;i++)  Panels[i].Init(NPoints);

  Packer=DefaultP;
}
//
//---------------------------------------------------------------------
//
void MultiPanel::SetCallBack(unsigned int Which,int ObjEnum,FL_CALLBACKPTR CB)
{
  FL_OBJECT *Obj;

  if (Which == ALLPANELS)
    for (unsigned int i=0;i<NPanels();i++)
      {
      Obj=Panels[i].GetObj(ObjEnum);
      fl_set_object_callback(Obj,CB,0);
    }
  else
    {
      assert(Which < NPanels());
      Obj=Panels[Which].GetObj(ObjEnum);
      fl_set_object_callback(Obj,CB,0);
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
}
//
//---------------------------------------------------------------------
//
void MultiPanel::PutText(const char* Text, double x, double y, 
			 FL_COLOR Col,int Align,int WhichPanel)
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
void MultiPanel::DeleteText(const char* Text, int WhichPanel)
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
void MultiPanel::Show(char *Title,FL_FORM *F,int BorderType)
{
  fl_show_form(F,FL_PLACE_FREE,BorderType,Title);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Hide(FL_FORM *F)
{
  fl_hide_form(F);
}
//
//---------------------------------------------------------------------
//
void MultiPanel::Redraw(int WhichPanel,int WhichOverlay)
{
  fl_freeze_form(GraphForm);

  if (WhichPanel==ALLPANELS) 
    for(int i=0;i<NPanels();i++)
      {
	Panels[i].Plot(WhichOverlay);
	Panels[i].YMaxCallback();
	Panels[i].YMinCallback();
      }
  else
    Panels[WhichPanel].Plot(WhichOverlay);

  fl_unfreeze_form(GraphForm);
}
//
//---------------------------------------------------------------------
//
FL_FORM *MultiPanel::MakeWindow(unsigned int Which,
				FL_COORD X0, FL_COORD Y0,
				FL_COORD W0, FL_COORD H0,
				int Disp, char *Title)
{
  int Height, Width,YSize;
  FL_OBJECT *obj;
  int X,Y,W,H;
  int CW,CH;
  int fd,NP;

  NP=NPanels();

  Height=H0*NP;
  fl_set_border_width(1);
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
  Height=CH>700?700:CH;  
  Width =CW>1000?1000:CW;
  //  if (FormBrowser==NULL)
  //  {
    FormBrowser = fl_bgn_form(FL_FRAME_BOX, Width, Height);
    BrowserObj = fl_add_formbrowser(FL_NORMAL_FORMBROWSER,
				    0,0,Width,Height-30,"");
    fl_set_object_boxtype(BrowserObj,FL_FRAME_BOX);
    fl_set_object_color(BrowserObj,FL_COL1,FL_MCOL);
    fl_set_object_lalign(BrowserObj,FL_ALIGN_CENTER);
  
    //
    // Make control buttons
    //
    obj = fl_add_button(FL_NORMAL_BUTTON,Width-50,Height-25,40,20,"Button");
          fl_set_object_color(obj,FL_TOMATO, FL_RED);
          fl_set_object_boxtype(obj,FL_FRAME_BOX);
          fl_set_object_callback(obj,mp_done_callback,(long)this);
          fl_set_object_label(obj,"Done");
	  fl_set_object_resize(obj,FL_RESIZE_NONE);
  
	  YSize = 60;
  
    obj = fl_add_button(FL_NORMAL_BUTTON,Width-YSize-60,Height-25,YSize,20,
		        "Button");
          fl_set_object_color(obj,FL_SLATEBLUE, FL_RED);
          fl_set_object_boxtype(obj,FL_FRAME_BOX);
          fl_set_object_callback(obj,mp_ctrlpanel_callback,(long)this);
          fl_set_object_label(obj,"CtrlPanel");
	  fl_set_object_resize(obj,FL_RESIZE_NONE);
  
  
    fl_end_form();
    //
    // Make the multipanel form
    //
    //    fl_get_formbrowser_area(BrowserObj,&X,&Y,&CW,&CH);
    GraphForm = fl_bgn_form(FL_FRAME_BOX,CW,CH);
  
    if (Which == ALLPANELS)
      {
	unsigned int i;
	for (i=0;i<NP;i++)
	  {
	    X=X0; Y=Y0;
	    W=W0; H=H0;
	    Packer(i,NP,&CW, &CH, &W, &H, &X, &Y);
	    Panels[i].Make(GraphForm,X,Y,W,H);
	  }
      }
    else
      {
	assert(Which < NPanels());
	Panels[Which].Make(GraphForm,X0,Y0,W0,H0);
      }
    //  fl_set_form_size(GraphForm,W0,Height);
    //  fl_set_form_size(GraphForm,CW,CH);
    fl_end_form();
    //
    // Add the graph form to the browser form
    //
    fl_addto_formbrowser(BrowserObj, GraphForm);
    //
    // Display the browser form (which carries with it the
    // graph form)
    //
    if (Disp) Show(Title,FormBrowser);
    return FormBrowser;
}
//
//---------------------------------------------------------------------
//
FL_FORM *MultiPanel::MakeCtrlPanel(FL_COORD X0, FL_COORD Y0, 
				   FL_COORD W0, FL_COORD H0, 
				   char *T)
{
  FL_OBJECT *obj;
  float Range[2];

  
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
      fl_set_object_callback(obj,mp_xmin_callback,(long)this);

      fl_set_slider_bounds(obj,(double)Range[0],(double)Range[1]);
      fl_set_slider_value(obj,(double)Range[0]);
      //
      // XMax slider
      //
      obj = fl_add_valslider(FL_HOR_NICE_SLIDER,
			     10,H0-80,W0-20,20,"XMax");
      fl_set_object_color(obj,FL_INDIANRED, FL_RED);
      fl_set_slider_return(obj, FL_RETURN_CHANGED);
      fl_set_object_callback(obj,mp_xmax_callback,(long)this);

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
      fl_set_object_callback(obj,mp_ymin_callback,(long)this);

      fl_set_slider_bounds(obj,(double)Range[1],(double)Range[0]);
      fl_set_slider_value(obj,(double)Range[0]);

      obj->u_ldata=(long)this;
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
      fl_set_object_callback(obj,mp_ymax_callback,(long)this);

      fl_set_slider_bounds(obj,(double)Range[1],(double)Range[0]);
      fl_set_slider_value(obj,(double)Range[1]);
      
      obj->u_ldata=(long)this;
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
      fl_set_object_callback(obj,cp_done_callback,(long)this);
      fl_set_object_label(obj,"Done");
      fl_set_object_resize(obj,FL_RESIZE_NONE);
      fl_end_form();
    }

  Show(T,CtrlPanel);//FL_TRANSIENT);
  return CtrlPanel;
}
//
//----------------------------------------------------------------
//
void MultiPanel::DoneCallback(FL_OBJECT *Ob, long data)
{
  //  throw ExitObj();
  exit(0);
}
//----------------------------------------------------------------
void MultiPanel::CPDoneCallback(FL_OBJECT *Ob, long data)
{
  fl_hide_form(CtrlPanel);
}
//----------------------------------------------------------------
void MultiPanel::CtrlPanelCallback(FL_OBJECT *Ob, long data)
{
  MakeCtrlPanel(1,1,500,250,"Control Panel");
}
//----------------------------------------------------------------
void MultiPanel::XMaxCallback(FL_OBJECT *Ob, long data)
{
  float Range[2];
  int i;
  
  operator[](0).GetRange(Range,0);
  
  Range[1] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,0);
}
//----------------------------------------------------------------
void MultiPanel::XMinCallback(FL_OBJECT *Ob, long data)
{
  float Range[2];
  int i;
  
  operator[](0).GetRange(Range,0);
  
  Range[0] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,0);
}
//----------------------------------------------------------------
void MultiPanel::YMaxCallback(FL_OBJECT *Ob, long data)
{
  float Range[2];
  int i;
  
  operator[](0).GetRange(Range,1);
  
  Range[1] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,1);
}
//----------------------------------------------------------------
void MultiPanel::YMinCallback(FL_OBJECT *Ob, long data)
{
  float Range[2];
  int i;
  
  operator[](0).GetRange(Range,1);
  
  Range[0] = (float)fl_get_slider_value(Ob);
  for (i=0;i<NPanels();i++)
    operator[](i).SetRange(Range,1);
}
//----------------------------------------------------------------
//   Event pre-handler to push the upper limit up or down for Max sliders.
//----------------------------------------------------------------
int MultiPanel::SliderPreHandlerUp(FL_OBJECT *ob, int Event,
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
	  }

      }
  return !FL_PREEMPT;
}
//----------------------------------------------------------------
// Event pre-hanlder to push the lower limit down or up for Min sliders.
//----------------------------------------------------------------
int MultiPanel::SliderPreHandlerDown(FL_OBJECT *ob, int Event,
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
	  case XK_Shift_L: Shift = !Shift;break;
	  case XK_Shift_R: Shift = !Shift;break;
	  }

      }
  return !FL_PREEMPT;
}
//
//----------------------------------------------------------------
//
#define EPSILON  1E-3
int MultiPanel::MaxSliderMove(FL_OBJECT *ob, float fac)
{
  double Range[2];
  float R[2];
    
  GetRange(R,1,0);
  Range[1] = (float)fl_get_slider_value(ob);/* Read the foucsed slider*/
    
  Range[1] += (Range[1]*fac+EPSILON); /* Modify the value */
    
  fl_set_slider_bounds(ob,(double)Range[1],(double)Range[0]);
  fl_set_slider_value(ob,(double)Range[1]);
    
  R[1]=Range[1];

  ((MultiPanel *)(ob->u_ldata))->SetRange(R,1);
    
  return FL_PREEMPT;
}
//
//----------------------------------------------------------------
//
// Reads the focused slider (ob), which is expected to be the Min
// slider, reduces it by a factor (fac) and sets the value and the
// bounds of all the selected Min sliders to this value.
//
int MultiPanel::MinSliderMove(FL_OBJECT *ob, float fac)
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
  operator[](Which).GetRange(R,Axis);
}
//
//---------------------------------------------------------------------
// C callbacks which in turn call the C++ object callback.  The
// callback action is ultimately in control of the C++ object.
//
extern "C" 
{
  void mp_done_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->DoneCallback(Ob,data);
    }

  void cp_done_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->CPDoneCallback(Ob,data);
    }

  void mp_ctrlpanel_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->CtrlPanelCallback(Ob,data);
    }

  void mp_xmax_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->XMaxCallback(Ob,data);
    }

  void mp_xmin_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->XMinCallback(Ob,data);
    }

  void mp_ymax_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->YMaxCallback(Ob,data);
    }

  void mp_ymin_callback(FL_OBJECT *Ob, long data)
    {
      ((MultiPanel *)data)->YMinCallback(Ob,data);
    }

  int mp_SliderPreHandlerDown(FL_OBJECT *ob, int Event,
			       FL_Coord Mx, FL_Coord My,
			       int Key, void *RawEvent)
  {
    return ((MultiPanel *)ob->u_ldata)->SliderPreHandlerDown(ob,Event,Mx,My,
							     Key,RawEvent);
  }

  int mp_SliderPreHandlerUp(FL_OBJECT *ob, int Event,
			     FL_Coord Mx, FL_Coord My,
			     int Key, void *RawEvent)
  {
    return ((MultiPanel *)ob->u_ldata)->SliderPreHandlerUp(ob,Event,Mx,My,
							   Key,RawEvent);
  }
};
//
//---------------------------------------------------------------------
//
