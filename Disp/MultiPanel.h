// -*- C++ -*-
// $Id$
#if !defined(MULTIPANEL_H)
#define      MULTIPANEL_H

#include <forms.h>
#include <XYPanel.h>
#include <vector>
#include <sys/types.h>
#include <unistd.h>
#include <scrollbuf.h>
#include <BitField.h>

typedef void (*PACKER_CALLBACK_PTR) (int n, int np, int *cw, int *ch, 
				     int *w, int *h, int *x, int *y);
typedef void (PACKER_CALLBACK) (int n, int np, int *cw, int *ch, 
				int *w, int *h, int *x, int *y);

extern "C" {
  void mp_done_callback(FL_OBJECT *,long data);
  void cp_done_callback(FL_OBJECT *,long data);
  void mp_ctrlpanel_callback  (FL_OBJECT *,long data);
  void mp_xmax_callback(FL_OBJECT *,long data);
  void mp_xmin_callback(FL_OBJECT *,long data);
  void mp_ymax_callback(FL_OBJECT *,long data);
  void mp_ymin_callback(FL_OBJECT *,long data);
  int mp_SliderPreHandlerDown(FL_OBJECT *ob, int Event,
			      FL_Coord Mx, FL_Coord My,
			      int Key, void *RawEvent);
  int mp_SliderPreHandlerUp(FL_OBJECT *ob, int Event,
			    FL_Coord Mx, FL_Coord My,
			    int Key, void *RawEvent);
  //  int MinSliderMove(FL_OBJECT *ob, float fac);
  //  int MinSliderMove(FL_OBJECT *ob, float fac);
  PACKER_CALLBACK DefaultPacker;

};

class MultiPanel {
public:

  MultiPanel(int argc, char **argv, char *Title=NULL,int N=0,int NPoints=0);
  ~MultiPanel() {};

  void Init(int N,int NPoints, PACKER_CALLBACK_PTR DefaultP=DefaultPacker);

  void Show(char *Title,FL_FORM *F,int BorderType=FL_FULLBORDER);
  void Hide(FL_FORM *F);

  FL_FORM *MakeWindow(unsigned int Which,
		      FL_COORD X0, 
		      FL_COORD Y0,
		      FL_COORD W0, 
		      FL_COORD H0,
		      int Disp=1,
		      char *Title="Plot Tool");

  FL_FORM *MakeCtrlPanel(FL_COORD X0, FL_COORD Y0,
			 FL_COORD W0,  FL_COORD H0,
			 char *Title);
  

  void SetCallBack(unsigned int Which, int Enum, FL_CALLBACKPTR CB);
  void SetCallBack(PACKER_CALLBACK_PTR PC) {Packer = PC;};
  void Redraw(int WhichPanel=ALLPANELS,int WhichOverlay=XYPanel::ALLOVERLAYS);
//
// A generalized SetAttribute method needs to be designed
//
  int SetAttribute(int WhichAttr, int Val, int WhichPanel=ALLPANELS,
		   int WhichOverlay=XYPanel::ALLOVERLAYS);
  int SetAttribute(int WhichAttr, int scale, double base,
		   int WhichPanel=ALLPANELS);
  int SetAttribute(int WhichAttr, char *Val0, char *Val1,
		   int WhichPanel=ALLPANELS,
		   int WhichOverlay=XYPanel::ALLOVERLAYS);
//
// These two routines should be part of the generalized SetAttribute
// method (which has not yet been written!).
//
  void PutText(const char* Text, double x, double y, 
	       FL_COLOR Col,int Align=FL_ALIGN_RIGHT, 
	       int WhichPanel=ALLPANELS);
 void DeleteText(const char* Text, int WhichPanel=ALLPANELS);

  void SetRange(float R[2], int Axis, int Which=ALLPANELS);
  void GetRange(float R[2], int Axis, int Which=ALLPANELS);
  int NPanels() {return Panels.size();}
  void Clear(int Which=ALLPANELS,int WhichOverlay=XYPanel::ALLOVERLAYS);
  XYPanel &operator[](int i) {return Panels[i];};

  //
  // Callbacks
  //
  void DoneCallback(FL_OBJECT *Ob, long data);      //Done button
  void CPDoneCallback(FL_OBJECT *Ob, long data);    //CtrlPanel done button
  void CtrlPanelCallback(FL_OBJECT *Ob, long data); //CtrlPanel button
  void XMaxCallback(FL_OBJECT *Ob, long data);      //XMax slider
  void XMinCallback(FL_OBJECT *Ob, long data);      //XMin slider
  void YMaxCallback(FL_OBJECT *Ob, long data);      //YMax slider
  void YMinCallback(FL_OBJECT *Ob, long data);      //YMin slider
  int MultiPanel::SliderPreHandlerUp(FL_OBJECT *ob, int Event,
				     FL_Coord Mx, FL_Coord My,
				     int Key, void *RawEvent);
  int MultiPanel::SliderPreHandlerDown(FL_OBJECT *ob, int Event,
				     FL_Coord Mx, FL_Coord My,
				     int Key, void *RawEvent);

  int MaxSliderMove(FL_OBJECT *ob, float fac);
  int MinSliderMove(FL_OBJECT *ob, float fac);
		  
  void prtdata()
    {
      int i;
      for (i=0;i<Panels.capacity();i++)
	Panels[i].prtdata();
    };


  enum {ALLPANELS=-1};

private:
  Display               *TheDisplay;
  vector<XYPanel>       Panels;
  FL_FORM               *GraphForm, *FormBrowser, *CtrlPanel;
  FL_OBJECT             *BrowserObj;
  PACKER_CALLBACK_PTR   Packer;
  BitField              ChosenOnes;
  unsigned short        NewPlot;
};

#endif
