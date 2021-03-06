// -*- C++ -*-
// $Id$
#if !defined(MULTIPANEL_H)
#define      MULTIPANEL_H

#include "./WidgetLib.h"
#include <XYPanel.h>
#include <vector>
#include <sys/types.h>
#include <unistd.h>
#include <scrollbuf.h>
#include <BitField.h>

typedef void (*PACKER_CALLBACK_PTR) (int n, int np, gfloat *cw, gfloat *ch, 
				     gfloat *w, gfloat *h, gfloat *x, gfloat *y);
typedef void (PACKER_CALLBACK) (int n, int np, gfloat *cw, gfloat *ch, 
				gfloat *w, gfloat *h, gfloat *x, gfloat *y);

extern "C" {
  void mp_done_callback(GtkWidget *,gpointer data);
  void cp_done_callback(GtkWidget *,gpointer data);
  void mp_ctrlpanel_callback  (GtkWidget *,gpointer data);
  void mp_xmax_callback(GtkWidget *,gpointer data);
  void mp_xmin_callback(GtkWidget *,gpointer data);
  void mp_ymax_callback(GtkWidget *,gpointer data);
  void mp_ymin_callback(GtkWidget *,gpointer data);
  int mp_SliderPreHandlerDown(GtkWidget *ob, int Event,
			      gint Mx, gint My,
			      int Key, void *RawEvent);
  int mp_SliderPreHandlerUp(GtkWidget *ob, int Event,
			    gint Mx, gint My,
			    int Key, void *RawEvent);
  //  int MinSliderMove(GtkWidget *ob, float fac);
  //  int MinSliderMove(GtkWidget *ob, float fac);
  PACKER_CALLBACK DefaultPacker;

};

class MultiPanel {
public:

  MultiPanel(int argc, char **argv, char *Title=NULL,int N=0,int NPoints=0);
  ~MultiPanel() {};

  void Init(int N,int NPoints, PACKER_CALLBACK_PTR DefaultP=DefaultPacker);

  void Show(char *Title,GtkWidget *F,int BorderType=0);
  void Hide(GtkWidget *F);

  GtkWidget *MakeWindow(unsigned int Which,
		      gfloat X0, 
		      gfloat Y0,
		      gfloat W0, 
		      gfloat H0,
		      int Disp=1,
		      char *Title="Plot Tool");

  GtkWidget *MakeCtrlPanel(gint X0, gint Y0,
			 gint W0,  gint H0,
			 char *Title);
  

  void SetCallBack(unsigned int Which, int Enum, GtkSignalFunc CB);
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
  void PutText(char* Text, double x, double y, 
	       gint Col,int Align=0,
	       int WhichPanel=ALLPANELS);
 void DeleteText(char* Text, int WhichPanel=ALLPANELS);

  void SetRange(float R[2], int Axis, int Which=ALLPANELS);
  void GetRange(float R[2], int Axis, int Which=ALLPANELS);
  int NPanels() {return Panels.size();}
  void Clear(int Which=ALLPANELS,int WhichOverlay=XYPanel::ALLOVERLAYS);
  XYPanel &operator[](int i) {return Panels[i];};

  //
  // Callbacks
  //
  void DoneCallback(GtkWidget *Ob, gpointer data);      //Done button
  void CPDoneCallback(GtkWidget *Ob, gpointer data);    //CtrlPanel done button
  void CtrlPanelCallback(GtkWidget *Ob, gpointer data); //CtrlPanel button
  void XMaxCallback(GtkWidget *Ob, gpointer data);      //XMax slider
  void XMinCallback(GtkWidget *Ob, gpointer data);      //XMin slider
  void YMaxCallback(GtkWidget *Ob, gpointer data);      //YMax slider
  void YMinCallback(GtkWidget *Ob, gpointer data);      //YMin slider
  int MultiPanel::SliderPreHandlerUp(GtkWidget *ob, int Event,
				     gint Mx, gint My,
				     int Key, void *RawEvent);
  int MultiPanel::SliderPreHandlerDown(GtkWidget *ob, int Event,
				     gint Mx, gint My,
				     int Key, void *RawEvent);

  int MaxSliderMove(GtkWidget *ob, float fac);
  int MinSliderMove(GtkWidget *ob, float fac);
		  
  void prtdata()
    {
      int i;
      for (i=0;i<(int)Panels.capacity();i++)
	Panels[i].prtdata();
    };


  enum {ALLPANELS=-1};

private:
  //  Display               *TheDisplay;
  vector<XYPanel>       Panels;
  GtkWidget             *TopLevelWin, *TopLevelVBox, *Canvas, *CtrlPanel;
  GtkWidget             *BrowserObj, *Scroll1, *Layout;
  GtkWidget             *QuitButton, *CtrlButton;
  PACKER_CALLBACK_PTR   Packer;
  BitField              ChosenOnes;
  unsigned short        NewPlot;
};

#endif
