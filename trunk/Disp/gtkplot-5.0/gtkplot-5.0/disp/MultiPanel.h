// -*- C++ -*-
// $Id$
#if !defined(MULTIPANEL_H)
#define      MULTIPANEL_H

#include "./WidgetLib.h"
#include "../gtkplotps.h"
#include <XYPanel.h>
#include <vector>
#include <sys/types.h>
#include <unistd.h>
#include <scrollbuf.h>
#include <BitField.h>
#include <namespace.h>
#include <MPPPacker.h>
typedef void (*PACKER_CALLBACK_PTR) (int n, int np, gfloat *cw, gfloat *ch, 
				     gfloat *w, gfloat *h, gfloat *x, gfloat *y);
typedef void (PACKER_CALLBACK) (int n, int np, gfloat *cw, gfloat *ch, 
				gfloat *w, gfloat *h, gfloat *x, gfloat *y);

#define AUTOSCALEX 1
#define AUTOSCALEY 2
extern "C" {
  void mp_done_callback(GtkWidget *,gpointer data);
  int  cp_done_callback(GtkWidget *,gpointer data);
  int  mp_ctrlpanel_callback(GtkWidget *,gpointer data);
  int  mp_rescale_callback(GtkWidget *,gpointer data);
  void mp_xmax_callback(GtkWidget *,gpointer data);
  void mp_xmin_callback(GtkWidget *,gpointer data);
  void mp_ymax_callback(GtkWidget *,gpointer data);
  void mp_ymin_callback(GtkWidget *,gpointer data);
  int ClickOnPlot_handler(GtkWidget *,gpointer data);
  int mp_key_press_handler(GtkWidget *,GdkEventKey *event, gpointer data);
  int configure_handler(GtkWidget *,GdkEvent *event, gpointer data);
  int SelectRegion_handler(GtkPlotCanvas *canvas,
			   gint cxoff, gint cyoff,
			   gint panelx0, gint panely0,
			   gint panelx1, gint panely1,
			   gdouble x1, gdouble y1,
			   gdouble x2, gdouble y2,
			   gpointer data) ;
  gint progress_timeout( gpointer data );
  //  int MinSliderMove(GtkWidget *ob, float fac);
  //  int MinSliderMove(GtkWidget *ob, float fac);
};

class MultiPanel {
public:

  MultiPanel() {};
  MultiPanel(int argc, char **argv, char *Title=NULL,int N=0,int NPoints=0);
  ~MultiPanel() {};

  void SetUp(int argc, char **argv, char *Title=NULL, int N=0, int NPoints=0);
  void Init(int N,int NPoints);
  void SetDefaults();

  void Show(char *Title,GtkWidget *F,int BorderType=0);
  void Hide(GtkWidget *F);

  GtkWidget *MakeWindow(unsigned int Which,
			gfloat X0, float Y0,
			gfloat W0, gfloat H0,
			int Disp=1, char *Title="Plot Tool",
			int makeYScrollBars=1);

  GtkWidget* MakePanels(unsigned int NP, 
			gfloat X0, gfloat Y0,
			gfloat W0, gfloat H0,
			int makeYScrollBars=1);
  GtkWidget *MakeCtrlPanel(gint X0, gint Y0,
			 gint W0,  gint H0,
			 char *Title);
  

  void SetCallBack(unsigned int Which, int Enum, GtkSignalFunc CB);
  void SetPacker(MPPPacker& PC) {PackerObj = PC;}
  void Redraw(int Mode=AUTOSCALEX|AUTOSCALEY,int WhichPanel=ALLPANELS,
	      int WhichOverlay=XYPanel::ALLOVERLAYS);
//
// A generalized SetAttribute method needs to be designed
//
  int SetAttribute(int WhichAttr, int *Val, int WhichPanel=ALLPANELS,
		   int WhichOverlay=XYPanel::ALLOVERLAYS);
  int SetAttribute(int WhichAttr, int Val, int WhichPanel=ALLPANELS,
		   int WhichOverlay=XYPanel::ALLOVERLAYS);
  int SetAttribute(int WhichAttr, gdouble &val0, gdouble &val1,
		   int WhichPanel=ALLPANELS);
  int SetAttribute(int WhichAttr, char *Val0, char *Val1,
		   int WhichPanel=ALLPANELS,
		   int WhichOverlay=XYPanel::ALLOVERLAYS);
//
// These two routines should be part of the generalized SetAttribute
// method (which has not yet been written!).
//
  void PutText(char* Text, double x, double y, 
	       char *Col=NULL,int Align=0,
	       int WhichPanel=ALLPANELS);
 void DeleteText(char* Text, int WhichPanel=ALLPANELS);

  void SetRange(float R[2], int Axis, int Which=ALLPANELS);
  void GetRange(float R[2], int Axis, int Which=ALLPANELS);
  inline int NPanels() {return Panels.size();}
  void Clear(int Which=ALLPANELS,int WhichOverlay=XYPanel::ALLOVERLAYS);
  inline XYPanel &operator[](int i) {return Panels[i];};

  //
  // Callbacks
  //
  void DoneCallback(GtkWidget *Ob, gpointer data);      //Done button
  int  CPDoneCallback(GtkWidget *Ob, gpointer data);    //CtrlPanel done button
  int  CtrlPanelCallback(GtkWidget *Ob, gpointer data); //CtrlPanel button
  int  RescaleAllPanelCallback(GtkWidget *Ob, gpointer data); //CtrlPanel button
  void XMaxCallback(GtkWidget *Ob, gpointer data);      //XMax slider
  void XMinCallback(GtkWidget *Ob, gpointer data);      //XMin slider
  void YMaxCallback(GtkWidget *Ob, gpointer data);      //YMax slider
  void YMinCallback(GtkWidget *Ob, gpointer data);      //YMin slider

  int KeyPressHandler(GtkWidget *Ob, GdkEventKey *event, gpointer data);
  int MaxSliderMove(GtkWidget *ob, float fac);
  int MinSliderMove(GtkWidget *ob, float fac);
		  
  int MapPointerToPanel(int X, int Y, bool isWindowPixelLocation=TRUE);
  void DisableProgressMeter();
  void EnableProgressMeter() ;
  void IterMainLoop() {while (g_main_iteration(TRUE));};
  void FreezeDisplay() {Hide(Canvas);};
  void UnFreezeDisplay() {Show(NULL,Canvas);};
  void prtdata()
    {
      int i;
      for (i=0;i<(int)Panels.capacity();i++)
	Panels[i].prtdata();
    };

  void print(string filename,gint paper=GTK_PLOT_LETTER, gint orientation=GTK_PLOT_PORTRAIT,
	     gint whichPanel=ALLPANELS,gint epsflags=0)
  {
    if (whichPanel==ALLPANELS)
      gtk_plot_layout_export_ps(GTK_PLOT_LAYOUT(Canvas),
				(char *)filename.c_str(),
				orientation,
				epsflags,
				paper);
    else
      {
	Panels[whichPanel].print(filename,paper,epsflags,orientation);
      }
  }

  enum {ALLPANELS=-1};
  
private:
  //  Display               *TheDisplay;
  vector<XYPanel>       Panels;
  vector< gdouble >     X;
  GtkWidget             *TopLevelWin, *TopLevelVBox, *HBox, *ProgressBar,*Canvas, *CtrlPanel, *menubar;
  GtkWidget             *Scroll1, *Layout;
  GtkAdjustment         *XMin_Adj, *XMax_Adj;
  BitField              ChosenOnes;
  unsigned short        NewPlot;
  int Timer;
  MPPPacker PackerObj;
};

#endif
