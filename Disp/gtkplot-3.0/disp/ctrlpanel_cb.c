#include <stdio.h>
#include <forms.h>
#include "xplot.h"
#include "ctrlpanel.h"

/* callbacks for form ctrlpanel */
extern FD_ctrlpanel *fd_ctrlpanel;
extern int GlobalNoOfOverlays;

void XMax_cb(FL_OBJECT *ob, long data)
{
  float max,min;
  int i;
  for(i=0;i<GlobalNoOfOverlays;i++)
    {
      fl_get_xyplot_xbounds(CHART[i],&min,&max);
      max = (float)fl_get_slider_value(ob);
      fl_set_xyplot_xbounds(CHART[i],(double)min,(double)max);
    }
}
void XMin_cb(FL_OBJECT *ob, long data)
{
  float max,min;
  int i;
  for(i=0;i<GlobalNoOfOverlays;i++)
    {
      fl_get_xyplot_xbounds(CHART[i],&min,&max);
      min = (float)fl_get_slider_value(ob);
      fl_set_xyplot_xbounds(CHART[i],(double)min,(double)max);
    }
}
void YMax_cb(FL_OBJECT *ob, long data)
{
  float max,min;
  int i;
  fprintf(stderr,"%ld\n",data);
  for(i=0;i<GlobalNoOfOverlays;i++)
    {
      fl_get_xyplot_ybounds(CHART[i],&min,&max);
      max = (float)fl_get_slider_value(ob);
      fl_set_xyplot_ybounds(CHART[i],(double)min,(double)max);
    }
}
void YMin_cb(FL_OBJECT *ob, long data)
{
  float max,min;
  int i;
  fprintf(stderr,"%ld\n",data);
  for(i=0;i<GlobalNoOfOverlays;i++)
    {
      fl_get_xyplot_ybounds(CHART[i],&min,&max);
      min = (float)fl_get_slider_value(ob);
      fl_set_xyplot_ybounds(CHART[i],(double)min,(double)max);
    }
}


/*
   If the control panel is on, reset the sliders.  This is done
   by asking the xyplot for it's current bounds.

   Hence, if the use sets the bounds from Glish to values other
   than what is set by default, the sliders will see the change.
*/
void ResetSliders()
{
  if (CtrlPanelOn)
    {
      float max,min;
      int i;
      fl_get_xyplot_xbounds(CHART[0],&min,&max);
      fl_set_slider_bounds(XMax_Slider,(double)min,(double)max);
      fl_set_slider_bounds(XMin_Slider,(double)min,(double)max);
      
      fl_set_slider_value(XMax_Slider,(double)max);
      fl_set_slider_value(XMin_Slider,(double)min);


      for (i=0;i<=GlobalNoOfOverlays;i++)
	{
	  fl_get_xyplot_ybounds(CHART[i],&min,&max);
	  fl_set_slider_bounds(YMax_Slider[i],(double)max,(double)min);
	  fl_set_slider_bounds(YMin_Slider[i],(double)max,(double)min);
	  
	  fl_set_slider_value(YMax_Slider[i],(double)max);
	  fl_set_slider_value(YMin_Slider[i],(double)min);
	}
    }
}

/*
   If the control panel is visible, hide it and then delete it.
*/
void Quit_cb(FL_OBJECT *ob, long data)
{
  CtrlPanelOn=0;
  fl_hide_form(fd_ctrlpanel->ctrlpanel);
  fl_free_form(fd_ctrlpanel->ctrlpanel);
}

/*
   Reset the bounds of the 2 axis to that allowed by the plotted
   data.  First reset the bounds in the XPLOT object and then
   reset the sliders (which talk to XYPLOT object to get the bounds).
*/
void Reset_cb(FL_OBJECT *ob, long data)
{
  int i;
  for (i=0;i<GlobalNoOfOverlays;i++)
    fl_set_xyplot_xbounds(CHART[i],0,0);
  /*  fl_set_xyplot_ybounds(CHART[0],0,0);*/
  ResetSliders();
}
/*
  Freeze the limits of the sliders to the current slider values.
  This will allow incrimental zoon-in and zoom-out
*/
void FreezX_cb(FL_OBJECT *ob, long data)
{
  if (CtrlPanelOn)
    {
      double max,min;
      max = fl_get_slider_value(XMax_Slider);
      min = fl_get_slider_value(XMin_Slider);
      fl_set_slider_bounds(XMax_Slider,min,max);
      fl_set_slider_bounds(XMin_Slider,min,max);
    }
}  

void FreezY_cb(FL_OBJECT *ob, long data)
{
  if (CtrlPanelOn)
    {
      double max,min;
      max = fl_get_slider_value(YMax_Slider);
      min = fl_get_slider_value(YMin_Slider);
      fl_set_slider_bounds(YMax_Slider,max,min);
      fl_set_slider_bounds(YMin_Slider,max,min);
    }
}  
