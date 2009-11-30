/* Form definition file generated with fdesign. */

#include <forms.h>
#include <stdlib.h>
#include "xplot.h"
#include "ctrlpanel.h"

FD_ctrlpanel *create_form_ctrlpanel(void)
{
  FL_OBJECT *obj;
  FD_ctrlpanel *fdui = (FD_ctrlpanel *) fl_calloc(1, sizeof(*fdui));
  int old_bw = fl_get_border_width();
  float min,max;

  fl_set_border_width(2);
  fdui->ctrlpanel = fl_bgn_form(FL_NO_BOX, 500, 100);
  obj = fl_add_box(FL_UP_BOX,0,0,500,100,"");

  XMax_Slider = obj = fl_add_valslider(FL_HOR_NICE_SLIDER,20,40,470,20,"XMax");
        fl_set_slider_return(obj, FL_RETURN_CHANGED);
        fl_set_object_callback(obj,XMax_cb,0);
        fl_get_xyplot_xbounds(CHART[0],&min,&max);
        fl_set_slider_value(obj,(double)max);
        fl_set_slider_bounds(obj,(double)min,(double)max);

  XMin_Slider = obj = fl_add_valslider(FL_HOR_NICE_SLIDER,20,10,470,20,"XMin");
        fl_set_slider_return(obj, FL_RETURN_CHANGED);
        fl_set_object_callback(obj,XMin_cb,0);
        fl_set_slider_value(obj,(double)min);
        fl_set_slider_bounds(obj,(double)min,(double)max);

	/*
  YMax_Slider = obj = fl_add_valslider(FL_VERT_NICE_SLIDER,80,80,20,150,"YMax");
        fl_set_slider_return(obj, FL_RETURN_CHANGED);
        fl_set_object_callback(obj,YMax_cb,0);
        fl_get_xyplot_ybounds(CHART[0],&min,&max);
        fl_set_slider_bounds(obj,(double)max,(double)min);
        fl_set_slider_value(obj,(double)max);

  YMin_Slider = obj = fl_add_valslider(FL_VERT_NICE_SLIDER,20,80,20,150,"YMin");
        fl_set_slider_return(obj, FL_RETURN_CHANGED);
        fl_set_object_callback(obj,YMin_cb,0);
        fl_set_slider_bounds(obj,(double)max,(double)min);
        fl_set_slider_value(obj,(double)min);

  obj = fl_add_button(FL_NORMAL_BUTTON,290,220,40,20,"Button");
        fl_set_object_callback(obj,FreezX_cb,0);
        fl_set_object_label(obj,"FreezX");
  
  obj = fl_add_button(FL_NORMAL_BUTTON,340,220,40,20,"Button");
        fl_set_object_callback(obj,FreezY_cb,0);
        fl_set_object_label(obj,"FreezY");
  

	*/  
  obj = fl_add_button(FL_NORMAL_BUTTON,390,70,40,20,"Button");
        fl_set_object_callback(obj,Reset_cb,0);
        fl_set_object_label(obj,"Reset");

  obj = fl_add_button(FL_NORMAL_BUTTON,440,70,40,20,"Button");
        fl_set_object_callback(obj,Quit_cb,0);
        fl_set_object_label(obj,"Done");

  fl_end_form();
  fl_set_border_width(old_bw);

  return fdui;
}
/*---------------------------------------*/

