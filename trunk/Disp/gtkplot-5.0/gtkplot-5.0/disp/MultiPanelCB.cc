#include <forms.h>
#include <MultiPanel.h>
#include <string.h>
#include <ExitObj.h>
extern "C" {
/*------------------------------------------------------------------*/
void ResetSliderCallBack(FL_OBJECT *Obj, long data) 
{
  printf("Reset\n");
}
/*------------------------------------------------------------------*/

void DoneCallBack(FL_OBJECT *Obj, long data)
{
	//  exit(0);
}
/*------------------------------------------------------------------*/
void CPDoneCallBack(FL_OBJECT *Obj, long data)
{
  fl_hide_form((FL_FORM *)Obj->u_ldata);
}
/*------------------------------------------------------------------*/
void CtrlPanelCallBack(FL_OBJECT *Obj, long data)
{
  float Range[2];
  MultiPanel *MP=  (MultiPanel *)Obj->u_ldata;
  MP->MakeCtrlPanel(1,1,500,250,"Control Panel");
}
/*------------------------------------------------------------------*/
void XMax_cb(FL_OBJECT *ob, long data)
{
  float Range[2];
  MultiPanel *MP=(MultiPanel *)ob->u_ldata;
  int i;
  
  MP->operator[](0).GetRange(Range,0);
  
  Range[1] = (float)fl_get_slider_value(ob);
  for (i=0;i<MP->NPanels();i++)
    MP->operator[](i).SetRange(Range,0);
}
/*------------------------------------------------------------------*/
void XMin_cb(FL_OBJECT *ob, long data)
{
  float Range[2];
  MultiPanel *MP=(MultiPanel *)ob->u_ldata;
  int i;
  
  MP->operator[](0).GetRange(Range,0);
  
  Range[0] = (float)fl_get_slider_value(ob);
  for (i=0;i<MP->NPanels();i++)
    MP->operator[](i).SetRange(Range,0);
}
/*------------------------------------------------------------------*/
void YMax_cb(FL_OBJECT *ob, long data)
{
  float Range[2];
  MultiPanel *MP=(MultiPanel *)ob->u_ldata;
  int i;
  
  MP->operator[](0).GetRange(Range,1);
  
  Range[1] = (float)fl_get_slider_value(ob);
  for (i=0;i<MP->NPanels();i++)
    MP->operator[](i).SetRange(Range,1);
}
/*------------------------------------------------------------------*/
void YMin_cb(FL_OBJECT *ob, long data)
{
  float Range[2];
  MultiPanel *MP=(MultiPanel *)ob->u_ldata;
  int i;
  
  MP->operator[](0).GetRange(Range,1);
  
  Range[0] = (float)fl_get_slider_value(ob);
  for (i=0;i<MP->NPanels();i++)
    MP->operator[](i).SetRange(Range,1);
}


};
