#include <XYPanel.h>
#include <vector>
#include <iostream.h>

extern "C" {
  void YMinCallBack(FL_OBJECT *Obj, long data);
  void YMaxCallBack(FL_OBJECT *Obj, long data);
  void ResetSliderCallBack(FL_OBJECT *Obj, long data);
  void GlobalActionCallBack(FL_OBJECT *Obj, long data);
  void io_callback(int, void *);
}

main(int argc, char **argv)
{
  vector<XYPanel> OnDisp;
  FL_OBJECT *CHART;
  FL_FORM *Form;
  char str[10];
  int N;
  float X[100],Y[100], Range[2];
  Display *TheDisplay;

  for (int i=0;i<100;i++)
    {
      X[i] = i;
      Y[i] = sin(2*3.1415*i/10);
    }
  Range[0] = Range[1] = 0.0;

  OnDisp.resize(1);

  OnDisp[0].SetAttribute(XYPanel::XTICS0,-1);
  OnDisp[0].SetAttribute(XYPanel::XTICS1,-1);

  Form = OnDisp[0].Make(Form,5,5,470,300,argc,argv,"PlotTool");
  
  OnDisp[0].Plot(X,Y,100);

  OnDisp[0].GetRange(Range,1);
  cerr << Range[0] << " " << Range[1] << endl;

  fl_do_forms();
}

