#include <MPPPacker.h>
#include <namespace.h>
#include <iostream>
void MPPPacker::Reset(int NPanels, float CanvasWidth, float CanvasHeight)
{
  PanelsPerPage_p=NPanels_p=NPanels; CanvasWidth_p = CanvasWidth; CanvasHeight_p = CanvasHeight;
}
vector<float> MPPPacker::GetPanelShape()
{
  vector<float> shape(2);
  shape[0] = CanvasWidth_p*0.87;
  shape[1] = CanvasHeight_p*0.98/NPanels_p;
  return shape;
}
//
// Compute the width, height and (x,y) location of the panel (whichPanel)
//
void MPPPacker::Geometry(int whichPanel, float &width, float &height, float& xloc, float& yloc)
{
  vector<float> pshape=GetPanelShape();
  width  = pshape[0];
  height = pshape[1];

  xloc   = CanvasWidth_p*0.08;
  float delta;
  delta  = (int)(whichPanel/PanelsPerPage_p)*GetSpaceBetweenPages();
  yloc   = height * whichPanel + delta;
}
//
// Retrun the shape of the canvas.
//
vector<float> MPPPacker::CanvasShape() 
{
  vector<float> shape(2); shape[0]=CanvasWidth_p;shape[1]=CanvasHeight_p;return shape;
}; 
//
//
//
vector<float> MPPPacker::PhysicalCanvasShape(bool SetShape) 
{
  vector<float> shape(2);
  float pw=GetPanelShape()[1];
  int n = NPanels_p%PanelsPerPage();

  int h = NumberOfPages()*(PanelsPerPage()*pw+GetSpaceBetweenPages()) + 
    n*pw;//+GetSpaceBetweenPages();
  shape[0]=CanvasWidth_p;
  shape[1]=h;
  if (SetShape)
    {
      CanvasWidth_p  = shape[0];
      CanvasHeight_p = shape[1];
    }
  return shape;
}
