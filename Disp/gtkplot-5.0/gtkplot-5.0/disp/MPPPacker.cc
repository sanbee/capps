#include <MPPPacker.h>
void MPPPacker::Reset(int NPanels, float CanvasWidth, float CanvasHeight)
{
  NPanels_p=NPanels; CanvasWidth_p = CanvasWidth; CanvasHeight_p = CanvasHeight;
}
//
// Compute the width, height and (x,y) location of the panel (whichPanel)
//
void MPPPacker::Geometry(int whichPanel, float &width, float &height, float& xloc, float& yloc)
{
  width  = CanvasWidth_p*0.87;
  height = CanvasHeight_p*0.98/NPanels_p;
  xloc   = CanvasWidth_p*0.08;
  float delta;
  delta  = (int)(whichPanel/PanelsPerPage_p)*50;
  yloc   = height * whichPanel + delta;
}
