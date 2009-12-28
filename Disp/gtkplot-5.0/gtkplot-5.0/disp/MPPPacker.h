// -*- C++ -*-
// $Id$
class MPPPacker
{
public:
  MPPPacker():PanelsPerPage_p(-1) {};
  ~MPPPacker() {};
  //
  // Set the total number of panels (NPanels), and the canvas width and height.
  //
  void Reset(int NPanels, float CanvasWidth, float CanvasHeight);
  //
  // Compute the width, height and (x,y) location of the panel (whichPanel)
  //
  void Geometry(int whichPanel, float &width, float &height, float& xloc, float& yloc);
  void SetPanelsPerPage(int n) {PanelsPerPage_p = n;};
  
private:
  int NPanels_p, PanelsPerPage_p;
  float CanvasWidth_p, CanvasHeight_p;
};
