// -*- C++ -*-
// $Id$
#if !defined(MPPPACKER_H)
#define MPPPACKER_H

#include <iostream>
#include <vector>
#include <math.h>
#include <namespace.h>
class MPPPacker
{
public:
  MPPPacker():PanelsPerPage_p(-1), SpaceBetweenPages_p(50) {};
  ~MPPPacker() {};
  //
  // Set the total number of panels (NPanels), and the canvas width and height.
  //
  void Reset(int NPanels, float CanvasWidth, float CanvasHeight);
  void SetPanelShape(int PanelW, int PanelH) {PanelWidth_p=PanelW; PanelHeight_p=PanelH;};
  //
  // Compute the width, height and (x,y) location of the panel (whichPanel)
  //
  void Geometry(int whichPanel, float &width, float &height, float& xloc, float& yloc);

  void SetPanelsPerPage(int n)      {PanelsPerPage_p = n;};
  int  PanelsPerPage()              {return PanelsPerPage_p;};
  void SetSpaceBetweenPages(int n)  {SpaceBetweenPages_p = n;};

  int  GetSpaceBetweenPages()       {return SpaceBetweenPages_p;};
  vector<float> GetPanelShape();
  int NumberOfPanels()              {return NPanels_p;};
  int NumberOfPages()               {return (int)ceil(((double)NPanels_p)/PanelsPerPage());}
  vector<float> CanvasShape() ;
  vector<float> PhysicalCanvasShape(bool SetShape=true);

private:
  int NPanels_p, PanelsPerPage_p, SpaceBetweenPages_p;
  float CanvasWidth_p, CanvasHeight_p, PanelWidth_p, PanelHeight_p;
};

#endif
