#include <stdlib.h>
#include <cl.h>
#include <clinteract.h>

#include <casa/aips.h>
#include <images/Images/ImageInterface.h>
#include <images/Images/PagedImage.h>
#include <casa/Arrays/Array.h>
#include <casa/Arrays/Vector.h>
#include <lattices/Lattices/LatticeExpr.h>
#include <coordinates/Coordinates/CoordinateSystem.h>
#include <coordinates/Coordinates/DirectionCoordinate.h>
#include <coordinates/Coordinates/SpectralCoordinate.h>
#include <coordinates/Coordinates/StokesCoordinate.h>

using namespace std;
using namespace casa;

void toCASAVector(std::vector<int>& stdv, IPosition& tmp)
{
  Int n=stdv.size();
  tmp.resize(n);
  for(Int i=0;i<n;i++) tmp(i) = stdv[i];
}

void toCASAVector(std::vector<float>& stdv, Vector<Double>& tmp)
{
  Int n=stdv.size();
  tmp.resize(n);
  for(Int i=0;i<n;i++) tmp(i) = stdv[i];
}

void toCASASVector(std::vector<string>& stdv, Vector<String>& tmp)
{
  Int n=stdv.size();
  tmp.resize(n);
  for(Int i=0;i<n;i++) tmp(i) = stdv[i];
}

void UI(Bool restart, int argc, char **argv, string& ImName, string& ImType, 
	Vector<String>& AxisNames, IPosition& ImShape, Vector<Double>& CellSize)
{
  if (!restart)
    {
      BeginCL(argc,argv);
      char TBuf[FILENAME_MAX];
      clgetConfigFile(TBuf,argv[0]);strcat(TBuf,".config");
      clloadConfig(TBuf);
      clInteractive(0);
    }
  else
   clRetry();
  try
    {
      VString options;
      int i;
      {
	i=1;clgetSValp("image", ImName,i);  

	ImType="Complex";
	options.resize(1);
	options[0]=ImType;
	i=1;clgetSValp("type", ImType,i);  
	clSetOptions("type",options);

	vector<string> tmpNames;
	tmpNames.resize(2);tmpNames[0]="RA"; tmpNames[1]="Dec";
	//	i=2;dbgclgetNSValp("axisnames",tmpNames,i);
	toCASASVector(tmpNames, AxisNames);

	vector<int> tmpShape;
	tmpShape.resize(2,512);
	i=2;clgetNIValp("imsize",tmpShape,i);
	toCASAVector(tmpShape, ImShape);

	vector<float> tmpCell;
	tmpCell.resize(2,1);
	i=2;clgetNFValp("cellsize",tmpCell,i);
	toCASAVector(tmpCell, CellSize);
      }
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}

CoordinateSystem mkCoords(IPosition& shape, Vector<Double>& cellsize, Vector<String>& axisNames)
{
  DirectionCoordinate dc;
  const Double toRad=C::pi/180.0/3600.0;
  Matrix<Double> xform(2,2);                                
  xform = 0.0; xform.diagonal() = 1.0;                    
  DirectionCoordinate radec(MDirection::J2000,            
                            Projection(Projection::SIN),    
                            135*C::pi/180.0, 60*C::pi/180.0,
                            -cellsize(0)*toRad, cellsize(1)*toRad,
                            xform,                          
                            shape(0)/2.0, shape(1)/2.0);                      

  CoordinateSystem coords;
  coords.addCoordinate(radec);
  return coords;
}

int main(int argc, char **argv)
{
  void UI(Bool, int, char**, string&, string&, Vector<String>&, IPosition&, Vector<Double>&);
  String ImName, ImType;
  Vector<String> AxisNames;
  IPosition ImShape;
  Vector<Double> CellSize;
  IPosition shape;
  int reStart=0;
  
  UI(reStart, argc, argv, ImName, ImType, AxisNames, ImShape, CellSize);
  
  CoordinateSystem coords(mkCoords(ImShape, CellSize, AxisNames));
  
  PagedImage<Complex> img(ImShape, coords, ImName.c_str());
  Array<Complex> data;

  data.resize(ImShape);
  IPosition shp(data.shape()),ndx(2);
  ndx=0;


  for(ndx(0)=0;ndx(0)<shp(0);ndx(0)++)
    for(ndx(1)=0;ndx(1)<shp(1);ndx(1)++)
      {
	Float x0=shp(0)/2, y0=shp(0)/2;
	Float x= 2*3.1415926*(ndx(0)-x0)/shp(0),
	  y= 2*3.1415926*(ndx(1)-y0)/shp(1);
	Float xval,yval,amp;
	if (x==0) xval=1.0; else xval=sin(x)/x;
	if (y==0) yval=1.0; else yval=sin(y)/y;
	amp=xval*yval;
	data(ndx) = Complex(amp*sin(0.2*3.1415296*ndx(0)/shp(0)),
			    amp*cos(0.2*3.1415926*ndx(1)/shp(1)));
      }

  img.put(data);
}
