#include <casa/Exceptions/Error.h>
#include <casa/iostream.h>
#include <msvis/MSVis/StokesVector.h>
#include <msvis/MSVis/VisBufferUtil.h>
#include <msvis/MSVis/VisSet.h>
#include <msvis/MSVis/VisibilityIterator.h>
#include <msvis/MSVis/VisBuffer.h>
#include <casa/OS/Timer.h>
#include <images/Images/ImageInterface.h>
#include <images/Images/TempImage.h>
using namespace casa;

namespace casa {
Timer modelIOTimer, correctedIOTimer, weightIOTimer;
Double totalModelIOTime=0, totalCorrectedIOTime=0, totalWeightIOTime=0, totalIOCalls=0;
};

void tstGridder(Array<Complex>& uvGrid, Int& nx, Int& ny, Int& nVis, Int& nPol, Int& nChan, Int &support)
{
  IPosition pos(4,nx/2,ny/2,0,0);
  for(Int r=0;r<nVis;r++)
    for(Int c=0;c<nChan; c++)
      for(Int p=0;p<nPol; p++)
	for(Int s0=0;s0<support;s0++)
	  for(Int s1=0;s1<support;s1++)
	    uvGrid(pos) += Complex(c*r,0);
}

int main(int argc, char **argv)
{
  if (argc < 2) cerr << "Usage: <MSName> [doWrite(0 or 1)]" << endl;

  MS ms(argv[1],Table::Update);
  Block<Int> sort(0);
  Int doWrite=0;
  if (argc > 2) sscanf(argv[2],"%d",&doWrite);

  if (doWrite) cout << "Testing Read & Write I/O: " << endl;
  else cout << "Testing Read I/O: " << endl;

  Matrix<Int> noselection;
  Double timeInterval=0;
  sort.resize(4);
  sort[0] = MS::ARRAY_ID;
  sort[1] = MS::FIELD_ID;
  sort[2] = MS::DATA_DESC_ID;
  sort[3] = MS::TIME;
  VisSet vs(ms,sort,noselection,timeInterval,False);
  VisIter& vi=vs.iter();
  VisBuffer vb(vi);
  Long rsize=0, wsize=0, nRows=0;
  Timer writeTimer;
  TempImage<Complex> uvGrid;
  Bool deleteIt;
  Int nx=2048, ny=2048;
  uvGrid.resize(IPosition(4,nx,ny,2,1));
  Array<Complex> uvGridArray(uvGrid.get());
  //  Complex *uvGridStorage = uvGridArray.getStorage(deleteIt);

  IPosition mshape;
  for (vi.originChunks();vi.moreChunks();vi.nextChunk()) 
    for (vi.origin(); vi.more(); vi++) 
      {
	Cube<Complex> model;
	Cube<Bool> flagCube;
	Vector<Float> weight;
	Vector<Double> time;
	Vector<RigidVector<Double,3> > uvw;
	//	if (nRows==0)
	  {
	    model=vb.modelVisCube();
	    flagCube = vb.flagCube();
	    weight = vb.weight();
	    time = vb.time();
	    uvw = vb.uvw();
	    mshape=model.shape();
	  }
	Int nVis=mshape(2), nChan=mshape(1), nPol=mshape(0), 
	  support=4;
	rsize += sizeof(Float)*2*mshape.product()
	  + sizeof(Bool)*flagCube.shape().product()
	  + sizeof(Float)*weight.shape().product()
	  + sizeof(Double)*time.shape().product()
	  + sizeof(Double)*3*uvw.shape().product();
	wsize += sizeof(Float)*2*model.shape().product();
	//	vb.setModelVisCube(Complex(0.0,0.0));
	model.set(Complex(0,0));
	//	tstGridder(uvGridArray,nx,ny,nVis, nPol, nChan, support);
	if (doWrite)  vi.setVis(model,VisibilityIterator::Model);
	nRows++;
      }
  if (doWrite)
    cout << "Read and wrote " << rsize << " , " << wsize << " bytes for " << nRows << " rows." << endl;
  else
    cout << "Read " << rsize << " bytes from " << nRows << "rows." << endl;
  modelIOTimer.show();
}
