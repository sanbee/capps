#include <images/Images/ImageInterface.h>
#include <images/Images/PagedImage.h>
#include <casa/Arrays/Array.h>
#include <casa/Arrays/Vector.h>
#include <lattices/Lattices/LatticeExpr.h>

using namespace casa;

int main(int argc, char **argv)
{
  PagedImage<Complex> pb(argv[1]);
  Array<Complex> data;
  //  data.resize(pb.shape());
  data=pb.get();
  IPosition shp(data.shape()),ndx(4);
  ndx=0;
  for(ndx(0)=0;ndx(0)<shp(0);ndx(0)++)
    for(ndx(1)=0;ndx(1)<shp(1);ndx(1)++)
      {
	IPosition n0(ndx),n1(ndx);
	n0(2) = 0; n1(2)=1;
	data(n0) = (data(n0) + data(n1))/2;
      }
  PagedImage<Complex> tpb(pb.shape(), pb.coordinates(), argv[2]);
  tpb.put(data);
  //  LatticeExpr<Complex> le(data);
  //  tpb.copyData(le);
}
