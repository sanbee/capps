#include <casa/aips.h>
#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/Cube.h>
#include <casa/Arrays/Slicer.h>
#include <casa/Containers/Stack.h>
#include <ms/MeasurementSets/MeasurementSet.h>
#include <measures/Measures/Stokes.h>
#include <measures/Measures/MeasConvert.h>
#include <casa/Quanta/MVDoppler.h>
#include <measures/Measures/MCDoppler.h>
#include <measures/Measures/MDoppler.h>
#include <tables/Tables/ArrayColumn.h>
#include <tables/Tables/ScalarColumn.h>
#include <casa/BasicSL/String.h>
#include <scimath/Mathematics/SquareMatrix.h>
#include <scimath/Mathematics/RigidVector.h>
#include <casa/Utilities/Assert.h>
#include <casa/Utilities/Sort.h>
#include <casa/Arrays/ArrayLogical.h>
#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/MaskedArray.h>

#include <ms/MeasurementSets/MSDerivedValues.h>
#include <msvis/MSVis/StokesVector.h>
#include <ms/MeasurementSets/MSIter.h>
#include <iostream>

using namespace casa;

int main(int argc, char **argv)
{
  if (argc < 2)
    {
      cerr << "Usage: " << argv[0] << " <TableName>" << endl;
      exit(0);
    }

  Int rowBlock=351, polLen=2;
  if (argc > 2) sscanf(argv[2],"%d",&rowBlock);
  if (argc > 3) sscanf(argv[3],"%d",&polLen);
  polLen = polLen < 0 ? 1 : polLen;

  try
    {
      Table tab(argv[1]),selTab;
      Int nrows = tab.nrow(),beginRow=0,endRow=-1;

      IPosition slicerStart(2), slicerLength(2);
      slicerStart     = 0;      slicerStart(1)  = 0; //slicerStart(1)  = 100;
      slicerLength(0) = polLen; slicerLength(1) = 1; //slicerLength(1) = 300;
      Slicer slicer(slicerStart, slicerLength);

      while (beginRow <= nrows)
	{
	  endRow = beginRow+rowBlock;
	  if (endRow >= nrows) endRow = nrows-1;
	  Vector<uInt> rows(rowBlock);
	  indgen(rows,uInt(beginRow));
	  selTab = tab(rows);
	  
	  ROArrayColumn<Complex> colVis;
	  Cube<Complex> data;
	  colVis.attach(selTab,"DATA");

	  colVis.getColumn(slicer,data,True);
	  //	  colVis.getColumn(data,True);
	  cout << "Data shape = " << data.shape() << ".  Row " << beginRow << " of " << nrows << endl;
	  beginRow=endRow;
	}
    }
  catch (AipsError &x)
    {
      cerr << x.what() << endl;
    }
}
