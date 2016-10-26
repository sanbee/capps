#include <casa/aips.h>
#include <casa/BasicSL/String.h>
#include <casa/Exceptions/Error.h>
#include <casa/sstream.h>
#include <strstream>

namespace casa 
{
using namespace casacore;
  Bool checkCASAEnv(String pathVar="CASAPATH")
  {
    if (!getenv(pathVar.c_str()))
      {
	ostringstream msg;
	msg << "Environment variable " << pathVar << " not found. " << endl
	    << "Perhaps you forgot to source casainit.sh/.csh?";
	
	throw(AipsError(msg.str()));
      }
    return True;
  }
};


