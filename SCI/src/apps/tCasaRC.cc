#include <casa/aips.h>
#include <casa/System/Casarc.h>
#include <casa/sstream.h>
#include <casa/Utilities/CountedPtr.h>
#include <synthesis/Utilities/AppRC.h>

using namespace std;
using namespace casa;

void test1()
{
  Int run=1;
  AppRC rc("/tmp/.imagerrc",False,True); rc.setID("RC1");
  AppRC rc2("/tmp/.imagerrc",False,True); rc2.setID("RC2");
  {

    while(run)
      {
	sleep(2);
	rc.get("run",run);
	if (run < 0) exit(0);
      }
    rc.put("run",100);
    rc2.get("run",run);
    cerr << "RC2: " << run << endl;
  }
}


int main(int argc, char **argv)
{
  Int i=0;
  while(1)
    {
      cerr << i++ << endl;
      test1();
    }
}
