#include <SockIO.h>
#include <ErrorObj.h>

int GetFreePorts(int& P0, int& P1)
{
  int Found=0;
  SockIO tmp1;
  P0=6000; P1=P0+1;
  while (!Found && P0<=10000)
    {
      cerr << "Trying port " << P0 << "..." << endl;
      tmp1.shut();
      try
	{
	  tmp1.init(P0);
	  Found=1;
	}
      catch (ErrorObj& x)
	{
	  cerr << x.what() << endl;
	  cerr << P0 << endl;
	  Found=0;
	  P0++;
	}
    }
  cerr << "P0="<<P0<<endl;
  tmp1.shut();
  if (!Found) P0=-1;
  Found=0;
  P1=P0+1;
  while (!Found && P1<=10000)
    {
      cerr << "Trying port " << P1 << "..." << endl;
      try
	{
	  tmp1.init(P1);
	  Found=1;
	  
	}
      catch (ErrorObj x)
	{
	  cerr << x.what() << endl;
	  cerr << P1 << endl;
	  Found=0;
	  P1++;
	}
    }
  if (!Found) P1=-1;
  cerr << "P1="<<P1<<endl;
  return 1;
}

main()
{
  int P0,P1;
  GetFreePorts(P0,P1);
}
