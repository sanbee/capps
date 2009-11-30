#include <iostream.h>
#include <math.h>

main()
{
  int n=0;
  while(1)
    {
      cout << n++ << " ";
      for (int i=0;i<60;i++)
	cout << sin(2*3.14*n/10) << " ";
      cout << endl;
    }
}
    
