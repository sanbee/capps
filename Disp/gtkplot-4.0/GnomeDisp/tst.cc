#include <iostream.h>

void tst(int &val)
{
  for (int i=0;i<10;i++)
    cerr << val[i] << endl;
}

main()
{
  int val[10];

  for (int i=0;i<10;i++) val[i]=i;

  tst(val);
}


