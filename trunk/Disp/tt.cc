#include <iostream.h>
#include <vector.h>

main()
{
  vector < vector<int> > tt;
  
  tt.resize(10);
  for (int i=0;i<10;i++) tt[i].resize(10);
  
  tt[5][5]=10;
}
