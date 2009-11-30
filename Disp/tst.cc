#include <string.h>
#include <string>
#include <iostream.h>
main()
{
  int i=0,p=0;
  string tst="This is a test",t;

  t=tst; 

  while ((i=t.find_first_of(" ",0,t.size()))>=0)
   {
    cout << t.substr(0,i) << endl;
    t=t.substr(i+1,t.size());
   }
  cout << t.substr(0,i) << endl;
}

