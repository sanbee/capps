#include <iostream>
#include <strstream>
#include <string>

main()
{
  strstream Msg;
  int i=10;

  Msg << "THis is a test " << i << ends;

  cout << Msg.str() << strlen(Msg.str()) << endl;
}
