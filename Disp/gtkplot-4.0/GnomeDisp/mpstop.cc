#include <stdio.h>
#include <mp.h>
#include <vector>
#include <string>

main(int argc, char **argv)
{
  string Msg="So long...";
  SockIO S0;

  if (argc < 2)
    {
      cerr << "Usage: " << argv[0] << " <remote hostname>" << endl;
      exit(-1);
    }
  //
  // Connect to the "well known" port.
  //
  S0.init(1858,0,argv[1]);
  S0.konnekt();

  cerr << "###Informational: Stopping MP server by confusing it completely!"
       << endl 
       << "                  Heeeaaa..ha..ha..ha...." 
       << endl;

  S0.Send(Msg);
}
