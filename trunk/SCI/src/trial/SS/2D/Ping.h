#include <iostream>
#include "Glish/Client.h"

class MyClient:public Client{
public:
  MyClient(int argc, char **argv):Client(argc,argv) {};
protected:
  void Client::HandlePing()
	{
	  cerr << "My handler" << endl;
	};
};
