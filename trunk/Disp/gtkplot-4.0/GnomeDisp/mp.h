#include <unistd.h>
#include <iostream.h>
#include <vector>
#include <string.h>
#include <Protocol.h>
#include <SockIO.h>
#include <ErrorObj.h>

int Dock(char *Host, SockIO &CmdS, SockIO &DataS);
int MakeColorList(vector<char *> &ColorList);
