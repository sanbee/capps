#include <iostream.h>
#include <PipeMgr.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

void main()
{
  int CmdStream, DataStream;
  PipeMgr PM;

  PM.Setup(CmdStream, DataStream);
}
