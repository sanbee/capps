#include <PipeMgr.h>
#include <ErrorObj.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

PipeMgr::~PipeMgr()
{
}

int PipeMgr::Setup(int& CmdS, int& DataS)
{
  Pid=getpid();
  char FName[128];

  sprintf(FName,"/tmp/mpcmd%d%c",Pid,'\0');
  if ((CmdS=mkfifo(FName,O_RDWR))<=0)
    throw(ErrorObj("In opening command pipe","###Error:",ErrorObj::Fatal));
  sprintf(FName,"/tmp/mpdata%d%c",Pid,'\0');
  if ((DataS=mkfifo(FName,O_RDWR))<=0)
    throw(ErrorObj("In opening command pipe","###Error:",ErrorObj::Fatal));

  return 1;
}

int PipeMgr::CmdSend(string& Msg)
{
  int n=Msg.size();
  write(CmdS,(const void *)&n,sizeof(int));

  return write(CmdS,(const void *)&Msg[0],n);
}

int PipeMgr::DataSend(string& Msg)
{
  int n=Msg.size();

  write(DataS,(const void *)&n,sizeof(int));

  return write(DataS,(const void *)&Msg[0],n);
}

int PipeMgr::CmdReceive(string& Msg)
{
  int n;
  read(CmdS,(const void *)&n,sizeof(int));

  Msg.resize(n);
  return read(CmdS,(const void *)&Msg[0],n);
}


int PipeMgr::DataReceive(string& Msg)
{
  int n;
  read(DataS,(const void *)&n,sizeof(int));

  Msg.resize(n);
  return read(DataS,(const void *)&Msg[0],n);
}
