#include <PipeMgr.h>
#include <ErrorObj.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

PipeMgr::~PipeMgr()
{
  if (InternalCmdS > 0)  {unlink(CmdPipeName.c_str());  CmdPipeName.resize(0);}
  if (InternalDataS > 0) {unlink(DataPipeName.c_str()); DataPipeName.resize(0);}
}

int PipeMgr::Setup(int& CmdS, int& DataS)
{
  Pid=getpid();
  char FName[128];

  sprintf(FName,"/tmp/mpcmd%d%c",Pid,'\0');
  CmdPipeName=FName;
  if (mkfifo(FName,O_RDWR)<0)
    throw(ErrorObj("In opening command pipe","###Error:",ErrorObj::Fatal));

  InternalCmdS = fopen(FName,O_RDWR);

  sprintf(FName,"/tmp/mpdata%d%c",Pid,'\0');
  DataPipeName=FName;
  if (mkfifo(FName,O_RDWR)<0)
    throw(ErrorObj("In opening command pipe","###Error:",ErrorObj::Fatal));

  InternalDataS = fopen(FName,O_RDWR);

  CmdS=InternalCmdS; DataS=InternalDataS;
  return 1;
}

int PipeMgr::CmdSend(string& Msg)
{
  int n=Msg.size();
  write(InternalCmdS,(const void *)&n,sizeof(int));

  return write(InternalCmdS,(void *)&Msg[0],n);
}

int PipeMgr::DataSend(string& Msg)
{
  int n=Msg.size();

  write(InternalDataS,(const void *)&n,sizeof(int));

  return write(InternalDataS,(const void *)&Msg[0],n);
}

int PipeMgr::CmdReceive(string& Msg)
{
  int n;
  read(InternalCmdS,(void *)&n,sizeof(int));

  Msg.resize(n);
  return read(InternalCmdS,(void *)&Msg[0],n);
}


int PipeMgr::DataReceive(string& Msg)
{
  int n;
  read(InternalDataS,(void *)&n,sizeof(int));

  Msg.resize(n);
  return read(InternalDataS,(void *)&Msg[0],n);
}
