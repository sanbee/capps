// -*-C++-*-
#if !defined(PIPEMGR_H)
#define PIPEMGR_H
#include <LinkMgr.h>
#include <string>
#include <stdlib.h>

class PipeMgr: public LinkMgr {
public:
  PipeMgr():LinkMgr() {};
  ~PipeMgr();
  
  int Setup(int &CmdS, int& DataS);
  int CmdSend(string& Msg);
  int CmdReceive(string& Msg);
  int DataSend(string& Msg);
  int DataReceive(string& Msg);
private:
  int Pid;
  FILE *InternalCmdS, *InternalDataS;
  string CmdPipeName, DataPipeName;
};

#endif
