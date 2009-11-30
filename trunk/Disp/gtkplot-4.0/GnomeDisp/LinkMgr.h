#if !defined(LINKMGR_H)
#define LINKMRG_H
#include <string>

class LinkMrg{
public:
  LinkMgr() {CmdStream=DataStream=-1;}
  ~LinkMgr() {};

  virtual int Setup(int &CmdS, int& DataS) {};
  virtual int Send(string& Msg) {};
  virtual int Receive(string& Msg) {};
private:
  int CmdStream, DataStream;
}

#endif

