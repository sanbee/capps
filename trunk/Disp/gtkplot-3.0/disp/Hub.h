//-*-C++-*-
//$Id$
#if !defined(HUB_H)
#define HUB_H
#include <SockIO.h>
#include <Protocol.h>
#include <MultiPanel.h>
#include <vector.h>
#include <exception>

extern  void MPDefaultErrorHandler();


#define MSG_BUFLEN 16384
extern "C" 
{
  void hub_cmd_io_callback(int fd, void *data);
  void hub_data_io_callback(int fd, void *data) ;
  void hub_ctrl_io_callback(int fd, void *data) ;
};

class Hub
{
public:
  Hub(MultiPanel *Do=NULL);
  ~Hub() {cleanup();}

  int cleanup(int ShutIOOnly=0);

  void reset(MultiPanel *Do) {Doer = Do;};
  int init(u_short Port=1857);

  int CtrlProcessor(int fd, void *data);
  int CmdProcessor(int fd, void *data);
  int DataProcessor(int fd, void *data);

  int GulpBinData();
  int GulpASCIIData();
  int PlotData(int ndx, float Data, int C);

private:
  SockIO CtrlPort;
  SockIO CmdIO, DataIO;
  MultiPanel  *Doer;
  int NoOfCols, ColAsX, NPoints, NoOfPanels, Counter;
  int **A;
  float *Data;
  vector<int> NoOfOverlays;
  int Scroll;
  short int Big_Endian;
  //  int (Hub::*)LoadData();
};

#endif

