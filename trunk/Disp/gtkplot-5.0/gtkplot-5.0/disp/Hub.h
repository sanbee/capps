//-*-C++-*-
//$Id$
#if !defined(HUB_H)
#define HUB_H
#include <SockIO.h>
#include <Protocol.h>
#include <MultiPanel.h>
#include <vector>
#include <exception>
#include "./WidgetLib.h"

extern  void MPDefaultErrorHandler();


#define MSG_BUFLEN 16384
extern "C" 
{
  void hub_cmd_io_callback(gpointer data,gint fd,GdkInputCondition condition);
  void hub_data_io_callback(gpointer data,gint fd,GdkInputCondition condition);
  void hub_ctrl_io_callback(gpointer data,gint fd,GdkInputCondition condition);
};

class Hub
{
public:
  Hub(MultiPanel *Do=NULL);
  ~Hub() {cleanup();}

  int cleanup(int ShutIOOnly=0);

  void reset(MultiPanel *Do) {Doer = Do;};
  //  int init(u_short Port=1857);
  int init(u_short CmdPort, u_short DataPort);

  //  int CtrlProcessor(int fd, void *data); // Not used anymore
  int CmdProcessor(int fd, void *data);
  int DataProcessor(int fd, void *data);

  int GulpBinData();
  int GulpASCIIData();
  int PlotData(int ndx, gdouble Data, int C);
  void Start();

private:
  SockIO CtrlPort;
  SockIO CmdIO, DataIO;
  //  int CmdIODescriptor, DataIODescriptor;
  MultiPanel  *Doer;
  int NoOfCols, ColAsX, NPoints, NoOfPanels, Counter;
  gint CtrlCallbackTag;
  int **A;
  float *Data;
  vector<int> NoOfOverlays;
  int Scroll;
  short int Big_Endian;
  gint DataMonitorTag, CmdMonitorTag;
  //  int (Hub::*)LoadData();
};

#endif

