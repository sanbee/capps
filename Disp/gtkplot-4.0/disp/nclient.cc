#include <stdio.h>
#include <mp.h>
#include <mach.h>

int nDock(char *Host, SockIO &CmdS, SockIO &DataS)
{
  SockIO S0;
  int cmdport,dataport;
  char Msg[128];
  int n;
  short int Big_Endian;

  MachineSpecType Mac;

  GetMachineSpec(&Mac);
  Big_Endian = Mac.big_endian;

  //
  // Open the "well known" port.
  //
  S0.init(1857,0,Host);
  S0.konnekt();

  n=S0.Send(GREETINGS,strlen(GREETINGS));
  //
  // Get the command port and dock on it
  //
  n=S0.Receive(Msg,128); Msg[n]='\0';
  //  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&cmdport);
  fprintf(stderr,"\nnclient: Got command port no. %d\n",cmdport);

  CmdS.init(cmdport,0,Host);
  CmdS.konnekt();
  sleep(1);
  //
  // Send the first command....announce the endian type of your 
  // client
  //
  sprintf(Msg,"Big_Endian %d\0",Big_Endian);
  n=CmdS.Send(Msg,strlen(Msg));

  cerr << "nclient: Connected for commands on port " << cmdport << endl;
  /*----------------------------------------------------------------*/
  //
  // Get the data port and dock on it
  //
  n=S0.Receive(Msg,128); Msg[n]='\0';
  //  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&dataport);
  fprintf(stderr,"\nnclient: Got data port no. %d\n",dataport);

  DataS.init(dataport,0,Host);
  DataS.konnekt();

  sleep(1);
  cerr << "nclient: Connected for data on port " << dataport << endl;

  S0.shutup(2);
  return 1;
}  

int main(int argc, char *argv[])
{
  int n;
  SockIO CmdSock, DataSock;

  nDock("localhost", CmdSock, DataSock);
  scanf("%d",&n);
}









