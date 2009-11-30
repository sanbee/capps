#include <stdio.h>
#include <mp.h>
#include <mach.h>

int Dock(char *Host, SockIO &CmdS, SockIO &DataS)
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
  sprintf(Msg,"Big_Endian %d\0",Big_Endian);
  n=S0.Send(Msg,strlen(Msg));
  //
  // Get the command port and dock on it
  //
  n=S0.Receive(Msg,128); Msg[n]='\0';
  //  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&cmdport);
  fprintf(stderr,"Got command port no. %d\n",cmdport);

  CmdS.init(cmdport,0,Host);
  CmdS.konnekt();
  S0.Send(SURE,strlen(SURE));
  sleep(1);
  //  n=CmdS.Send(GREETINGS,strlen(GREETINGS));
  //  n=CmdS.Receive(Msg,strlen(SURE));

  /*----------------------------------------------------------------*/
  //
  // Get the data port and dock on it
  //
  n=S0.Receive(Msg,128); Msg[n]='\0';
  //  fprintf(stderr,"Got...%s\n",Msg);
  sscanf(Msg,"%d",&dataport);
  fprintf(stderr,"Got data port no. %d\n",dataport);

  DataS.init(dataport,0,Host);
  DataS.konnekt();
  S0.Send(SURE,strlen(SURE));
  sleep(1);

  S0.shutup(2);
  return 1;
}  


