#if !defined(COMMH_H)
#define COMMH_H
#include <SockIO.h>
#include <fcntl.h>
#include <sys/stat.h>

int CommH(int CmdPort, int DataPort,SockIO& CmdSock, SockIO& DataSock);
int nnCommH(int CmdPort, int DataPort,int& CmdDesc, int& DataDesc);
int nDock(char *Host, SockIO &CmdS, SockIO &DataS);

#endif
