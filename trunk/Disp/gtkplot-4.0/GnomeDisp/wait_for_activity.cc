#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <vector>
#include <namespace.h>

int mygetdtablehi();

int wait_for_activity(vector<int>& descriptors)
{
  fd_set fdset;
  int n;
  FD_ZERO(&fdset);
  for (unsigned int i=0;i<descriptors.size();i++)
    FD_SET(descriptors[i],&fdset);
  n=select(mygetdtablehi(), &fdset, NULL, NULL, NULL);
  if (n<0) {perror("select");exit(-1);}
  else if (n>0)
      {
      for (unsigned int i=0;i<descriptors.size();i++)
	if (FD_ISSET(descriptors[i],&fdset)) return descriptors[i];
      return -1;
      }
  else return -1;
}
