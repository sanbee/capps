//
//----------------------------------------------------------------
// Get the value of the highest file id used by this process.
//
int mygetdtablehi()
{
  int n=0;
  for (int i=0;i<getdtablesize();i++)
    if (FD_ISSET(i,&fdset)) n=i;
  return n+1;
}
