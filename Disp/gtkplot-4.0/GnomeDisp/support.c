/* $Id: support.c,v 2.3 1999/01/25 04:03:33 sanjay Exp $ */
#include <stdio.h>
#include <string.h>
/*#include <shell.h>*/
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define  getmem(a,b)    calloc(1,(a))

#ifdef __cplusplus
extern "C" {
#endif
/*---------------------------------------------------------------------------*/
/*----------------------------------------------------------------------
   Puts a NULL character after the first alpha-numeric character 
   that it finds in buf, starting from the back.
----------------------------------------------------------------------*/
void cltruncateFromBack(char *buf)
{
  int n;
  n=strlen(buf)-1;
  
  while(((buf[n] == '\t') || (buf[n] == ' ') || (buf[n] == '\n')) && (n >= 0))
    n--;
  n++;
  if (n>0) buf[n] = '\0';
}
/*----------------------------------------------------------------------
   Loads the source in the target buffer.  The mem. for target
   is allocated inside by malloc.  If source==NULL, the source
   is filled with default given by def.
----------------------------------------------------------------------*/
void clloadBuf(char **target, char *source, char *def)
{
  int n;
  n = strlen(source);
  if (n<=0) 
    {
      if (def != NULL) 
	{
	  n = strlen(def);
	  if (*target != NULL) free(*target);
	  *target = (char *)getmem(sizeof(char)*(n+1),"loadBuf");
	  strcpy(*target,def);
	}
      else
	{
	  if (*target != NULL) free(*target);
	  *target = NULL;
	}
    }
  else
    {
      if (*target != NULL) free(*target);
      *target = (char *)getmem(sizeof(char)*(n+1),"loadBuf");
      strcpy(*target,source);
    }
  source[0]='\0';
}
/*----------------------------------------------------------------------
   Strip any leading white spaces (' ',TAB).  This is useful while 
   reading strings typed by humans.
----------------------------------------------------------------------*/
void stripwhite (char *string)
{
  register int i = 0;
  
  if (string!=NULL)
    {
      while (string[i] == ' ' || string[i]=='\t') i++;
      if (i) strcpy (string, string + i);
      i = strlen (string) - 1;
      while (i > 0 && (string[i]==' '||string[i]=='\t')) i--;
      string[++i] = '\0';
    }
}
/*----------------------------------------------------------------------
  Break a string of the type <Name>=<Value> into Name and Value.
----------------------------------------------------------------------*/
int BreakStr(char *str, char **Name, char **val)
{
  char *t,*off;

  if ((off = strchr(str,'='))) off++;
  
  if ((t=strtok(str,"="))!=NULL)
    {
      stripwhite(t);
      *Name = (char *)getmem(strlen(t)+1,"BreakStr");
      strcpy(*Name,t);
    }
  else *Name = NULL;
  if (off)
    {
      stripwhite(off);
      *val = (char *)getmem(strlen(off)+1,"BreakStr");
      strcpy(*val,off);
    }
  else *val = NULL;
  return 1;
}
#ifdef __cplusplus
}
#endif
