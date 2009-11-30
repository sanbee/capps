/* $Id: ltaDefaultErrorHandler.cc,v 1.2 1999/03/23 05:46:19 sanjay Exp $ */
#include <stdio.h>
#ifdef __cplusplus
#include <iostream.h>
#include <stdlib.h>
#include <ErrorObj.h>
#include <ExitObj.h>

void Groan(ErrorObj &x, char *Throes, char *Aaahhh, char *LastWords, 
	   char *Type)
{
  x << Throes << Type << endl;
  x << Aaahhh << endl;
  x << "\t\"" << x.what() << "\"" << endl;
  x << LastWords << endl;
}

void MPDefaultErrorHandler()
{
  char *Throes    = "###Fatal: Uncaught exception of type ",
       *Aaahhh    = "###Fatal: The exceptional message was:",
       *LastWords = "###Fatal: Exiting the application.";

  try
    {
      throw;
    }
  catch(ExitObj& x)
    {
      Groan(x, Throes, Aaahhh, LastWords, "ExitObj");
    }
  catch(ErrorObj& x)
    {
      Groan(x, Throes, Aaahhh, LastWords, "ErrorObj");
    }
  catch (...)
    {
      ErrorObj x;
      x << Throes << "UNKNOWN" << endl;
      x << LastWords << endl;
    }
  exit(-1);
}


#endif
