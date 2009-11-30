#if !defined(CL_H)
/* $Id: cl.h,v 1.14 1998/09/08 09:34:12 sanjay Exp sanjay $ */
#define CL_H
#include <clshelldefs.h>
/* #include <clinteract.h> */
#include <setjmp.h>
#include <stdio.h>
#include <math.h>
#ifdef HANDLE_EXCEPTIONS
#undef HANDLE_EXCEPTIONS
#endif

#ifdef __cplusplus
#include <clError.h>

#ifdef	__cplusplus
extern "C" {
#endif

#define HANDLE_EXCEPTIONS(str)  try         \
                                  {         \
                                    str     \
                                  }         \
                                 catch(...) \
                                  {         \
                                    throw;  \
                                  }
                                

#else

#define HANDLE_EXCEPTIONS(str)  str
#endif

#define FAIL -1
#define HIST_DEFAULT ".g_hist"
#define HIST_LIMIT   100
#define PROCEED      1
#define GOBACK       2
#define CL_INFORMATIONAL -30
#define CL_SEVERE    -20
#define CL_FATAL     -10
/*
   If the lib. is compiled for use in FORTRAN, the function names
   should all be lower case with a "_" appended to it.
*/
#ifdef FORTRAN
#define BeginCL          begincl_
#define EndCL            endcl_
#define clCmdLineFirst   clcmdlinefirst_
#define clgetOpt         clgetopt_
#define clgetNOpts       clgetnopts_
#define clgetNVals       clgetnvals_
#define clgetFull        clgetfull_
#define clgetFullVal     clgetfullval_
#define clgetIVal        clgetival_
#define clgetFVal        clgetfval_
#define clgetSVal        clgetsval_
#define clgetNIVal       clgetnival_
#define clgetNFVal       clgetnfval_
#define clgetNSVal       clgetnsval_
#define dbgclgetIVal     dbgclgetival_
#define dbgclgetFVal     dbgclgetfval_
#define dbgclgetSVal     dbgclgetsval_
#define dbgclgetNIVal    dbgclgetnival_
#define dbgclgetNFVal    dbgclgetnfval_
#define dbgclgetNSVal    dbgclgetnsval_
#define clRestartShell   clrestartshell_
#define clfInteractive   clfinteractive_
#define clgetCommandLine clgetcommandline_
#define clstrtStream     clstrtstream_
#define clgetConfigFile  clgetconfigfile_
#endif
/*
   CL package's internal stuff
*/
Symbol   *IntallSymb(char *, char *, Symbol *);
Symbol   *SearchQSymb(char *Name, char *Type);
Symbol   *SearchVSymb(char *Name, Symbol *Tab);
Symbol   *AddQKey(char *Name, char *Type, 
		  Symbol **Head, Symbol **Tail);
int       ParseCmdLine(int, char **);
/*
   Programmer's interface to CL package
*/
int       BeginCL(int argc, char **argv);
int       EndCL();
int       clgetNOpts();
void      clCmdLineFirst();
int       clTgetOpt(char *Name, char *Type);
int       clgetOpt(char *Name);
int       clgetNVals(char *Name);
int       clgetFull(char  *Arg,  int   *N);
int       clgetFullVal(char  *Name,  char **Val);
int       clgetIVal(char  *Name, int   *Val, int *N);
int       dbgclgetIVal(char  *Name, int   *Val, int *N);
int       clgetFVal(char  *Name, float *Val, int *N);
int       dbgclgetFVal(char  *Name, float *Val, int *N);
int       clgetSVal(char  *Name, char  *Val, int *N);
int       dbgclgetSVal(char  *Name, char  *Val, int *N);
int       clgetNIVal(char *Key,  int   *Val, int *m);
int       dbgclgetNIVal(char *Key,  int   *Val, int *m);
int       clgetNFVal(char *Name, float *Val, int *N);
int       dbgclgetNFVal(char *Name, float *Val, int *N);
int       clgetNSVal(char *Name, char **Val, int *N);
int       dbgclgetNSVal(char *Name, char **Val, int *N);
void      clRestartShell();
void      clStartInteractive(jmp_buf *, int);
int       clfInteractive(int *);
void      cltruncateFromBack(char *);
void      clloadBuf(char **target, char *source, char *def);
char      *clgetCommandLine();
char      *clgetInputFile();
char      *clgetOutputFile();
int       clgetOptsList(char ***);
int       clclearOptsList(char ***,int);
int       clloadConfig(char *);
void      clCleanUp();
void      clReset();
void      clRetry();
FILE      *clstrtstream_(char *, char *, char *);
void      stripwhite (char *);
int       redirect(char *, char *);
void      yyerror();
int       clgetConfigFile(char *, char *);
int       AddCmd(char *Name, char *Doc, int (*func)(char *), 
		 CmdSymbol **Head, CmdSymbol **Tail);
int       BreakStr(char *, char **, char **);
int       dogo(char *);
int       dogob(char *);
int       docd(char *);
int       doinp(char *);
int       doquit(char *);
int       doedit(char *);
int       dohelp(char *);
int       doexplain(char *);
int       dosave(char *);
int       doload(char *);
int       dotypehelp(char *);
int       doademo(char *);
int       doprintdoc(char *);
int       loadDefaults();
int       clparseVal(Symbol *, int *, double *);
int       PrintVals(FILE *,Symbol *);

void mkfilename(char *out,char *envvar,char *name,char *type);
void save_hist(char *EnvVar, char *Default);
void limit_hist(char *EnvVar, int Default);
void load_hist(char *EnvVar, char *Default);
int  InstallSymb();
int  sh_parse();
int  UnsetVar(Symbol *);
int  SetVar(char *, char *, Symbol *,short int);
void SetVal(char *, Symbol *, int);
int  CopyVSymb(Symbol *, Symbol *,int);
int  FreeVSymb(Symbol *);
int  FreeCSymb(CmdSymbol *);
int  DeleteCmd(char *, CmdSymbol**, CmdSymbol**);
void clSaneExit(int);
int  clThrowUp(const char *, const char *,int);

void clDefaultErrorHandler();
void clSigHandler(int);
#ifdef	__cplusplus
}
#endif
#if !defined(FORTRAN)
#include <clconvert.h>
#endif
#endif

