# $Id: makefile,v 18.0 2002/06/07 21:30:21 aips2adm Exp $

#XLIBLIST:= MIRLIB GLISH PGPLOT LAPACK F77 WCSLIB IDL Motif XPM X11

# Use the generic AIPS++ application makefile.
#---------------------------------------------
#include $(word 1, $(AIPSPATH))/code/install/makefile.app
#XTRNLIBS += -L/usr/local/lib -lfftw
FFTWINCL = -I/usr/local/include
FFTWLIB  = /usr/lib64

BASE_DIR = /home/aips++/daily

GLIBRARIES = 

GINCLUDES = -I$(BASE_DIR)/code/aips/glish/include


C++ = g++32 -W -m64
CC = gcc32 -m64
#-DDEBUG=1
GSLHOME=/home/rishi/sanjay/local
GSLL=$(GSLHOME)/lib
GSLI=$(GSLHOME)/include/gsl

OPT = -O3 -L/export/home/langur/sanjay/lib
INCL = -I. $(FFTWINCL) -I$(GSLI) $(GINCLUDES) -L$(BASE_DIR)/linux_amd64/lib
CFLAGS = -static $(OPT) $(GSLL)/libgsl.a -lglish -lsos -lnpd $(GSLL)/libgslcblas.a -lm $(FFTWLIB)/libfftw3.a
#-lrfftw -lfftw

#CFLAGS = $(OPT) -g -lgsl -lglish -lsos -lnpd -lgslcblas -lm -lrfftw -lfftw

LIBS = vector_bfgs.o Convolver.o

ErrorObj.o: ErrorObj.cc ErrorObj.h
	$(C++) $(INCL) -c ErrorObj.cc

Array2D.o: Array2D.cc Array2D.h
	$(C++) $(INCL) -c Array2D.cc

Convolver.o: Convolver.n.cc Convolver.n.h
	$(C++) $(OPT) $(INCL) -c Convolver.n.cc -o Convolver.o

tConvolver: tConvolver.cc Convolver.o Array2D.o
	$(C++) $(INCL)  tConvolver.cc  -o tConvolver Convolver.o $(CFLAGS)

tFluxon2D: tFluxon2D.cc regFluxon2D.h Fluxon2D.h $(LIBS) Array2D.h
	$(C++) $(INCL) tFluxon2D.cc -o tFluxon2D $(LIBS) $(CFLAGS) 

tFluxonComplist: tFluxonComplist.cc $(LIBS) 
	$(C++) $(INCL) $(CFLAGS) tFluxonComplist.cc $(LIBS) -o tFluxonComplist

tstExp: tstExp.cc Exp.h $(LIBS)
	$(C++) $(INCL) $(CFLAGS) tstExp.cc $(LIBS) -o tstExp

psfdecomp: psfdecomp.cc Fluxon2D.h FluxonCompList.h $(LIBS)
	$(C++) $(INCL)  psfdecomp.cc  -o psfdecomp $(LIBS) $(CFLAGS)

lmdr_psfdecomp: lmdr_psfdecomp.cc Fluxon2D.h FluxonCompList.h $(LIBS)
	$(C++) $(INCL)  lmdr_psfdecomp.cc  -o lmdr_psfdecomp $(LIBS) $(CFLAGS)

ssdeconv: ssdeconv.cc Fluxon2D.h FluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) ssdeconv.cc -o ssdeconv $(LIBS) $(CFLAGS) 

ssdeconv2: ssdeconv2.cc Fluxon2D.h FluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) ssdeconv2.cc -o ssdeconv2 $(LIBS) $(CFLAGS) 

tstclient: tstclient.cc Fluxon2D.h FluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) tstclient.cc -o tstclient $(LIBS) $(CFLAGS) 

nssdeconv2.n: nssdeconv2.atlas.2.cc Fluxon2D.h FluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) nssdeconv2.atlas.2.cc -o nssdeconv2 $(LIBS) $(CFLAGS) 

nssdeconv2: nssdeconv2.cc Fluxon2D.h FluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) nssdeconv2.cc -o nssdeconv2 $(LIBS) $(CFLAGS) 

regss: regss.cc regFluxon2D.h regFluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) regss.cc -o regss $(LIBS) $(CFLAGS) 

tfft: tfft.cc $(LIBS)
	$(C++) $(INCL) $(CFLAGS) tfft.cc -o tfft $(LIBS) 

fpsf: fpsf.cc Fluxon2D.h FluxonCompList.h $(LIBS) vector_bfgs.o
	$(C++) $(INCL) $(CFLAGS) fpsf.cc vector_bfgs.o -o fpsf $(LIBS) 

gsslm: gsslm.cc Fluxon2D.h FluxonCompList.h $(LIBS)
	$(C++) $(INCL) $(CFLAGS) gsslm.cc -o gsslm $(LIBS) 

directional_minimize.o: directional_minimize.c 
	$(CC) -O2 $(INCL) -c directional_minimize.c

vector_bfgs.o: vector_bfgs.c directional_minimize.c
	$(C++) -O2 $(INCL) -c vector_bfgs.c

clean:
	\rm -f ss gss tFluxon *.o
