#
# Most of the user defined variables are here.  More complicated
# compiler and linker variables are in the makefile.  It is hoped that
# the end-user will need to set up only this file - but again, it is
# only a hope. :-)
#
ARCH         = linux_gnu
CASAHOME     = /export/home/langur/sbhatnag
CLHOME       = /export/home/langur/sbhatnag/SCI/linux_gnu
SCIHOME      = /export/home/langur/sbhatnag/SCI
CASADISTRO   = casa_daily
SYNTHESISPKG = casa_daily
COREPKG      = casa_daily
DISPLAYPKG   = casa_daily

C++          = /usr/bin/g++
#DEFS64       = -DAIPS_64B
#OPT64        = -m64
USRLIB       = /usr/lib
X11LIB       = /usr/X11R6/lib
OPT          = -g $(OPT64)

SYNTHROOT    = $(CASAHOME)/$(SYNTHESISPKG)
COREROOT     = $(CASAHOME)/$(COREPKG)
DISPLAYROOT  = $(CASAHOME)/$(DISPLAYPKG)

CASAROOT     = $(CASAHOME)/$(CASADISTRO)
#QT4HOME      = /usr/local/Trolltech/Qt-4.1.3
#QT4HOME      = /usr/local/lib
QT4HOME      = /usr/
QT4LIBD      = $(QT4HOME)/lib
SCIBIN       = $(SCIHOME)/$(ARCH)/bin
