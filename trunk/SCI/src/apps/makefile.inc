#
# Most of the user defined variables are here.  More complicated
# compiler and linker variables are in the makefile.  It is hoped that
# the end-user will need to set up only this file - but again, it is
# only a hope. :-)
#
ARCH         = linux_gnu
CASAHOME     = /home/rishi/sanjay
CLHOME       = /home/rishi/sanjay/Projects/parafeed/linux_amd64
SCIHOME      = /home/rishi/sanjay/SCI
CASADISTRO   = casa_rnd64
SYNTHESISPKG = casa_rnd64
COREPKG      = casa_rnd64
DISPLAYPKG   = casa_rnd64

C++          = /usr/bin/g++
DEFS64       = -DAIPS_64B
OPT64        = -m64
USRLIB       = /usr/lib64
X11LIB       = /usr/X11R6/lib64
OPT          = -g $(OPT64)

SYNTHROOT    = $(CASAHOME)/$(SYNTHESISPKG)
COREROOT     = $(CASAHOME)/$(COREPKG)
DISPLAYROOT  = $(CASAHOME)/$(DISPLAYPKG)

CASAROOT     = $(CASAHOME)/$(CASADISTRO)
#QT4HOME      = /usr/local/Trolltech/Qt-4.1.3
#QT4HOME      = /usr/local/lib64
QT4HOME      = /usr/lib64/qt4
QT4LIBD      = $(QT4HOME)/lib64
SCIBIN       = $(SCIHOME)/$(ARCH)/bin
