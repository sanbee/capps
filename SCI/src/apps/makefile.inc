#
# Most of the user defined variables are here.  More complicated
# compiler and linker variables are in the makefile.  It is hoped that
# the end-user will need to set up only this file - but again, it is
# only a hope. :-)
#
ARCH         = linux_64b
CASAHOME     = /export/home/langur/sbhatnag
CLHOME       = /export/home/langur/sbhatnag/SCI/linux_gnu
SCIHOME      = /export/home/langur/sbhatnag/SCI
CASADISTRO   = casa_ccore/
SYNTHESISPKG = casa_ccore/
COREPKG      = casa_ccore/
DISPLAYPKG   = casa_ccore/

C++          = /usr/bin/g++
DEFS64       = -DAIPS_64B
#OPT64        = -m64
USRLIB       = /usr/lib64
X11LIB       = /usr/X11R6/lib
OPT          = -g $(OPT64)

SYNTHROOT    = $(CASAHOME)/$(SYNTHESISPKG)/linux_64b
COREROOT     = $(CASAHOME)/$(COREPKG)/linux_64b
DISPLAYROOT  = $(CASAHOME)/$(DISPLAYPKG)/linux_64b

CASAROOT     = $(CASAHOME)/$(CASADISTRO)
#QT4HOME      = /usr/local/Trolltech/Qt-4.1.3
#QT4HOME      = /usr/local/lib
QT4HOME      = /usr/
QT4LIBD      = $(QT4HOME)/lib
SCIBIN       = $(SCIHOME)/linux_gnu/bin
