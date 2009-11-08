# -*- makefile -*-
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
SCISRC       = /export/home/langur/sbhatnag/Projects/GoogleCode/capps/SCI
SCILIBDIR    = $(SCISRC)/src/lib
SCIINCLDIR   = $(SCISRC)/src/include
CASADISTRO   = casa_ccore/
SYNTHESISPKG = casa_ccore/
COREPKG      = casa_ccore/
DISPLAYPKG   = casa_ccore/

CASACORENAME = casacore
WCSPKG       = $(CASAROOT)/$(ARCH)/lib
PGPLOTPKG    = $(CASAROOT)/$(ARCH)/lib
WCSPKG       = /usr/lib64
PGPLOTPKG    = /usr/lib64

C++          = /usr/bin/g++
DEFS64       = -DAIPS_64B
#OPT64        = -m64
USRLIB       = /usr/lib64
X11LIB       = /usr/X11R6/lib
OPT          = -g $(OPT64)

SYNTHROOT    = $(CASAHOME)/$(SYNTHESISPKG)/$(ARCH)
COREROOT     = $(CASAHOME)/$(COREPKG)/$(ARCH)
DISPLAYROOT  = $(CASAHOME)/$(DISPLAYPKG)/$(ARCH)

CASAROOT     = $(CASAHOME)/$(CASADISTRO)
#QT4HOME      = /usr/local/Trolltech/Qt-4.1.3
#QT4HOME      = /usr/local/lib
QT4HOME      = /usr/
QT4LIBD      = $(QT4HOME)/lib
SCIBIN       = $(SCIHOME)/linux_gnu/bin