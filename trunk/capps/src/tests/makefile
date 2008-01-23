C++          = /usr/bin/g++
OPT          = -g
CASAHOME     = /export/home/langur/sbhatnag
ARCH         = linux_gnu
CASADISTRO   = casa_rnd
SYNTHESISPKG = casa_rnd
COREPKG      = casa_rnd
DISPLAYPKG   = casa_rnd

SYNTHROOT    = $(CASAHOME)/$(SYNTHESISPKG)
COREROOT     = $(CASAHOME)/$(COREPKG)
DISPLAYROOT  = $(CASAHOME)/$(DISPLAYPKG)

CASAROOT     = $(CASAHOME)/$(CASADISTRO)
QT4HOME      = /usr/local/Trolltech/Qt-4.1.3
CLHOME       = /export/home/langur/sbhatnag/SCI

SCIROOT      = ~/SCI
SCIBIN       = $(SCIROOT)/$(ARCH)/bin

CPPFLAGS = $(OPT) -Wno-deprecated                           \
           -DHAVE_QT4                                       \
           -DAIPS_NEEDS_RETHROW                             \
           -DAIPS_LINUX                                     \
           -DAIPS_LITTLE_ENDIAN                             \
           -DAIPS_STDLIB                                    \
           -DAIPS_NO_LEA_MALLOC                             \
           -D_GLIBCPP_DEPRECATED                            \
	   -DAIPS_AUTO_STL                                  \
	   -DAIPS_DEBUG                                     \
	   -DAIPS_ARRAY_INDEX_CHECK                         \
	   -DSIGNBIT                                        \
	   -DAIPS_NO_TEMPLATE_SRC                           \
	   -Wall -pipe -Wno-non-template-friend             \
	   -Woverloaded-virtual -Wcast-align                \
	   -Wno-comment -D_FILE_OFFSET_BITS=64              \
	   -D_LARGEFILE_SOURCE  


PYFLAGS = -DPYTHONPATH=\"/usr/lib/casapy/bin//usr/lib/casapy/bin/python\" \
	  -DPYTHONVER=\"2.5\"                                             \
	  -DPYTHONLIB=\"/export/home/langur/sbhatnag/casa_rnd/linux_gnu/lib:/export/home/langur/sbhatnag/casa_rnd/linux_gnu/libexec/python\"

PYINCL  = -I/usr/lib/casapy/include/python2.5 -I/usr/lib/casapy/lib/python2.5/site-packages/numpy/core/include

PYLIBS  =  -L/usr/lib/casapy/lib/python2.5/config -lpython2.5  -L/usr/lib/casapy/lib -lCCM_Python -lWX_Utils_error -lWX_Utils_code -lWX_Utils_types

INCL = -I.                                                  \
       -I$(CLHOME)/$(ARCH)/include/parafeed                 \
       -I/export/home/langur/sbhatnag/CASADEVL/rnd/code/include \
       -I$(CASAROOT)/code/include                           \
       -I$(CASAROOT)/code/casa                              \
       -I/usr/include/cfitsio                               \
       -I$(QT4HOME)/include                                 \
       -I$(QT4HOME)/include/QtCore                          \
       -I$(QT4HOME)/include/QtGui                           \
       -I$(QT4HOME)/include/QtXml

DISPLAYLIBS = $(DISPLAYROOT)/$(ARCH)/lib/libdisplay.a       \
	  $(DISPLAYROOT)/$(ARCH)/lib/libgraphics.a         

SYNTHESISLIBS = $(SYNTHROOT)/$(ARCH)/lib/libsynthesis.a     \
	  $(SYNTHROOT)/$(ARCH)/lib/libcalibration.a      
#	  $(SYNTHROOT)/$(ARCH)/lib/libgraphics.a         

CORELIBS   = $(COREROOT)/$(ARCH)/lib/libmsfits.a        \
	  $(COREROOT)/$(ARCH)/lib/libmsvis.a            \
	  $(COREROOT)/$(ARCH)/lib/libms.a               \
	  $(COREROOT)/$(ARCH)/lib/libimages.a           \
	  $(COREROOT)/$(ARCH)/lib/libcomponents.a       \
	  $(COREROOT)/$(ARCH)/lib/libcoordinates.a      \
	  $(COREROOT)/$(ARCH)/lib/liblattices.a         \
	  $(COREROOT)/$(ARCH)/lib/libfits.a             \
	  $(COREROOT)/$(ARCH)/lib/libmeasures.a         \
	  $(COREROOT)/$(ARCH)/lib/libmeasures_f.a       \
	  $(COREROOT)/$(ARCH)/lib/libtables.a           \
	  $(COREROOT)/$(ARCH)/lib/libscimath.a          \
	  $(COREROOT)/$(ARCH)/lib/libscimath_f.a        \
	  $(COREROOT)/$(ARCH)/lib/libcasa.a 


CASALIBS = $(DISPLAYLIBS) $(SYNTHESISLIBS) $(CORELIBS)

UILIBS = $(CLHOME)/$(ARCH)/lib/libshccl.a                 \
         $(CLHOME)/$(ARCH)/lib/libsh.a                    \
	 $(CLHOME)/$(ARCH)/lib/libcalc.a                  \
	 -lreadline -lhistory -ltermcap -lm -lfl -lc -lg2c  

LINKEROPTS = -Xlinker -rpath -Xlinker                       \
	     $(CASAROOT)/$(ARCH)/lib:/usr/lib:/usr/X11R6/LessTif/Motif2.1/lib:/usr/X11R6/lib:/usr/lib/casapy/lib:/usr/local/lib: \
	     -Wl,-rpath -Wl,/usr/local/Trolltech/Qt-4.1.3/lib 

#	     -Xlinker -rpath -Xlinker                       \
#	     $(CASAROOT)/$(ARCH)/lib:/usr/lib/casapy/lib:/usr/lib:/usr/X11R6/LessTif/Motif2.1/lib:/usr/X11R6/lib:/usr/local/lib: \

LIBOPTS = -L$(CASAROOT)/$(ARCH)/lib                     \
	  -L/usr/lib                                    \
	  -L/usr/lib -L/usr/lib                         \
	  -L/usr/local/Trolltech/Qt-4.1.3/lib -lmir     \
	  -lcfitsio -llapack -lblas                     \
	  -lg2c -L/usr/local/lib -lcpgplot -lpgplot     \
	  -L/usr/X11R6/lib -lXt -lXmu -lX11 -lXext      \
	  $(CASAROOT)/$(ARCH)/lib/libwcs.a              \
	  -lQtCore -lQtGui -lQtXml 
#
#-------------------------------------------------------------------------------------------------
# Variables that control the generic targets below
#
#
# The list of source file names 
#
SOURCE  = timager.cc tEPJones.cc imasking.cc mssplit.cc tableinfo.cc
SOURCE  = $(wildcard *.cc)
#
# The list of applications to be built derived from the list of source files in $(SOURCE)
#
APPS    = $(patsubst %.cc,%,$(SOURCE))
#
# The list object files derived from the list of application in $(APPS)
#
OBJECTS = $(patsubst %,%.o,$(APPS))
#
#-------------------------------------------------------------------------------------------------
#
# The actual tagets
#
all: $(APPS)	

install: all
	strip $(APPS); cp $(APPS) $(SCIBIN)

$(OBJECTS): %.o: %.cc 
	 $(C++) -c $(PYFLAGS) $(CPPFLAGS) $(INCL) $(PYINCL) $< -o $@

$(APPS): % : %.o $(CASALIBS)
	 $(C++) $< $(PYFLAGS) $(CPPFLAGS) $(LINKEROPTS) $(CASALIBS) $(LIBOPTS) $(PYLIBS) $(UILIBS) -o $@
#
#-------------------------------------------------------------------------------------------------
#
.PHONY: clean allclean

depend:
	makedepend $(INCL) $(SOURCE)

clean: 
	\rm -rf *.o

allclean: clean
	\rm -f $(APPS)
