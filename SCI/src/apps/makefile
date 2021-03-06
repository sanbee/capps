# -*- makefile -*-
# ifeq (0,${MAKELEVEL})
# whoami    := $(shell whoami)
# host-type := $(shell arch)
# HOST := $(shell hostname)
# endif
#
# HOSTNAME is defined by make with the OS command as the value, which
# when executed, returns the name of the host.
#
HOST := $(shell $(HOSTNAME))
include makefile.casacore.inc.$(HOST)

CPPFLAGS = $(OPT) -Wno-deprecated                           \
           $(DEFS64)                                        \
           -DHAVE_QT4                                       \
           -DAIPS_NEEDS_RETHROW                             \
           -DAIPS_LINUX                                     \
           -DAIPS_LITTLE_ENDIAN                             \
           -DAIPS_STDLIB                                    \
           -DAIPS_NO_LEA_MALLOC                             \
           -D_GLIBCPP_DEPRECATED                            \
	   -DAIPS_AUTO_STL                                  \
#	   -DAIPS_DEBUG                                     \
	   -DAIPS_ARRAY_INDEX_CHECK                         \
	   -DSIGNBIT                                        \
	   -DAIPS_NO_TEMPLATE_SRC                           \
	   -Wall -pipe -Wno-non-template-friend             \
	   -Woverloaded-virtual -Wcast-align                \
	   -Wno-comment -D_FILE_OFFSET_BITS=64              \
	   -D_LARGEFILE_SOURCE  


PYFLAGS = -DPYTHONPATH=\"$(USRLIB)/casapy/bin/$(USRLIB)/casapy/bin/python\" \
	  -DPYTHONVER=\"2.5\"                                               \
	  -DPYTHONLIB=\"/export/home/langur/sbhatnag/casa_rnd/linux_gnu/lib:/export/home/langur/sbhatnag/casa_rnd/linux_gnu/libexec/python\"

#PYINCL  = -I$(USRLIB)/casapy/include/python2.5 -I$(USRLIB)/casapy/lib/python2.5/site-packages/numpy/core/include
#PYINCL  = -I$(USRLIB)/casapy/include/python2.5 -I$(USRLIB)/casapy/include/boost_python -I$(USRLIB)/casapy/lib/python2.5/site-packages/numpy/core/include

#PYLIBS  =  -L$(USRLIB)/casapy/lib/python2.5/config -lpython2.5  -L$(USRLIB)/casapy/lib -lCCM_Python -lWX_Utils_error -lWX_Utils_code -lWX_Utils_types
PYLIBS  =  -L$(USRLIB)/casapy/lib/python2.5/config -L$(USRLIB)/casapy/lib

INCL = -I.                                   \
       -I$(SCIINCLDIR)                       \
       -I$(CLHOME)/include                   \
       -I$(COREROOT)/include/$(CASABASE) \
       -I$(COREROOT)/include/$(CASACORENAME) \
       -I$(CASAROOT)/include/$(CASACODENAME) \
       -I$(CASAROOT)/include \
       -I/usr/include/cfitsio                \
       -I/usr/include/ATM                \
       -I$(QT4HOME)/include                  \
       -I$(QT4HOME)/include/QtCore           \
       -I$(QT4HOME)/include/QtGui            \
       -I$(QT4HOME)/include/QtXml            \
       -I$(QT4HOME)/include/qwt

#DISPLAYLIBS = -lefence $(DISPLAYROOT)/lib/libdisplay.a       \
#	  $(DISPLAYROOT)/lib/libcasaqt.a            \
#	  $(DISPLAYROOT)/lib/libgraphics.a \
#	  $(WCSPKG)/libwcs.a              \
#	  $(PGPLOTPKG)/libpgsbox.so              
#
# Only PlotMS appears to depend on libgraphics.a - which appears to have a free/delete problem
#
#DISPLAYLIBS = $(DISPLAYROOT)/lib/libdisplay.a \
#	  $(DISPLAYROOT)/lib/libcasaqt.a      \
#	  $(WCSPKG)/libwcs.a                  \
#	  $(PGPLOTPKG)/libpgsbox.so              

DISPLAYLIBS = -L$(DISPLAYROOT)/lib -ldisplay \
	  -lcasaqt \
	  $(WCSPKG)/libwcs.so                  \
	  $(PGPLOTPKG)/libpgsbox.so              


# SYNTHESISLIBS = $(SYNTHROOT)/lib/libsynthesis.a \
# 	  $(SYNTHROOT)/lib/libcasadbus.a     \
# 	  $(SYNTHROOT)/lib/libcalibration.a     \
# 	  $(SYNTHROOT)/lib/libcasadbus.a \
# 	  -laatm
# #	  $(SYNTHROOT)/lib/libgraphics.a         
SYNTHESISLIBS = -L$(SYNTHROOT)/lib \
		-lsynthesis -lcasadbus
#		-lsynthesis -lcasadbus -laatm
#		-lsynthesis -lcasadbus -lcalibration -lcasadbus -laatm


# CORELIBS   = $(COREROOT)/lib/libcasa_msfits.so   \
# 	  $(COREROOT)/lib/libmsvis.so            \
# 	  $(COREROOT)/lib/libcasa_ms.so          \
# 	  $(COREROOT)/lib/libcasa_images.so      \
# 	  $(COREROOT)/lib/libcasa_components.so  \
# 	  $(COREROOT)/lib/libcasa_coordinates.so \
# 	  $(COREROOT)/lib/libcasa_lattices.so    \
# 	  $(COREROOT)/lib/libcasa_fits.so        \
# 	  $(COREROOT)/lib/libcasa_measures.so    \
# 	  $(COREROOT)/lib/libcasa_measures_f.so  \
# 	  $(COREROOT)/lib/libcasa_tables.so      \
# 	  $(COREROOT)/lib/libcasa_scimath.so     \
# 	  $(COREROOT)/lib/libcasa_scimath_f.so   \
# 	  $(COREROOT)/lib/libcasa_casa.so 

#CORELIBS = -lcasa_msfits -lmsvis -lcasa_ms -lcasa_images        \
#            -lcomponents -lcasa_coordinates -lcasa_lattices \
#            -lcasa_fits -lcasa_measures \
#	    -lcasa_tables -lcasa_scimath -lcasa_scimath_f        \
#            -lcasa_casa -lflagging -lcasaqt
CORELIBS = -lcasa_msfits -lcasa_ms -lcasa_images -limageanalysis       \
            -lcomponents -lcasa_coordinates -lcasa_lattices \
            -lcasa_fits -lcasa_measures \
	    -lcasa_tables -lcasa_scimath -lcasa_scimath_f        \
            -lcasa_casa -lflagging -lcasaqt

CASALIBS = $(CORELIBS) $(SYNTHESISLIBS) $(DISPLAYLIBS)

CAAPSLIBOBJS = casaChecks.o
ifneq ($(strip $(SCILIBDIR)),)
CAAPSLIB = $(SCILIBDIR)/libcaaps.a
endif
#CASAAPPSLIBS = 

ifneq ($(strip $(CLHOME)),)
UILIBS = $(CLHOME)/lib/libshccl.a                 \
         $(CLHOME)/lib/libsh.a                    \
         $(CLHOME)/lib/libshccl.a                 \
	 $(CLHOME)/lib/libcalc.a                  \
	 -lreadline -lhistory -ltermcap -lm -lc
#	 -lreadline -lhistory -ltermcap -lm -lfl -lc
endif
# -lg2c  

LINKEROPTS = -Xlinker -rpath -Xlinker                       \
	     $(CASAROOT)/$(ARCH)/lib:$(USRLIB):$(X11LIB):$(USRLIB)/casapy/lib:/usr/local/lib: \
	     -Wl,-rpath -Wl,$(QT4LIBD)


LIBOPTS = -L$(COREROOT)/lib -L$(CASAROOT)/lib    \
	  -L$(USRLIB)                            \
	  -L$(QT4LIBD)                           \
	  -lcfitsio -llapack -lblas              \
	  -L/usr/local/lib -lcpgplot -lpgplot    \
	  -L$(X11LIB) -lXt -lXmu -lX11 -lXext    \
	  $(WCSPKG)/libwcs.so                     \
	  $(PGPLOTPKG)/libpgsbox.so              

# INCLUDE THE FOLLOWING FOR IMAGER!

#	  -lQtCore -lQtGui -lQtDBus -lQtDBus_debug -ldbus-c++-1 -lQtXml -lqwt -lgomp

#	  -lg2c -L/usr/local/lib -lcpgplot -lpgplot     \
#
#	  /usr/lib64/libcfitsio.so.0 -llapack -lblas    \
#	  -lcfitsio -llapack -lblas                     \
#-------------------------------------------------------------------------------------------------
# Variables that control the generic targets below
#
#
# The list of source file names 
#
SOURCE  = timager.cc tEPJones.cc imasking.cc mssplit.cc tableinfo.cc 
SOURCE  = $(wildcard *.cc)
ifneq ($(strip $(SCILIBDIR)),)
CAAPSLIBSRC = $(wildcard $(SCILIBDIR)/*.cc)
endif
#
# The list of applications to be built derived from the list of source files in $(SOURCE)
#
APPS    = $(patsubst %.cc,%,$(SOURCE))
#
# The list object files derived from the list of application in $(APPS)
#
OBJECTS = $(patsubst %,%.o,$(APPS))
ifneq ($(strip $(SCILIBDIR)),)
CAAPSLIBOBJS = $(patsubst %.cc, %.o, $(CAAPSLIBSRC))
endif
#
#-------------------------------------------------------------------------------------------------
#
# The actual tagets
#
all: $(APPS)	

install: all
	strip $(APPS); cp $(APPS) $(SCIBIN)

$(CAAPSLIB): $(CAAPSLIBOBJS) 
ifneq ($(strip $(SCILIBDIR)),)
	ar r $(CAAPSLIB) $(CAAPSLIBOBJS)
endif
$(CAAPSLIBOBJS): %.o: %.cc
	$(C++) -c $(PYFLAGS) $(CPPFLAGS) $(INCL) $(PYINCL) $< -o $@

$(OBJECTS): %.o: %.cc 
	 $(C++) -c $(PYFLAGS) $(CPPFLAGS) $(INCL) $(PYINCL) $< -o $@

$(APPS): % : %.o $(CAAPSLIB)
	 $(C++) $< $(PYFLAGS) $(CPPFLAGS) $(LINKEROPTS) $(LIBOPTS) \
	 $(CASALIBS) $(PYLIBS) $(UILIBS) $(CAAPSLIB) -o $@
#	 $(C++) $< $(PYFLAGS) $(CPPFLAGS) $(LINKEROPTS) $(CASALIBS) $(LIBOPTS) $(PYLIBS) $(UILIBS) -o $@
#
#-------------------------------------------------------------------------------------------------
#
.PHONY: clean allclean

depend:
	makedepend $(INCL) $(SOURCE)

clean: 
	\rm -rf *.o
	cd $(SCILIBDIR); \rm -rf *.{o,a}

allclean: clean
	\rm -f $(APPS)
