#-*-makefile-*-
HOST := $(shell $(HOSTNAME))

ifeq ($(HOST),langur)
CLHOME           = /export/home/langur/sbhatnag/SCI/linux_gnu
endif

ifeq ($(HOST),rishi)
CLHOME           = /home/rishi/sanjay/Projects/parafeed/linux_gnu
endif

ifeq ($(HOST),atlas)
CLHOME           = /home/rishi/sanjay/Projects/parafeed/linux_gnu
endif

GTK_PLOT_VERSION = 5.0
OPTFLAGS         = -Wall -g 
CC               = gcc $(CFLAGS) $(OPTFLAGS)
C++              = g++ $(CFLAGS) $(OPTFLAGS)

LDFLAGS          = $(CLIBS) $(shell gtk-config --libs) -L/usr/X11R6/lib

#GNOME_LIBS       = -rdynamic -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lesd -laudiofile -lm -lglib -ldl
GNOME_LIBS       = -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lesd -laudiofile -lm -lglib -ldl
#GNOME_LIBS       = -rdynamic -lgnomeui -lart_lgpl -lgdk_imlib -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lgnome -lgnomesupport -lesd -laudiofile -lm -lglib -ldl
#GNOME_LIBS       = -rdynamic -lgnomeui -lart_lgpl -lgdk_imlib -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lgnome -lgnomesupport -lesd -laudiofile -lm -ldb1 -lglib -ldl

CLIBS            = libmp.a libderr.a  ../libgtkplot.a
ERROBJS          = ErrorObj.o MPDefaultErrorHandler.o
LIBOBJS          = scrollbuf.o MultiPanel.o XYPanel.o SockIO.o nHub.o Dock.o \
	           CommH.o MakeColorList.o 
LIBOBJS_C        = clstring.o mach.o hvs.o
#	           GnomeWin.o callback.o

UILIBS           = $(CLHOME)/lib/libshccl.a                 \
                   $(CLHOME)/lib/libsh.a                    \
                   $(CLHOME)/lib/libcalc.a                  \
                   -lreadline -lhistory -ltermcap -lm -lfl -lc
UIINCL           = -I$(CLHOME)/include
SYSLIBS          = $(UILIBS) $(READLINELIB) -lX11 -ltermcap -lm -lfl
CFLAGS           = -I. $(UIINCL) -I/usr/include/g++ $(shell gtk-config --cflags) 

SOURCE           = tstclient nmpsrvr tstMultiPanel5 tstscroll mpdisp
#SOURCE = $(wildcard *.cc)
APPS             = $(patsubst $.cc,%,$(SOURCE))
APPOBJS          = $(patsubst %,%.o,$(APPS))
#---------------------------------------------------------------------
# Application targets
#---------------------------------------------------------------------
all: $(APPS)

$(APPOBJS): %.o: %.cc
	$(C++) -c $< -o $@

$(LIBOBJS): %.o: %.cc
	$(C++) -c $< -o $@

$(LIBOBJS_C): %.o: %.c
	$(C++) -c $(NEWFLAGS) $< -o $@

$(ERROBJS): %.o: %.cc
	$(C++) -c $< -o $@

$(APPS): % : %.o $(CLIBS) $(LIBOBJS)
	$(C++) $< $(LDFLAGS) $(CFLAGS) $(UILIBS) -o $@

libmp.a: $(LIBOBJS) $(LIBOBJS_C)
	ar r libmp.a $(LIBOBJS) $(LIBOBJS_C)

libderr.a: $(ERROBJS)
	ar r libderr.a $(ERROBJS)

#
#-------------------------------------------------------------------------------------------------
#
.PHONY: clean 

depend:
	makedepend $(INCL) $(SOURCE)

clean: 
	\rm -f *.{o,a} *~ $(APPS)
