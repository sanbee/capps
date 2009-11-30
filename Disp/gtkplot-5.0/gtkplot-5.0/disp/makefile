#-*-makefile-*-
GTK_PLOT_VERSION = 5.0

OPTFLAGS         = -Wall -g 
#XFORMSPATH       = /usr/local/xforms/FORMS
XFORMSPATH        = /usr/include
CC               = gcc $(CFLAGS) $(OPTFLAGS)
C++              = g++ $(CFLAGS) $(OPTFLAGS)

LDFLAGS          = $(CLIBS) $(shell gtk-config --libs) -L/usr/X11R6/lib

#GNOME_LIBS       = -rdynamic -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lesd -laudiofile -lm -lglib -ldl
GNOME_LIBS       = -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lesd -laudiofile -lm -lglib -ldl
#GNOME_LIBS       = -rdynamic -lgnomeui -lart_lgpl -lgdk_imlib -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lgnome -lgnomesupport -lesd -laudiofile -lm -lglib -ldl
#GNOME_LIBS       = -rdynamic -lgnomeui -lart_lgpl -lgdk_imlib -lSM -lICE -lgtk -lgdk -lgmodule -lXext -lX11 -lgnome -lgnomesupport -lesd -laudiofile -lm -ldb1 -lglib -ldl

CLIBS            = libmp.a libderr.a  ../libgtkplot.a
ERROBJS          = ErrorObj.o MPDefaultErrorHandler.o
OBJS             = scrollbuf.o MPDefaultErrorHandler.o MultiPanel.o \
	           XYPanel.o SockIO.o nHub.o clstring.o Dock.o \
	           mach.o hvs.o CommH.o MakeColorList.o 
#	           GnomeWin.o callback.o

CLHOME           = /home/rishi/sanjay/Projects/parafeed/linux_gnu
CLHOME=/export/home/langur/sbhatnag/SCI/linux_gnu
UILIBS           = $(CLHOME)/lib/libshccl.a                 \
                   $(CLHOME)/lib/libsh.a                    \
                   $(CLHOME)/lib/libcalc.a                  \
                   -lreadline -lhistory -ltermcap -lm -lfl -lc
UIINCL = -I$(CLHOME)/include
SYSLIBS          = $(UILIBS) $(READLINELIB) -lX11 -ltermcap -lm -lfl
PREFIX=j_

CFLAGS           = -I. $(UINCL) -I/usr/include/g++ $(shell gtk-config --cflags) 

ALL = tstclient nmpsrvr tstMultiPanel4 tstMultiPanel4 tstscroll mpdisp
#---------------------------------------------------------------------
# Application targets
#---------------------------------------------------------------------
all: $(ALL)

hvs.o: hvs.c
	$(C++) -c hvs.c 

wait_for_activity.o: wait_for_activity.cc
	$(C++) $(NEWCFLAGS) -c wait_for_activity.cc

GnomeWin.o: GnomeWin.cc
	$(C++) $(NEWCFLAGS) -I/usr/lib/gnome-libs/include -c GnomeWin.cc

callback.o: callback.cc
	$(C++) $(NEWCFLAGS) -I/usr/lib/gnome-libs/include -c callback.cc

MultiPanelCB.o: MultiPanelCB.cc
	$(C++) $(NEWCFLAGS) -c MultiPanelCB.cc

SockIO.o: SockIO.cc SockIO.h
	$(C++) $(NEWCFLAGS) -c SockIO.cc

MakeColorList.o: MakeColorList.cc 
	$(C++) $(NEWCFLAGS) -c MakeColorList.cc

mach.o: mach.c mach.h
	$(C++) $(NEWCFLAGS) -c mach.c

Dock.o: Dock.cc SockIO.h
	$(C++) $(NEWCFLAGS) -c Dock.cc

CommH.o: CommH.cc SockIO.h
	$(C++) $(NEWCFLAGS) -c CommH.cc

support.o: support.c
	$(C++) $(NEWCFLAGS) -c support.c

clstring.o: clstring.c
	$(C++) $(NEWCFLAGS) -c clstring.c

Hub.o: Hub.cc SockIO.h XYPanel.h
	$(C++) $(NEWCFLAGS) -c Hub.cc

nHub.o: nHub.cc SockIO.h XYPanel.h
	$(C++) $(NEWCFLAGS) $(UIINCL) -c nHub.cc

MultiPanel.o: MultiPanel.cc MultiPanel.h XYPanel.h
	$(C++) $(NEWCFLAGS) -c MultiPanel.cc

XYPanel.o: XYPanel.cc XYPanel.h
	$(C++) $(NEWCFLAGS) -c XYPanel.cc

MPDefaultErrorHandler.o: MPDefaultErrorHandler.cc ErrorObj.h
	$(C++) $(NEWCFLAGS) -c MPDefaultErrorHandler.cc

ErrorObj.o: ErrorObj.cc ErrorObj.h
	$(C++) $(NEWCFLAGS) -c ErrorObj.cc

tstMultiPanel4: tstMultiPanel4.cc $(CLIBS) $(OBJ) 
	$(C++) tstMultiPanel4.cc $(NEWCFLAGS) $(LDFLAGS) $(UIINCL) -o tstMultiPanel4 \
	$(SYSLIBS) $(GNOME_LIBS) 

tstMultiPanel5: tstMultiPanel5.cc $(CLIBS) $(OBJ) 
	$(C++) tstMultiPanel5.cc $(NEWCFLAGS) $(LDFLAGS) $(UIINCL) -o tstMultiPanel5 \
	$(SYSLIBS) $(GNOME_LIBS) 

demo1: demo1.cc $(CLIBS) $(OBJ) 
	$(C++) demo1.cc $(NEWCFLAGS) $(LDFLAGS) $(UIINCL) -o demo1 \
	$(SYSLIBS) $(GNOME_LIBS) 

mpsrvr: mpsrvr.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o mpsrvr mpsrvr.cc \
	$(CLIBS) $(SYSLIBS) $(LDFLAGS)

mppsrvr: mppsrvr.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o mppsrvr mppsrvr.cc \
	$(CLIBS) $(SYSLIBS) $(LDFLAGS)

schild: schild.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o schild schild.cc \
	$(CLIBS) $(SYSLIBS) $(LDFLAGS)

nmpsrvr: nmpsrvr.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o nmpsrvr nmpsrvr.cc \
	$(CLIBS) $(SYSLIBS) $(LDFLAGS)

nclient: nclient.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o nclient nclient.cc \
	$(CLIBS) $(SYSLIBS) $(LDFLAGS)

tstsock: tstsock.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o tstsock tstsock.cc \
	libmp.a libderr.a -lm -lc 

tstclient: tstclient.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o tstclient tstclient.cc \
	libmp.a libderr.a -lm -lc 

tstXYPanel: tstXYPanel.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) -o tstXYPanel tstXYPanel.cc \
	libmp.a libderr.a -lm -lc $(LDFLAGS) 

tstscroll: tstscroll.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o tstscroll tstscroll.cc $(CLIBS) \
	$(LDFLAGS) -lm -lc 

mpscroll: mpscroll.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o mpscroll mpscroll.cc $(CLIBS) \
	$(LDFLAGS) -lm -lc 

nmpscroll: nmpscroll.cc $(CLIBS)
	$(C++) $(NEWCFLAGS) -o nmpscroll nmpscroll.cc $(CLIBS) \
	$(LDFLAGS) -lm -lc 

colorsel.o: colorsel.c
	$(CC) $(NEWCFLAGS) -c colorsel.c
colorsel: colorsel.c $(CLIBS)
	$(CC) $(NEWCFLAGS) -o colorsel colorsel.c $(CLIBS) \
	$(LDFLAGS) -lm -lc 

tt: tt.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o tt tt.cc \
	libmp.a libderr.a -lm -lc 

mpdisp: mpdisp.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o mpdisp mpdisp.cc \
	libmp.a libderr.a -lm -lc 

tstdisp.o: tstdisp.cc 
	$(C++) $(NEWCFLAGS) -c tstdisp.cc

scrollbuf.o:scrollbuf.cc scrollbuf.h
	$(C++) -c $(NEWCFLAGS) $(INCLUDE) scrollbuf.cc

tst.o: tst.cc
	$(CC) $(NEWCFLAGS) -c tst.cc

libmp.a: $(OBJS)
	ar r libmp.a $(OBJS)

libderr.a: $(ERROBJS)
	ar r libderr.a $(ERROBJS)

clean: 
	rm -f *.o *~ core *.a $(ALL)

