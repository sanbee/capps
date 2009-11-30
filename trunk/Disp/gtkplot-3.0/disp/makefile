#-*-makefile-*-
GTK_PLOT_VERSION = 2.0

OPTFLAGS         = -Wall -g
#XFORMSPATH       = /usr/local/xforms/FORMS
XFORMSPATH        = /usr/include
CC               = gcc $(CFLAGS) $(OPTFLAGS)
C++              = g++ $(CFLAGS) $(OPTFLAGS)

CFLAGS           = -I.  $(shell gtk-config --cflags) 

LDFLAGS          = $(shell gtk-config --libs) -L/usr/X11R6/lib

CLIBS            = libmp.a libderr.a  
ERROBJS          = ErrorObj.o MPDefaultErrorHandler.o
OBJS             = scrollbuf.o MPDefaultErrorHandler.o MultiPanel.o \
	           XYPanel.o SockIO.o Hub.o support.o clstring.o Dock.o \
	           mach.o

SYSLIBS          = $(READLINELIB) ../libgtkplot.a -lX11 -ltermcap $(CLIBS) -lm -lfl
PREFIX=j_
ALL = tstclient mpsrvr
#---------------------------------------------------------------------
# Application targets
#---------------------------------------------------------------------
all: $(ALL)

MultiPanelCB.o: MultiPanelCB.cc
	$(C++) $(NEWCFLAGS) -c MultiPanelCB.cc

SockIO.o: SockIO.cc SockIO.h
	$(C++) $(NEWCFLAGS) -c SockIO.cc

mach.o: mach.c mach.h
	$(C++) $(NEWCFLAGS) -c mach.c

Dock.o: Dock.cc SockIO.h
	$(C++) $(NEWCFLAGS) -c Dock.cc

support.o: support.c
	$(C++) $(NEWCFLAGS) -c support.c

clstring.o: clstring.c
	$(C++) $(NEWCFLAGS) -c clstring.c

Hub.o: Hub.cc SockIO.h XYPanel.h
	$(C++) $(NEWCFLAGS) -c Hub.cc

MultiPanel.o: MultiPanel.cc MultiPanel.h XYPanel.h
	$(C++) $(NEWCFLAGS) -c MultiPanel.cc

XYPanel.o: XYPanel.cc XYPanel.h
	$(C++) $(NEWCFLAGS) -c XYPanel.cc

MPDefaultErrorHandler.o: MPDefaultErrorHandler.cc ErrorObj.h
	$(C++) $(NEWCFLAGS) -c MPDefaultErrorHandler.cc

ErrorObj.o: ErrorObj.cc ErrorObj.h
	$(C++) $(NEWCFLAGS) -c ErrorObj.cc

tstMultiPanel: tstMultiPanel.cc libmp.a libderr.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o tstMultiPanel tstMultiPanel.cc \
	libmp.a libderr.a $(SYSLIBS)

mpsrvr: mpsrvr.cc libmp.a libderr.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o mpsrvr mpsrvr.cc \
	libmp.a libderr.a $(SYSLIBS)

tstclient: tstclient.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o tstclient tstclient.cc \
	libmp.a libderr.a -lm -lc 

tstscroll: tstscroll.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) $(LDFLAGS) -o tstscroll tstscroll.cc \
	libmp.a libderr.a -lm -lc 

tstdisp.o: tstdisp.cc 
	$(C++) $(NEWCFLAGS) -c tstdisp.cc

scrollbuf.o:scrollbuf.cc scrollbuf.h
	$(CC) -c $(NEWCFLAGS) $(INCLUDE) scrollbuf.cc

tst.o: tst.cc
	$(CC) $(NEWCFLAGS) -c tst.cc

libmp.a: $(OBJS)
	ar r libmp.a $(OBJS)

libderr.a: $(ERROBJS)
	ar r libderr.a $(ERROBJS)

clean: 
	rm -f *.o *~ core *.a $(ALL)

