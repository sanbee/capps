#-*-makefile-*-
OPTFLAGS         = -O3
#XFORMSPATH       = /usr/local/xforms/FORMS
XFORMSPATH        = /home/sanjay/tmp/XForms-88.9/xforms/FORMS
CFLAGS           = -I. -I$(XFORMSPATH) -I/home/sanjay/include \
	           -L$(XFORMSPATH) -I/usr/openwin/include -L/usr/X11R6/lib
CC               = egcs $(CFLAGS) $(OPTFLAGS)
C++              = g++ $(CFLAGS) $(OPTFLAGS)

CLIBS            = libmp.a libderr.a  
ERROBJS          = ErrorObj.o MPDefaultErrorHandler.o
OBJS             = scrollbuf.o MPDefaultErrorHandler.o MultiPanel.o \
	           XYPanel.o SockIO.o Hub.o support.o clstring.o Dock.o \
	           mach.o

SYSLIBS          = $(READLINELIB) -lforms -lX11 -ltermcap $(CLIBS) -lm -lfl
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
	$(C++) $(NEWCFLAGS) -o tstMultiPanel tstMultiPanel.cc \
	libmp.a libderr.a $(SYSLIBS)

mpsrvr: mpsrvr.cc libmp.a libderr.a
	$(C++) $(NEWCFLAGS) -o mpsrvr mpsrvr.cc \
	libmp.a libderr.a $(SYSLIBS)

tstclient: tstclient.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) -o tstclient tstclient.cc \
	libmp.a libderr.a -lm -lc 

tstscroll: tstscroll.cc libderr.a libmp.a
	$(C++) $(NEWCFLAGS) -o tstscroll tstscroll.cc \
	libmp.a libderr.a -lm -lc 

tstdisp.o: tstdisp.cc 
	$(C++) $(NEWCFLAGS) -c tstdisp.cc

scrollbuf.o:scrollbuf.cc scrollbuf.h
	$(CC) -c $(NEWCFLAGS) $(INCLUDE) scrollbuf.cc

libmp.a: $(OBJS)
	ar r libmp.a $(OBJS)

libderr.a: $(ERROBJS)
	ar r libderr.a $(ERROBJS)

clean: 
	rm -f *.o *~ core *.a $(ALL)

