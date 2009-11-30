CC = gcc  -g

AR = ar clq
RM = rm -f
RANLIB = ranlib

GTK_PLOT_VERSION = 5.0

SRCS = gtkpsfont.c \
       gtkplot.c \
       gtkplotlayout.c \
       gtkplotcanvas.c \
       gtkplotpc.c \
       gtkplotprint.c \
       gtkplotps.c \
       demo.c 
OBJS = gtkpsfont.o \
       gtkplot.o \
       gtkplotlayout.o \
       gtkplotcanvas.o \
       gtkplotpc.o \
       gtkplotprint.o \
       gtkplotps.o 

LIBDIRS = $(shell gtk-config --libs)
CFLAGS = -Wall $(shell gtk-config --cflags)

all:    demo realdemo libgtkplot.a 

libgtkplot.a : $(OBJS)
	$(RM) $@
	$(AR) $@ $(OBJS) 
	$(RANLIB) $@

demo: $(OBJS) demo.o 
	$(CC) $(CFLAGS) $(LIBDIRS) $(OBJS) demo.o -o demo

realdemo: $(OBJS) realdemo.o 
	$(CC) $(CFLAGS) $(LIBDIRS) $(OBJS) realdemo.o -o realdemo

demo.o: gtkplotpc.h \
        gtkpsfont.h \
        gtkplotprint.h \
        gtkplot.h \
        gtkplotlayout.h \
        gtkplotcanvas.h \
        gtkplotps.h 

realdemo.o: gtkplotpc.h \
        gtkplotprint.h \
        gtkplotps.h \
        gtkplot.h \
        gtkplotlayout.h \
        gtkplotcanvas.h \
        gtkpsfont.h 


clean:
	rm -f $(OBJS) demo.o demo realdemo.o realdemo core libgtkplot.a plotdemo.ps

