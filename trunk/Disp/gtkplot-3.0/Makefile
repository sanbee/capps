CC = gcc -g

AR = ar clq
RM = rm -f
RANLIB = ranlib

GTK_PLOT_VERSION = 2.0

SRCS = gtkplotfont.c \
       gtkplotps.c \
       gtkplot.c \
       gtkplotlayout.c \
       gtkplotcanvas.c \
       demo.c 
OBJS = gtkplotfont.o \
       gtkplotps.o \
       gtkplot.o \
       gtkplotlayout.o \
       gtkplotcanvas.o

LIBDIRS = $(shell gtk-config --libs)
CFLAGS = -Wall $(shell gtk-config --cflags)

all:    demo libgtkplot.a 

libgtkplot.a : $(OBJS)
	$(RM) $@
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

demo: $(OBJS) demo.o 
	$(CC) $(CFLAGS) $(LIBDIRS) $(OBJS) demo.o -o demo

demo.o: gtkplot.h gtkplotlayout.h gtkplotcanvas.h gtkplotps.h gtkplotfont.h

demo1: $(OBJS) demo1.o 
	$(CC) $(CFLAGS) $(LIBDIRS) $(OBJS) demo1.o -o demo1

demo1.o: gtkplot.h gtkplotlayout.h gtkplotcanvas.h gtkplotps.h gtkplotfont.h

clean:
	rm -f $(OBJS) demo.o demo core libgtkplot.a plotdemo.ps

