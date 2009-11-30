CC = gcc -g

AR = ar clq
RM = rm -f
RANLIB = ranlib

GTK_PLOT_VERSION = 2.0

SRCS = gtkplotfont.c \
       gtkplot.c \
       gtkplotlayout.c \
       gtkplotcanvas.c \
       gtkplotpc.c \
       gtkplotprint.c \
       gtkplotps.c \
       demo.c 
OBJS = gtkplotfont.o \
       gtkplot.o \
       gtkplotlayout.o \
       gtkplotcanvas.o \
       gtkplotpc.o \
       gtkplotprint.o \
       gtkplotps.o 

LIBDIRS = $(shell gtk-config --libs)
CFLAGS = -Wall $(shell gtk-config --cflags)

all:    demo libgtkplot.a 

libgtkplot.a : $(OBJS)
	$(RM) $@
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

demo: $(OBJS) demo.o 
	$(CC) demo.o $(OBJS) $(CFLAGS) $(LIBDIRS) -o demo

demo.o: gtkplotpc.h \
        gtkplotprint.h \
        gtkplotps.h \
        gtkplot.h \
        gtkplotlayout.h \
        gtkplotcanvas.h \
        gtkplotps.h \
        gtkplotfont.h

clean:
	rm -f $(OBJS) demo.o demo core libgtkplot.a plotdemo.ps

