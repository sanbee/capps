#! /bin/bash
# GROOT points ot the root directory in which lib, bin and doc 
# directories will be placed. 
# GSRTROOT points to the root directory containing the source 
# code.  This is used by makefiles in various directories to
# load the standard settings from $GSRCROOT/Setup/Setup.$GOSDIR
#
GROOT=~/SCI/
GSRCROOT=$GROOT
#
# Set the name of the OS in GOSDIR
#
OSNAME=`uname -a | cut -c 1-4`
if [ $OSNAME = "SunO" ]; then
  GOSDIR="Solaris"
elif [ $OSNAME = "IRIX" ]; then
  GOSDIR="IRIX"
elif [ $OSNAME = "Linu" ]; then
  GOSDIR="linux_gnu"
fi
#
# The path for lib, bin, doc, conf directories
#
GLIB=~/CSI/$GOSDIR/lib
GINCLUDE=~/CSI/$GOSDIR/include
GBIN=$GROOT/$GOSDIR/bin
GDOC=$GSRCROOT/doc
GCONF=$GROOT/config
GDEFAULTS=$GROOT/defaults
MPCOLOURS=$GLIB/MPColours.dat

if [ `echo $PATH | grep $GBIN` ]; then echo $PATH > /dev/null; else PATH=$PATH:$GBIN; fi
if [ `echo $LD_LIBRARY_PATH | grep $GLIB` ]; then echo $LD_LIBRARY_PATH > /dev/null; else export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/astro/local/lib; fi
export GROOT
export GDOC
export GBIN
export GLIB
export GOSDIR
export PATH
export GCONF
export GDEFAULTS
export GSRCROOT
export MPCOLOURS

casa rnd
