//# tTablePlot.cc: Test program for the TablePlot classes
//# Copyright (C) 1994,1995,1996,1997,1998,1999,2000,2001,2002
//# Associated Universities, Inc. Washington DC, USA.
//#
//# This program is free software; you can redistribute it and/or modify it
//# under the terms of the GNU General Public License as published by the Free
//# Software Foundation; either version 2 of the License, or (at your option)
//# any later version.
//#
//# This program is distributed in the hope that it will be useful, but WITHOUT
//# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
//# more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with this program; if not, write to the Free Software Foundation, Inc.,
//# 675 Massachusetts Ave, Cambridge, MA 02139, USA.
//#
//# Correspondence concerning AIPS++ should be addressed as follows:
//#        Internet email: aips2-request@nrao.edu.
//#        Postal address: AIPS++ Project Office
//#                        National Radio Astronomy Observatory
//#                        520 Edgemont Road
//#                        Charlottesville, VA 22903-2475 USA
//#
//# $Id$

#include <casa/OS/RegularFile.h>
#include <casa/OS/File.h>
#include <casa/Utilities/Assert.h>
#include <casa/Exceptions/Error.h>
#include <casa/iostream.h>
#include <casa/stdio.h>

#include <tables/TablePlot/TablePlot.h>
#include <tables/TablePlot/FlagVersion.h>

#include <tables/Tables/TableDesc.h>
#include <tables/Tables/TableInfo.h>
#include <tables/Tables/SetupNewTab.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableRecord.h>
#include <tables/Tables/ScaColDesc.h>
#include <tables/Tables/ArrColDesc.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/ArrayColumn.h>
#include <tables/Tables/ExprNode.h>
#include <casa/BasicSL/Complex.h>
#include <casa/Arrays/Vector.h>
#include <casa/Arrays/Matrix.h>
#include <casa/Arrays/ArrayIO.h>
#include <casa/Arrays/Cube.h>
#include <casa/Arrays/MaskedArray.h>
#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/ArrayLogical.h>
#include <casa/Arrays/ArrayIO.h>
#include <casa/Containers/Record.h>

#include <casa/namespace.h>

#include <Python.h>

#include <cl.h>
#include <clinteract.h>

//template void Array<Bool>::operator+=(Array<Bool> &, Array<Bool> const &);
//template void Array<Bool>::operator+=(Array<Bool> const &);

// <summary>
// Test program for the TablePlot classes
// </summary>

// This program tests the class TablePlot and related classes BasePlot,
// CrossPlot and TPPlotter.
// The results are written to stdout. 

/* create a virtual table */
int maketable(Table &tab, String tname, Int flagnum)
{
	    TableDesc td("", "1", TableDesc::Scratch);
	    td.comment() = "Test for a virtual table in tableplot";
	    td.addColumn (ScalarColumnDesc<Int>("XAXIS"));
	    td.addColumn (ArrayColumnDesc<Float>("YAXIS"));
	    td.addColumn (ArrayColumnDesc<Bool> ("FLAG"));
	    td.addColumn (ScalarColumnDesc<Bool> ("FLAG_ROW"));
	   
	    Int nrows = 10;
	    
	    SetupNewTable aNewTab("TBVTtest", td, Table::New);

	    tab = Table(aNewTab, Table::Memory, nrows);
	    
	    Array<Float> y(IPosition(2,3,nrows));
	    Array<Bool> fg(IPosition(2,3,nrows));
	    for(Int i=0;i<nrows;i++)
	    {
		    y(IPosition(2,0,i)) = (i-5)*(i-5);
		    y(IPosition(2,1,i)) = (i-2)*(i-2);
		    y(IPosition(2,2,i)) = (i-6)*(i-6);
		    fg(IPosition(2,0,i)) = False;
		    fg(IPosition(2,1,i)) = False;
		    fg(IPosition(2,2,i)) = False;
	    }

	    /* Setting the flags for values for X = 4 */
            if(flagnum==1)
            {
	    fg(IPosition(2,0,3)) = True;
	    fg(IPosition(2,1,3)) = True;
	    fg(IPosition(2,2,3)) = True;
            }
	    
	    ScalarColumn<Bool> flagRowCol(tab, "FLAG_ROW");
            
	    flagRowCol.putColumn(Vector<Bool>(nrows,False));
	    
	    ArrayColumn<Bool> flag(tab, "FLAG");
	    flag.putColumn(fg);
	    
	    Vector<Int> x(nrows);
	    for(Int i=0;i<nrows;i++) x[i] = i+1;
	    ScalarColumn<Int> xvals(tab, "XAXIS");
	    xvals.putColumn(x);
	    
	    ArrayColumn<Float> yvals(tab, "YAXIS");
	    yvals.putColumn(y);

	    cout << " y shape : " << yvals.shape(0) << endl;

	    tab.flush();
            
	    return 0;
}

/* Tests overplot with different plot options */
int demo1(TablePlot &TP)
{
	    TP.clearPlot(0,0,0);

	    Vector<String> tnames(1);
	    tnames[0] = "s1.ms";
	    Vector<String> tselstr(1);
	    tselstr[0] = "all";
	    
	    Vector<Table> tvec(1);
	    tvec[0] = Table(tnames[0],Table::Update);
	    //tvec[0] = Table("/home/ballista3/rurvashi/data/IC1262cal.ms",Table::Update);

	    TP.setTableT(tvec,tnames,tselstr);
	    
	    PlotOptions pop;
	    pop.PlotSymbol = String("g.");

	    Vector<String> iterstr(0);
	    Vector<String> datastr(2);
	    datastr[0] = "SQRT(SUMSQUARE(UVW[1:2]))";
	    datastr[1] = "AMPLITUDE(DATA)";
	    
	    pop.Title = String("UVdist plot");
	    pop.XLabel = String("uvdist");
	    pop.YLabel = String("amplitude");

	    Vector<String> Err;
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastr,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    TP.plotData(pop,datastr);
	    	    
	    pop.PlotSymbol = String("c+");
	    pop.OverPlot = True;
	    
	    datastr[1] = "MEAN(AMPLITUDE(IIF(FLAG,0.0,DATA)))";
	    
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastr,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    TP.plotData(pop,datastr);
	    
	    /* Flag a region with diskwrite=1*/
	    Vector<Double> reg(4);
	    reg[0] = 100.0;
	    reg[1] = 400.0;
	    reg[2] = 0.1;
	    reg[3] = 0.15;
	    TP.markRegions(1,1,1,reg);
	    TP.flagData(1);
	    
	    TP.runPlotCommand(String("pl.savefig('demo1.png')"));

	    return 0;
}

/* Tests multiple panels with overplot and flagging */
int demo2(TablePlot &TP)
{
	    TP.clearPlot(0,0,0);

	    Vector<String> tnames(1);
	    tnames[0] = "s1.ms";
	    Vector<String> tselstr(1);
	    tselstr[0] = "all";
	    Vector<Table> tvec(1);
	    tvec[0] = Table(tnames[0],Table::Update);

	    TP.setTableT(tvec,tnames,tselstr);

	    PlotOptions pop;
	    pop.PlotSymbol = String("r,");
	    pop.PanelMap[0] = 2;
	    pop.PanelMap[1] = 1;
	    pop.PanelMap[2] = 1;

	    Vector<String> iterstr(0);
	    Vector<String> datastr(2);
	    datastr[0] = "SQRT(SUMSQUARE(UVW[1:2]))";
	    datastr[1] = "AMPLITUDE(DATA[1,1:5])";
	    
	    pop.Title = String("UVdist plot");
	    pop.XLabel = String("uvdist");
	    pop.YLabel = String("amplitude");

	    Vector<String> Err;
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastr,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    TP.plotData(pop,datastr);
	    
	    pop.PlotSymbol = String("g,");
	    pop.PanelMap[2] = 2;
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastr,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    datastr[1] = "AMPLITUDE(DATA[1,6:10])";
	    
	    TP.plotData(pop,datastr);
	    
	    Vector<Double> reg(4);
	    reg[0] = 100.0;
	    reg[1] = 400.0;
	    reg[2] = 0.1;
	    reg[3] = 0.15;
	    TP.markRegions(2,1,1,reg);
	    TP.flagData(1);
	    
	    TP.runPlotCommand(String("pl.savefig('demo2.png')"));

	    return 0;
}

/* Tests on a virtual table */
int demo3(TablePlot &TP)
{
	    TP.clearPlot(0,0,0);
	    
	    TableDesc td("", "1", TableDesc::Scratch);
	    td.comment() = "Test for a virtual table in tableplot";
	    td.addColumn (ScalarColumnDesc<Int>("XAXIS"));
	    td.addColumn (ArrayColumnDesc<Float>("YAXIS"));
	    td.addColumn (ArrayColumnDesc<Bool> ("FLAG"));
	    td.addColumn (ScalarColumnDesc<Bool> ("FLAG_ROW"));
	   
	    Int nrows = 10;
	    
	    SetupNewTable aNewTab("TBVTtest", td, Table::New);

	    Vector<String> tnames(1);
	    tnames[0] = "TBCTest";
	    Vector<String> tselstr(1);
	    tselstr[0] = "all";
	    Vector<Table> tvec(1);
	    tvec[0] = Table (aNewTab, Table::Memory, nrows);
	    
	    Array<Float> y(IPosition(2,3,nrows));
	    Array<Bool> fg(IPosition(2,3,nrows));
	    for(Int i=0;i<nrows;i++)
	    {
		    y(IPosition(2,0,i)) = (i-5)*(i-5);
		    y(IPosition(2,1,i)) = (i-2)*(i-2);
		    y(IPosition(2,2,i)) = (i-6)*(i-6);
		    fg(IPosition(2,0,i)) = False;
		    fg(IPosition(2,1,i)) = False;
		    fg(IPosition(2,2,i)) = False;
	    }

	    /* Setting the flags for values for X = 4 */
	    fg(IPosition(2,0,3)) = True;
	    fg(IPosition(2,1,3)) = True;
	    fg(IPosition(2,2,3)) = True;
	    
	    ScalarColumn<Bool> flagRowCol(tvec[0], "FLAG_ROW");
	    flagRowCol.putColumn(Vector<Bool>(nrows,False));
	    
	    ArrayColumn<Bool> flag(tvec[0], "FLAG");
	    flag.putColumn(fg);
	    
	    Vector<Int> x(nrows);
	    for(Int i=0;i<nrows;i++) x[i] = i+1;
	    ScalarColumn<Int> xvals(tvec[0], "XAXIS");
	    xvals.putColumn(x);
	    
	    ArrayColumn<Float> yvals(tvec[0], "YAXIS");
	    yvals.putColumn(y);

	    //cout << " y shape : " << yvals.shape(0) << endl;

	    tvec[0].flush();
	    
	    TP.setTableT(tvec,tnames,tselstr);

	    PlotOptions pop;
	    Vector<String> iterstr(0);
	    Vector<String> datastr(2);
	    Vector<String> labels(3);
	    
	    pop.PlotSymbol = String("ro");
	    pop.MultiColour = True;
            pop.Connect = String("cellcol");

	    datastr[0] = "XAXIS";
	    datastr[1] = "YAXIS[1:3]";
	    
	    pop.Title = String("Virtual table test");
	    pop.XLabel = String("XAXIS");
	    pop.YLabel = String("YAXIS");
	    
	    Vector<String> Err;
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastr,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    TP.plotData(pop,datastr);
	    
	    /* Flag a region with diskwrite=1*/
	    Vector<Double> reg(4);
	    reg[0] = 6.2;
	    reg[1] = 7.8;
	    reg[2] = -1.0;
	    reg[3] = 10.0;
	    TP.markRegions(1,1,1,reg);
	    TP.flagData(1);
	    
	    /* Resetting the table, so that it's re-read */
	    TP.setTableT(tvec,tnames,tselstr);
	    TP.plotData(pop,datastr);
	    
	    TP.runPlotCommand(String("pl.savefig('demo3.png')"));

	    return 0;
}

/* Tests on multiple virtual table */
int demo6(TablePlot &TP)
{
	    TP.clearPlot(0,0,0);
	    
	    Vector<String> tnames(2);
	    tnames[0] = "TEST1";
	    tnames[1] = "TEST2";
	    Vector<String> tselstr(2);
	    tselstr[0] = "all";
	    Vector<Table> tvec(2);
	    maketable(tvec[0],tnames[0],1);
	    maketable(tvec[1],tnames[1],2);
	    
	    TP.setTableT(tvec,tnames,tselstr);

	    PlotOptions pop;
	    Vector<String> iterstr(0);
	    Vector<Vector<String> > datastrvector(2);
	    Vector<String> labels(3);
	    
	    pop.PlotSymbol = String("ro");
	    //pop.MultiColour = True;
            pop.TableMultiColour = False;
            pop.Connect = String("cellcol");

            datastrvector[0].resize(2);
	    (datastrvector[0])[0] = "XAXIS";
	    (datastrvector[0])[1] = "YAXIS[1:2]";
            datastrvector[1].resize(2);
	    (datastrvector[1])[0] = "XAXIS";
	    (datastrvector[1])[1] = "YAXIS[3]";
	    
	    pop.Title = String("Multiple table/taql test");
	    pop.XLabel = String("XAXIS");
	    pop.YLabel = String("YAXIS");
	    
	    Vector<String> Err;
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastrvector,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    TP.plotData(pop,datastrvector);
#if 1 
	    /* Flag a region with diskwrite=1*/
	    Vector<Double> reg(4);
	    reg[0] = 6.2;
	    reg[1] = 7.8;
	    reg[2] = -1.0;
	    reg[3] = 10.0;
	    TP.markRegions(1,1,1,reg);
	    TP.flagData(0);
	    
	    /* Resetting the table, so that it's re-read */
	    TP.setTableT(tvec,tnames,tselstr);
	    TP.plotData(pop,datastrvector);
#endif	    
	    TP.runPlotCommand(String("pl.savefig('demo6.png')"));

	    return 0;
}

/* Tests on a virtual table with crossplot, and a Convert function */
int demo4(TablePlot &TP)
{
	    TP.clearPlot(0,0,0);
	    
	    TableDesc td("", "1", TableDesc::Scratch);
	    td.comment() = "Test for a virtual table in tableplot";
	    td.addColumn (ScalarColumnDesc<Int>("XAXIS"));
	    td.addColumn (ArrayColumnDesc<Float>("YAXIS"));
	    td.addColumn (ArrayColumnDesc<Bool> ("FLAG"));
	    td.addColumn (ScalarColumnDesc<Bool> ("FLAG_ROW"));
	   
	    Int nrows = 10;
	    
	    SetupNewTable aNewTab("TBVTtest", td, Table::New);

	    Vector<String> tnames(1);
	    tnames[0] = "TBVtest";
	    Vector<String> tselstr(1);
	    tselstr[0] = "all";
	    Vector<Table> tvec(1);
	    // what about "update" ?
	    tvec[0] = Table (aNewTab, Table::Memory, nrows);
	    
	    Array<Float> y(IPosition(2,3,nrows));
	    Array<Bool> fg(IPosition(2,3,nrows));
	    for(Int i=0;i<nrows;i++)
	    {
		    y(IPosition(2,0,i)) = (i-5)*(i-5);
		    y(IPosition(2,1,i)) = (i-2)*(i-2);
		    y(IPosition(2,2,i)) = (i-6)*(i-6);
		    fg(IPosition(2,0,i)) = False;
		    fg(IPosition(2,1,i)) = False;
		    fg(IPosition(2,2,i)) = False;
	    }

	    ScalarColumn<Bool> flagRowCol(tvec[0], "FLAG_ROW");
	    flagRowCol.putColumn(Vector<Bool>(nrows,False));
	    
	    ArrayColumn<Bool> flag(tvec[0], "FLAG");
	    flag.putColumn(fg);
	    
	    Vector<Int> x(nrows);
	    for(Int i=0;i<nrows;i++) x[i] = i+1;
	    ScalarColumn<Int> xvals(tvec[0], "XAXIS");
	    xvals.putColumn(x);
	    
	    ArrayColumn<Float> yvals(tvec[0], "YAXIS");
	    yvals.putColumn(y);

	    cout << " y shape : " << yvals.shape(0) << endl;
	    cout << " flag shape : " << flag.shape(0) << endl;

	    tvec[0].flush();
	    
	    TP.setTableT(tvec,tnames,tselstr);

	    PlotOptions pop;
	    Vector<String> iterstr(0);
	    Vector<String> datastr(2);
	    Vector<String> labels(3);
	    
	    pop.PlotSymbol = String("ro");
	    pop.MultiColour = True;
	    pop.ColumnsXaxis = False;

	    datastr[0] = "CROSS";
	    datastr[1] = "YAXIS[1:3]";
	    
	    pop.Title = String("Virtual table test");
	    pop.XLabel = String("XAXIS (MHz) ");
	    pop.YLabel = String("YAXIS");

	    pop.Convert = new TPConvertChanToFreq();
	    
	    Vector<String> Err;
	    Err = pop.validateParams();
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    Err = TP.checkInputs(pop,datastr,iterstr);
	    if(Err[0].length()>0){cout << Err << endl; return -1;}
	    
	    TP.plotData(pop,datastr);
	    
	    TP.runPlotCommand(String("pl.savefig('demo4.png')"));
 
 	    if(pop.Convert != NULL) delete pop.Convert;

	    return 0;
}

int demo5(TablePlot &TP)
{
/*
	File fil("./tempdir/thefile");
	
	cout << "exists : " << fil.exists() << endl;
	
	cout << "exists : " << fil.exists() << endl;
	
	fil.touch();
	cout << "exists : " << fil.exists() << endl;
*/	
	    
	    Table tab("./s1.ms",Table::Update);

	    FlagVersion FV("./s1.ms",String("FLAG"),String("FLAG_ROW"));
	    ArrayColumn<Bool> flags;
	    ScalarColumn<Bool> rowflags;
	    Array<Bool> theflags;

	    cout << " Version list : " << FV.getVersionList() << endl;

	    FV.attachFlagColumns(String("one"),rowflags,flags,tab);
	    
	    FV.saveFlagVersion(String("one"),String("comment1"),String("replace"));
	    cout << " Version list : " << FV.getVersionList() << endl;

	    FV.attachFlagColumns(String("one"),rowflags,flags,tab);
	    theflags = flags.getColumn();
	    theflags.set(False);
	    flags.putColumn(theflags);
	    
	    FV.saveFlagVersion(String("two"),String("comment2"),String("replace"));
	    cout << " Version list : " << FV.getVersionList() << endl;
	    


	    return 0;
}
fd_set fdset;

//
//----------------------------------------------------------------
// A warpper for select() system call.  The EINTR return status 
// (system call interrupted) is trapped and select simply called
// again.  This can happen since if a child proccess exits while
// the server is waiting on select() (which is where this server
// spends most of it's time).
//
int  Select(int  n,  fd_set  *readfds,  fd_set  *writefds,
       fd_set *exceptfds, struct timeval *timeout)
{
  int m;
  do
    {
      m=select(n, readfds, writefds, exceptfds, timeout);
    }
  while((m<0) && (errno==EINTR));
  return m;
}
//
//----------------------------------------------------------------
// Get the value of the highest file id used by this process.
//
int mygetdtablehi()
{
  int n=0;
  for (int i=0;i<getdtablesize();i++)
    if (FD_ISSET(i,&fdset)) n=i;
  return n+1;
}

int main (int argc, char** argv)
{
    try 
    {
	    Py_Initialize();

// 	    BeginCL(argc,argv);
// 	    char TBuf[FILENAME_MAX];
// 	    clgetConfigFile(TBuf,argv[0]);strcat(TBuf,".config");
// 	    clloadConfig(TBuf);
// 	    clInteractive(0);
// 	    EndCL();

	    PyRun_SimpleString("import pylab as pl");
	    PyRun_SimpleString("print dir");
	    PyRun_SimpleString("import sys");
	    PyRun_SimpleString("print sys.path");
	    PyRun_SimpleString("sys.path.append('/export/home/langur/sbhatnag/casa_daily/linux_gnu/python/2.5/')");

	    TablePlot TP;
	    // TP.setGui(False);

	    //demo1(TP);
            //sleep(2);
            //TP.clearAllFlags(True);
	    //demo2(TP);
            //sleep(2);
	    demo3(TP);
	    Int n;
	    while(1)
	    {
	      FD_ZERO(&fdset);
	      FD_SET(1,&fdset);

	      do
		{
		  n=select(mygetdtablehi(), &fdset, NULL, NULL, NULL);
		}
	      while (n > 0);
	      if (FD_ISSET(0,&fdset)) 	      cout << "Selected 0" << endl;
	      if (FD_ISSET(1,&fdset)) 	      cout << "Selected 1" << endl;
	      if (FD_ISSET(2,&fdset)) 	      cout << "Selected 2" << endl;

	    }
      //	    clRetry();
	    /*
            sleep(2);
            demo4(TP);
            sleep(2);
	    //demo5(TP);
            //sleep(2);
	    demo6(TP);
            sleep(2);
	    */

    } 
    catch (AipsError x) 
    {
	cout << "Caught an exception: " << x.getMesg() << endl;
	return 1;
    } 
    return 0;                           // exit with success status
}

