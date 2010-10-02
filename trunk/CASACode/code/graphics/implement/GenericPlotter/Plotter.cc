//# Plotter.cc: Highest level plotting object that holds one or more canvases.
//# Copyright (C) 2009
//# Associated Universities, Inc. Washington DC, USA.
//#
//# This library is free software; you can redistribute it and/or modify it
//# under the terms of the GNU Library General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or (at your
//# option) any later version.
//#
//# This library is distributed in the hope that it will be useful, but WITHOUT
//# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public
//# License for more details.
//#
//# You should have received a copy of the GNU Library General Public License
//# along with this library; if not, write to the Free Software Foundation,
//# Inc., 675 Massachusetts Ave, Cambridge, MA 02139, USA.
//#
//# Correspondence concerning AIPS++ should be addressed as follows:
//#        Internet email: aips2-request@nrao.edu.
//#        Postal address: AIPS++ Project Office
//#                        National Radio Astronomy Observatory
//#                        520 Edgemont Road
//#                        Charlottesville, VA 22903-2475 USA
//#
//# $Id: $
#include <graphics/GenericPlotter/Plotter.h>

#include <casa/OS/Time.h>

#include <ctype.h>
#include <iomanip>

namespace casa {

/////////////////////////
// PLOTTER DEFINITIONS //
/////////////////////////

// Static //

const String Plotter::DEFAULT_DATE_FORMAT = "%y/%m/%d\n%h:%n:%s";
const String Plotter::DEFAULT_RELATIVE_DATE_FORMAT = "%h:%n:%s";

String Plotter::formattedDateString(const String& format, double value,
            PlotAxisScale scale, bool isRelative) {
    stringstream ss;
    
    // Values to be used in formatted String.
    Time t;
    unsigned long hours = 0, minutes = 0;
    double seconds = 0;
    
    // Calculate relative values, if needed.
    if(isRelative) {
        // Put - in front for negative value.
        if(value < 0) ss << '-';
        if(scale == DATE_MJ_DAY) {
            double temp = value * 24;
            hours = (unsigned long)temp;
            temp -= hours;
            temp *= 60;
            minutes = (unsigned long)temp;
            temp *= 60;
            seconds = temp;
            
        } else if(scale == DATE_MJ_SEC) {
            hours = (unsigned long)(value / 3600);
            double temp = value - (hours * 3600);
            minutes = (unsigned long)(temp / 60);
            temp -= minutes * 60;
            seconds = temp;
        }
        
    // Calculate absolute values, if needed.
    } else {
        if(scale == DATE_MJ_SEC) {
            t = Time((value / 86400) + 2400000.5);
            seconds = modf(value, &seconds);
        } else if(scale == DATE_MJ_DAY) {
            t = Time(value + 2400000.5);
            seconds = modf(value * 86400, &seconds);
        }
    }
    
    int defPrec = ss.precision();
    char c;
    for(unsigned int i = 0; i < format.length(); i++) {
        c = format[i];
        if(c == '%' && i < format.length() - 1) {
            c = format[++i];
            switch(c) {
            case 'y': case 'Y':
                if(!isRelative) ss << t.year();
                break;
                
            case 'm': case 'M':
                if(!isRelative) {
                    if(t.month() < 10) ss << '0';
                    ss << t.month();
                }
                break;
                
            case 'd': case 'D':
                if(!isRelative) {
                    if(t.dayOfMonth() < 10) ss << '0';
                    ss << t.dayOfMonth();
                }
                break;
                
            case 'h': case 'H':
                if(!isRelative) {
                    if(t.hours() < 10) ss << '0';
                    ss << t.hours();
                } else {
                    if(hours < 10) ss << '0';
                    ss << hours;
                }
                break;
                
            case 'n': case 'N':
                if(!isRelative) {
                    if(t.minutes() < 10) ss << '0';
                    ss << t.minutes();
                } else {
                    if(minutes < 10) ss << '0';
                    ss << minutes;
                }
                break;
                
            case 's': case 'S':
                if(!isRelative) {
                    if(t.seconds() < 10) ss << '0';
                    ss << (t.seconds() + seconds);
                } else {
                    if(seconds < 10) ss << '0';
                    ss << seconds;
                }
                break;
                
            case 'p': case 'P': {
                if(i >= format.length() - 1 || (!isdigit(format[i + 1]) &&
                   format[i + 1] != '-' && format[i + 1] != '+')) break;

                i++;
                unsigned int j = i + 1;
                for(; j < format.length() && isdigit(format[j]); j++);
                    
                String sprec = format.substr(i, j - i);
                int prec;
                sscanf(sprec.c_str(), "%d", &prec);
                if(prec < 0) prec = defPrec;
                ss.precision(prec);
                
                if(prec >= 0) ss << setprecision(prec);
                break; }
                
            default: ss << format[i - 1] << c; break;
            }
        } else ss << c;
    }

    return ss.str();
}


// Non-Static //

Plotter::Plotter() { }

Plotter::~Plotter() { }


PlotCanvasPtr Plotter::canvasAt(const PlotLayoutCoordinate& coord) {
    PlotCanvasLayoutPtr layout = canvasLayout();
    if(!layout.null()) return layout->canvasAt(coord);
    else               return PlotCanvasPtr();
}

PlotCanvasPtr Plotter::canvas() {
    PlotCanvasLayoutPtr layout = canvasLayout();
    if(!layout.null()) return layout->canvas();
    else               return PlotCanvasPtr();
}

void Plotter::setCanvas(PlotCanvasPtr canvas) {
    if(!canvas.null())
        setCanvasLayout(PlotCanvasLayoutPtr(new PlotLayoutSingle(canvas)));
}

LogMessage::Priority Plotter::logFilterMinPriority() const {
    return logger()->filterMinPriority(); }
void Plotter::setLogFilterMinPriority(PlotLogMessage::Priority minPriority) {
    logger()->setFilterMinPriority(minPriority); }

bool Plotter::logFilterEventFlag(int flag) const {
    return logger()->filterEventFlag(flag); }
void Plotter::setLogFilterEventFlag(int flag, bool on) {
    logger()->setFilterEventFlag(flag, on); }

int Plotter::logFilterEventFlags() const{ return logger()->filterEventFlags();}
void Plotter::setLogFilterEventFlags(int flags) {
    logger()->setFilterEventFlags(flags); }

PlotLoggerPtr Plotter::logger() const {
    if(m_logger.null())
        const_cast<PlotLoggerPtr&>(m_logger) = new PlotLogger(
                const_cast<Plotter*>(this));
    return m_logger;
}

}
