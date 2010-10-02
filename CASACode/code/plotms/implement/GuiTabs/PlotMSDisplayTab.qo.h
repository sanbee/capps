//# PlotMSDisplayTab.qo.h: Plot tab to manage plot display parameters.
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
#ifndef PLOTMSDISPLAYTAB_QO_H_
#define PLOTMSDISPLAYTAB_QO_H_

#include <plotms/GuiTabs/PlotMSDisplayTab.ui.h>

#include <plotms/GuiTabs/PlotMSPlotTab.qo.h>
#include <plotms/Plots/PlotMSPlotParameterGroups.h>

#include <casa/namespace.h>

namespace casa {

//# Forward declarations.
class PlotSymbolWidget;
class QtIndexChooser;
class QtLabelWidget;


// Subclass of PlotMSPlotSubtab to manage plot display parameters.
class PlotMSDisplayTab : public PlotMSPlotSubtab, Ui::DisplayTab {
    Q_OBJECT
    
    
public:

	
    // Constructor which takes the parent tab and plotter.
    PlotMSDisplayTab(PlotMSPlotTab* plotTab, PlotMSPlotter* parent);
    
    // Destructor.
    ~PlotMSDisplayTab();
    
    
    // Implements PlotMSTab::tabName().
    QString tabName() const { return "Display"; }
    
    // Implements PlotMSPlotSubtab::getValue().  WARNING: for now, only works
    // with PlotMSSinglePlotParameters.
    void getValue(PlotMSPlotParameters& params) const;
    
    // Implements PlotMSPlotSubtab::setValue().  WARNING: for now, only works
    // with PlotMSSinglePlotParameters.
    void setValue(const PlotMSPlotParameters& params);
    
    // Implements PlotMSPlotSubtab::update().  WARNING: for now, only works
    // with PlotMSSinglePlotParameters.
    void update(const PlotMSPlot& plot);
    
    
    // Hides the index chooser at the top.
    void hideIndex();
    
    // Uses the index chooser at the top, with the given number of rows and
    // columns, to manage multi-plot display parameters.
    void setIndexRowsCols(unsigned int nRows, unsigned int nCols);
    
private:
    // Index chooser.
    QtIndexChooser* itsIndexChooser_;
    
    // Label widget for title.
    QtLabelWidget* itsTitleWidget_;
    
    // Symbol widgets for unflagged and flagged points, respectively.
    PlotSymbolWidget* itsSymbolWidget_, *itsMaskedSymbolWidget_;
    
    // Display parameters.
    PMS_PP_Display itsPDisplay_;
    
private slots:
    // Slot for when the index changes.
    void indexChanged(unsigned int index);
};

}

#endif /* PLOTMSDISPLAYTAB_QO_H_ */
