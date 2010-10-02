import os
import time
from taskinit import *

def plotms(vis=None, xaxis=None, xdatacolumn=None, yaxis=None,
           ydatacolumn=None,
           selectdata=None, field=None, spw=None,
           timerange=None, uvrange=None, antenna=None, scan=None,
           correlation=None, array=None, msselect=None,
           averagedata=None,
           avgchannel=None, avgtime=None, avgscan=None, avgfield=None,
           avgbaseline=None, avgantenna=None, avgspw=None, scalar=None,
           transform=None,
           freqframe=None,restfreq=None,veldef=None,shift=None,
           extendflag=None,
           extcorr=None, extchannel=None,
           plotfile=None, format=None,
           highres=None, interactive=None, overwrite=None
):

# we'll add these later
#           extspw=None, extantenna=None,
#           exttime=None, extscans=None, extfield=None,

    """
    
            Task for plotting and interacting with visibility data.  A variety
        of axes choices (including data column) along with MS selection and
        averaging options are provided for data selection.  Flag extension
        parameters are also available for flagging operations in the plotter.
        
            All of the provided parameters can also be set using the GUI once
        the application has been launched.  Additional and more specific
        operations are available through the GUI and/or through the plotms
        tool (pm).
        

    Keyword arguments:
    vis -- input visibility dataset
           default: ''
    xaxis, yaxis -- what to plot on the two axes
                    default: '' (uses PlotMS defaults/current set).
      &gt;&gt;&gt; xaxis, yaxis expandable parameters
        xdatacolumn, ydatacolumn -- which data column to use for data axes
                                    default: '' (uses PlotMS default/current
                                    set).
    
    selectdata -- data selection parameters flag
                  (see help par.selectdata for more detailed information)
                  default: False
      &gt;&gt;&gt; selectdata expandable parameters
        field -- select using field ID(s) or field name(s)
                 default: '' (all).
        spw -- select using spectral window/channels
               default: '' (all)
        timerange -- select using time range
                     default: '' (all).
        uvrange -- select using uvrange
                   default: '' (all).
        antenna -- select using antenna/baseline
                   default: '' (all).
        scan -- select using scan number
                default: '' (all).
        correlation -- select using correlations
                       default: '' (all).
        array -- select using (sub)-array range
                 default: '' (all).
        msselect -- TaQL selection expression
                    default: '' (all).
    
    averagedata -- data averaing parameters flag
                   default: False.
      &gt;&gt;&gt; averagedata expandable parameters
        avgchannel -- average over channel?  either blank for none, or a value
                      in channels.
                      default: '' (none).
        avgtime -- average over time?  either blank for none, or a value in
                   seconds.
                   default: '' (none).
        avgscan -- average over scans?  only valid if time averaging is turned
                   on.
                   default: False.
        avgfield -- average over fields?  only valid if time averaging is
                    turned on.
                    default: False.
        avgbaseline -- average over all baselines?  mutually exclusive with
                       avgantenna.
                       default: False.
        avgantenna -- average by per-antenna?  mutually exclusive with
                      avgbaseline.
                      default: False.
        avgspw -- average over all spectral windows?
                  default: False.
    
    extendflag -- have flagging extend to other data points?
                  default: False.
      &gt;&gt;&gt; extendflag expandable parameters
        extcorr -- extend flags based on correlation?  blank = none.
                          default: ''.
        extchannel -- extend flags based on channel?
                      default: False.
        extspw -- extend flags based on spw?
                  default: False.
        extantenna -- extend flags based on antenna?  should be either blank,
                      'all' for all baselines, or an antenna-based value.
                      default: ''.
        exttime -- extend flags based on time (within scans)?
                   default: False.
        extscans -- extend flags based on scans?  only valid if time extension
                    is turned on.
                    default: False.
        extfield -- extend flags based on field?  only valid if time extension
                    is turned on.
                    default: False.
    """
    # Check if DISPLAY environment variable is set.
    if os.getenv('DISPLAY') == None:
        casalog.post('ERROR: DISPLAY environment variable is not set!  Cannot open plotms.', 'SEVERE')
        return False
    
    if (plotfile and os.path.exists(plotfile) and not overwrite):
        casalog.post("Plot file " + plotfile + " exists and overwrite is false, cannot write the file", "SEVERE")
        return False

    try:            
        # Check synonyms

        synonyms = {}
        synonyms['timeinterval'] = synonyms['timeint'] = 'time_interval'
        synonyms['chan'] = 'channel'
        synonyms['freq'] = 'frequency'
        synonyms['vel'] = 'velocity'
        synonyms['correlation'] = 'corr'
        synonyms['ant1'] = 'antenna1'
        synonyms['ant2'] = 'antenna2'
        synonyms['uvdistl'] = 'uvdist_l'
        synonyms['amplitude'] = 'amp'
        synonyms['imaginary'] = 'imag'
        synonyms['ant'] = 'antenna'
        synonyms['parallacticangle'] = 'parang'
        synonyms['hourangle'] = 'hourang'
        synonyms['ant-hourangle'] = 'ant-hourang'
        synonyms['ant-parallacticangle'] = 'ant-parang'
        
        if(synonyms.has_key(xaxis)): xaxis = synonyms[xaxis]
        if(synonyms.has_key(yaxis)): yaxis = synonyms[yaxis]
        
        # Set filename and axes

        pm.setPlotMSFilename(vis, False)
        pm.setPlotAxes(xaxis, yaxis, xdatacolumn, ydatacolumn, False)

        # Set selection
        if (selectdata):
            pm.setPlotMSSelection(field, spw, timerange, uvrange, antenna, scan, correlation, array, msselect, False)
        else:
            pm.setPlotMSSelection('','','','','','','','','',False)
            
        # Set averaging
        if not averagedata:
            avgchannel = avgtime = ''
            avgscan = avgfield = avgbaseline = avgantenna = avgspw = False
           
            scalar = False
            
        pm.setPlotMSAveraging(avgchannel, avgtime, avgscan, avgfield, avgbaseline, avgantenna, avgspw, scalar, False)

        # Set transformations
        if not transform:
            freqframe=''
            restfreq=''
            veldef='RADIO'
            shift=[0.0,0.0]
        
        pm.setPlotMSTransformations(freqframe,veldef,restfreq,shift[0],shift[1],False)
        
        # Set flag extension
        # for now, some options here are not available:
        # pm.setFlagExtension(extendflag, extcorrelation, extchannel, extspw, extantenna, exttime, extscans, extfield)
        extcorrstr=''
        if extcorr:
            extcorrstr='all'
        pm.setFlagExtension(extendflag, extcorrstr, extchannel)
        # Update and show
        pm.update()
        pm.show()
        
        # write file if requested
        if(plotfile != ""):
            while (pm.isDrawing()):
                casalog.post("Waiting until drawing of the plot has completed before saving it",'NORMAL')
                time.sleep(0.5)
            pm.save(plotfile, format, highres, interactive)
    
    except Exception, instance:
        print "Exception during plotms task: ", instance
        
    return True
