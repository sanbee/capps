from taskinit import *
import time
import os
import sys

debug = False
def flagdata(vis = None,
             flagbackup = None,
             mode = None,
             spw = None, field = None,
             selectdata = None,
             antenna = None,
             uvrange = None,
             timerange = None,
             correlation = None,
             scan = None,
             feed = None, array = None,
             clipexpr = None, clipminmax = None,
             clipcolumn = None, clipoutside = None, channelavg = None,
             quackinterval = None, quackmode = None, quackincrement = None,
             autocorr = None,
             unflag = None, algorithm = None,
             column = None, expr = None,
             thr = None, window = None,
             diameter = None,
             time_amp_cutoff = None,
             freq_amp_cutoff = None,
             freqlinefit = None,
             auto_cross = None,
             num_time = None,
             start_chan = None,
             end_chan = None,
             bs_cutoff = None,
             ant_cutoff = None,
             flag_level = None,
             minrel = None,
             maxrel = None,
             minabs = None,
             maxabs = None):

        casalog.origin('flagdata')

        fglocal = casac.homefinder.find_home_by_name('flaggerHome').create()
        mslocal = casac.homefinder.find_home_by_name('msHome').create()
        
        try: 
                if ((type(vis)==str) & (os.path.exists(vis))):
                        fglocal.open(vis)
                else:
                        raise Exception, 'Visibility data set not found - please verify the name'

                if mode == 'manualflag':
                        # In manualflag and quack modes,
                        # filter out the parameters which are not used

                        manualflag_quack(fglocal, mode, selectdata, flagbackup,
                                         autocorr=autocorr,
                                         unflag=unflag,
                                         clipexpr=clipexpr,       # manualflag only
                                         clipminmax=clipminmax,   # manualflag only
                                         clipcolumn=clipcolumn,   # manualflag only
                                         clipoutside=clipoutside, # manualflag only
                                         channelavg=channelavg,   # manualflag only
                                         spw=spw,
                                         field=field,
                                         antenna=antenna,
                                         timerange=timerange,
                                         correlation=correlation,
                                         scan=scan,
                                         feed=feed,
                                         array=array,
                                         uvrange=uvrange)
                elif mode == 'quack':
                        manualflag_quack(fglocal, mode, selectdata, flagbackup,
                                         autocorr=autocorr,
                                         unflag=unflag,
                                         clipminmax=[], clipoutside=False,
                                         channelavg=False,
                                         quackinterval=quackinterval,   # quack only
                                         quackmode=quackmode,           # quack only
                                         quackincrement=quackincrement, # quack only
                                         spw=spw,
                                         field=field,
                                         antenna=antenna,
                                         timerange=timerange,
                                         correlation=correlation,
                                         scan=scan,
                                         feed=feed,
                                         array=array,
                                         uvrange=uvrange)
                elif ( mode == 'shadow' ):
                        fglocal.setdata()
                        fglocal.setshadowflags( \
                                field = field, \
                                spw = spw, \
                                array = array, \
                                feed = feed, \
                                scan = scan, \
                                baseline = antenna, \
                                uvrange = uvrange, \
                                time = timerange, \
                                correlation = correlation, \
                                diameter = diameter)

                        if flagbackup:
                                backup_flags(fglocal, mode)
                        fglocal.run()
                elif ( mode == 'autoflag' ):
                        fglocal.setdata(field = field, \
                                   spw = spw, \
                                   array = array, \
                                   feed = feed, \
                                   scan = scan, \
                                   baseline = antenna, \
                                   uvrange = uvrange, \
                                   time = timerange, \
                                   correlation = correlation)
                        rec = fglocal.getautoflagparams(algorithm=algorithm)
                        rec['expr'] = expr
                        rec['thr'] = thr
                        #rec['rowthr'] = rowthr;
                        rec['hw'] = window
                        #rec['rowhw'] = rowhw;
                        #if( algorithm == 'uvbin' ):
                        #     rec['nbins'] = nbins;
                        #     rec['minpop'] = minpop;
                        rec['column'] = column
                        fglocal.setautoflag(algorithm = algorithm,
                                       parameters = rec)
                        
                        if flagbackup:
                                backup_flags(fglocal, mode)
                        fglocal.run()

                elif mode == 'rfi':
                        fglocal.setdata(field = field, \
                                   spw = spw, \
                                   array = array, \
                                   feed = feed, \
                                   scan = scan, \
                                   baseline = antenna, \
                                   uvrange = uvrange, \
                                   time = timerange, \
                                   correlation = correlation)

                        # Get the detault parameters for a particular algorithm,
                        # then modify them
                        
                        par = fglocal.getautoflagparams(algorithm='tfcrop')
                        #print "par =", par
                        
                        ## True : Show plots of each time-freq chunk.
                        ## Needs 'gnuplot'
                        ## Needs "ds9 &" running in the background (before starting casapy)
                        ## Needs xpaset, xpaget, etc.. accessible in the path (for ds9)
                        par['showplots']=False
			## jmlarsen: Do not show plots. There's no way for the user to interrupt
			## a lengthy sequence of plots (CAS-1655)
                        
                        ## channel range (1 based)
                        par['start_chan']=start_chan 
                        par['end_chan']=end_chan
                        
                        ## number of time-steps in each chunk
                        par['num_time']=num_time

                        ## flag on cross-correlations and auto-correlations. (0 : only autocorrelations)
                        par['auto_cross']= auto_cross   
                        
                        ## Flag Level :
                        ## 0: flag only what is found. 
                        ## 1: extend flags one timestep before and after
                        ## 2: 1 and extend flags one channel before/after.
                        par['flag_level']=flag_level

                        ## data expression on which to flag.
                        par['expr']=clipexpr

                        ## data column to use.
                        par['column']=clipcolumn

                        ## False : Fit the bandpass with a piecewise polynomial
                        ## True : Fit the bandpass with a straight line.
                        par['freqlinefit']=freqlinefit

                        ## Flagging thresholds ( N sigma ), where 'sigma' is the stdev of the "fit".
                        #par['freq_amp_cutoff']=3
                        #par['time_amp_cutoff']=4
                        par['freq_amp_cutoff']=freq_amp_cutoff
                        par['time_amp_cutoff']=time_amp_cutoff
                        
                        # Tell the 'fg' tool which algorithm to use, and set the parameters.
                        # Note : Can set multiple instances of this (will be done one after the other)
                        #
                        fglocal.setautoflag(algorithm='tfcrop', parameters=par)

                        if flagbackup:
                                backup_flags(fglocal, mode)

                        fglocal.run()

                elif ( mode == 'summary' ):
                        fglocal.setdata()
                        fglocal.setflagsummary(field=field, \
                                          spw=spw, \
                                          array=array, \
                                          feed=feed, \
                                          scan=scan, \
                                          baseline=antenna, \
                                          uvrange=uvrange, \
                                          time=timerange, \
                                          correlation=correlation)
                        
                        # do not backup existing flags
                        stats = fglocal.run()

                        # filter out baselines/antennas/fields/spws/...
                        # which do not fall within limits
                        if type(stats) is dict:
                            for x in stats.keys():
                                if type(stats[x]) is dict:
                                    for xx in stats[x].keys():
                                        flagged = stats[x][xx]
                                        assert type(flagged) is dict
                                        assert flagged.has_key('flagged')
                                        assert flagged.has_key('total')
                                        if flagged['flagged'] < minabs or \
                                           (flagged['flagged'] > maxabs and maxabs >= 0) or \
                                           flagged['flagged']*1.0/flagged['total'] < minrel or \
                                           flagged['flagged']*1.0/flagged['total'] > maxrel:
                                                del stats[x][xx]
                        
                        return stats

        except Exception, instance:
                print '*** Error ***', instance
        
        #write history
        mslocal.open(vis,nomodify=False)
        mslocal.writehistory(message='taskname = flagdata', origin='flagdata')
        mslocal.writehistory(message='vis      = "' + str(vis) + '"', origin='flagdata')
        mslocal.writehistory(message='mode     = "' + str(mode) + '"', origin='flagdata')
        mslocal.close()

        return

#
# Handle mode = 'manualflag' and mode = 'quack'
#
def manualflag_quack(fglocal, mode, selectdata, flagbackup, **params):
        if debug: print params

        if not selectdata:
                params['antenna'] = params['timerange'] = params['correlation'] = params['scan'] = params['feed'] = params['array'] = params['uvrange'] = ''
        
        vector_mode = False         # Are we in vector mode?
        vector_length = -1          # length of all vectors
        vector_var = ''             # reference parameter
        is_vector_spec = {}         # is a variable a vector specification?
        for x in params.keys():
                is_vector_spec[x] = False
                #print x, params[x], type(params[x])
                if x != 'clipminmax':
                        if type(params[x]) == list:
                                is_vector_spec[x] = True

                else:
                        # clipminmax is a special case
                        if type(params[x]) == list and \
                                len(params[x]) > 0 and \
                                type(params[x][0]) == list:
                                is_vector_spec[x] = True

                if is_vector_spec[x]:
                        vector_mode = True
                        vector_length = len(params[x])
                        vector_var = x
                        if debug: print x, "is a vector => vector mode, length", vector_length
                else:
                        if debug: print x, "is not a vector"

        if not vector_mode:
                fglocal.setdata()
                rename_params(params)
                fglocal.setmanualflags(**params)
        else:
                # Vector mode
                plural_s = ''
                if vector_length > 1:
                        plural_s = 's'
                casalog.post('In parallel mode, will apply the following ' + str(vector_length) + \
                             ' flagging specification' + plural_s)
                
                # Check that parameters are consistent,
                # i.e. if they are vectors, they must have the same length
                for x in params.keys():
                        if is_vector_spec[x]:
                                l = len(params[x])

                                if debug: print x, "has length", l
                                if l != vector_length:
                                        raise Exception(str(x) + ' has length ' + str(l) + \
                                                        ', but ' + str(vector_var) + ' has length ' + str(vector_length))
                        else:
                                # vectorize this parameter (e.g.  '7' -> ['7', '7', '7']
                                params[x] = [params[x]] * vector_length

                if debug: print params
                
                # Input validation done.
                # Now call setmanualflags for every specification

                fglocal.setdata()
                for i in range(vector_length):
                        param_i = {}
                        param_list = ''
                        for e in params.keys():
                                param_i[e] = params[e][i]
                                if param_i[e] != '':
                                        if param_list != '':
                                                param_list += '; '
                                        param_list = param_list + e + ' = ' + str(param_i[e])

                        casalog.post(param_list)
                        rename_params(param_i)
                        if debug: print param_i
                        fglocal.setmanualflags(**param_i)

        if flagbackup:
                backup_flags(fglocal, mode)
        fglocal.run()

# rename some parameters,
# in order to match the interface of fglocal.tool
#
# validate parameter quackmode

def rename_params(params):
        if params.has_key('quackmode') and \
          not params['quackmode'] in ['beg', 'endb', 'end', 'tail']:
                raise Exception, "Illegal value '%s' of parameter quackmode, must be either 'beg', 'endb', 'end' or 'tail'" % (params['quackmode'])
        
        params['baseline']        = params['antenna']     ; del params['antenna']
        params['time']            = params['timerange']   ; del params['timerange']
        params['autocorrelation'] = params['autocorr']    ; del params['autocorr']
        params['cliprange']       = params['clipminmax']  ; del params['clipminmax']
        params['outside']         = params['clipoutside'] ; del params['clipoutside']

def backup_flags(fglocal, mode):

        # Create names like this:
        # before_manualflag_1,
        # before_manualflag_2,
        # before_manualflag_3,
        # etc
        #
        # Generally  before_<mode>_<i>, where i is the smallest
        # integer giving a name, which does not already exist
       
        existing = fglocal.getflagversionlist(printflags=False)

	# remove comments from strings
	existing = [x[0:x.find(' : ')] for x in existing]
	i = 1
	while True:
		versionname = mode +"_" + str(i)

		if not versionname in existing:
			break
		else:
			i = i + 1

        time_string = str(time.strftime('%Y-%m-%d %H:%M:%S'))

        casalog.post("Saving current flags to " + versionname + " before applying new flags")

        fglocal.saveflagversion(versionname=versionname,
                           comment='flagdata autosave before ' + mode + ' on ' + time_string,
                           merge='replace')

