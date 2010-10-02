from taskutil import get_global_namespace
import numpy
import os
import shutil
import stat

def make_labelled_ms(srcms, outputms, labelbases, ow=False, debug=False,
                     whichdatacol='DATA'):
    """
    Transform one measurement set into another with the same (ideally small but
    nontrivial) shape and reference frames, but data set to a sequence
    according to labelbases.  The output MS is useful for testing tasks or
    tools that read and write MSes, since it is more obvious where each bit of
    data came from, and what has been done to it.  Think of it as an
    alternative to making certain bits of data fluoresce under UV light.

    Arguments:

    srcms:    The template MS.
    
    outputms: The output MS.

    labelbases: A dictionary of quant: number pairs, where quant is an index to
    label, and number is how quickly the label changes with the index.  quant
    should be one of 'SCAN_NUMBER', 'DATA_DESC_ID', 'ANTENNA1', 'ANTENNA2',
    'ARRAY_ID', 'FEED1', 'FEED2', 'FIELD_ID', 'OBSERVATION_ID', 'PROCESSOR_ID',
    'STATE_ID', 'time', 'time_centroid', 'chan(nel)', or 'pol(whatever)'
    (case insensitive), but you will be let off with a warning if it isn't.
    TIME and TIME_CENTROID tend to hover around 4.7e9, so they are offset and scaled
    by subtracting the first value and dividing by the first integration interval.
    Example labelbases:
	labelbases = {'channel': 1.0, 'antenna1': complex(0, 1)}
        The data column in the output will be complex(channel index, antenna1).

	labelbases = {'SCAN_NUMBER': 1.0,
                      'FIELD_ID':    0.1,
                      'DATA_DESC_ID': complex(0, 1)}
        The data column in the output will go like complex(scan.field, spw) as
        long as field < 10.

    ow: Whether or not outputms can be overwritten if it exists.

    debug: If True and it hits an error, it will try to return what it has so
           far instead of raising an exception.

    whichdatacol: Which of DATA, MODEL_DATA, or CORRECTED_DATA to modify in the output.
                  Case insensitive.

    Returns True or False as a success guess.
    """
    # Make sure we have tb, casalog, clearcal, and ms.
    have_tools = False
    try:
        # Members that likely they have, but imposters (if any) do not.
        have_tools = hasattr(tb, 'colnames') and hasattr(ms, 'selectchannel')
        have_tools &= hasattr(casalog, 'post') and hasattr(clearcal, 'check_params')
    except NameError:
        pass  # through to next clause...
    if not have_tools:
        try:
            from taskutil import get_global_namespace
            my_globals = get_global_namespace()
            tb = my_globals['tb']
            ms = my_globals['ms']
            casalog = my_globals['casalog']
            clearcal = my_globals['clearcal']
        except NameError:
            print "Could not find tb and ms.  my_globals =",
            print "\n\t".join(my_globals.keys())

    casalog.origin("make_labelled_ms")

    whichdatacol = whichdatacol.upper()
    if whichdatacol not in ['DATA', 'MODEL_DATA', 'CORRECTED_DATA']:
        casalog.post(whichdatacol + "is not one of DATA, MODEL_DATA, or CORRECTED_DATA.",
                     'EXCEPTION')

    try:
        if outputms != srcms:
            if os.path.isdir(outputms):
                if ow:
                    shutil.rmtree(outputms)
                else:
                    print "Use ow=True if you really want to overwrite", outputms
                    return False
            shutil.copytree(srcms, outputms)
    except Exception, instance:
        casalog.post("*** Error %s copying %s to %s." % (instance,
                                                         srcms, outputms),
                     'SEVERE')
        return

    make_writable_recursively(outputms)
        
    tb.open(outputms, nomodify=False)

    if whichdatacol not in tb.colnames():
        casalog.post("Adding scratch columns to " + outputms, 'INFO')
        tb.close()
        clearcal(outputms)
        tb.open(outputms, nomodify=False)

    # Setup rowcols, polbase, and chanbase
    polbase = 0.0
    chanbase = 0.0
    rowcols = {}
    for quant in labelbases:
        try:
            if quant.upper() in ['SCAN_NUMBER', 'DATA_DESC_ID',
                                 'ANTENNA1', 'ANTENNA2', 'ARRAY_ID',
                                 'FEED1', 'FEED2', 'FIELD_ID',
                                 'OBSERVATION_ID', 'PROCESSOR_ID',
                                 'STATE_ID']:
                rowcols[quant] = tb.getcol(quant.upper())
            elif quant[:4].upper() == 'TIME':
                rowcols[quant] = tb.getcol(quant.upper())
                print quant, "is a timelike quantity, so it will be offset and scaled"
                print "by subtracting the first value and dividing by the first integration"
                print "interval."
                rowcols[quant] -= rowcols[quant][0]
                rowcols[quant] /= tb.getcell('INTERVAL')
            elif quant[:3].upper() == 'POL':
                polbase = labelbases[quant]
            elif quant[:4].upper() == 'CHAN':
                chanbase = labelbases[quant]
            else:
                casalog.post("Do not know how to label %s." % quant, 'WARN')
        except Exception, e:
            print "Error getting", quant
            if debug:
                return rowcols
            else:
                raise e

    dat = numpy.array(tb.getcol('DATA'))
    for rowind in xrange(dat.shape[2]):
        rowlabel = 0
        for q in rowcols:
            rowlabel += rowcols[q][rowind] * labelbases[q]
					
        for polind in xrange(dat.shape[0]):
            pollabel = rowlabel + polbase * polind
				
            for chanind in xrange(dat.shape[1]):
                label = pollabel + chanind * chanbase
                dat[polind, chanind, rowind] = label
    tb.putcol(whichdatacol, dat.tolist())
    tb.close()

    try:
        addendum = srcms + " labelled by labelbases = {\n"
        qbs = []
        for q, b in labelbases.items():
            qb = "\t%16s: " % ("'" + q + "'")
            if type(b) == complex:
                if b.real != 0.0:
                    qb += "%.1g" % b.real
                if b.imag != 0.0:
                    qb += "%+.1gi" % b.imag
            else:
                qb += "%.1g" % b
            qbs.append(qb)
        addendum += ",\n".join(qbs)
        addendum += "\n}"
        ms.open(outputms, nomodify=False)
        # The parms parameter is a false lead - ms.listhistory and listhistory
        # don't show it in the logger.
        ms.writehistory(addendum, origin="make_labelled_ms")
        ms.close()
    except Exception, instance:
        casalog.post("*** Error %s updating %s's history." % (instance,
                                                              outputms),
                     'SEVERE')
    return True


def make_writable_recursively(dir):
    """
    Unfortunately neither os nor shutil make operating on permissions as easy
    as it should be.
    """
    def walkable_chmod(mode, dirname, fnames):
        "Thanks to Fabian Steiner on the python list."
        dmode = os.lstat(dirname)[stat.ST_MODE]
        os.chmod(dirname, dmode | stat.S_IWUSR)
        for fname in fnames:
            dfname = os.path.join(dirname, fname)
            fmode = os.lstat(dfname)[stat.ST_MODE]
            os.chmod(dfname, fmode | stat.S_IWUSR)

    os.path.walk(dir, walkable_chmod, None)
