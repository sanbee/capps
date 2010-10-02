import os
from taskinit import *

def importuvfits(fitsfile,vis,antnamescheme=None):
	"""Convert a UVFITS file to a CASA visibility data set (MS):

	Keyword arguments:
	fitsfile -- Name of input UV FITS file
		default: <unset>; example='3C273XC1.fits'
	vis -- Name of output visibility file (MS)
		default: <unset>; example: vis='3C273XC1.ms'


	"""

	#Python script
	try:
		casalog.origin('importuvfits')
		casalog.post("")
		ms.fromfits(vis,fitsfile,antnamescheme=antnamescheme)
		ms.close()
		# save original flagversion
		ok=fg.open(vis)
		ok=fg.saveflagversion('Original',comment='Original flags at import into CASA', merge='replace')
		ok=fg.done()
	        # write history
                if ((type(vis)==str) & (os.path.exists(vis))):
                        ms.open(vis,nomodify=False)
                else:
                        raise Exception, 'Visibility data set not found - please verify the name'
        	ms.writehistory(message='taskname     = importuvfits',origin='importuvfits')
        	ms.writehistory(message='fitsfile     = "'+str(fitsfile)+'"',origin='importuvfits')
        	ms.writehistory(message='vis          = "'+str(vis)+'"',origin='importuvfits')
        	ms.writehistory(message='antnamescheme= "'+str(antnamescheme)+'"',origin='importuvfits')
        	ms.close()

	except Exception, instance: 
		print '*** Error ***',instance
		raise Exception, instance


