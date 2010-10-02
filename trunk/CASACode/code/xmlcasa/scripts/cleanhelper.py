##loolooaaa
import casac
import os
import commands
import math
#import pdb
import numpy
import shutil

###some helper tools
mstool = casac.homefinder.find_home_by_name('msHome')
ms = mstool.create()
tbtool = casac.homefinder.find_home_by_name('tableHome')
tb = tbtool.create()
qatool = casac.homefinder.find_home_by_name('quantaHome')
qa = qatool.create()
metool = casac.homefinder.find_home_by_name('measuresHome')
me = metool.create()
rgtool=casac.homefinder.find_home_by_name('regionmanagerHome')
rg = rgtool.create()
iatool=casac.homefinder.find_home_by_name('imageHome')
ia = iatool.create()

class cleanhelper:
    def __init__(self, imtool='', vis='', usescratch=False, casalog=None):
        """
        Contruct the cleanhelper object with an imager tool
        like so:
        a=cleanhelper(im, vis)
        """
        if((type(imtool) != str) and (len(vis) !=0)):
            self.initsinglems(imtool, vis, usescratch)
        self.maskimages={}
        self.finalimages={}
        self.usescratch=usescratch
        self.dataspecframe='LSRK'
        self.usespecframe='' 
        # to use phasecenter parameter in initChannelizaiton stage
        # this is a temporary fix need. 
        self.srcdir=''
        if not casalog:  # Not good!
            loghome =  casac.homefinder.find_home_by_name('logsinkHome')
            casalog = loghome.create()
            #casalog.setglobal(True)
        self._casalog = casalog
        
        
    def initsinglems(self, imtool, vis, usescratch):
        self.im=imtool
        self.vis=vis
        if ((type(vis)==str) & (os.path.exists(vis))):
            self.im.open(vis, usescratch=usescratch)
        else:
            raise Exception, 'Visibility data set not found - please verify the name'
        self.phasecenter=''
        self.spwindex=-1
        self.fieldindex=-1
        self.outputmask=''
        self.csys={}

    def defineimages(self, imsize, cell, stokes, mode, spw, nchan, start,
                     width, restfreq, field, phasecenter, facets=1, outframe='',
                     veltype='radio'):
        if((type(cell)==list) and (len(cell)==1)):
            cell.append(cell[0])
        elif ((type(cell)==str) or (type(cell)==int) or (type(cell)==float)):
            cell=[cell, cell]
        elif (type(cell) != list):
            raise TypeError, "parameter cell %s is not understood" % str(cell)
        cellx=qa.quantity(cell[0], 'arcsec')
        celly=qa.quantity(cell[1], 'arcsec')
        if(cellx['unit']==''):
            #string with no units given
            cellx['unit']='arcsec'
        if(celly['unit']==''):
            #string with no units given
            celly['unit']='arcsec'
        if((type(imsize)==list) and (len(imsize)==1)):
            imsize.append(imsize[0])
        elif(type(imsize)==int):
            imsize=[imsize, imsize]
        elif(type(imsize) != list):
            raise TypeError, "parameter imsize %s is not understood" % str(imsize)
            
	elstart=start
        if(mode=='frequency'):
        ##check that start and step have units
            if(qa.quantity(start)['unit'].find('Hz') < 0):
                raise TypeError, "start parameter %s is not a valid frequency quantity " % str(start)
            ###make sure we respect outframe
            if(self.usespecframe != ''):
                elstart=me.frequency(self.usespecframe, start)
            if(qa.quantity(width)['unit'].find('Hz') < 0):
                raise TypeError, "width parameter %s is not a valid frequency quantity " % str(width)	
        elif(mode=='velocity'): 
        ##check that start and step have units
            if(qa.quantity(start)['unit'].find('m/s') < 0):
                raise TypeError, "start parameter %s is not a valid velocity quantity " % str(start)
            ###make sure we respect outframe
            if(self.usespecframe != ''):
                elstart=me.radialvelocity(self.usespecframe, start)
            if(qa.quantity(width)['unit'].find('m/s') < 0):
                raise TypeError, "width parameter %s is not a valid velocity quantity " % str(width)	
        else:
            if((type(width) != int) or 
               (type(start) != int)):
                raise TypeError, "start (%s), width (%s) have to be integers with mode %s" % (str(start),str(width),mode)

        #####understand phasecenter
        if(type(phasecenter)==str):
            ### blank means take field[0]
            if (phasecenter==''):
                fieldoo=field
                if(fieldoo==''):
                    fieldoo='0'
                phasecenter=int(ms.msseltoindex(self.vis,field=fieldoo)['field'][0])
            else:
                tmppc=phasecenter
                try:
                    if(len(ms.msseltoindex(self.vis, field=phasecenter)['field']) > 0):
                        tmppc = int(ms.msseltoindex(self.vis,
                                                    field=phasecenter)['field'][0])
                    ##succesful must be string like '0' or 'NGC*'
                except Exception, instance:
                    ##failed must be a string 'J2000 18h00m00 10d00m00'
                    tmppc = phasecenter
                phasecenter = tmppc
        self.phasecenter = phasecenter
        #print 'cell', cellx, celly, restfreq
        ####understand spw
        if spw in (-1, '-1', '*', '', ' '):
            spwindex = -1
        else:
            spwindex=ms.msseltoindex(self.vis, spw=spw)['spw'].tolist()
            if(len(spwindex) == 0):
                spwindex = -1
        self.spwindex = spwindex
        ##end spwindex
        self.im.defineimage(nx=imsize[0],      ny=imsize[1],
                            cellx=cellx,       celly=celly,
                            mode=mode,         nchan=nchan,
                            start=elstart,       step=width,
                            spw=spwindex,      stokes=stokes,
                            restfreq=restfreq, outframe=outframe,
                            veltype=veltype, phasecenter=phasecenter,
                            facets=facets)

    def definemultiimages(self, rootname, imsizes, cell, stokes, mode, spw,
                          nchan, start, width, restfreq, field, phasecenters,
                          names=[], facets=1, outframe='', veltype='radio',
                          makepbim=False, checkpsf=False):
        #pdb.set_trace()
        #will do loop in reverse assuming first image is main field
        if not hasattr(imsizes, '__len__'):
            imsizes = [imsizes]
        self.nimages=len(imsizes)
        if((len(imsizes)<=2) and ((type(imsizes[0])==int) or
                                  (type(imsizes[0])==long))):
            self.nimages=1
            if(len(imsizes)==2):
                imsizes=[(imsizes[0], imsizes[1])]
            else:
                imsizes=[(imsizes[0], imsizes[0])]

        self._casalog.post('Number of images: ' + str(self.nimages), 'DEBUG1')
        #imagelist is to have the list of image model names
        self.imagelist={}
        #imageids is to tag image to mask in aipsbox style file 
        self.imageids={}
        if(type(phasecenters) == str):
            phasecenters=[phasecenters]
        if(type(phasecenters) == int):
            phasecenters=[phasecenters]
        self._casalog.post('Number of phase centers: ' + str(len(phasecenters)),
                           'DEBUG1')

        if((self.nimages==1) and (type(names)==str)):
            names=[names]
        if((len(phasecenters)) != (len(imsizes))):
            errmsg = "Mismatch between the number of phasecenters (%d), image sizes (%d) , and images (%d)" % (len(phasecenters), len(imsizes), self.nimages)
            self._casalog.post(errmsg, 'SEVERE')
            raise ValueError, errmsg
        self.skipclean=False
        lerange=range(self.nimages)
        lerange.reverse()
        for n in lerange:
            self.defineimages(list(imsizes[n]), cell, stokes, mode, spw, nchan,
                              start, width, restfreq, field, phasecenters[n],
                              facets,outframe,veltype)
            if(len(names)==self.nimages):
                self.imageids[n]=names[n]
                if(rootname != ''):
                    self.imagelist[n]=rootname+'_'+names[n]
                else:
                    self.imagelist[n]=names[n]
            else:
                self.imagelist[n]=rootname+'_'+str(n)
            ###make the image only if it does not exits
            ###otherwise i guess you want to continue a clean
            if(not os.path.exists(self.imagelist[n])):
                self.im.make(self.imagelist[n])
	    #if(makepbim and n==0):
	    if(makepbim):
    		##make .flux image 
                # for now just make for a main field 
                ###need to get the pointing so select the fields
                self.im.selectvis(field=field)
                # set to default minpb(=0.1), should use input minpb?
                self.im.setmfcontrol()
                self.im.setvp(dovp=True)
                self.im.makeimage(type='pb', image=self.imagelist[n]+'.flux',
                                  compleximage="", verbose=False)
		self.im.setvp(dovp=False, verbose=False)
            # Moved the following to a seperate function
            #if(checkpsf):
                # make sure psf can be created
                #self.im.makeimage(type='psf', image=self.imagelist[n]+'.test.psf')
                #ia.open(self.imagelist[n]+'.test.psf')
                #imdata=ia.getchunk()
                #if self.skipclean:
                #    pass
                #elif imdata.sum()==0.0:
                #    self.skipclean=True
            #    print "checkpsf DONE"
                
    def checkpsf(self,chan):
        """
        a check to make sure selected channel plane is not entirely flagged
        (for chinter=T interactive clean)
        """
        lerange=range(self.nimages)
        lerange.reverse()
        for n in lerange:
            #self.getchanimage(self.finalimages[n]+'_template.psf',self.imagelist[n]+'.test.psf',chan)
            self.getchanimage(self.finalimages[n]+'.psf',self.imagelist[n]+'.test.psf',chan)
            ia.open(self.imagelist[n]+'.test.psf')
            imdata=ia.getchunk()
            if self.skipclean:
                pass
            elif imdata.sum()==0.0:
                self.skipclean=True

    def makeEmptyimages(self):
        """
        Create empty images (0.0 pixel values) for 
        image, residual, psf
        must run after definemultiimages()
        and it is assumed that definemultiimages creates 
        empty images (self.imagelist). 
        """ 
        lerange=range(self.nimages)
        for n in lerange:
            os.system('cp -r '+self.imagelist[n]+' '+self.imagelist[n]+'.image')
            os.system('cp -r '+self.imagelist[n]+' '+self.imagelist[n]+'.residual')
            os.system('cp -r '+self.imagelist[n]+' '+self.imagelist[n]+'.psf')
            os.system('cp -r '+self.imagelist[n]+' '+self.imagelist[n]+'.model')
            os.system('cp -r '+self.imagelist[n]+' '+self.imagelist[n]+'.mask')

    
    def makemultifieldmask(self, maskobject=''):
        """
        This function assumes that the function definemultiimages has been run and thus
        self.imagelist is defined
        if single image use the single image version
        """
        if((len(self.maskimages)==(len(self.imagelist)))):
            if(not self.maskimages.has_key(self.imagelist[0])):
                self.maskimages={}
        else:
            self.maskimages={}
        masktext=[]
        if( (len(maskobject)==0) or (maskobject==[''])):
            return
        if(type(maskobject)==str):
            maskobject=[maskobject]
        if(type(maskobject) != list):
            ##don't know what to do with this
            raise TypeError, 'Dont know how to deal with mask object'
        n=0
        for masklets in maskobject:
            if(type(masklets)==str):
                    if(os.path.exists(masklets)):
                        if(commands.getoutput('file '+masklets).count('directory')):
                            self.maskimages[self.imagelist[n]]=masklets
                            n=n+1
                        elif(commands.getoutput('file '+masklets).count('text')):
                            masktext.append(masklets)
                        else:
                            raise TypeError, 'Can only read text mask files or mask images'
                    else:
                       raise TypeError, masklets+' seems to be non-existant' 
        if(len(masktext) > 0):
            circles, boxes=self.readmultifieldboxfile(masktext)
            if(len(self.maskimages)==0):
                for k in range(len(self.imageids)):
                    if(not self.maskimages.has_key(self.imagelist[k])):
                        self.maskimages[self.imagelist[k]]=self.imagelist[k]+'.mask'
            for k in range(len(self.imageids)):
                ###initialize mask if its not there yet
                if(not (os.path.exists(self.maskimages[self.imagelist[k]]))):
                    ia.fromimage(outfile=self.maskimages[self.imagelist[k]],
                                 infile=self.imagelist[k])
                    ia.open(self.maskimages[self.imagelist[k]])
                    ia.set(pixels=0.0)
                    ia.done(verbose=False)
                if(circles.has_key(self.imageids[k]) and boxes.has_key(self.imageids[k])):
                    self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[k]],
                                              boxes=boxes[self.imageids[k]],
                                              circles=circles[self.imageids[k]])
                elif(circles.has_key(self.imageids[k])):
                    self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[k]],
                                              circles=circles[self.imageids[k]])
                elif(boxes.has_key(self.imageids[k])):
                    self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[k]],
                                              boxes=boxes[self.imageids[k]])
                else:
                    ###need to make masks that select that whole image
                    ia.open(self.maskimages[self.imagelist[k]])
                    ia.set(pixels=1.0)
                    ia.done(verbose=False)

    def makemultifieldmask2(self, maskobject='',slice=-1):
        """
        New makemultifieldmask to accomodate different kinds of masks supported
        in clean with flanking fields (added by TT)
        required: definemultiimages has already run so that imagelist is defined 
        """
        if((len(self.maskimages)==(len(self.imagelist)))):
            if(not self.maskimages.has_key(self.imagelist[0])):
                self.maskimages={}
        else:
            self.maskimages={}

        #print "makemultifieldmask2: intial self.imagelist=",self.imagelist
        if((len(maskobject)==0) or (maskobject==[''])):
            return
        # determine number of input elements
        if (type(maskobject)==str):
            maskobject=[maskobject]
        if(type(maskobject) != list):
            ##don't know what to do with this
            raise TypeError, 'Dont know how to deal with mask object'
        if(type(maskobject[0])==int or type(maskobject[0])==float):
            maskobject=[maskobject] 
        if(type(maskobject[0][0])==list):
            if(type(maskobject[0][0][0])!=int and type(maskobject[0][0][0])!=float):        
                maskobject=maskobject[0]
                    
        # define maskimages
        if(len(self.maskimages)==0):
            for k in range(len(self.imageids)):
                if(not self.maskimages.has_key(self.imagelist[k])):
                    self.maskimages[self.imagelist[k]]=self.imagelist[k]+'.mask'
        # initialize maskimages
        # --- use outframe or dataframe for mask creation
        if self.usespecframe=='': 
            maskframe=self.dataspecframe
        else:
            maskframe=self.usespecframe
        for k in range(len(self.imagelist)):
            if(not (os.path.exists(self.maskimages[self.imagelist[k]]))):
                ia.fromimage(outfile=self.maskimages[self.imagelist[k]],
                        infile=self.imagelist[k])
                ia.open(self.maskimages[self.imagelist[k]])
                ia.set(pixels=0.0)
                mcsys=ia.coordsys().torecord()
                mcsys['spectral2']['conversion']['system']=maskframe
                ia.setcoordsys(mcsys)
                ia.done(verbose=False)

        # assume a file name list for each field
        masktext=[]
        # take out extra []'s
        maskobject=self.flatten(maskobject)
        for maskid in range(len(maskobject)):
            masklist=[]
            tablerecord=[]
            if(type(maskobject[maskid]))==str:
                if(maskobject[maskid])=='':
                    #skipped
                    continue
                else:
                    maskobject[maskid]=[maskobject[maskid]]
            for masklets in maskobject[maskid]:
                if(type(masklets)==int):
                    masklist.append(maskobject[maskid])
                if(type(masklets)==str):
                    if(masklets==''):
                        #skip
                        continue
                    if(os.path.exists(masklets)):
                        if(commands.getoutput('file '+masklets).count('directory')):
                            if(self.maskimages[self.imagelist[maskid]] == masklets):
                                self.maskimages[self.imagelist[maskid]]=masklets
                            else:
                                # make a copy
                                ia.open(self.imagelist[maskid])
                                self.csys=ia.coordsys().torecord()
                                shp = ia.shape()
                                ia.done(verbose=False)
                                if slice>-1:
                                    self.getchanimage(masklets,masklets+'chanim',slice)
                                    self.copymaskimage(masklets+'chanim',shp,'__tmp_mask')
                                    ia.removefile(maskets+'chanim')
                                else:
                                    self.copymaskimage(masklets, shp, '__tmp_mask')
                                #self.copymaskimage(masklets,shp,'__tmp_mask')
                                ia.open(self.maskimages[self.imagelist[maskid]])
                                ia.calc(pixels='+ __tmp_mask')
                                ia.done(verbose=False)
                                ia.removefile('__tmp_mask')

                        elif(commands.getoutput('file '+masklets).count('text')):
                            masktext.append(masklets)
                        else:
                            tablerecord.append(masklets)
                            #raise TypeError, 'Can only read text mask files or mask images'
                    else:
                        raise TypeError, masklets+' seems to be non-existant'

                if(type(masklets)==list):
                    masklist.append(masklets)
            # initialize mask
            #if (len(self.maskimages) > 0):
            #    if(not (os.path.exists(self.maskimages[self.imagelist[maskid]]))):
            #        ia.fromimage(outfile=self.maskimages[self.imagelist[maskid]],
            #                     infile=self.imagelist[maskid])
            #        ia.open(self.maskimages[self.imagelist[maskid]])
            #        ia.set(pixels=0.0)
            #        ia.done(verbose=False)
            #        print "INITIALIZED: ",self.maskimages[self.imagelist[maskid]]

            # handle boxes in lists
            if(len(masklist) > 0):
                self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[maskid]], boxes=masklist)
            if(len(tablerecord) > 0 ):
                reg={}
                for tabl in tablerecord:
                    reg.update({tabl:rg.fromfiletorecord(filename=tabl, verbose=False)})
                if(len(reg)==1):
                    reg=reg[reg.keys()[0]]
                else:
                    reg=rg.makeunion(reg)
                self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[maskid]], region=reg)
        #boxfile handling done all at once
        if(len(masktext) > 0):
            # fill for all fields in boxfiles
            circles, boxes=self.readmultifieldboxfile(masktext)
            # doit for all fields
            for k in range(len(self.imageids)):
                if(circles.has_key(self.imageids[k]) and boxes.has_key(self.imageids[k])):
                    self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[k]],
                                              boxes=boxes[self.imageids[k]],
                                              circles=circles[self.imageids[k]])
                elif(circles.has_key(self.imageids[k])):
                    self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[k]],
                                              circles=circles[self.imageids[k]])
                elif(boxes.has_key(self.imageids[k])):
                    self.im.regiontoimagemask(mask=self.maskimages[self.imagelist[k]],
                                                   boxes=boxes[self.imageids[k]])
        # set unused mask images to 1 for a whole field
        for key in self.maskimages:
            if(os.path.exists(self.maskimages[key])):
                ia.open(self.maskimages[key])
                fsum=ia.statistics()['sum']
                if(fsum[0]==0.0):
                    ia.set(pixels=1.0)
                ia.done(verbose=False)

    def make_mask_from_threshhold(self, imagename, thresh, outputmask=None):
        """
        Makes a mask image with the same coords as imagename where each
        pixel is True if and only if the corresponding pixel in imagename
        is >= thresh.

        The mask will be named outputmask (if provided) or imagename +
        '.thresh_mask'.  The name is returned on success, or False on failure.
        """
        if not outputmask:
            outputmask = imagename + '.thresh_mask'

        # im.mask would be a lot shorter, but it (unnecessarily) needs im to be
        # open with an MS.
        # I am not convinced that im.mask should really be using Quantity.
        # qa.quantity(quantity) = quantity.
        self.im.mask(imagename, outputmask, qa.quantity(thresh))
        
        ## # Copy imagename to a safe name to avoid problems with /, +, -, and ia.
        ## ia.open(imagename)
        ## shp = ia.shape()
        ## ia.close()
        ## self.copymaskimage(imagename, shp, '__temp_mask')
        
        ## self.copymaskimage(imagename, shp, outputmask)
        ## ia.open(outputmask)
        ## ###getchunk is a mem hog
        ## #arr=ia.getchunk()
        ## #arr[arr>0.01]=1
        ## #ia.putchunk(arr)
        ## #inpix="iif("+"'"+outputmask.replace('/','\/')+"'"+">0.01, 1, 0)"
        ## #ia.calc(pixels=inpix)
        ## ia.calc(pixels="iif(__temp_mask>" + str(thresh) + ", 1, 0)")
        ## ia.close()
        ## ia.removefile('__temp_mask')
        return outputmask
        
    def makemaskimage(self, outputmask='', imagename='', maskobject=[], slice=-1):
        """
        This function is an attempt to convert all the kind of 'masks' that
        people want to throw at it and convert it to a mask image to be used
        by imager...For now 'masks' include
        
        a)set of previous mask images
        b)lists of blc trc's
        c)record output from rg tool for e.g
        """
        if( (len(maskobject)==0) or (maskobject==[''])):
            return
        maskimage=[]
        masklist=[]
        masktext=[]
        maskrecord={}
        tablerecord=[]
        # relax to allow list input for imagename 
        if(type(imagename)==list):
           imagename=imagename[0] 

        if(type(maskobject)==dict):
            maskrecord=maskobject
            maskobject=[]
        if(type(maskobject)==str):
            maskobject=[maskobject]
        
        if(type(maskobject) != list):
            ##don't know what to do with this
            raise TypeError, 'Dont know how to deal with maskobject'
        if((type(maskobject[0])==int) or  (type(maskobject[0])==float)):
            masklist.append(maskobject)
        else:
            for masklets in maskobject:
                
                if(type(masklets)==str):
                    if(os.path.exists(masklets)):
                        if(commands.getoutput('file '+masklets).count('directory')):
                            maskimage.append(masklets)
                        elif(commands.getoutput('file '+masklets).count('text')):
                            masktext.append(masklets)
                        else:
                            tablerecord.append(masklets)
                    else:
                       raise TypeError, masklets+' seems to be non-existant' 
                if(type(masklets)==list):
                    masklist.append(masklets)
                if(type(masklets)==dict):
                    maskrecord=masklets
        if(len(outputmask)==0):
            outputmask=imagename+'.mask'
        if(os.path.exists(outputmask)):
            self.im.make('__temp_mask')    
            ia.open('__temp_mask')
            shp=ia.shape()
            self.csys=ia.coordsys().torecord()
            ia.close()
            ia.removefile('__temp_mask')
            ia.open(outputmask)
            ia.regrid(outfile='__temp_mask',shape=shp,axes=[0,1], csys=self.csys,overwrite=True)
            ia.done(verbose=False)
            ia.removefile(outputmask)
            os.rename('__temp_mask',outputmask)
        else:
            self.im.make(outputmask)
        ia.open(outputmask)
        shp=ia.shape()
        self.csys=ia.coordsys().torecord()
        # respect dataframe or outframe
        if self.usespecframe=='': 
            maskframe=self.dataspecframe
        else:
            maskframe=self.usespecframe
        self.csys['spectral2']['conversion']['system']=maskframe
        ia.setcoordsys(self.csys)
        ia.close()
        if(len(maskimage) > 0):
            for ima in maskimage :
                if slice>-1:
                    self.getchanimage(ima, ima+'chanim',slice)
                    self.copymaskimage(ima+'chanim',shp,'__temp_mask')
                    ia.removefile(ima+'chanim')
                else:
                    self.copymaskimage(ima, shp, '__temp_mask')
                #ia.open(ima)
                #ia.regrid(outfile='__temp_mask',shape=shp,csys=self.csys,
                #          overwrite=True)
                #ia.done(verbose=False)
                os.rename(outputmask,'__temp_mask2')
                ia.imagecalc(outfile=outputmask,
                             pixels='__temp_mask + __temp_mask2',
                             overwrite=True)
                ia.done(verbose=False)
                ia.removefile('__temp_mask')
                ia.removefile('__temp_mask2')
            outputmask = self.make_mask_from_threshhold(outputmask, 0.01,
                                                        outputmask)
        #pdb.set_trace()
        #### This goes when those tablerecord goes
        if(len(tablerecord) > 0):
            reg={}
            for tabl in tablerecord:
                reg.update({tabl:rg.fromfiletorecord(filename=tabl, verbose=False)})
            if(len(reg)==1):
                reg=reg[reg.keys()[0]]
            else:
                reg=rg.makeunion(reg)
            self.im.regiontoimagemask(mask=outputmask, region=reg)
        ###############
        if((type(maskrecord)==dict) and (len(maskrecord) > 0)):
            self.im.regiontoimagemask(mask=outputmask, region=maskrecord)
        if(len(masktext) >0):
            for textfile in masktext :
                polydic,listbox=self.readboxfile(textfile)
                masklist.extend(listbox)
                if(len(polydic) > 0):
                    self.im.regiontoimagemask(mask=outputmask, region=polydic)
        if((type(masklist)==list) and (len(masklist) > 0)):
            self.im.regiontoimagemask(mask=outputmask, boxes=masklist)
        if(os.path.exists(imagename) and (len(rg.namesintable(imagename)) !=0)):
            regs=rg.namesintable(imagename)
            if(type(regs)==str):
                    regs=[regs]
            for reg in regs:
                elrec=rg.fromtabletorecord(imagename, reg)
                self.im.regiontoimagemask(mask=outputmask, region=elrec)

        self.outputmask=outputmask
            
        #Done with making masks
    def datselweightfilter(self, field, spw, timerange, uvrange, antenna,scan,
                           wgttype, robust, noise, npixels, mosweight,
                           innertaper, outertaper, calready, nchan=-1, start=0, width=1):
        rmode='none'
        weighting='natural';
        if(wgttype=='briggsabs'):
            weighting='briggs'
            rmode='abs'
        elif(wgttype=='briggs'):
            weighting='briggs'
            rmode='norm'
        else:
            weighting=wgttype
            
        self.fieldindex=ms.msseltoindex(self.vis,field=field)['field'].tolist()
        if(len(self.fieldindex)==0):
            tb.open(self.vis+'/FIELD')
            self.fieldindex=range(tb.nrows())
            tb.close()
        #weighting and tapering should be done together
        if(weighting=='natural'):
            mosweight=False
        self.im.selectvis(nchan=nchan,start=start,step=width,field=field,spw=spw,time=timerange,
                              baseline=antenna, scan=scan, uvrange=uvrange, usescratch=calready)
        self.im.weight(type=weighting,rmode=rmode,robust=robust, npixels=npixels, noise=qa.quantity(noise,'Jy'), mosaic=mosweight)
        if((type(outertaper)==list) and (len(outertaper) > 0)):
            if(len(outertaper)==1):
                outertaper.append(outertaper[0])
                outertaper.append('0deg')
            if(qa.quantity(outertaper[0])['value'] > 0.0):    
                self.im.filter(type='gaussian', bmaj=outertaper[0],
                               bmin=outertaper[1], bpa=outertaper[2])

    def setrestoringbeam(self, restoringbeam):
        if((restoringbeam == ['']) or (len(restoringbeam) ==0)):
            return
        resbmaj=''
        resbmin=''
        resbpa='0deg'
        if((type(restoringbeam) == list)  and len(restoringbeam)==1):
            restoringbeam=restoringbeam[0]
        if((type(restoringbeam)==str)):
            if(qa.quantity(restoringbeam)['unit'] == ''):
                restoringbeam=restoringbeam+'arcsec'
            resbmaj=qa.quantity(restoringbeam, 'arcsec')
            resbmin=qa.quantity(restoringbeam, 'arcsec')
        if(type(restoringbeam)==list):			
            resbmaj=qa.quantity(restoringbeam[0], 'arcsec')
            resbmin=qa.quantity(restoringbeam[1], 'arcsec')
            if(resbmaj['unit']==''):
                resbmaj=restoringbeam[0]+'arcsec'
            if(resbmin['unit']==''):
                resbmin=restoringbeam[1]+'arcsec'
            if(len(restoringbeam)==3):
                resbpa=qa.quantity(restoringbeam[2], 'deg')
                if(resbpa['unit']==''):
                    resbmin=restoringbeam[2]+'deg'
        if((resbmaj != '') and (resbmin != '')):
           self.im.setbeam(resbmaj, resbmin, resbpa)
        
    def convertmodelimage(self, modelimages=[], outputmodel=''):
        modelos=[]
        maskelos=[]
        if((modelimages=='') or (modelimages==[])):
            return
        if(type(modelimages)==str):
            modelimages=[modelimages]
        k=0
        for modim in modelimages:
            ia.open(modim)
            modelos.append('modelos_'+str(k))
            if( (ia.brightnessunit().count('/beam')) > 0):
                maskelos.append(modelos[k]+'.sdmask')
                self.im.makemodelfromsd(sdimage=modim,modelimage=modelos[k],maskimage=maskelos[k])
            else:
                ##assuming its a model image already then just regrid it
                self.im.make(modelos[k])
                ia.open(modelos[k])
                newcsys=ia.coordsys()
                newshape=ia.shape()
                ia.open(modim)
                ib=ia.regrid(outfile=modelos[k], shape=newshape, axes=[0,1,3], csys=newcsys.torecord(), overwrite=True)
                ib.done(verbose=False)
                
            k=k+1
            ia.close()
        #########
        if((len(maskelos)==1) and  (self.outputmask == '')):
            self.outputmask=modelimages[0]+'.mask'
            if(os.path.exists(self.outputmask)):
                ia.removefile(self.outputmask)
            os.rename(maskelos[0],self.outputmask)
        elif(len(maskelos) > 0):
            if(self.outputmask == ''):
                self.outputmask=modelimages[0]+'.mask'
                
            else:
                outputmask=self.outputmask
            ##okay if outputmask exists then we need to do an "and" with
            ##the sdmask one
            doAnd=False;
            if(os.path.exists(outputmask)):
                ia.open(outputmask)
                if((ia.statistics()['max'].max()) > 0.00001):
                    doAnd=True
                ia.close()
            if(doAnd):
                tmpomask='__temp_o_mask'
                self.makemaskimage(outputmask=tmpomask, maskobject=maskelos)
                os.rename(outputmask, '__temp_i_mask')
                ia.imagecalc(outfile=outputmask, pixels='__temp_o_mask * __temp_i_mask', overwrite=True)
                ia.removefile('__temp_o_mask')
                ia.removefile('__temp_i_mask')
                self.outputmask=outputmask
            else:
                self.makemaskimage(outputmask=outputmask, maskobject=maskelos)
        for ima in maskelos:
            if(os.path.exists(ima)):
                ia.removefile(ima)
        if(not (os.path.exists(outputmodel))):
            self.im.make(outputmodel)
        for k in range(len(modelos)):
            os.rename(outputmodel,'__temp_model2')
            ia.imagecalc(outfile=outputmodel,
                             pixels=modelos[k]+' + '+'__temp_model2',
                             overwrite=True)
            
            ia.removefile('__temp_model2')
            ia.removefile(modelos[k])
            
    
    def readboxfile(self, boxfile):
        """ Read a file containing clean boxes (compliant with AIPS BOXFILE)

        Format is:
        #FIELDID BLC-X BLC-Y TRC-X TRC-Y
        0       110   110   150   150 
        or
        0       hh:mm:ss.s dd.mm.ss.s hh:mm:ss.s dd.mm.ss.s
        
        Note all lines beginning with '#' are ignored.
        
        """
        union=[]
        polyg={}
        f=open(boxfile)
        temprec={}
        counter=0
        while 1:
            try:
                counter=counter+1
                line=f.readline()
                if(len(line)==0):
                    raise Exception
                if (line.find('#')!=0):
                    if(line.count('[')==2):
                        ##its an output from qtclean
                        line=line.replace('\n','')
                        line=line.replace('\t',' ')
                        line=line.replace('[',' ')
                        line=line.replace(']',' ')
                        line=line.replace(',',' ')
                        splitline=line.split()
                        if(len(splitline)==5):
                            ##its box
                            if(int(splitline[4]) > 0):
                                ##it was a "mask" region not "erase"
                                boxlist=[int(splitline[0]),int(splitline[1]),
                                         int(splitline[2]),int(splitline[3])]
                                union.append(boxlist)
                        else:
                            #its a polygon
                            x=[]
                            y=[]
                            if(int(splitline[len(splitline)-1]) > 0):
                                ###ignore erase regions
                                nnodes=(len(splitline)-1)/2
                                for kk in range(nnodes):
                                    x.append(splitline[kk]+'pix')
                                    y.append(splitline[kk+nnodes]+'pix')
                                elreg=rg.wpolygon(x=x, y=y, csys=self.csys)
                                temprec.update({counter:elreg})
                                
                    else:
                        ### its an AIPS boxfile
                        splitline=line.split('\n')
                        splitline2=splitline[0].split()
                        if (len(splitline2)<6):
                            if(int(splitline2[1])<0):
                                ##circle
                                #circlelist=[int(splitline2[2]),
                                #     int(splitline2[3]),int(splitline2[4])]
                                #circles[splitline2[0]].append(circlelist)
                                continue
                            else:
                                boxlist=[int(splitline2[1]),int(splitline2[2]),
                                         int(splitline2[3]),int(splitline2[4])]
                                union.append(boxlist)
                        else:
                           ## Don't know what that is
                           ## might be a facet definition 
                           continue
    
                       
    
            except:
                break

        f.close()
        if(len(temprec)==1):
            polyg=temprec[temprec.keys()[0]]
        elif (len(temprec) > 1):
            polyg=rg.dounion(temprec)
        return polyg,union

    def readmultifieldboxfile(self, boxfiles):
        circles={}
        boxes={}
        for k in range(len(self.imageids)):
            circles[self.imageids[k]]=[]
            boxes[self.imageids[k]]=[]
        for boxfile in boxfiles:
            f=open(boxfile)
            while 1:
                try:
                    line=f.readline()
                    if(len(line)==0):
                        raise Exception
                    if (line.find('#')!=0):
                        ### its an AIPS boxfile
                        splitline=line.split('\n')
                        splitline2=splitline[0].split()
                        #print "splitline2=",splitline2
                        if (len(splitline2)<6):
                            ##circles
                            if(int(splitline2[1]) <0):
                                circlelist=[int(splitline2[2]),
                                            int(splitline2[3]),int(splitline2[4])]
                                #circles[splitline2[0]].append(circlelist)
                                circles[self.imageids[int(splitline2[0])]].append(circlelist)
                            else:
                                #boxes
                                boxlist=[int(splitline2[1]),int(splitline2[2]),
                                         int(splitline2[3]),int(splitline2[4])]
                                #boxes[splitline2[0]].append(boxlist)
                                boxes[self.imageids[int(splitline2[0])]].append(boxlist)
                        else:
                           ## Don't know what that is
                           ## might be a facet definition 
                            continue
    
                       
    
                except:
                    break

            f.close()
        ###clean up the records
        for k in range(len(self.imageids)):
            if(circles[self.imageids[k]]==[]):
                circles.pop(self.imageids[k])
            if(boxes[self.imageids[k]]==[]):
                boxes.pop(self.imageids[k])
                
        return circles,boxes

    def readoutlier(self, outlierfile):
        """ Read a file containing clean boxes (kind of
        compliant with AIPS FACET FILE)
            
        Format is:
         col0    col1   col2  col3 col4 col5 col6 col7  col8   col9
          C    FIELDID SIZEX SIZEY RAHH RAMM RASS DECDD DECMM DECSS      
        why first column has to have C ... because its should
        not to be A or B ...now D would be a totally different thing

        Note all lines beginning with '#' are ignored.
        
        """
        imsizes=[]
        phasecenters=[]
        imageids=[]
        f=open(outlierfile)
        while 1:
            try:
                line=f.readline()
                    
                if(len(line)==0):
                    raise Exception
                if (line.find('#')!=0):
                    splitline=line.split('\n')
                    splitline2=splitline[0].split()
                    if (len(splitline2)==10):
                        if(splitline2[0]=='C'):
                            imageids.append(splitline2[1])
                            imsizes.append((int(splitline2[2]),int(splitline2[3])))
                            mydir='J2000  '+splitline2[4]+'h'+splitline2[5]+'m'+splitline2[6]+'  '+splitline2[7]+'d'+splitline2[8]+'m'+splitline2[9]
                            phasecenters.append(mydir)
    
            except:
                break

        f.close()
        return imsizes,phasecenters,imageids

    def copymaskimage(self, maskimage, shp, outfile):
        if outfile == maskimage:     # Make it a no-op,
            return                   # this is more than just peace of mind.
        #pdb.set_trace() 
        ia.open(maskimage)
        oldshp=ia.shape()
        if((len(oldshp) < 4) or (shp[2] != oldshp[2]) or (shp[3] != oldshp[3])):
            #take the first plane of mask
            tmpshp=oldshp
            tmpshp[0]=shp[0]
            tmpshp[1]=shp[1]
            ib=ia.regrid(outfile='__looloo', shape=tmpshp, axes=[0,1], csys=self.csys, overwrite=True)
            #dat=ib.getchunk()
            ib.done(verbose=False)
            ia.fromshape(outfile=outfile, shape=shp, csys=self.csys, overwrite=True)
            ##getchunk is a massive memory hog
            ###so going round in a funny fashion
            #arr=ia.getchunk()
            #for k in range(shp[2]):
            #    for j in range(shp[3]):
            #        if(len(dat.shape)==2):
            #            arr[:,:,k,j]=dat
            #        elif(len(dat.shape)==3):
            #            arr[:,:,k,j]=dat[:,:,0]
            #        else:
            #            arr[:,:,k,j]=dat[:,:,0,0]
            #ia.putchunk(arr)
            ia.calc('__temp_mask[index3 in [0]]+__looloo') 
            ia.done(verbose=False)
            ia.removefile('__looloo')
        else:
            ib=ia.regrid(outfile=outfile ,shape=shp, axes=[0,1], csys=self.csys, overwrite=True)
            ia.done(verbose=False)
            ib.done(verbose=False)


    def flatten(self,l):
        """
        A utility function to flatten nested lists 
        but allow nesting of [[elm1,elm2,elm3],[elm4,elm5],[elm6,elm7]]
        to handle multifield masks.
        This does not flatten if an element is a list of int or float. 
        And also leave empty list as is.
        """ 
        retlist = []
        l = list(l)
        #print 'l=',l
        for i in range(len(l)):
            #print "ith l=",i, l[i] 
            if isinstance(l[i],list) and l[i]:
                # and (not isinstance(l[i][0],(int,float))):
                #print "recursive l=",l
                if isinstance(l[i][0],list) and isinstance(l[i][0][0],list):
                   retlist.extend(self.flatten(l[i]))
                else:
                   retlist.append(l[i])
            else:
                retlist.append(l[i])
        return retlist 


    def getchanimage(self,cubeimage,outim,chan):
        """
        create a slice of channel image from cubeimage
        """
        #pdb.set_trace()
        ia.open(cubeimage)
        modshape=ia.shape()
        if modshape[3]==1:
          return False
        if modshape[3]-1 < chan:
          return False
        blc=[0,0,modshape[2]-1,chan]
        trc=[modshape[0]-1,modshape[1]-1,modshape[2]-1,chan]
        sbim=ia.subimage(outfile=outim, region=rg.box(blc,trc), overwrite=True)
        sbim.close()
        ia.close()
        return True

    def putchanimage(self,cubimage,inim,chan):
        """
        put channel image back to a pre-exisiting cubeimage
        """
        ia.open(inim)
        inimshape=ia.shape()
        imdata=ia.getchunk()
        immask=ia.getchunk(getmask=True)
        ia.close()
        blc=[0,0,inimshape[2]-1,chan]
        trc=[inimshape[0]-1,inimshape[1]-1,inimshape[2]-1,chan]
        ia.open(cubimage)
        cubeshape=ia.shape()
        if not (cubeshape[3] > (chan+inimshape[3]-1)):
            return False
        rg0=ia.setboxregion(blc=blc,trc=trc)
        if inimshape[0:3]!=cubeshape[0:3]: 
            return False
        #ia.putchunk(pixels=imdata,blc=blc)
        ia.putregion(pixels=imdata,pixelmask=immask, region=rg0)
        ia.close()
        return True


    def oldsetChannelization(self,mode,spw,field,nchan,start,width,frame,veltype,restf):
        """
        determine appropriate values for channelization
        parameters when default values are used
        for mode='velocity' or 'frequency' or 'channel'
        """
        #pdb.set_trace()
        #if (mode!='frequency' and mode!='velocity') or (nchan!=-1 and start!='' and width!=''):
        instartunit=''
        inwidthunit=''
        if(mode=='channel'):
            ###for mode channel ignore the frame to use the data frame
            #if(frame != ''):
            #    self._casalog.post('Note: in channel mode, the outframe parameter is ignored.', 'WARN')
            #frame=''
            if(type(start)!=int):
                raise TypeError, "Wrong type for start parameter. Int is expected for the channel mode." 
            if(type(width)!=int):
                raise TypeError, "Wrong type for width parameter. Int is expected for the channel mode." 
        elif(mode=='frequency' or mode=='velocity'):
            if(type(start)!=str or type(width)!=str):
                if type(start)==int:
                    if start==0:
                        #assume the default is not properly set
                        start=''
                if type(width)==int:
                    if width==1:
                       # assume the default is not properly set 
                        width=''
                else:      
                    raise TypeError, "Start and width parameters must be given in strings, for mode=%s" % mode

        ####use the frame defined by user
        if(frame != ''):
            self.usespecframe=frame    
        if(nchan!=-1 and start!='' and width!=''):
            # do nothing
            retnchan=nchan
            retstart=start
            retwidth=width 
            ##do a dummy run to get the frame mainly
            (dumfreqlist,dumfinc)=self.getfreqs(1,spw,'','', True)
            if(self.usespecframe == ''):
                self.usespecframe=self.dataspecframe
            return retnchan,retstart,retwidth

        if(mode=='channel'):
            if(nchan==-1):
                (freqlist,finc)=self.getfreqs(nchan,spw,start,width)
                retnchan=len(freqlist)
            else:
                retnchan=nchan
            retstart=start
            retwidth=width 
            return retnchan,retstart,retwidth

        if(mode=='velocity'): 
            # keep original unit
            if(qa.quantity(start)['unit'].find('m/s') > -1):
                instartunit=qa.quantity(start)['unit']
            else:
                if start!='': 
                    raise TypeError, "Unrecognized unit for start parameter in velocity mode"
            if(qa.quantity(width)['unit'].find('m/s') > -1):
                inwidthunit=qa.quantity(start)['unit']
            else:
                if width!='':
                    raise TypeError, "Unrecognized unit for width parameter in velocity mode"
            if((start=='' or width=='') or nchan==-1):
                if(veltype!='radio'):
                    raise TypeError, "Currently default nchan, start and width for velocity mode work with the default veltype(='radio') only "
                # convert to frequency
            if(start!=''):
                start=self.convertvf(start,frame,field,restf)     
        if(mode=='frequency'): 
            # keep original unit
            if(qa.quantity(start)['unit'].find('Hz') > -1):
                instartunit=qa.quantity(start)['unit']
            else:
                if start!='': 
                    raise TypeError, "Unrecognized unit for start parameter in frequency mode"
            if(qa.quantity(width)['unit'].find('Hz') > -1):
                inwidthunit=qa.quantity(width)['unit']
            else:
                if width!='': 
                    raise TypeError, "Unrecognized unit for width parameter in frequency mode"
        # deal with frequency or velocity mode
        if(start==''):
            if(width==''):
                loc_width=1
            else:
                if(qa.quantity(width)['unit'].find('m/s') > -1):
                    loc_width=1
                elif(qa.quantity(width)['unit'].find('Hz') < 0):
                    raise TypeError, "width parameter is not a valid frequency quantity "
                else:
                    loc_width=width
            if(loc_width==1 and frame != ''):
                self._casalog.post('Note: in frequency and velocity mode, the default width is the original channel width\n  and is not converted to the output reference frame.', 'WARN')
            (freqlist, finc)=self.getfreqs(nchan,spw,0,loc_width)
            ###use the bloody frame of the data to define the start for defaults
            self.usespecframe=self.dataspecframe
            retnchan = len(freqlist)
        #    if(mode=='velocity' and nchan==-1):
        #       vmin=self.convertvf(str(freqlist[-1])+'Hz',frame,field) 
        #       vmax=self.convertvf(str(freqlist[0])+'Hz',frame,field) 
        #       if(width==''):
        #           vwidth=qa.sub(qa.quantity(vmax),qa.quantity(self.convertvf(str(freqlist[1])+'Hz',frame,field)))
        #       else:
        #           vwidth=qa.convert(width,'m/s')
        #       vrange=qa.sub(qa.quantity(vmax),qa.quantity(vmin))
        #       retnchan=min(int(math.ceil(qa.div(vrange,qa.abs(qa.quantity(vwidth)))['value']))+1,retnchan)
        else:
            if(width==''):
                if(frame != ''):
                    self._casalog.post('Note: in frequency and velocity mode, the default width is the original channel width\n  and is not converted to the output reference frame.', 'WARN')
                # need be done in a better way... 
                if(mode=='velocity'):
                    (freqlist,finc)=self.getfreqs(nchan,spw,start,-1)
                else: 
                    (freqlist,finc)=self.getfreqs(nchan,spw,start,1)
            else:
                if(mode=='velocity'):
                    loc_width=1
                else:
                    loc_width=width
                if(loc_width==1 and frame != ''):
                    self._casalog.post('Note: in frequency and velocity mode, the default width is the original channel width\n  and is not converted to the output reference frame.', 'WARN')
                (freqlist,finc)=self.getfreqs(nchan,spw,start,loc_width)
            ###at this stage it is safe to declare now that the user want the data frame
            if(self.usespecframe==''):
                self.usespecframe=self.dataspecframe
            retnchan = len(freqlist)

        if(mode=='frequency'):
            # returned values are in Hz
            if instartunit=='':
                retstart = str(freqlist[0])+'Hz'
            else:
                retstart = self.qatostring(qa.convert(str(freqlist[0])+'Hz',instartunit))

            if inwidthunit=='':
                retwidth = str(finc)+'Hz'
            else:
                retwidth = self.qatostring(qa.convert(str(finc)+'Hz',inwidthunit))
             
        elif(mode=='velocity'):
            #convert back to velocities (take max freq for min vel )
            if start=='':
                retstart = self.convertvf(str(freqlist[-1])+'Hz',frame,field,restf)
            else:
                retstart = self.convertvf(start,frame,field,restf)

            if nchan ==-1:
            # re-caluculate nchan
                if retnchan > 1:
                    vmin=self.convertvf(str(freqlist[-1])+'Hz',frame,field,restf) 
                    vmax=self.convertvf(str(freqlist[0])+'Hz',frame,field,restf) 
                    if(width==''):
                        vwidth=qa.sub(qa.quantity(vmax),qa.quantity(self.convertvf(str(freqlist[1])+'Hz',frame,field,restf)))
                    else:
                        vwidth=qa.convert(width,'m/s')
                    vrange=qa.sub(qa.quantity(vmax),qa.quantity(vmin))
                else:
                    vwidth=qa.quantity(str(finc)+'m/s')
                    vrange=vwidth
                retnchan=min(int(math.ceil(qa.div(vrange,qa.abs(qa.quantity(vwidth)))['value']))+1,retnchan)
                retwidth=vwidth

            else:
                if(width==''):
                    # width should be determined from last freq channels
                    v1 = self.convertvf(str(freqlist[-2])+'Hz',frame,field,restf)
                    v0 = self.convertvf(str(freqlist[-1])+'Hz',frame,field,restf)
                    retwidth=str(qa.quantity(qa.sub(qa.quantity(v1),qa.quantity(v0)))['value'])+'m/s'
                else:
                    retwidth=width
        else:
            raise TypeError, "Specified mode is not support"

        return retnchan, retstart, retwidth

    def qatostring(self,q):
        """
        return a quantity in string
        """
        if not q.has_key('unit'):
            raise TypeError, "Does not seems to be quantity"
        return str(q['value'])+q['unit']

    def convertvf(self,vf,frame,field,restf,veltype='radio'):
        """
        returns doppler(velocity) or frequency in string
        currently use first rest frequency
        Assume input vf (velocity or fequency) and output are
        the same 'frame'.
        """
        #pdb.set_trace()
        docalcf=False
        #if(frame==''): frame='LSRK' 
        #Use datasepcframe, it is cleanhelper initialized to set
        #to LSRK
        if(frame==''): frame=self.dataspecframe
        if(qa.quantity(vf)['unit'].find('m/s') > -1):
            docalcf=True
        elif(qa.quantity(vf)['unit'].find('Hz') > -1):
            docalcf=False
        else:
            if vf !=0:
                raise TypeError, "Unrecognized unit for the velocity or frequency parameter"
        fldinds=ms.msseltoindex(self.vis, field=field)['field'].tolist()
        if(len(fldinds) == 0):
            fldid0=0
        else:
            fldid0=fldinds[0]
        if restf=='':
            tb.open(self.vis+'/FIELD')
            srcid=tb.getcell('SOURCE_ID',fldid0)
            tb.close()
            if fldid0==-1:
                raise TypeError, "Rest frequency info is not supplied"
            tb.open(self.vis+'/SOURCE')
            rfreq=tb.getcell('REST_FREQUENCY',fldid0)
            if(rfreq<=0):
                raise TypeError, "Rest frequency does not seems to be properly set, check the data"
            tb.close()
        else:
            if type(restf)==str: restf=[restf]
            if(qa.quantity(restf[0])['unit'].find('Hz') > -1):
                rfreq=[qa.convert(qa.quantity(restf[0]),'Hz')['value']] 
                #print "using user input rest freq=",rfreq
            else:
                raise TypeError, "Unrecognized unit or type for restfreq"
        if(vf==0):
            # assume just want to get a restfrequecy from the data
            ret=str(rfreq[0])+'Hz'
        else:
            if(docalcf):
                dop=me.doppler(veltype, qa.quantity(vf)) 
                rvf=me.tofrequency(frame, dop, qa.quantity(rfreq[0],'Hz'))
            else:
                frq=me.frequency(frame, qa.quantity(vf))
                rvf=me.todoppler(veltype, frq, qa.quantity(rfreq[0],'Hz')) 
            ret=str(rvf['m0']['value'])+rvf['m0']['unit']
        return ret 


    def getfreqs(self,nchan,spw,start,width, dummy=False):
        """
        returns a list of frequencies to be used in output clean image
        if width = -1, start is actually end (max) freq 
        """
        #pdb.set_trace()
        freqlist=[]
        finc=1
        loc_nchan=0

        if spw in (-1, '-1', '*', '', ' '):
            spwinds = -1
        else:
            spwinds=ms.msseltoindex(self.vis, spw=spw)['spw'].tolist()
            if(len(spwinds) == 0):
                spwinds = -1

        if(spwinds==-1):
            # first row
            spw0=0
        else:
            spw0=spwinds[0]
        tb.open(self.vis+'/SPECTRAL_WINDOW')
        chanfreqscol=tb.getvarcol('CHAN_FREQ')
        chanwidcol=tb.getvarcol('CHAN_WIDTH')
        spwframe=tb.getcol('MEAS_FREQ_REF');
        tb.close()
        # assume spw[0]  
        elspecframe=["REST",
                     "LSRK",
                     "LSRD",
                     "BARY",
                     "GEO",	    
                     "TOPO",
                     "GALACTO",
                     "LGROUP",
                     "CMB"]
        self.dataspecframe=elspecframe[spwframe[spw0]];
        if(dummy):
            return freqlist, finc
        #DP extract array from dictionary returned by getvarcol
        chanfreqs1dx = numpy.array([])
        chanfreqs=chanfreqscol['r'+str(spw0+1)].transpose()
        chanfreqs1dx = chanfreqs[0]
        if(spwinds!=-1):
            for ispw in range(1,len(spwinds)):
                chanfreqs=chanfreqscol['r'+str(spwinds[ispw]+1)].transpose()            
                chanfreqs1dx = numpy.concatenate((chanfreqs1dx, chanfreqs[0]))
        chanfreqs1d = chanfreqs1dx.flatten()        
        #RI this is woefully inadequate assuming the first chan's width
        #applies to everything selected, but we're going to replace all
        #this with MSSelect..
        chanwids=chanwidcol['r'+str(spw0+1)].transpose()
        chanfreqwidth=chanwids[0][0]
        
        if(type(start)==int or type(start)==float):
            if(start > len(chanfreqs1d)):
                raise TypeError, "Start channel is outside the data range"
            startf = chanfreqs1d[start]
        elif(type(start)==str):
            if(qa.quantity(start)['unit'].find('Hz') > -1):
                startf=qa.convert(qa.quantity(start),'Hz')['value']
            else:
                raise TypeError, "Unrecognized start parameter"
        if(type(width)==int or type(width)==float):
            if(type(start)==int or type(start)==float):
                #finc=(chanfreqs1d[start+1]-chanfreqs1d[start])*width
                finc=(chanfreqwidth)*width
                # still need to convert to target reference frame!
            elif(type(start)==str):
                if(qa.quantity(start)['unit'].find('Hz') > -1):
                   # assume called from setChannelization with local width=1
                   # for the default width(of clean task parameter)='' for
                   # velocity and frequency modes. This case set width to 
                   # first channel width (for freq) and last one (for vel) 
                   if width==-1:
                       finc=chanfreqs1d[-1]-chanfreqs1d[-2]
                   else:
                       finc=chanfreqs1d[1]-chanfreqs1d[0]

                   # still need to convert to target reference frame!
        elif(type(width)==str):
            if(qa.quantity(width)['unit'].find('Hz') > -1):
                finc=qa.convert(qa.quantity(width),'Hz')['value']
        if(nchan ==-1):
            if(qa.quantity(start)['unit'].find('Hz') > -1):
                if width==-1: # must be in velocity order (i.e. startf is max)
                    bw=startf-chanfreqs1d[0]
                else:
                    bw=chanfreqs1d[-1]-startf
            else:
                bw=chanfreqs1d[-1]-chanfreqs1d[start]
            if(bw < 0):
                raise TypeError, "Start parameter is outside the data range"
            if(qa.quantity(width)['unit'].find('Hz') > -1):
                qnchan=qa.convert(qa.div(qa.quantity(bw,'Hz'),qa.quantity(width)))
                #DP loc_nchan=int(math.ceil(qnchan['value']))+1
                loc_nchan=int(round(qnchan['value']))+1
            else:
                #DP loc_nchan=int(math.ceil(bw/finc))+1
                loc_nchan=int(round(bw/finc))+1
        else:
            loc_nchan=nchan
        for i in range(int(loc_nchan)):
            if(i==0): 
                freqlist.append(startf)
            else:
                freqlist.append(freqlist[-1]+finc) 
        return freqlist, finc

    def setChannelization(self,mode,spw,field,nchan,start,width,frame,veltype,restf):
        """
        determine appropriate values for channelization
        parameters when default values are used
        for mode='velocity' or 'frequency' or 'channel'
        """

        tb.open(self.vis+'/SPECTRAL_WINDOW')
        chanfreqscol=tb.getvarcol('CHAN_FREQ')
        chanwidcol=tb.getvarcol('CHAN_WIDTH')
        spwframe=tb.getcol('MEAS_FREQ_REF');
        tb.close()

        # first parse spw parameter:

        # use MSSelect if possible
        if spw in (-1, '-1', '*', '', ' '):
            # spwinds = -1
            # chaninds = -1
            # get nchan of all spw from mssel- after this spwinds and chaninds should always 
            # be defined
            # spw = range(len(chanfreqscol))
            spw="*"

        sel=ms.msseltoindex(self.vis, spw=spw)
        # spw returned by msseletoindex, spw='0:5~10;10~20' 
        # will give spw=[0] and len(spw) not equal to len(chanids)
        # so get spwids from chaninds instead.
        chaninds=sel['channel'].tolist()
        spwinds=[]
        for k in range(len(chaninds)):
            spwinds.append(chaninds[k][0])
        if(len(spwinds) == 0):
            raise Exception, 'unable to parse spw parameter '+spw;
            
        # the first selected spw 
        spw0=spwinds[0]

        # set dataspecframe:
        elspecframe=["REST",
                     "LSRK",
                     "LSRD",
                     "BARY",
                     "GEO",	    
                     "TOPO",
                     "GALACTO",
                     "LGROUP",
                     "CMB"]
        self.dataspecframe=elspecframe[spwframe[spw0]];

        # set usespecframe:  user's frame if set, otherwise data's frame
        if(frame != ''):
            self.usespecframe=frame
        else:
            self.usespecframe=self.dataspecframe


        # extract array from dictionary returned by getvarcol
        # and accumulate selected chan freqs in chanfreqs1d (sorted by flatten)

        chanfreqs=chanfreqscol['r'+str(spw0+1)].transpose()
        # ALL chans in first spw - keep this because later if in chan mode, start will 
        # be an index in this list, and width will also calculate based on this
        chanfreqs0 = chanfreqs[0]  
        if len(chanfreqs0)<1:
            raise Exception, 'spw parameter '+spw+' selected spw '+str(spw0+1)+' that has no frequencies - SPECTRAL_WINDOW table may be corrupted'

        
        # start accumulating channels:
        chanind0=chaninds[0]
                
        chanfreqs1dx = numpy.array(chanfreqs0[chanind0[1]])
        for ci in range(chanind0[1],chanind0[2]+1,chanind0[3])[1:]:
            chanfreqs1dx = numpy.append(chanfreqs1dx,chanfreqs0[ci])

        # while we're here get width of first selected channel of first selected spw
        chanwids0=chanwidcol['r'+str(spw0+1)].transpose()
        chan0freqwidth=chanwids0[0][chanind0[1]]

        # more spw?:
        # spw returned by msseletoindex, spw='0:5~10;10~20' 
        # will give spw=[0] and len(spw) not equal to len(chanids)
        chanindi=chanind0
        spwi=spw0
        unsorted=False
        for isel in range(1,len(chaninds)):
            chanindi=chaninds[isel]
            spwi=chanindi[0]
            chanfreqs=chanfreqscol['r'+str(spwi+1)].transpose()
            chanfreqsi = chanfreqs[0]  
            if len(chanfreqsi)<1:
                raise Exception, 'spw parameter '+spw+' selected spw '+str(spwinds[isel]+1)+' that has no frequencies - SPECTRAL_WINDOW table may be corrupted'
            if chanfreqsi[0] < chanfreqs0[0]: 
                unsorted = True 
            for ci in range(chanindi[1],chanindi[2]+1,chanindi[3]):
                chanfreqs1dx = numpy.append(chanfreqs1dx,chanfreqsi[ci])
         
        # get width of last selected channel of last spw (that could be used for width in vel mode)
        chanwidsN=chanwidcol['r'+str(spwi+1)].transpose()
        chanNfreqwidth=chanwidsN[0][chanindi[2]]


        # flatten:
        chanfreqs1d = chanfreqs1dx.flatten()        
        # flatten() does not sort.
        chanfreqs1d.sort()
        chanfreqs0.sort()
                
        # now we have a list of the selected channel freqs in the data, 
        # and we can start to parse start/width/nchan in a mode-dependent way:

        # noninteractive clean(mode="freq") passes start=0 as default.
        if (mode=="frequency" or mode=="velocity") and start==0:            
            start=""
 
        # copy these params - we may change them
        locstart=start
        locwidth=width
        inwidthunit=''
        instartunit=''
        
        # if start is float or int, will interpret as channel index.  otherwise
        # otherwise convert start/width from vel to freq, save original units.
        # do the conversion in the USER-SPECIFIED outframe 
        # if frame='', convertvf will use dataspecframe

        if type(start)==str:  
            instartunit=qa.quantity(start)['unit']            
            if(qa.quantity(start)['unit'].find('m/s') > -1):                
                locstart=self.convertvf(start,frame,field,restf,veltype=veltype)

        if type(width)==str:  
            inwidthunit=qa.quantity(width)['unit']
            if(qa.quantity(width)['unit'].find('m/s') > -1):            
                if veltype!="radio":
                    self._casalog.post('Note: the specified width '+width+' in frame '+veltype+' is being converted to frequency at the rest frequency, as in the radio frame', 'WARN')
                    # the only other choice I could think of for freq at which to do the 
                    # conversion would be at the start freq or end freq...
                tmprestf=restf
                if restf=="":
                    # run a convertvf to get rest freq from the data
                    tmprestf=self.convertvf(0,frame,field,restf,veltype=veltype)
                locwidth=self.qatostring(qa.sub(self.convertvf(width,frame,field,restf,veltype=veltype),tmprestf))
                
        # now locstart and locwidth are either strings in freq, or numbers = chan indices, or ""

        # next, convert chan indices into freqs, and convert string/quantities into numbers in Hz:
        # some of this is mode-dependent for the interpretation of channels and defaults:

        if(type(locstart)==int or type(locstart)==float):
            # the user must mean start to refer to a channel
            # we reference the start channel to the *unselected* channels in the first selected spw
            if locstart>=len(chanfreqs0) or locstart<0:
                raise TypeError, "Start channel is outside the first spw"
            if type(locstart)==int:
                # in noninteractive mode, default start=0 instead of ""
                if mode=="velocity" and locstart==0:
                    fstart = chanfreqs1d[-1]
                else:
                    fstart = chanfreqs0[locstart]
            else:
                raise TypeError, "clean cannot use a fractional start channel at this time.  If you intended a start frequency or velocity please set start as a string with units"
        elif(type(locstart)==str):
            if(qa.quantity(locstart)['unit'].find('Hz') > -1):                
                fstart=qa.convert(qa.quantity(locstart),'Hz')['value']
            elif len(locstart)<=0:  # start not specified by user:
                if mode=="velocity":
                    fstart=chanfreqs1d[-1] # last selected channel
                else:
                    fstart=chanfreqs1d[0] # first selected channel
            else:
                raise TypeError, "Unrecognized start parameter"


        if(type(locwidth)==str):
            if(qa.quantity(locwidth)['unit'].find('Hz') > -1):
                finc=qa.convert(qa.quantity(locwidth),'Hz')['value']
            elif len(locwidth)<=0: # width not specified by user
                locwidth=1
                # Now the frame conversion is done in the end, hopefully this
                # warning is  no longer needed
                #if frame!="":
                #    self._casalog.post('Note: in frequency and velocity mode, the default width is the original channel width\n  and is not converted to the output reference frame.', 'WARN')
            else:
                raise TypeError, "Unrecognized width parameter"

        if(type(locwidth)==int or type(locwidth)==float):
            # the user must mean width to refer to channels
            # can be fractional or negative;  be careful about width=-1 which is valid, doesn't mean default.
            #if frame!="":
            #    self._casalog.post('Note: in frequency and velocity mode, the default width is the original channel width\n  and is not converted to the output reference frame.', 'WARN')

            if locwidth==0:  # this should not happen
                locwidth=1
                
            # if more than one channel, use the difference between 0 and 1 as the default width, 
            # except in vel mode we'll use the last (highest freq) width of the last spw selected 
            if mode=="velocity":
                if len(chanfreqs1d)>1:
                    defchanwidth = chanfreqs1d[-2]-chanfreqs1d[-1]  # negative inc
                else:
                    defchanwidth = -chanNfreqwidth
                # if the user puts in "2" for width, they mean 2 chan width, in vel mode that's 
                # a negative increment.  if they put in -2 they mean to go opposite the natural
                # direction for some reason, i.e. a positive freq increment.

            else:
                if len(chanfreqs1d)>1:
                    defchanwidth = chanfreqs1d[1]-chanfreqs1d[0]
                else:
                    defchanwidth = chan0freqwidth
                # in freq or chan mode, a width of "2" means 2 chans, positive increment in freq.
                    
            self._casalog.post('default chan width = %f' % defchanwidth, 'DEBUG')
            finc=defchanwidth*locwidth
            # XXX still need to convert to target reference frame!
            



        self._casalog.post('fstart = %f, finc = %f' % (fstart,finc), 'DEBUG')
        #print 'fstart = %f, finc = %f' % (fstart,finc)
        # we now have fstart and finc, the start and increment in Hz.  finc can be <0        

        if nchan<=0:            
            if finc>=0:  # (shouldn't be 0)
                bw = chanfreqs1d[-1]-fstart
            else:
                bw = fstart-chanfreqs1d[0]

            if(bw < 0): # I think this is already take care of above but just in case
                raise TypeError, "Start parameter is outside the data range"
            # now we implicitly (in the +1) use the default chan width at both 
            # beginning and end of the calculated bw.  if the last channel has a 
            # vastly different chan width than the first one, and if the first chan 
            # width was being used to calculate the output chan width, then part of the 
            # last chan may be cut off.
            nchan = int(round(bw/abs(finc)))+1   # XXX could argue for ceil here

        # sanity checks:
        # unsorted case
        if mode=="channel" and unsorted and start!='':
            fend = fstart - finc*(nchan-1)
        else:
           fend = fstart + finc*(nchan-1) 
        if fend >= (chanfreqs1d[-1]+abs(finc)):
            self._casalog.post("your values of spw, start, and width appear to exceed the spectral range available.  Blank channels may result","WARN")
        if fend <= 0:
            if finc<=0:
                nchan = int(floor(fstart/abs(finc)))
            else:
                raise TypeError, "values of spw, start, width, and nchan have resulted in negative frequency output channels"
        if fend <= (chanfreqs1d[0]-abs(finc)):
            self._casalog.post("your values of spw, start, and width appear to exceed the spectral range available.  Blank channels may result","WARN")
            
        if nchan<=0:
            raise TypeError, "values of spw, start, width, and nchan result in no output cube"


        # here are the output channels in Hz
        freqlist = numpy.array(range(nchan)) * finc + fstart
        retnchan=len(freqlist)
    
        #print bw,freqlist[0],freqlist[-1]
        
        if mode=="channel":
            # XXX depending on how this gets used, we probably needs checks here to convert
            # from quantity strings to channel indices.  Check for strange behaviour.
            if start=="":
                # here, we can make the default start be the first *selected* channel
                retstart = chanind0[1]
                # or we can make it be zero:
                # retstart=0
            else:
                retstart=start
            if width=="":
                retwidth=1
            else:
                retwidth=width

        elif(mode=='frequency'):
            # freqlist are in Hz - could use fstart here too.
            if instartunit=='':
                retstart = str(freqlist[0])+'Hz'
            else:
                retstart = self.qatostring(qa.convert(str(freqlist[0])+'Hz',instartunit))

            if inwidthunit=='':
                # convert frame when default width is used 
                if width=="" and frame!='':
                    finc=self.convertframe(finc,frame,field)
                retwidth = str(finc)+'Hz'
            else:
                retwidth = self.qatostring(qa.convert(str(finc)+'Hz',inwidthunit))

        elif(mode=='velocity'):
            # convert back to velocities (take max freq for min vel )
            # use USER-SPECIFIED frame again (or default LSRK) XXX 
            if(qa.quantity(start)['unit'].find('m/s') > -1):
                # retstart = self.convertvf(start,frame,field,restf)
                # since this is the same frame as we converted to freq, we can just give the 
                # original start parameter back:
                retstart = start
            else:
                # start means start. 
                # start="" case, do in data frame
                ##retstart = self.convertvf(str(fstart)+'Hz',frame,field,restf,veltype=veltype)
                retstart = self.convertvf(str(fstart)+'Hz',self.dataspecframe,field,restf,veltype=veltype)
                                
            if(qa.quantity(width)['unit'].find('m/s') > -1):
                # since this is the same frame as we converted to freq, we can just give the 
                # original start parameter back:
                retwidth=width
            else:
                if retnchan>1:
                # watch out for the sign of finc.
                    if finc>0:
                        if frame =="":
                            # here use data frame
                            # v1 = self.convertvf(str(freqlist[-2])+'Hz',frame,field,restf,veltype=veltype)
                            # v0 = self.convertvf(str(freqlist[-1])+'Hz',frame,field,restf,veltype=veltype)                
                            v1 = self.convertvf(str(freqlist[-2])+'Hz',self.dataspecframe,field,restf,veltype=veltype)
                            v0 = self.convertvf(str(freqlist[-1])+'Hz',self.dataspecframe,field,restf,veltype=veltype)                
                        else:
                            # do frame conversion while it is still in freq
                            f1 = self.convertframe(feqlist[-2], frame, field) 
                            f0 = self.convertframe(feqlist[-1], frame, field) 
                            v1 = self.convertvf(str(f1)+'Hz',self.datasepcframe,field,restf,veltype=veltype)
                            v0 = self.convertvf(str(f0)+'Hz',self.dataspecframe,field,restf,veltype=veltype)                
                        retwidth = str(qa.quantity(qa.sub(qa.quantity(v0),qa.quantity(v1)))['value'])+'m/s'
                    else:
                        if frame =="":
                            # here use data frame
                            # v1 = self.convertvf(str(freqlist[1])+'Hz',frame,field,restf,veltype=veltype)
                            # v0 = self.convertvf(str(freqlist[0])+'Hz',frame,field,restf,veltype=veltype)
                            v1 = self.convertvf(str(freqlist[1])+'Hz',self.dataspecframe,field,restf,veltype=veltype)
                            v0 = self.convertvf(str(freqlist[0])+'Hz',self.dataspecframe,field,restf,veltype=veltype)
                        else:
                            f1 = self.convertframe(freqlist[1],frame,field)
                            f0 = self.convertframe(freqlist[0],frame,field)
                            v1 = self.convertvf(str(f1)+'Hz',self.dataspecframe,field,restf,veltype=veltype)
                            v0 = self.convertvf(str(f0)+'Hz',self.dataspecframe,field,restf,veltype=veltype)
                        retwidth = str(qa.quantity(qa.sub(qa.quantity(v1),qa.quantity(v0)))['value'])+'m/s'
                else:
                    self._casalog.post("your parameters result in one channel - in vel mode the calculated width may not be accurate","WARN")
                    retwidth = self.convertvf(str(finc)+'Hz',self.dataspecframe,field,restf,veltype=veltype)
        else:
            raise TypeError, "Specified mode is not supported"

        # XXX do we need to set usespecframe=dataspecframe, overriding the user's, if start was ""?
        # the old code seemed to do that.
        # use data frame for default start (will be passed to defineimage in
        # data frame, do the conversion in imager)
        if start=="":
            self.usespecframe=self.dataspecframe

        return retnchan, retstart, retwidth


    def convertframe(self,fin,frame,field):
        """
        convert freq frame in dataframe to specfied frame, assume fin in Hz
        retruns converted freq in Hz (value only)
        """
        # assume set to phasecenter before initChanelization is called
        pc=self.srcdir
        if(type(pc)==str):
            if (pc==''):
                fieldused = field
                if (fieldused ==''):
                    fieldused ='0'
                dir = int(ms.msseltoindex(self.vis,field=fieldused)['field'][0])
            else:
                tmpdir = phasecenter
                try:
                    if(len(ms.msseltoindex(self.vis, field=pc)['field']) > 0):
                        tmpdir  = int(ms.msseltoindex(self.vis,field=pc)['field'][0])
                except Exception, instance:
                    tmpdir = pc
                dir = tmpdir
        if type(dir)==str:
            try:
                mrf, ra, dec = dir.split()
            except Exception, instance:
                raise TypeError, "Error in a string format  for phasecenter"
            mdir = me.direction(mrf, ra, dec)
        else:
            tb.open(self.vis+'/FIELD')
            srcdir=tb.getcell('DELAY_DIR',dir)
            mrf=tb.getcolkeywords('DELAY_DIR')['MEASINFO']['Ref']
            tb.close()
            mdir = me.direction(mrf,str(srcdir[0][0])+'rad',str(srcdir[1][0])+'rad')
            tb.open(self.vis+'/OBSERVATION')
        telname=tb.getcell('TELESCOPE_NAME',0)
        # use time in main table instead?
        tmr=tb.getcell('TIME_RANGE',0)
        tb.close()
        #print "direction=", me.direction(mrf,str(srcdir[0][0])+'rad',str(srcdir[1][0])+'rad')
        #print "tmr[1]=",tmr[1]
        #print "epoch=", me.epoch('utc',qa.convert(qa.quantity(str(tmr[1])+'s'),'d'))
        me.doframe(me.epoch('utc',qa.convert(qa.quantity(str(tmr[0])+'s'),'d')))
        me.doframe(me.observatory(telname))
        me.doframe(mdir)
        f0 = me.frequency(self.dataspecframe, str(fin)+'Hz')
        #print "frame=", frame, ' f0=',f0
        fout = me.measure(f0,frame)['m0']['value']
        return fout

    def setspecframe(self,spw):
        """
        set spectral frame for mfs to data frame based
        on spw selection 
        (part copied from setChannelization)
        """
        tb.open(self.vis+'/SPECTRAL_WINDOW')
        spwframe=tb.getcol('MEAS_FREQ_REF');
        tb.close()

        # first parse spw parameter:

        # use MSSelect if possible
        if spw in (-1, '-1', '*', '', ' '):
            spw="*"

        sel=ms.msseltoindex(self.vis, spw=spw)
        # spw returned by msseletoindex, spw='0:5~10;10~20' 
        # will give spw=[0] and len(spw) not equal to len(chanids)
        # so get spwids from chaninds instead.
        chaninds=sel['channel'].tolist()
        spwinds=[]
        for k in range(len(chaninds)):
            spwinds.append(chaninds[k][0])
        if(len(spwinds) == 0):
            raise Exception, 'unable to parse spw parameter '+spw;
            
        # the first selected spw 
        spw0=spwinds[0]

        # set dataspecframe:
        elspecframe=["REST",
                     "LSRK",
                     "LSRD",
                     "BARY",
                     "GEO",	    
                     "TOPO",
                     "GALACTO",
                     "LGROUP",
                     "CMB"]
        self.dataspecframe=elspecframe[spwframe[spw0]];
        return 

    def initChaniter(self,nchan,spw,start,width,imagename,mode,tmpdir='_tmpimdir/'):
        """
        initialize for channel iteration in interactive clean
        --- create a temporary directory, get frequencies for
        mode='channel'
        """
        # create a temporary directory to put channel images
        tmppath=[]
        freqs=[]
        finc=0
        newmode=mode
        for imname in imagename:
            if os.path.dirname(imname)=='':
                tmppath.append(tmpdir)
            else:
                tmppath.append(os.path.dirname(imname)+'/'+tmpdir)
            # clean up old directory
            if os.path.isdir(tmppath[-1]):
                shutil.rmtree(tmppath[-1])
            os.mkdir(tmppath[-1])
        #internally converted to frequency mode for mode='channel'
        #to ensure correct frequency axis for output image
        #if mode == 'channel':
        #    freqs, finc = self.getfreqs(nchan, spw, start, width)
        #    newmode = 'frequency'
        if mode == 'channel':
            # get spectral axis info from the dirty image
            ia.open(imagename[0]+'.image')
            imcsys=ia.coordsys().torecord()
            ia.close()
            cdelt=imcsys['spectral2']['wcs']['cdelt']
            crval=imcsys['spectral2']['wcs']['crval']
            for i in range(nchan):
                if i==0: freqs.append(crval)
                freqs.append(freqs[-1]+cdelt)
            finc = cdelt
            newmode = 'frequency'
        return freqs,finc,newmode,tmppath


    def makeTemplateCubes(self, imagename,outlierfile, field, spw, selectdata, timerange,
          uvrange, antenna, scan, mode, facets, cfcache, interpolation, 
          imagermode, localFTMachine, mosweight, locnchan, locstart, locwidth, outframe,
          veltype, imsize, cell, phasecenter, restfreq, stokes, weighting,
          robust, uvtaper, outertaper, innertaper, modelimage, restoringbeam,
          calready, noise, npixels, padding):
        """
        make template cubes to be used for chaniter=T interactive clean
        """
        imageids=[]
        imsizes=[]
        phasecenters=[]
        rootname=''
        multifield=False

        if len(outlierfile) != 0:
            imsizes,phasecenters,imageids=self.readoutlier(outlierfile)
            if type(imagename) == list:
                rootname = imagename[0]
            else:
                rootname = imagename
            if len(imageids) > 1:
                multifield=True
        else:
            imsizes=imsize
            phasecenters=phasecenter
            #imageids=imagename+'_template'
            imageids=imagename

        # readoutlier need to be run first....
        #pdb.set_trace() 
        self.definemultiimages(rootname=rootname,imsizes=imsizes,cell=cell,
                                stokes=stokes,mode=mode,
                               spw=spw, nchan=locnchan, start=locstart,
                               width=locwidth, restfreq=restfreq,
                               field=field, phasecenters=phasecenters,
                               names=imageids, facets=facets,
                               outframe=outframe, veltype=veltype,
                               makepbim=False, checkpsf=False)

        self.datselweightfilter(field=field, spw=spw,
                                 timerange=timerange, uvrange=uvrange,
                                 antenna=antenna, scan=scan,
                                 wgttype=weighting, robust=robust,
                                 noise=noise, npixels=npixels,
                                 mosweight=mosweight,
                                 innertaper=innertaper,
                                 outertaper=outertaper,
                                 calready=calready, nchan=-1,
                                 start=0, width=1)
       
        #localAlgorithm = getAlgorithm(psfmode, imagermode, gridmode, mode,
        #                             multiscale, multifield, facets, nterms,
        #                             'clark');

        #localAlgorithm = 'clark'
        #print "localAlogrithm=",localAlgorithm

        #self.im.setoptions(ftmachine=localFTMachine,
        #                     wprojplanes=wprojplanes,
        #                     freqinterp=interpolation, padding=padding,
        #                     cfcachedirname=cfcache, pastep=painc,
        #                     epjtablename=epjtable,
        #                     applypointingoffsets=False,
        #                     dopbgriddingcorrections=True)
        self.im.setoptions(ftmachine=localFTMachine,
                             freqinterp=interpolation, padding=padding,
                             cfcachedirname=cfcache)

        modelimages=[]
        restoredimage=[]
        residualimage=[]
        psfimage=[]
        fluximage=[]
        for k in range(len(self.imagelist)):
            ia.open(self.imagelist[k])
            if (modelimage =='' or modelimage==[]) and multifield:
                ia.rename(self.imagelist[k]+'.model',overwrite=True)
            else:
                ia.remove(verbose=False)
            ia.close()
            modelimages.append(self.imagelist[k]+'.model')
            restoredimage.append(self.imagelist[k]+'.image')
            residualimage.append(self.imagelist[k]+'.residual')
            psfimage.append(self.imagelist[k]+'.psf')
            if(imagermode=='mosaic'):
                fluximage.append(self.imagelist[k]+'.flux')

        self.im.clean(algorithm='clark', niter=0,
                   model=modelimages, residual=residualimage,
                   image=restoredimage, psfimage=psfimage,
                   mask='', interactive=False)


    def setChaniterParms(self,finalimagename, spw,chan,start,width,freqs,finc,tmppath):
        """
        set parameters for channel by channel iterations
        """
        retparms={}
        self.maskimages={}
        retparms['imagename']=[tmppath[indx]+os.path.basename(imn)+'.ch'+str(chan)
                   for indx, imn in enumerate(finalimagename)]

        #print "Processing channel %s " % chan
        #self._casalog.post("Processing channel %s "% chan)

        # Select only subset of vis data if possible.
        # It does not work well for multi-spw so need
        # to select with nchan=-1
        retparms['imnchan']=1
        retparms['chanslice']=chan
        qat=qatool.create();
        q = qat.quantity

        # 2010-08-18 note: disable this. Has the problem 
        # getting imaging weights correctly when the beginning 
        # channels were flagged.
        #if type(spw)==int or len(spw)==1:
        #    if width>1:
        #        visnchan=width
        #    else:
        #        visnchan=1
        #else:
        #    visnchan=-1

        visnchan=-1
        retparms['visnchan']=visnchan
        visstart=0

        if type(start)==int:
            # need to convert to frequencies
            # to ensure correct frequencies in
            # output images(especially for multi-spw)
            # Use freq list instead generated in initChaniter
            imstart=q(freqs[chan],'Hz')
            width=q(finc,'Hz')
        elif start.find('m/s')>0:
            imstart=qat.add(q(start),qat.mul(chan,q(width)))
        elif start.find('Hz')>0:
            imstart=qat.add(q(start),qat.mul(chan,q(width)))
        retparms['width']=width
        retparms['imstart']=imstart
        retparms['visstart']=visstart

        #
        return retparms

    def defineChaniterModelimages(self,modeimage,chan,tmppath):
        chanmodimg=[]
        if type(modelimage)==str:
            modelimage=[modelimage]
        for modimg in modelimage:
            if type(modimg)==list:
                chanmodimg=[]
                for img in modimg:
                    if os.path.dirname(img) != '':
                        chanmodimg.append(tmppath[0] + '_tmp.' +
                                           os.path.basename(img))
                    else:
                        chanmodimg.append(tmppath[0] + '_tmp.' + img)
                    self.getchanimage(cubeimage=img, outim=chanmodimg[-1], chan=chan)
                self.convertmodelimage(modelimages=chanmodimg,
                                        outputmodel=self.imagelist.values()[0]+'.model')
                chanmodimg=[]
            else:
                if os.path.dirname(modimg) != '':
                    chanmodimg.append(tmppath[0] + '_tmp.' + os.path.basename(modimg))
                else:
                    chanmodimg.append(tmppath[0] + '_tmp.' + modimg)
                self.getchanimage(cubeimage=modimg, outim=chanmodimg[-1],chan=chan)

                self.convertmodelimage(modelimages=chanmodimg,
                                        outputmodel=self.imagelist.values()[0]+'.model')
            # clean up temporary channel model image
            self.cleanupTempFiles(chanmodimg)

    def storeCubeImages(self,cubeimageroot,chanimageroot,chan,imagermode):
        """
        put channel images back into CubeImages
        """
        imagext = ['.image','.model','.flux','.residual','.psf','.mask']
        if imagermode=='mosaic':
            imagext.append('.flux.pbcoverage')
        lerange=range(self.nimages)
        for n in lerange:
            cubeimagerootname=cubeimageroot[n]
            chanimagerootname=chanimageroot[n]
        for ext in imagext:
            nomaskim=False
            cubeimage=cubeimagerootname+ext
            chanimage=chanimagerootname+ext
            if not os.path.exists(cubeimage):
                if os.path.exists(chanimage):
                    outim=ia.newimagefromimage(cubeimagerootname+'.model',cubeimage)
                elif ext=='.mask':
                    # unless mask image is given or in interactive mode
                    # there is no mask image
                    nomaskim=True
            if not nomaskim: 
                self.putchanimage(cubeimage, chanimage,chan)

    def cleanupTempFiles(self, tmppath):
        """
        Remove the directories listed by tmppath.
        """
        # Created to deal with temporary dirs created by chaniter=T clean,
        # now used elsewhere too.
        for dir in tmppath:
            if os.path.exists(dir):
               shutil.rmtree(dir)


def getFTMachine(gridmode, imagermode, mode, wprojplanes, userftm):
    """
    A utility function which implements the logic to determine the
    ftmachine name to be used in the under-laying tool.
    """
#    ftm = userftm;
    ftm='ft';
    if ((gridmode == 'widefield') and(wprojplanes > 1)): ftm = 'wproject';
    elif (gridmode == 'aprojection'):                    ftm = 'pbwproject';
    elif (imagermode == 'csclean'):                      ftm = 'ft';
    elif (imagermode == 'mosaic'):                       ftm = userftm;
    return ftm;

def getAlgorithm(psfmode, imagermode, gridmode, mode, 
                 multiscale, multifield, facets, nterms, useralg):
    """
    A utility function which implements the logic to determine the
    deconvolution algorithm to be used in the under-laying tool.
    """
    alg=useralg
    addMultiField=False;

    if((type(multiscale)==list) and 
       (len(multiscale) > 0) and
       (sum(multiscale) > 0)): alg = 'multiscale';
    elif ((psfmode == 'clark') or (psfmode == 'hogbom')): alg=psfmode;

    if ((imagermode == '') and (multifield)): addMultiField=True;
    if (imagermode == 'mosaic'):              addMultiField=True;
    if (imagermode == 'csclean'):             addMultiField = True; #!!!!

    if ((mode == 'mfs') and (nterms > 1)): 
        alg = 'msmfs';
        if(imagermode == 'mosaic'): 
               raise Exception, 'msmfs (nterms>1) not allowed with imagermode=' + imagermode + '. For now, msmfs automatically performs cs-clean type iterations';
        if (multifield): 
		addMultiField = True;
		raise Exception, 'For now, msmfs (nterms>1) is not allowed in multi-field mode. Please supply a single image name.'

#    if (gridmode == 'widefield'): alg='mfclark';

    if (gridmode == 'widefield'):
        addMultiField=True;
        if (facets > 1):
            if(alg.count('multiscale') > 0):
                raise Exception, 'multiscale with facets > 1 not allowed for now';
            if (psfmode==''): psfmode='clark';
            if((psfmode == 'clark') or (psfmode == 'hogbom')):
                alg='wf'+psfmode;
                addMultiField=False;
            else:
                addMultiField=True;
#            addMultiField=False;

#
# if facets > 1 && mutliscale ==> fail


    if (addMultiField and (alg[0:2] != 'mf') and (alg != 'msmfs')):  alg = 'mf' + alg;
    return alg;

# Function to compute Calculate alpha
def msmfsCalcAlphaBeta(imtemplate="",taylorlist=[],namealpha="",namebeta="",threshold=0.001):
    nterms = len(taylorlist);
    if(nterms>1):
     if(not os.path.exists(namealpha)):
       cpcmd = 'cp -r ' + imtemplate + ' ' + namealpha;
       os.system(cpcmd);
    if(nterms>2):
     if(not os.path.exists(namebeta)):
       cpcmd = 'cp -r ' + imtemplate + ' ' + namebeta;
       os.system(cpcmd);
    if(nterms>0):
     ia.open(taylorlist[0]);
     ptay0 = ia.getchunk();
     ia.close();
    if(nterms>1):
     ia.open(taylorlist[1]);
     ptay1 = ia.getchunk();
     ia.close();
     ia.open(namealpha);
     alpha = ia.getchunk();
     alpha.fill(0.0);
     ia.close();
    if(nterms>2):
     ia.open(taylorlist[2]);
     ptay2 = ia.getchunk();
     ia.close();
     ia.open(namebeta);
     beta = ia.getchunk();
     beta.fill(0.0);
     ia.close();
   # Calc alpha,beta from ptay0,ptay1,ptay2
    N = ptay0.shape[0];
    if(nterms>1):
     for ii in range(0,N):
       for jj in range(0,N):
         if(ptay0[ii,jj,0,0]>threshold):
	    mtay0 = ptay0[ii,jj,0,0];
	    mtay1 = ptay1[ii,jj,0,0];
	    alphaval = mtay1/mtay0;
	    alpha[ii,jj,0,0] = alphaval;
	    if(nterms>2):
	       mtay2 = ptay2[ii,jj,0,0];
	       beta[ii,jj,0,0] = (mtay2/mtay0) - 0.5*alphaval*(alphaval-1);
    if(nterms>1):
     ia.open(namealpha);
     ia.putchunk(alpha);
     ia.setbrightnessunit(unit=' ');
     ia.close();
    if(nterms>2):
     ia.open(namebeta);
     ia.putchunk(beta);
     ia.setbrightnessunit(unit=' ');
     ia.close();
