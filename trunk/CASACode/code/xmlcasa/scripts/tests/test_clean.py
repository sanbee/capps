import os
import sys
import shutil
import commands
import numpy
from __main__ import default
from tasks import *
from taskinit import *
import unittest

'''
Unit tests for task clean. It tests the following parameters:
    vis:           wrong and correct values
    imagename:     if output exists
    field:         wrong field type; non-default value
    spw:           wrong value; non-default value
    mode:          empty value; non-default values
    gridmode:      unsupported value; non-default values
    niter:         wrong type; non-default values
    psfmode:       unsupported value; non-default values
    imagermode:    unsupported value; non-default values
    imsize:        zero value; non-default value
    stokes:        unsupported value; non-default value
    weighting:     unsupported value; non-default values
    selectdata:    True; subparameters:
                     timerange:    non-default value
                     antenna:      unsupported value; non-default value
    
    Other tests: check the value of a pixel.
'''
class clean_test1(unittest.TestCase):

    # Input and output names
#    msfile = 'ngc7538_ut.ms'
    msfile = 'ngc7538_ut1.ms'
    res = None
    img = 'cleantest1'
    img2 = '0-cleantest1'
    msk = 'cleantest1.in.mask'

    def setUp(self):
        self.res = None
        default(clean)
        if (os.path.exists(self.msfile)):
            os.system('rm -rf ' + self.msfile)
            
        datapath = os.environ.get('CASAPATH').split()[0] + '/data/regression/unittest/clean/'
        shutil.copytree(datapath+self.msfile, self.msfile)
    
    def tearDown(self):
        if (os.path.exists(self.msfile)):
            os.system('rm -rf ' + self.msfile)

        os.system('rm -rf ' + self.img+'*')
     
    def getpixval(self,img,pixel):
        ia.open(img)
        px = ia.pixelvalue(pixel)
        ia.close()
        return px['value']['value']
        
    def test1(self):
        '''Clean 1: Default values'''
        self.res = clean()
        self.assertFalse(self.res)
        
    def test2(self):
        """Clean 2: Wrong input should return False"""
        msfile = 'badfilename'
        self.res = clean(vis=msfile, imagename=self.img)
        self.assertFalse(self.res)
        
    def test3(self):
        """Clean 3: Good input should return None"""
        self.res = clean(vis=self.msfile,imagename=self.img)
        self.assertEqual(self.res,None)
        
    def test4(self):
        """Clean 4: Check if output exists"""
        self.res = clean(vis=self.msfile,imagename=self.img)
        self.assertTrue(os.path.exists(self.img+'.image'))
        
    def test5(self):
        """Clean 5: Wrong field type"""
        self.res = clean(vis=self.msfile,imagename=self.img,field=0)
        self.assertFalse(self.res)
        
    def test6(self):
        """Clean 6: Non-default field value"""
        self.res = clean(vis=self.msfile,imagename=self.img,field='0~1')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))           
        
    def test7(self):
        """Clean 7: Wrong spw value"""
        self.res = clean(vis=self.msfile,imagename=self.img,spw='10')
        self.assertFalse(os.path.exists(self.img+'.image'))
       
    def test8(self):
        """Clean 8: Non-default spw value"""
        self.res = clean(vis=self.msfile,imagename=self.img,spw='0')
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test9(self):
        """Clean 9: Empty mode value"""
        self.res = clean(vis=self.msfile,imagename=self.img,mode='')
        self.assertFalse(self.res)

    # FIXME: CHANGE SUBPARAMETERS FOR DIFFERENT MODES
    def test10(self):
        """Clean 10: Non-default mode channel"""
        self.res = clean(vis=self.msfile,imagename=self.img,mode='channel')
        self.assertEqual(self.res,None)
        self.assertTrue(os.path.exists(self.img+'.image'),'Image %s does not exist'%self.img)

    def test11(self):
        """Clean 11: Non-default mode velocity"""
        retValue = {'success': True, 'msgs': "", 'error_msgs': '' }            
        res = clean(vis=self.msfile,imagename=self.img,mode='velocity',restfreq='23600MHz')
        if(res != None):
            retValue['success']=False
            retValue['error_msgs']=retValue['error_msgs']\
                     +"\nError: Failed to run in velocity mode."
        if(not os.path.exists(self.img+'.image')):
            retValue['success']=False
            retValue['error_msgs']=retValue['error_msgs']\
                     +"\nError: Failed to create output image."
                         
        # Verify if there are blank planes at the edges
        vals = imval(self.img+'.image')
        size = len(vals['data'])
        if (vals['data'][0]==0 or vals['data'][size-1]==0):
            retValue['success']=False
            retValue['error_msgs']=retValue['error_msgs']\
                     +"\nError: There are blank planes in the edges of the image."
                    

        self.assertTrue(retValue['success'],retValue['error_msgs'])
        
    def test12(self):
        """Clean 12: Non-default mode frequency"""
        self.res = clean(vis=self.msfile,imagename=self.img,mode='frequency')
        self.assertEqual(self.res,None)
        self.assertTrue(os.path.exists(self.img+'.image'),'Image %s does not exist'%self.img)
        
    def test13(self):
        """Clean 13: Unsupported gridmode"""
        self.res = clean(vis=self.msfile,imagename=self.img,gridmode='grid')
        self.assertFalse(self.res)
        
    def test14(self):
        """Clean 14: Non-default gridmode widefield"""
        self.res = clean(vis=self.msfile,imagename=self.img,gridmode='widefield',imsize=[20])
        self.assertEqual(self.res, None) 
        self.assertTrue(os.path.exists(self.img+'.image'),'Image %s does not exist'%self.img)

    # Takes too long!!!
#    def test11(self):
#        """Clean 11: Non-default gridmode aprojection"""
#        self.res = clean(vis=self.msfile,imagename=self.img,gridmode='aprojection',imsize=[10],niter=1)
#        self.assertEqual(self.res, None) 
#        self.assertTrue(os.path.exists(self.img+'.image'),'Image %s does not exist'%self.img)
                     
    def test15(self):
        """Clean 15: Wrong niter type"""
        self.res = clean(vis=self.msfile,imagename=self.img,niter='1')
        self.assertFalse(self.res)
        
    def test16(self):
        """Clean 16: Non-default niter values"""
        for n in range(10,400,50):
            self.res = clean(vis=self.msfile,imagename=self.img,niter=n)
            self.assertEqual(self.res,None,'Failed for niter = %s' %n)
    
    def test17(self):
        """Clean 17: Unsupported psfmode"""
        self.res = clean(vis=self.msfile,imagename=self.img,psfmode='psf')
        self.assertFalse(self.res)
        
    def test18(self):
        """Clean 18: Non-default psfmode hogbom"""
        self.res = clean(vis=self.msfile,imagename=self.img,psfmode='hogbom')
        self.assertEqual(self.res, None)            
        self.assertTrue(os.path.exists(self.img+'.image'))
        
    def test19(self):
        """Clean 19: Non-default psfmode clarkstokes"""
        self.res = clean(vis=self.msfile,imagename=self.img,psfmode='clarkstokes')
        self.assertEqual(self.res, None)            
        self.assertTrue(os.path.exists(self.img+'.image'))
      
    def test20(self):
        """Clean 20: Unsupported imagermode"""
        self.res = clean(vis=self.msfile,imagename=self.img,imagermode='clark')
        self.assertFalse(self.res)      

    def test21(self):
        '''Clean 21: Non-default imagermode csclean'''
        self.res = clean(vis=self.msfile,imagename=self.img,imagermode='csclean')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test22(self):
        '''Clean 22: Non-default imagermode mosaic'''
        self.res = clean(vis=self.msfile,imagename=self.img,imagermode='mosaic',imsize=[20])
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test23(self):
        """Clean 23: Zero value of imsize"""
        self.res = clean(vis=self.msfile,imagename=self.img,imsize=0)
        self.assertFalse(os.path.exists(self.img+'.image'))

    def test24(self):
        '''Clean 24: Non-default imsize values'''
        self.res = clean(vis=self.msfile,imagename=self.img,imsize=[80,80])
        self.assertEqual(self.res,None)
        self.assertTrue(os.path.exists(self.img+'.image'),'Image %s does not exist' %self.img)

    def test25(self):
        """Clean 25: Non-default cell values"""
        self.res = clean(vis=self.msfile,imagename=self.img, cell=12.5)
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))
        
    def test26(self):
        """Clean 26: Unsupported Stokes parameter"""
        self.res = clean(vis=self.msfile,imagename=self.img, stokes='V')
        self.assertFalse(self.res)
        
    def test27(self):
        """Clean 27: Non-default Stokes parameter"""
        self.res = clean(vis=self.msfile,imagename=self.img, stokes='RR')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))
        
    def test28(self):
        '''Clean 28: Unsupported weighting mode'''
        self.res = clean(vis=self.msfile,imagename=self.img, weighting='median')
        self.assertFalse(self.res)
        
    def test29(self):
        '''Clean 29: Non-default weighting uniform'''
        self.res = clean(vis=self.msfile,imagename=self.img, weighting='uniform')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test30(self):
        '''Clean 30: Non-default weighting briggs'''
        self.res = clean(vis=self.msfile,imagename=self.img, weighting='briggs')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test31(self):
        '''Clean 31: Non-default weighting radial'''
        self.res = clean(vis=self.msfile,imagename=self.img, weighting='radial')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))
        
    def test32(self):
        '''Clean 32: Non-default subparameters of selectdata'''
#        self.res = clean(vis=self.msfile,imagename=self.img,selectdata=True,
#                         timerange='>11:30:00',antenna='VA12')
        self.res = clean(vis=self.msfile,imagename=self.img,selectdata=True,
                         timerange='>11:30:00',antenna='VA01')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test33(self):
        '''Clean 33: Wrong antenna subparameter of selectdata'''
        self.res = clean(vis=self.msfile,imagename=self.img,selectdata=True,
                         antenna='88')
        self.assertFalse(os.path.exists(self.img+'.image'))

    def test34(self):
        '''Clean 34: Verify the value of pixel 50'''
        #run clean with some parameters
        self.res = clean(vis=self.msfile,imagename=self.img,selectdata=True,
                         timerange='>11:28:00',field='0~2',imsize=[100,100],niter=10)
        
        os.system('cp -r ' + self.img + '.image' + ' myimage.im')
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img + '.image'))
#        ref = 0.007161217276006937
#        ref = 0.011824539862573147
        ref = 0.009637                                 # active rev. 12908
        value = self.getpixval(self.img+'.image', 50)
        diff = abs(ref - value)
        self.assertTrue(diff < 10e-4,
                        'Something changed the pixel brightness. ref_val=%s, new_val=%s'
                                        %(ref,value))
        
    def test35(self):
        '''Clean 35: Wrong type of phasecenter'''
        self.res=clean(vis=self.msfile,imagename=self.img,phasecenter=4.5)
        self.assertFalse(os.path.exists(self.img+'.image'))
        
    def test36(self):
        '''Clean 36: Non-default value of phasecenter'''
        self.res=clean(vis=self.msfile,imagename=self.img,phasecenter=2)
        self.assertTrue(os.path.exists(self.img+'.image'))

    def test37(self):
        '''Clean 37: Test box mask'''
        self.res=clean(vis=self.msfile,imagename=self.img,mask=[115,115,145,145])
        self.assertTrue(os.path.exists(self.img+'.image') and
			os.path.exists(self.img+'.mask'))

    def test38(self):
        '''Clean 38: Test freeing of resource for mask (checks CAS-954)'''
        self.res=clean(vis=self.msfile,imagename=self.img,mask=[115,115,145,145])
        cmd='/usr/sbin/lsof|grep %s' % self.img+'.mask'
        output=commands.getoutput(cmd)
        ret=output.find(self.img+'.mask')
        self.assertTrue(ret==-1)

    def test39(self):
        '''Clean 39: Input mask image specified'''
        datapath = os.environ.get('CASAPATH').split()[0] + '/data/regression/unittest/clean/'
        shutil.copytree(datapath+self.msk, self.msk)
        self.res=clean(vis=self.msfile,imagename=self.img2,mask=self.msk)
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img2+'.image'))
        # cleanup
        os.system('rm -rf ' + self.msk + ' '+ self.img2+'*')

    def test40(self):
        '''Clean 40: Test chaniter=T clean with flagged channels'''
        # test CAS-2369 bug fix 
        flagdata(vis=self.msfile,mode='manualflag',spw='0:0~0')
        self.res=clean(vis=self.msfile,imagename=self.img,mode='channel',spw=0)
        self.assertEqual(self.res, None)
        self.assertTrue(os.path.exists(self.img+'.image'))
         
    def test41(self):
        '''Clean 41: Test nterms=2 and ref-freq > maximum-frequency'''
        # This tests if negative-weights are being correctly allowed through the gridders
        self.res=clean(vis=self.msfile,imagename=self.img,nterms=2,reffreq='25.0GHz',niter=5);
        self.assertEqual(self.res,None);

    def test42(self):
        '''Clean42: Test nterms=2, with only one channel whose frequency is same as ref-freq'''
        # This tests if a numerical failure-mode is detected and returned without fuss.
        self.res=clean(vis=self.msfile,imagename=self.img,nterms=2,reffreq='23691.4682MHz',spw='0:0');
        self.assertFalse(self.res);


class clean_test2(unittest.TestCase):
    
    # Input and output names
    msfile = 'split1scan.ms'
    res = None
    img = 'cleantest2'

    def setUp(self):
        self.res = None
        default(clean)
        if (os.path.exists(self.msfile)):
            os.system('rm -rf ' + self.msfile)
            
        datapath = os.environ.get('CASAPATH').split()[0] + '/data/regression/unittest/clean/'
        shutil.copytree(datapath+self.msfile, self.msfile)
    
    def tearDown(self):
        if (os.path.exists(self.msfile)):
            os.system('rm -rf ' + self.msfile)
        
    def test1a(self):
        """Clean 1a: Non-default mode velocity"""
        retValue = {'success': True, 'msgs': "", 'error_msgs': '' }            
        res = clean(vis=self.msfile,imagename=self.img,mode='velocity',restfreq='231901MHz')
        if(res != None):
            retValue['success']=False
            retValue['error_msgs']=retValue['error_msgs']\
                     +"\nError: Failed to run in velocity mode."
        if(not os.path.exists(self.img+'.image')):
            retValue['success']=False
            retValue['error_msgs']=retValue['error_msgs']\
                     +"\nError: Failed to create output image."
                         
        # Verify if there are blank planes at the edges
        vals = imval(self.img+'.image')
        size = len(vals['data'])
        if (vals['data'][0]==0.0 or vals['data'][size-1]==0.0):
            retValue['success']=False
            retValue['error_msgs']=retValue['error_msgs']\
                     +"\nError: There are blank planes in the edges of the image."
                    

        self.assertTrue(retValue['success'],retValue['error_msgs'])

class clean_test3(unittest.TestCase):

    # Input and output names#    msfile = 'ngc7538_ut.ms'
    #msfile = '0556kf.ms'
    msfile = 'outlier_ut.ms'
    res = None
    img = ['cleantest3a','cleantest3b']

    def setUp(self):
        self.res = None
        default(clean)
        if (os.path.exists(self.msfile)):
            os.system('rm -rf ' + self.msfile)

        datapath = os.environ.get('CASAPATH').split()[0] + '/data/regression/unittest/clean/'
        shutil.copytree(datapath+self.msfile, self.msfile)

    def tearDown(self):
        if (os.path.exists(self.msfile)):
            os.system('rm -rf ' + self.msfile)

        #os.system('rm -rf ' + self.img[0]+'* ')
        #os.system('rm -rf ' + self.img[1]+'* ')
	for imext in ['.image','.model','.residual','.psf']:
	    shutil.rmtree(self.img[0]+imext)
	    shutil.rmtree(self.img[1]+imext)

    #def testCAS1972(self):
    #    """Clean test3:test bug fixes: CAS-1972"""
    #    self.res= clean(vis=self.msfile,imagename=self.img,mode="mfs",interpolation="linear",
    #                    niter=100, psfmode="clark", 
    #                    imsize=[[512, 512], [512, 512]],
    #                    cell="0.0001arcsec", phasecenter=['J2000 05h59m32.03313 23d53m53.9267', 
    #                                                      'J2000 05h59m32.03313 23d53m53.9263'],
    #                    weighting="natural",pbcor=False,minpb=0.1)
    #
    #    self.assertEqual(self.res,None)
    #    for im in self.img:
    #        self.assertTrue(os.path.exists(im+'.image'))
    #    stat0 = imstat(self.img[0]+'.model')
    #    stat1 = imstat(self.img[1]+'.model')
    #    self.assertEqual(stat0['max'],stat1['max'])
    #    self.assertTrue(all(stat0['maxpos']==numpy.array([256,256,0,0])) and
    #           all(stat1['maxpos']==numpy.array([256,260,0,0])))

    def testCAS1972(self):
        """Clean test3:test CAS-1972 bug fixes ver2 (with smaller dataset)"""
        self.res= clean(vis=self.msfile,
                        imagename=self.img,
                        mode="mfs",
                        interpolation="linear",
                        niter=100, psfmode="clark",
                        mask=[[250, 250, 262, 262], [250, 350, 262, 362]],
                        imsize=[[512, 512], [512, 512]],
                        cell="0.0001arcsec", 
                        phasecenter=['J2000 05h59m32.03313 23d53m53.9267', 'J2000 05h59m32.03313 23d53m53.9167'],
                        weighting="natural",
                        pbcor=False,
                        minpb=0.1)

        self.assertEqual(self.res,None)
        # quick check on the peaks apear at the location as expected
        for im in self.img:
            self.assertTrue(os.path.exists(im+'.image'))
        stat0 = imstat(self.img[0]+'.model')
        stat1 = imstat(self.img[1]+'.model')
        #self.assertEqual(stat0['max'],stat1['max'])
        self.assertTrue(all(stat0['maxpos']==numpy.array([256,256,0,0])) and
               all(stat1['maxpos']==numpy.array([256,356,0,0])))


def suite():
    return [clean_test1]


