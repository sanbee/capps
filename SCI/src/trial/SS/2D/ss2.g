include 'pixpsf.g'
#include 'simpsf.g'

sscl:=0;
ss:=function(suspend=F) 
{
    global sscl;
    sscl:=client('ssdeconv2',suspend=suspend); return sscl;
}
function sendimg(val FileName,blc=[0,0],trc=[0,0])
{
    local Shape,Arr;

    im:=image(FileName);
#    Arr:=im.getchunk(blc=[896,896],trc=[1151,1151]);
    Arr:=im.getchunk(blc=[0,0],trc=[0,0]);
    Shape:=Arr::shape;
    NX:=Shape[1]; NY:=Shape[2];
    print "###Info: Sending a ",NX,"x",NY," image";

    return Arr;
}
setncomp :=function(n=1,cl=sscl)               {cl->ncomp(n);}
setniter :=function(n=200,cl=sscl)             {cl->niter(n);}
setimsize:=function(n=256,cl=sscl)             {cl->imsize(n);}
rdpsf    :=function(file='psf.comp1000',cl=sscl)         {cl->psff(file);}
truepsf  :=function(file='truepsf.im',cl=sscl)           {cl->truepsf(sendimg(file));}
rddimg   :=function(file='testDirtyImage.im',cl=sscl)    {cl->mod(sendimg(file));}
done     :=function(cl=sscl)                   {if (is_record(sscl) && sscl.active==1) cl->done();}
go       :=function(n=0,cl=sscl)               {cl->deconv(n);}

setup:=function(ncomp,niter,imsize,PSFFile,ModelFile)
{
    setncomp(ncomp);
    setniter(niter);
    setimsize(imsize);
    rdpsf(PSFFile);
    rddimg(ModelFile);
}
