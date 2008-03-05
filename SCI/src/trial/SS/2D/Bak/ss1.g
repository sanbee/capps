include 'image.g'


sscl:=0;
ss:=function(suspend=F) 
{
    global sscl;
    sscl:=client('ssdeconv1',suspend=suspend); return sscl;
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
gsetncomp :=function(n=1,cl=sscl)               {cl->ncomp(n);}
gsetniter :=function(n=200,cl=sscl)             {cl->niter(n);}
gsetimsize:=function(n=256,cl=sscl)             {cl->imsize(n);}
grdpsf    :=function(file='psf.comp1000',cl=sscl)         {cl->psff(file);}
gtruepsf  :=function(file='truepsf.im',cl=sscl)           {cl->truepsf(sendimg(file));}
grddimg   :=function(file='testDirtyImage.im',cl=sscl)    {cl->mod(sendimg(file));}
gdone     :=function(cl=sscl)                   {if (is_record(sscl) && sscl.active==1) cl->done();}
go       :=function(n=0,cl=sscl)               {cl->deconv(n);}

gsetup:=function(ncomp=2,niter=1000,sigma=10.0,gain=1.0,scale=0.1,imsize=512,
		FPSFFile="psf.comp1000",DirtyFile="sim/sim.dirty",TruePSF="sim/sim.psf")
{
    sscl->init();
    gsetimsize(imsize);
    gsetncomp(ncomp);
    gsetniter(niter);
    grdpsf(FPSFFile);
    grddim(ModelFile);
    gtruepsf(TruePSF);
    sscl->gain(gain);
    sscl->sigma(sigma);
    sscl->scale(scale);
}
