include 'images.g'
psfcl:=0;
pss:=function(suspend=F) 
{
    global psfcl;
    psfcl:=client('lmdr_psfdecomp',suspend=suspend); return psfcl;
}
psetncomp :=function(n=1,cl=psfcl)               {cl->ncomp(n);}
psetniter :=function(n=200,cl=psfcl)             {cl->niter(n);}
psetimsize:=function(n=2048,cl=psfcl)            {cl->imsize(n);}

function g2(val x, val y, val a, val x0, val y0, val s)
{
    local arg;

    arg := ((x-x0)*(x-x0) + (y-y0)*(y-y0))*s*s/2;
    return exp(-arg);
}

function psendpsf(val FileName,blc=[896,896],trc=[1151,1151],cl=psfcl)
{
    local Shape,Arr;

    im:=image(FileName);
#    Arr:=im.getchunk(blc=[896,896],trc=[1151,1151]);
    Arr:=im.getchunk(blc,trc);
    Shape:=Arr::shape;
    NX:=Shape[1]; NY:=Shape[2];
    print "###Info: Sending a ",NX,"x",NY," image as PSF";

    cl->rpsf(Arr);
}
prdpsf    :=function(file='fpsf2.dat',cl=psfcl)  {cl->rpsf(file);}
prdmodel  :=function(file='model.dat',cl=psfcl)  {cl->modf(file);}
pdone     :=function(cl=psfcl)                   {if (is_record(psfcl) && psfcl.active==1) cl->done();}
pgo       :=function(n=0,cl=psfcl)               {cl->decomp(n);}

pdoit:=function(suspend=F)
{
    p:=pss(suspend);
    setimsize(2048);
    p->ncomp(5000);
    sendpsf('DirtyBeam.im');
    return p;
}

psetup:=function(ncomp,niter,imsize,PSFFile,ModelFile)
{
    psetncomp(ncomp);
    psetniter(niter);
    psetimsize(imsize);
    prdpsf(PSFFile);
    prdmodel(ModelFile);
}
