include 'image.g'
include 'stopwatch.g'

ImSize:=0;
sscl:=0;
ri:=array(0,1);
ci:=array(0,1);
restoredimg:=array(0,1);
st:=0;
#
# The constructor...;-)
#
ss:=function(client_pgm='nssdeconv2',suspend=F) 
{
    global sscl;
    sscl:=client(client_pgm,suspend=suspend);

    callbacks(sscl);

    return sscl;
}
#
# The callbacks installer...
#
callbacks:=function(c)
{
    global ri,ci,ImSize,restoredimg;

    whenever c->finished do
    {
	st.show();
	st.stop();
    }

    whenever c->resimg do
    {
	ri := $value;
	if (length(ri) < ImSize*ImSize)
          print "###Error: resimg event needed ",ImSize*ImSize, " numbers but got ", length(ri);
	else ri::shape:=[ImSize,ImSize];
    }

    whenever c->ccimg do
    {
	ci := $value;
	if (length(ri) < ImSize*ImSize)
          print "###Error: resimg event needed ",ImSize*ImSize, " numbers but got ", length(ri);
	else ci::shape:=[ImSize,ImSize];
    }

    whenever c->restoredimg do
    {
	restoredimg := $value;
	if (length(restoredimg) < ImSize*ImSize)
          print "###Error: resimg event needed ",ImSize*ImSize, " numbers but got ", length(ri);
	else restoredimg::shape:=[ImSize,ImSize];
    }

    whenever c->imsize do
    {
	ImSize := $value;
    }
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
gsetimsize:=function(n=256,cl=sscl)             {global ImSize; ImSize:=n;cl->imsize(n);}
grdpsf    :=function(file='psf.comp1000',cl=sscl)         {cl->psff(file);}
gtruepsf  :=function(file='truepsf.im',cl=sscl)           {cl->truepsf(sendimg(file));}
grddimg   :=function(file='testDirtyImage.im',cl=sscl)    {cl->mod(sendimg(file));}
gdone     :=function(cl=sscl)                   {if (is_record(sscl) && sscl.active==1) cl->done();}
go       :=function(n=0,cl=sscl)               {st.zero();st.start();cl->deconv(n);}

gsetup:=function(ncomp=2,niter=1000,sigma=10.0,gain=1.0,scale=0.1,imsize=512,
		FPSFFile="psf.comp1000",DirtyFile="sim/sim.dirty",TruePSF="sim/sim.psf")
{
    global st;
    sscl->init();
    gsetimsize(imsize);
    gsetncomp(ncomp);
    gsetniter(niter);
    grdpsf(FPSFFile);
    grddimg(DirtyFile);
    gtruepsf(TruePSF);
    sscl->gain(gain);
    sscl->sigma(sigma);
    sscl->scale(scale);
    st := stopwatch();
    st.stop();st.zero();
}

const exist:= function(tool=dv,varname='dv',toolname='viewer')
{
    if (!is_defined(varname) ||	        # if the var. is not even defined, return False. Else..
	(!is_record(varname) ||         # ...if the var. is defined, and is a record...
	 (has_field(varname, 'type') &&	# ...and if it has the field 'type'...
	  is_function(tool.type()) &&   # ...and if the tool.type is a function... 
	  (pgp.type() != toolname))))   # ...and if tool.type() is not equal to tool name...
	      return F;	                # ...THEN return False.  Pheeew!!!
    return T;                           # ...else return T;
}

function gui()
{
    include 'viewer.g';
    if (!exist()) dv.gui()

    top       := frame(side='top');
    ci_frame  := frame(top,side='left');
    ri_frame  := frame(top,side='left');
    res_frame := frame(top,side='left');

    ci_b:=button(ci_frame,'Asp model image');
    ci_d:=button(ci_frame,'Delete');

    ri_b:=button(ri_frame,'Residual image');
    ri_d:=button(ri_frame,'Delete');

    res_b:=button(res_frame,'Restored image');
    res_d:=button(res_frame,'Delete');


    ri_data:=0;
    ci_data:=0;
    res_data:=0;

    whenever ri_b->press do
    {
	deleteData(ri_data);
	ri_data:=dv.loaddata(ri,'raster',T);
	if (is_record(ri_data)) ri_d->background('red');
    }
    whenever ri_d->press do
    {
	deleteData(ri_data);
	ri_d->background('lightgray');
    }

    whenever ci_b->press do
    {
	deleteData(ci_data);
	ci_data:=dv.loaddata(ci,'raster',T);
	if (is_record(ci_data)) ci_d->background('red');
    }
    whenever ci_d->press do
    {
	deleteData(ci_data);
	ci_d->background('lightgray');
    }

    whenever res_b->press do
    { 
	deleteData(res_data);
	res_data:=dv.loaddata(restoredimg,'raster',T);
	if (is_record(res_data)) res_d->background('red');
    }
    whenever res_d->press do
    {
	deleteData(res_data);
	res_d->background('lightgray');
    }
}

function deleteData(ref d)
{
    if (is_record(d)) dv.deletedata(d);
    val d:=0;
}
