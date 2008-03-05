include 'functionals.g'
include 'pgplotter.g'
include 'fitting.g'
include 'image.g'
include 'mathematics.g'
include 'statistics.g'
include 'fftserver.g'

ft:=fftserver();
RMS:=array(0,1);
#
#-----------------------------------------------------------------------------
#
function pixmodel(val Peak=1.0, val Pos=0.0, val Sigma=1.0) 
{
#    t:=dfs.compiled("(p[1]*abs(p[3])/sqrt(2*pi))*exp(-(x-p[2])*(x-p[2])*(p[3]*p[3])/2)");
    t:=dfs.compiled("p[1]*exp(-(x-p[2])*(x-p[2])*p[3]*p[3]/2)");
    t.setparameters([Peak,Pos,1/Sigma]);
   return t;
}
#
#-----------------------------------------------------------------------------
# Return a Pixel model which is a multiplication of the two Pixel models P1 and P2
#
function mul(val P1, val P2)
{
    t:=pixmodel();

    par1:=P1.parameters();
    par2:=P2.parameters();

    sqS1:=par1[3]*par1[3]; 
    sqS2:=par2[3]*par2[3];
    s12:=sqrt(sqS1 + sqS2);
    x12:=(par1[2]*sqS1 + par2[2]*sqS2)/(s12*s12);
#    a12:=par1[1]*par2[1]*par1[3]*par2[3]*
#	exp(-(sqS1*sqS2/2)*(par1[2]-par2[2])*(par1[2]-par2[2])/(sqS1 + sqS2))/(2*pi)
    a12:=par1[1]*par2[1]*
	exp(-(sqS1*sqS2/2)*(par1[2]-par2[2])*(par1[2]-par2[2])/(sqS1 + sqS2));

    t.setparameters([a12,x12,s12]);
    return t;
}
#
#-----------------------------------------------------------------------------
# Return a pixel model which is a convolution of the two pixel models P1 and P2
#
function conv(val P1, val P2, val shift)
{
    t:=pixmodel();
    par1:=P1.parameters();
    par2:=P2.parameters();
 
    s:=sqrt((par1[3]*par1[3] + par2[3]*par2[3]));
    s12:=par1[3]*par2[3]/s;
    x12:=(par2[2] + par1[2])-shift;
#    a12:=par1[1]*par2[1];
    a12:=par1[1]*par2[1]*sqrt(2*pi)/s;

    t.setparameters([a12,x12,s12]);
    return t;
}
#
#-----------------------------------------------------------------------------
# Area under the product of two pixel models P1 and P2
#
function mul_area(val P1, val P2)
{
    par1:=P1.parameters();
    par2:=P2.parameters();

    x1:=par1[2]; s1:=par1[3];
    x2:=par2[2]; s2:=par2[3];
    a1:=par1[1]; a2:=par2[1];

    s12:=s1*s1 + s2*s2;
    p12:=(s1*s2)*(s1*s2);

#    return sqrt(p12/(2*pi*s12))*exp(-((p12*(x1-x2)^2)/(2*s12)));
    return a1*a2*sqrt(2*pi/s12)*exp(-((p12*(x1-x2)^2)/(2*s12)));
}
#
#-----------------------------------------------------------------------------
# Value of the derivative of Chisq w.r.t. x
#
function dchisq_x1(val chisq, val x1, val x2, val s1, val s2)
{
    s12:=s1*s1 + s2*s2;
    p12:=(s1*s2)*(s1*s2);

    return (p12/s12)*chisq*(x1-x2);
}
#
#-----------------------------------------------------------------------------
# Value of the derivative of Chisq w.r.t. sigma
#
function dchisq_s1(val chisq, val x1, val x2, val s1, val s2)
{
    s12:=s1*s1 + s2*s2;
    p12:=(s1*s2)*(s1*s2);

    return (chisq/s1)*(s2^2/s12)*(1-(p12/s12)*(x1-x2)^2);
}
#
#-----------------------------------------------------------------------------
#
function functionalpixmodel(val BlobCC, ref BlobFM)
{
    val BlobFM:=dfs.compound();
    t:=pixmodel();
#
# Make the strings and produce a list of functionals.  
# Then add them together to make the compund functional 
# and solve for the shape of all the constituent Blob
# shapes.
#
    for (i in [1:BlobCC::shape[1]])
    {
	t.setparameters(BlobCC[i,]);
	val BlobFM.add(t);
    }
    BlobFM.state();
}
#
#-----------------------------------------------------------------------------
#
function makepixmodel(ref Img, val Blobs)
{
    x:=[1:length(Img)];
    Img[x]:=0;
    t:=pixmodel();
    N:=Blobs::shape[1];
    for(i in 1:N)
    {
	t.setparameters(Blobs[i,]);
	val Img:=Img+t.f(x);
    }
}
#
#-----------------------------------------------------------------------------
#
function decomp(ref Img, ref Pix, ref RMS, val MaxPix=100, val Gain=1.0,
		val Mask=[T,T,T])
{
    Res:=Img;
    t:=pixmodel();
    NBlobs:=1;
    xl:=[1:Res::shape];
    ImSize:=Res::shape;
    val Pix:=array(0,MaxPix,3);
    val RMS:=array(0,MaxPix);
    Center:=Res::shape/2+1;
    ImgMask:=array(T,Res::shape);
    Edge:=1;
    ImgMask[1:Edge]:=F;
    x1:=ImSize-Edge;
    ImgMask[x1:ImSize]:=F;

    while(NBlobs <= MaxPix)
    {
	Peak:=max_with_location(abs(Res),pos,ImgMask);
	Peak:=Res[pos];		

	t.setparameters([Peak,pos,1.0]);
	t.setmasks(Mask);

	x1:=pos-20; x2:=pos+20;
	if (x1<1) x1:=1;
	if (x2>ImSize) x2:=ImSize;

	x0:=x1:x2;
	dfit.reset();dfit.functional(t,xl,Res);
	Sol:=dfit.solution();
	if ((abs(Sol[1]) < 10.0) && (abs(Sol[3]) < 2))
	{
	    Pix[NBlobs,]:=Sol;
	    t.setparameters(Sol);
	    #
	    # Subtract the current blob from the image
	    # upto N*Sigma 
	    #
	    sz:=7/abs(Sol[3]);
	    r0:=as_integer(Sol[2]-sz);
	    r1:=as_integer(Sol[2]+sz);

	    if (r0<1) r0:=1;
	    if (r1>ImSize) r1:=ImSize;
	    rr:=[r0:r1];
	    val Res[rr]:=Res[rr]-Gain*t.f(rr);

	    print "Blob: ", NBlobs, ".Initial: [",Peak,",",pos,"]",   "Final: [",Pix[NBlobs,];
	    NBlobs:=NBlobs+1;

#	    dx:=Center-Sol[2];
#	    if (abs(dx)>1) 
#		{
#		    Sol[2]:=Center+dx;
#		    Pix[NBlobs,]:=Sol;
#		    t.setparameters(Sol);
#		    val Res[rr]:=Res[rr]-Gain*t.f(rr);
#		    print "Blob: ", NBlobs, ".Initial: [",Peak,",",pos,"]",   "Final: [",Pix[NBlobs,];
#		    NBlobs:=NBlobs+1;
#		}
	    
	    RMS[NBlobs]:=stddev(Res);
	}
	else
	{
	    t.setparameters([Peak,pos,01]);
	    Pix[NBlobs,]:=[Peak,pos,01];
	    val Res:=Res-Gain*t.f(xl);
	    RMS[NBlobs]:=stddev(Res);
	    print "0Blob: ", NBlobs, ".Initial: [",Peak,",",pos,"]",   "Final: [",Pix[NBlobs,];
	    NBlobs:=NBlobs+1;

#	    dx:=Center-pos;
#	    if (abs(dx)>1) 
#		{
#		    S:=[Peak,p,0.10];
#		    S[2]:=Center+dx;
#		    Pix[NBlobs,]:=S;
#		    t.setparameters(S);
#		    val Res:=Res-Gain*t.f(xl);
#		    print "0Blob: ", NBlobs, ".Initial: [",Peak,",",pos,"]",   "Final: [",Pix[NBlobs,];
#		    NBlobs:=NBlobs+1;
#		}
	}
    }
#    Pix::shape[1]:=NBlobs-1;
}
#
#--------------------------------------------------------------------------
#
function makepc(val Peak, val Pos, val Sigma)
{
    N:=length(Peak);
    t:=array(0,N,3);
    for (i in 1:N) t[i,]:=[Peak[i],Pos[i],1/Sigma[i]];
    return t;
}
#
#--------------------------------------------------------------------------
#
function makedi(ref DI, ref PSF, val Comps)
{
#
# Given a list of components (Comps) and the pixelated PSF, make the dirty image
#
#    N:=Comps::shape[1];
#    x:=1:length(DI);
#    DI[x]:=0;

    makepixmodel(DI,Comps);

#    for (i in 1:N) 
#    {
#	t:=pixmodel(Comps[i,1],Comps[i,2],Comps[i,3]);
#	t.state();
#	val DI:=DI+t.f(x);
#    }
    val DI:=ft.convolve(DI,PSF);
}

function makepdi(ref DI, ref Blobs, val Comps)
{
#
# Given a list of components (Comps) and the decomposed PSF (Blobs),
# make the dirty image
#
    NC:=Comps::shape[1];
    NB:=Blobs::shape[1];
    Shift:=length(DI)/2+1;
    ImSize:=length(DI);
    x:=1:ImSize;
    DI[x]:=0;
    for (i in 1:NC) 
    {
	cc:=pixmodel(Comps[i,1],Comps[i,2],Comps[i,3]);
	for (j in 1:NB)	
	{
	    bb:=pixmodel(Blobs[j,1],Blobs[j,2],1/Blobs[j,3]);
	    t:=conv(cc,bb,Shift);
	    sz:=5/t.par(3);
	    r0:=as_integer(t.par(2)-sz);
	    r1:=as_integer(t.par(2)+sz);
	    if (r0<1) r0:=1;
	    if (r1>ImSize) r1:=ImSize;
	    rr:=[r0:r1];
	    val DI[rr]:=DI[rr]+t.f(rr);
	}
    }
}
#psf2d:=image('~sanjay/Data/FITS/DIRTY_BEAM.FITS');
#s:=psf2d.shape();
#psf1d:=psf2d.getchunk([1,s[2]/2+1,1],[s[1],s[2]/2+1,1]);
#
#--------------------------------------------------------------------------
#
function plottool()
{
    global pgp;

    if (!is_defined('pgp') ||	# if pgp is not even defined, start the pgplotter(). Else..
	(!is_record(pgp) ||		# ...if pgp is defined, and is a record...
	 (has_field(pgp, 'type') &&	# ...and if it has the field 'type'...
	  is_function(pgp.type()) && # ...and if the pgp.type is a function... 
	  (pgp.type() != 'pgplotter')))) # ...and if pgp.type() is not equal to 'pgplotter'...
	pgp:=pgplotter();		# ...THEN start a fresh pgplotter().  Pheeew!!!
    return pgp;
}

#Res:=psf1d[,1,1];
Res:=0;
di:=Res;
dim:=Res;
mod:=Res;
RMS:=Res;
Blobs:=0;
CC:=array(0,1,3);
CC[1,]:=[1,1024,0.5];

function test(val n)
{
    if (Blobs::shape[1] < n)
    {
	decomp(Res, Blobs, RMS, n, 1.0, [T,T,T]);
	makepixmodel(mod,Blobs);
    }
    makedi(di,mod,CC);
    makepdi(dim,Blobs,CC);

    pgp.ploty(di);
    pgp.ploty(dim);
}

function wrtpixpsf(val FileName, val PSF, val m=1.0)
{
    str := spaste(">",FileName);
    fd := open(str);
    N := PSF::shape[1];
    fprintf(fd,"%d\n",N);
    write(fd);

    for (i in 1:N)
    {
	fprintf(fd,"%f %f %f\n", PSF[i,1]/m, PSF[i,2]-1, PSF[i,3]);
	write(fd);
    }
}

function rdpixpsf(val FileName, ref PSF)
{
    str := spaste("<",FileName);
    fd := open(str);
    N := as_integer(read(fd));
    print "No. of comps found",N;
    val PSF:=array(0,N,3);
    for (i in 1:N)
    {
	v:=read(fd);
	PSF[i,]:=as_float(split(v));
	PSF[i,2]:=PSF[i,2]+1;
    }
}

function wrtimg(val FileName, val Img)
{
    str := spaste(">",FileName);
    fd := open(str);
    N := length(Img);
    fprintf(fd,"%d\n",N);
    write(fd);
    for (i in 1:N)
    {
	fprintf(fd,"%f\n", Img[i]);
	write(fd);
    }
}

function rdimg(val FileName)
{
    str := spaste("<",FileName);
    fd := open(str);
    v:=read(fd);
    N := as_integer(v);
    img := array(0,N);
    for (i in 1:N)
    {
	v := read(fd);
	img[i]:=as_float(v);
    }
    close(fd);
    return img;
}

function rdrealimg(val FileName)
{
   str := spaste("<",FileName);
   fd := open(str);
   
   N := as_integer(read(fd));
   print N

#   PSF := array(0,N);
   s:=psf2d.shape();
   PSF:=psf2d.getchunk([1,s[2]/2+1,1],[s[1],s[2]/2+1,1]);

   for (i in 1:N)
   {
       v := read(fd);
       PSF[i] := as_float(v);
   }
   return PSF;
}
