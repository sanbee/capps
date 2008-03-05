include 'simulator.g';

# pass in an image and simulate away;
const sim:=function(modfile='', noise='0.0Jy', sim=T, algorithms='msclean')
{

  include 'logger.g';
  dl.purge(0);
  include 'webpublish.g';

  testdir := 'sim';
  
  if(sim) {
    note('Cleaning up directory ', testdir);
    ok := shell(paste("rm -fr ", testdir));
    if (ok::status) { throw('Cleanup of ', testdir, 'fails!'); };
    ok := shell(paste("mkdir", testdir));
    if (ok::status) { throw("mkdir", testdir, "fails!") };
  }

  resultsdir := spaste(testdir, '/results');
  webpub:=webpublish(resultsdir, 'AIPS++ simulation of Multiscale clean');
  if(is_fail(webpub)) fail;
		     
  webpub.comments('This simulates processing of VLA observations using Multiscale clean.</p>');

  msname   := spaste(testdir, '/',testdir, '.ms');
  simmodel := spaste(testdir, '/',testdir, '.model');
  simsmodel:= spaste(testdir, '/',testdir, '.smodel');
  simtemp  := spaste(testdir, '/',testdir, '.temp');
  simpsf   := spaste(testdir, '/',testdir, '.psf');
  simempty := spaste(testdir, '/',testdir, '.empty');
  simmask  := spaste(testdir, '/',testdir, '.mask');
  simvp    := spaste(testdir, '/',testdir, '.vp');

  dir0 := dm.direction('j2000',  '0h0m0.0', '45.00.00.00');
  reftime := dm.epoch('iat', '2001/01/01');

  if(sim) {

    note('Create the empty measurementset');
    
    mysim := simulator();
    
    mysim.setspwindow(row=1, spwname='XBAND', freq='8.0GHz', deltafreq='50.0MHz',
		      freqresolution='50.0MHz', nchannels=1, stokes='RR LL');
    
      note('Simulating VLA');
      posvla := dm.observatory('vla');
#
#  Define VLA C array by hand, local coordinates
#
      xx := [41.1100006,134.110001,268.309998,439.410004,644.210022,880.309998,
	     1147.10999,1442.41003,1765.41003,-36.7900009,-121.690002,-244.789993,
	     -401.190002,-588.48999,-804.690002,-1048.48999,-1318.48999,-1613.98999,
	     -4.38999987,-11.29,-22.7900009,-37.6899986,-55.3899994,-75.8899994,
	     -99.0899963,-124.690002,-152.690002];
      yy := [3.51999998,-39.8300018,-102.480003,-182.149994,-277.589996,-387.839996,
	     -512.119995,-649.76001,-800.450012,-2.58999991,-59.9099998,-142.889999,
	     -248.410004,-374.690002,-520.599976,-685,-867.099976,-1066.42004,77.1500015,
	     156.910004,287.980011,457.429993,660.409973,894.700012,1158.82996,1451.43005,
	     1771.48999];
      zz := [0.25,-0.439999998,-1.46000004,-3.77999997,-5.9000001,-7.28999996,
	     -8.48999977,-10.5,-9.56000042,0.25,-0.699999988,-1.79999995,-3.28999996,
	     -4.78999996,-6.48999977,-9.17000008,-12.5299997,-15.3699999,1.25999999,
	     2.42000008,4.23000002,6.65999985,9.5,12.7700005,16.6800003,21.2299995,
	     26.3299999];
      
      diam := 0.0 * [1:27] + 25.0;
      mysim.setconfig(telescopename='VLA', x=xx, y=yy, z=zz,
		      dishdiameter=diam, 
		      mount='alt-az', antname='VLA',
		      coordsystem='local', referencelocation=posvla);
      mysim.setfield(sourcename='M31SIM', sourcedirection=dir0,
		     integrations=1, xmospointings=1, ymospointings=1,
		     mosspacing=1.0);
      mysim.settimes('60s', '300s', T, '-14400s', '+14400s');
      mysim.create(newms=msname, shadowlimit=0.001, 
		   elevationlimit='8.0deg', autocorrwt=0.0);
    }
    
    mysim.done();
    
    note('Make an empty image from the MS, and fill it with the');
    note('the model image;  this is to get all the coordinates to be right');
    
    myimg1:=F;
    if(modfile=='') {
      include 'sysinfo.g';
      sysroot := sysinfo().root();
      modfile:=spaste(sysroot, '/data/demo/M31.model.fits');
      myimg1 := imagefromfits(infile=modfile, zeroblanks=T);
    }
    else {
      myimg1 := image(modfile);
    }
    if(is_fail(myimg1)) fail;
    imgshape := myimg1.shape();
    note('The model is ', modfile);

    if(is_fail(myimg1)) fail;
    imgshape := myimg1.shape();
    imsize := imgshape[1];
    
    myimager := imager(msname);
    myimager.setdata(mode="none" , nchan=1, start=1, step=1,
		     mstart="0km/s" , mstep="0km/s" , spwid=1, fieldid=1);
    myimager.setimage(nx=imsize, ny=imsize, cellx="0.5arcsec" , celly="0.5arcsec" ,
		      stokes="I" , fieldid=1, facets=1, doshift=T,
		      phasecenter=dir0);
    myimager.make(simmodel);
    myimager.done();
    
    myimg2 := image(simmodel);  #  this is the dummy image with correct coordinates
    arr1 := myimg1.getchunk();
    myimg2.putchunk( arr1 );      #  now this image has the model pixels and 
    #  the correct coordinates
    myimg1.done();
    myimg2.done();
    note('Made model image with correct coordinates');
    note('Read in the MS again and predict from this new image');
    
    mysim := simulatorfromms(msname);
    mysim.predict(simmodel);
    
    if(noise!='0.0Jy') {
      note('Add noise');
      mysim.setnoise(mode='simplenoise', simplenoise=noise);
      mysim.corrupt();
    }
    mysim.done();
    
  
  webpub.comments('<p>The original model (including a copy in FITS format )</p>');
  for (name in [simmodel]) {
    if(tableexists(name)) webpub.image(name, name, dofits=T, dodeep=T);
  }
  webpub.flush();

  cell:="0.5arcsec"; imsize:=imgshape[1];
  if(imsize%2) imsize+:=3;

  myimager := imager(msname);
  myimager.setdata(mode="none" , nchan=1, start=1, step=1,
		   mstart="0km/s" , mstep="0km/s" , spwid=1, fieldid=1);
  myimager.setimage(nx=2*imsize, ny=2*imsize, cellx=cell , celly=cell ,
		    stokes="I" , fieldid=1, facets=1, doshift=T,
		    phasecenter=dir0);

  myimager.weight(type="uniform");

  myimager.make(simempty);
  myimager.approximatepsf(model=simempty, psf=simpsf);
  bmaj:=F; bmin:=F; bpa:=F;
  myimager.fitpsf(simpsf, bmaj, bmin, bpa);

  im:=image(simempty);
  shape:=im.shape();
  cs:=im.coordsys();
  cs.summary()
  im.done();
  myimager.smooth(simmodel, simtemp, F, bmaj, bmin, bpa, normalize=F);
  im:= image(simtemp);
  im.regrid(outfile=simsmodel, shape=shape, csys=cs, axes=[1,2]);
  im.done();

  myimager.regionmask(simmask, drm.quarter());

  for (algorithm in algorithms) {

    simimage := spaste(testdir, '/', testdir, '.', algorithm);
    simrest  := spaste(testdir, '/', testdir, '.', algorithm, '.restored');
    simresid := spaste(testdir, '/', testdir, '.', algorithm, '.residual');
    simerror := spaste(testdir, '/', testdir, '.', algorithm, '.error');
    
    tabledelete(simrest);
    tabledelete(simresid);
    tabledelete(simimage);
    
    if(algorithm=='clark') {
      myimager.clean(algorithm='clark', niter=100000, gain=0.1,
		     displayprogress=F,
		     model=simimage, image=simrest, residual=simresid,
		     mask=simmask);
    }
    else if(algorithm=='multiscale'){
      myimager.setscales('uservector', uservector=[0, 6, 12, 24]);
      myimager.clean(algorithm='multiscale', niter=2000, gain=0.7,
	displayprogress=F,
		     model=simimage , image=simrest, residual=simresid,
		     mask=simmask);
      
    }
    else if(algorithm=='entropy'){
      myimager.mem(algorithm='entropy', niter=30, displayprogress=F,
		   model=simimage , image=simrest, residual=simresid,
		   mask=simmask);
    }
    else if(algorithm=='emptiness'){
      myimager.mem(algorithm='emptiness', niter=30, displayprogress=F,
		   model=simimage , image=simrest, residual=simresid,
		   mask=simmask);
    }
    else {
      myimager.make(simempty);
      myimager.residual(model=simempty, image=simimage);
    }
    if(tableexists(simrest)) {
      imagecalc(simerror, spaste('"', simrest, '" - "', simsmodel, '"')).done();
    }
    webpub.comments(spaste('<p>The images for ', algorithm, ' image processing</p>'));
    for (name in [simimage, simrest, simresid, simerror]) {
      if(tableexists(name)) webpub.image(name, name, dodeep=T);
    }
    webpub.flush();
  }

  webpub.comments('<p>The smoothed model, and psf</p>');
  for (name in [simsmodel, simpsf]) {
    if(tableexists(name)) webpub.image(name, name, dodeep=T);
  }
  webpub.flush();

  myimager.done();

  webpub.comments('<p>Processing script and log</p>');
  webpub.script('sim.g');
  webpub.log();
  webpub.done();
}

include 'logger.g';
dl.purge(0);
dl.screen();
sim(sim=T, algorithms="dirty multiscale clark entropy emptiness");

