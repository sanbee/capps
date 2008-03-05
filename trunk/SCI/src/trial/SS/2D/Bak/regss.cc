#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Glish/Client.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <regFluxonCompList.h>
#include <vector.h>
#include <Exp.h>
//#include <ErrorObj.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_blas.h>
#include <Params.h>
#include <Convolver.h>

#define TOL 1E-4
#define PTOL 1E-02
#define FIRST_STEP 0.1
#define EDGE 128

FTYPE GAIN=1.0;
int PARAMS_PER_FLUXON;
FTYPE DEFAULT_SCALE=0.1, DefaultSigma=4.0;

template<class T> T getPeak(IMAGETYPE<T>& Img, int& PosX,int &PosY);
template <class T> T getDefaultScale() {return DEFAULT_SCALE;};

Convolver<FTYPE> Convo;


void Normalize(IMAGETYPE<FTYPE>& Img, FluxonCompList<FTYPE>& CL)
{
  int PosX,PosY;
  FTYPE Peak;
  Peak = getPeak<FTYPE>(Img,PosX,PosY);

  for(int i=0;i<Img.size();i++)
    for(int j=0;j<Img.size();j++)
      Img.setVal(i,j,Img(i,j)/Peak);

  for(int i=0;i<CL.size();i++)
    CL[i].setAmp(CL[i].getAmp()/Peak);
}

template <class T> void MakeImages(FluxonCompList<T>&ModelDI,
				   IMAGETYPE<T> &DImg,
				   IMAGETYPE<T> &MImg,
				   IMAGETYPE<T> &ResImg)
{
  int N;
  //  MImg.assign(0.0);
  //  MImg   += ModelDI;
  //  Normalize(MImg,ModelDI);
  N = MImg.size();
  
  

  for(int i=EDGE;i<(N-EDGE);i++) 
    for(int j=EDGE;j<(N-EDGE);j++) 
      ResImg.setVal(i,j,(DImg(i,j) - MImg(i,j)));
  /*
  {
    std::ofstream tmp("di.dat");
    std::ofstream tmp1("mi.dat");
    std::ofstream tmp2("ri.dat");
    tmp << DImg ;
    tmp1 << MImg;
    tmp2 << ResImg;
    exit(0);
  }
  */
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static T my_f (const gsl_vector *v, void *params)
{
  T ChiSq=0,A,Px,Py,S;
  int NX,NY, k;
  IMAGETYPE<T> *ResImg, *DImg, *MImg;
  vector<int> *Pj;
  FluxonCompList<T> *ModelDI, *MComp, *PSF;
  gsl_multimin_fdfminimizer *s;
  //
  // Extract the params from the void pointer...
  //
  Params<FTYPE> MyP(params);
  ResImg  = MyP.ResImg();
  MComp   = MyP.ModComp();   // CC list
  s       = MyP.Minimizer();
  Pj      = MyP.CompList();
  DImg    = MyP.DImg();
  MImg    = MyP.MImg();      // PSFImage
  ModelDI = MyP.ModelDI();   // (Fluxon PSF) * (CC List) = Fluxon Model Dirty Image
  PSF     = MyP.PSF();       // Fluxon PSF
  //
  // Make the model and residual image at the current location on the
  // Chisq surface
  //
  NX = (*Pj).size();

  k=0;
  for(int j=0;j<NX;j++)
    if ((*Pj)[j] == 1)
    {
      int m;

      //      k=j;
      m=PARAMS_PER_FLUXON*k;
      A  = (T)gsl_vector_get(v,m);
      Px = (T)gsl_vector_get(v,m+1);
      Py = (T)gsl_vector_get(v,m+2);
      S  = (T)gsl_vector_get(v,m+3);

      (*MComp)[k].setAmp(A);
      (*MComp)[k].setXPos(Px);
      (*MComp)[k].setYPos(Py);
      (*MComp)[k].setScale(S);
      //      cerr << "Params in my_f: " << (*MComp)[k] << endl;
      k++;
    }

  (*MImg).size(NX,NY);
  (*MImg).assign(0);

  (*ModelDI) = (*MComp).convolve((*PSF));
  (*MImg) += (*ModelDI);
  /*
  (*MImg) += (*MComp);
  Convo.convolve((*MImg));
  */
  //  (*MImg).assign(0.0);
  //  (*MImg) += (*MComp);
  MakeImages(*ModelDI, *DImg, *MImg, *ResImg);


  for (int i=EDGE;i<(NX-EDGE);i++) 
    for (int j=EDGE;j<(NY-EDGE);j++) 
      ChiSq += (*ResImg)(i,j)*(*ResImg)(i,j);

  //
  // Debugging code...
  //
  /*
  gsl_vector *dp0;
  dp0 = gsl_multimin_fdfminimizer_gradient(s);

  N = (*Pj).size();
  for(int j=0;j<N;j++)
    if ((k = (*Pj)[j]) > -1)
    {
      int m;
      m=PARAMS_PER_FLUXON*k;

      cerr << "my_f: " << k << " " << (*MComp)[k] << " " << ChiSq << " " 
	   << fabs(ChiSq-gsl_multimin_fdfminimizer_minimum(s)) << " "
	   << gsl_vector_get(dp0,m) << " " << gsl_vector_get(dp0,m+1) << " " 
	   << gsl_vector_get(dp0,m+2) << endl;
    }
  */
  //  cerr << "ChiSq: " << ChiSq << endl;

  return ChiSq;
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static void my_df (const gsl_vector *v, void *params, gsl_vector *g)
{
  T dA, dPx, dPy, dS,tmp;
  int Nx,Ny,NP,k;
  IMAGETYPE<T> *ResImg;
  vector<int> *Pj;
  FluxonCompList<T> *MComp, *ModelDI, *PSF;
  //
  // Recover the pointers to the parameters... :-| (Phew!)
  //
  Params<FTYPE> MyP(params);
  ResImg = MyP.ResImg();
  MComp  = MyP.ModComp();
  Pj     = MyP.CompList();
  ModelDI= MyP.ModelDI();
  PSF    = MyP.PSF();

  ResImg->size(Nx,Ny);

  k=0;
  NP=(*Pj).size();
  //  cerr << "Params in my_df: " << (*MComp)[k] << " " << NP << endl;

  for(int j=0;j<NP;j++)
    if ((*Pj)[j] == 1)
      {
	int m,XRange0,XRange1,YRange0,YRange1;
	Fluxon2D<T> tF;
	//	k=j;
	
	dA = dPx = dPy = dS = 0.0;
	for (int i=0;i<Nx;i++)
	  for (int j=0;j<Ny;j++)
	  {
	    tF = (*MComp)[k];
	    //
	    // Derivatives are evaluated only for the region bound by (XRange0,YRange0)
	    // and (XRange1, YRange1).
	    // This range as of now is not strictly accurate for derivative computation
	    // (specially for dChiSq/dScale) - should be improved to take into account 
	    // the multiplication with the parabolid for the derivative w.r.t scale.
	    //
	    tF.getRange(Nx,XRange0,XRange1,YRange0,YRange1);
	    if (tF.inRange(i,j,XRange0,XRange1,YRange0,YRange1))
	      {
		tF.evalDerivatives(i,j);
	
		tmp = (*ResImg)(i,j);
		dA  += tmp*tF.dF(tF.DA);
		dPx += tmp*tF.dF(tF.DXP);
		dPy += tmp*tF.dF(tF.DYP);
		dS  += tmp*tF.dF(tF.DS);
	      }
	  }

	//	if (dA==0) {junk=1;goto AGAIN;}
	m=PARAMS_PER_FLUXON*k;
	gsl_vector_set(g,m  ,dA);
	gsl_vector_set(g,m+1,dPx);
	gsl_vector_set(g,m+2,dPy);
	gsl_vector_set(g,m+3,dS);
	//	cerr << "G: " << dA << " " << dPx << " " << dPy << " " << dS << " " << (*MComp)[k] << endl;
	k++;
      }
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static void my_fdf (const gsl_vector *v, void *params, T *f, gsl_vector *df)
{
  *f = my_f<T>(v,params);
  my_df<T>(v,params,df);
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> T getPeak(IMAGETYPE<T>& Img, int& PosX, int &PosY)
{
  int n=Img.size();
  int Window=100;
  T P=Window;
  PosX=PosY=Window;

  for(int i=Window;i<(n-Window);i++) 
    for(int j=Window;j<(n-Window);j++) 
      if (fabs(Img(i,j)) > fabs(Img(PosX,PosY)))  
	{PosX = i; PosY=j;}

  PosX = PosX-1;
  return Img(PosX,PosY);
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> void setupSolver(gsl_multimin_fdfminimizer **Solver, 
				   const gsl_multimin_fdfminimizer_type *MinimizerType,
				   gsl_vector *x,
				   gsl_multimin_function_fdf *my_func,
				   int NPixons,
				   void *par[],
				   vector<int> *PIndex,
				   IMAGETYPE<T>      *ResImg,
				   IMAGETYPE<T>      *DImg,
				   IMAGETYPE<T>      *MImg,
				   FluxonCompList<T> *ModelDI,
				   FluxonCompList<T> *FCList,
				   FluxonCompList<T> *PSFList
				   )
{
  int NParams;


  NParams = PARAMS_PER_FLUXON*(NPixons);
  //
  // Tell GSL to use only first NParams out of 
  // 3*NP to solve the current problem
  //
  x->size = NParams;

  if (*Solver) gsl_multimin_fdfminimizer_free(*Solver);
  *Solver = gsl_multimin_fdfminimizer_alloc(MinimizerType, NParams);
  cerr << "###Using " << gsl_multimin_fdfminimizer_name(*Solver) 
       << " minimization" << " for " << NParams << " params" 
       << endl;

  //  (*PIndex)[NP] = NP;

  par[0]         = (void *)ResImg;
  par[1]         = (void *)FCList;
  par[2]         = (void *)*Solver;
  par[3]         = (void *)PIndex;
  par[4]         = (void *)DImg;
  par[5]         = (void *)MImg;
  par[6]         = (void *)ModelDI;
  par[7]         = (void *)PSFList;

  my_func->f      = &my_f<FTYPE>;
  my_func->df     = &my_df<FTYPE>;
  my_func->fdf    = &my_fdf<FTYPE>;
  my_func->n      = NParams;
  my_func->params = par;
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> int findComponent(int NIter, gsl_multimin_fdfminimizer *s, 
				    gsl_vector *x, 
				    gsl_multimin_function_fdf *func, 
				    FluxonCompList<T> &MI, 
				    int P0,int Pi, int DoInit=1,int RestartAt=300)
{
  int iter = 0,m,status;
  gsl_vector *x0, *p0;
  T A,PX,PY,S;

  FTYPE Min0=0, Min=0;
  do
    {
      if ((iter % RestartAt) == 0)
	{
	  gsl_vector *x0;
	  T InitStep;
	  
	  x0 = gsl_multimin_fdfminimizer_x(s);
	  InitStep = gsl_blas_dnrm2 (x0);
	  //	  cerr << "####Restarting: " << InitStep << endl;
	  gsl_multimin_fdfminimizer_set(s, func, x0, InitStep, TOL);

	  gsl_multimin_fdfminimizer_restart(s);
	}
      //
      // 
      x0 = gsl_multimin_fdfminimizer_x(s);
      p0 = gsl_multimin_fdfminimizer_gradient(s);
      if (DoInit)
	for (int j=P0;j<=Pi;j++)
	  {
	    m=PARAMS_PER_FLUXON*(j-P0);
	    m=PARAMS_PER_FLUXON*(j);
	    A  = gsl_vector_get(x0,m);
	    PX = gsl_vector_get(x0,m+1);
	    PY = gsl_vector_get(x0,m+2);
	    S  = gsl_vector_get(x0,m+3);
	    MI[j].setAmp(A);	  
	    MI[j].setXPos(PX);	  
	    MI[j].setYPos(PY);	  
	    MI[j].setScale(S);
	  }

      Min0 = Min;	  Min = gsl_multimin_fdfminimizer_minimum(s);
      for (int j=P0;j<=Pi;j++)
	{
	  int mm;
	  mm=PARAMS_PER_FLUXON*(j-P0);
	  cerr << iter                    << " "
	       << Min                     << " "
	       << gsl_blas_dnrm2(s->dx)   << " "
	       << j                       << " "
	       << MI[j]                   << " "
	       << gsl_vector_get(p0,mm)   << " "
	       << gsl_vector_get(p0,mm+1) << " "
	       << gsl_vector_get(p0,mm+2) << " "
	       << gsl_vector_get(p0,mm+3) << " "
	       << endl;
	}
      //
      // Make the move!
      //
      status = gsl_multimin_fdfminimizer_iterate(s);
      if (status == GSL_ENOPROG) gsl_multimin_fdfminimizer_restart(s);
      if (status)  
	{
	  cerr << "###Info: " << gsl_strerror(status) << endl;
	  break;
	}
      status = gsl_multimin_test_gradient(s->gradient, 1E-3);
      
      p0 = gsl_multimin_fdfminimizer_gradient(s);
      
      if (iter%20 == 0)
	cerr << "###Intermediate results " << iter                    << " "
	     << Min                     << " "
	     << gsl_blas_dnrm2(s->dx)   << " "
	     << endl;
      iter++;
    }
  while(status == GSL_CONTINUE && iter < NIter);// && fabs(Min-Min0) > TOL);
  cerr << "###Info: " << gsl_strerror(status) << endl;
  
  return status;
}
//
//---------------------------------------------------------------------------------------------
//
template <class T> void ssdeconvolve(int NP, 
				     int NIter, 
				     IMAGETYPE<T>      &TruePSF,
				     FluxonCompList<T> &PSF, 
				     IMAGETYPE<T>      &DImg, 
				     FluxonCompList<T> &MI,
				     int N0, int N1,int PSFSize)
{
  int N=1, NX, NY;
  T Normalization, CleanBeamScale, InitStep;
  T Peak;
  IMAGETYPE<T> PSFImg,MImg,ResImg;
  int PosX,PosY;
  FluxonCompList<T> ModelDI;
  Fluxon2D<T> f0;
  vector<int> PIndex;


  //  Convo.makeXFR(TruePSF);

  PARAMS_PER_FLUXON = f0.getNoOfParams();

  DImg.size(NX,NY);

  PSFImg.resize(NX,NY);  PSFImg.assign(0.0);
  MImg.resize(NX,NY);    MImg.assign(0.0);
  ResImg.resize(NX,NY);
  
  PIndex.resize(NP);     PIndex.assign(NP,-10);

  if (MI.size() > 0)
    {
      cerr << "###Computing model data..." << endl << MI << endl;
      MImg.assign(0.0);

      //      ModelDI = MI.convolve(PSF);
      //      MImg   += ModelDI;
      MImg += MI;
      Convo.convolve(MImg);
      
      for (int i=0;i<MI.size();i++) PIndex[i]=1;
    }
  else
    MI.resize(0);

  CleanBeamScale = PSF[0].getScale();

  //
  // Convert all component images to normal images
  //
  PSFImg += PSF;

  cerr << "###Computing residual..." << endl;
  for (int i=0;i<NX;i++) 
    for (int j=0;j<NY;j++) 
      ResImg.setVal(i,j,(DImg(i,j) - MImg(i,j)));

  //
  // MI is the model component image.  PIndex is list of 
  // Fluxons which are to be used in the problem.
  //
  // Begin adding components to the comp. list (MI)
  //
  //  MI.resize(0);
  //  PIndex.resize(NP);
  //  PIndex.assign(PIndex.size(),-1);
  //
  // The array of pixel model parameters.  We find one component
  // at a time right now.
  //
  gsl_vector *x=NULL;
  x = gsl_vector_alloc(PARAMS_PER_FLUXON*NP);
  gsl_vector_set_zero(x);
  
  gsl_vector *x0;
  static void* par[8];
  int status,m,NParams;
  //  const gsl_multimin_fdfminimizer_type *MT;
  static gsl_multimin_fdfminimizer *s=NULL,*s1=NULL;
  static gsl_multimin_function_fdf my_func,one_fluxon;


  Normalization  = getDefaultScale<T>()*getDefaultScale<T>();
  Normalization  = (2*M_PI/(CleanBeamScale*CleanBeamScale + Normalization));
        

      IMAGETYPE<T> SmoothedRes;
      SmoothedRes = ResImg;
      int Smoothed=0;
      T SmoothingScale=8;

  for (int Pi=N0; Pi<N1; Pi++)
    {
      if (Pi & (Pi%10 == 0))
	{
	  cerr << "###Info: Writing " << Pi << " components to the disk (comps.tmp)" << endl;
	  std::ofstream tmp("comps.tmp");
	  tmp << MI;
	}	  
      size_t iter = 0;

      NParams = PARAMS_PER_FLUXON*(Pi+1);

      // T = gsl_multimin_fdfminimizer_conjugate_fr;
      // T = gsl_multimin_fdfminimizer_conjugate_pr;
      // T = gsl_multimin_fdfminimizer_steepest_descent;
      // T = gsl_multimin_fdfminimizer_vector_bfgs;
      setupSolver(&s, 
		  gsl_multimin_fdfminimizer_vector_bfgs,
		  x,
		  &my_func,
		  Pi+1,
		  par, 
		  &PIndex, 
		  &ResImg, 
		  &DImg, 
		  &MImg, 
		  &ModelDI, 
		  &MI, 
		  &PSF);
      //
      // Set the initial guess
      //
      for(int i=0;i<=Pi;i++)
	if (MI.size() > i)
	  {
	    m = PARAMS_PER_FLUXON*i;
	    gsl_vector_set(x, m,   MI[i].getAmp());
	    gsl_vector_set(x, m+1, MI[i].getXPos());
	    gsl_vector_set(x, m+2, MI[i].getYPos());
	    gsl_vector_set(x, m+3, MI[i].getScale());
	  }
      //
      // Get the peak and it's position from the residual image.
      // Setup a Fluxon with these values and add to the model
      // component image (MI).
      //
      //      if (!Restart)
      {
	gsl_vector *tx=NULL;
	
	
	tx = gsl_vector_alloc(PARAMS_PER_FLUXON);

	SmoothedRes.assign(0);
	if (Pi > N1) Smoothed=1;
	if (!Smoothed) SmoothedRes = ResImg;
	else
	  {
	    IMAGETYPE<T> tmp;
	    tmp.resize(ResImg.size(),ResImg.size());
	    Fluxon2D<T> tf;
	    tf = PSF[0];
	    tf.setScale(PSF[0].getScale()/SmoothingScale);
	    tf.setAmp(tf.getScale()/tf.getAmp());
	    cerr << "###Info: Smoothing residuals to " << tf.getScale() << " scale" << endl;
	    tmp.assign(0);
	    tmp += tf;
	    Convolver<T> tConvo(tmp);
	    SmoothedRes = ResImg;
	    tConvo.convolve(SmoothedRes);
	    Smoothed = 1;
	  }
	//	Peak = getPeak<T>(ResImg, PosX, PosY);
	Peak = getPeak<T>(SmoothedRes, PosX, PosY);
	Peak = ResImg(PosX,PosY);

	//	for (int i=0;i<Peak.size();i++) Peak[i] /= PSF[0].getAmp()*Normalization;
	Peak /= PSF[0].getAmp()*Normalization;

	cerr << "###Peak = " << Peak << " " << "@(" << PosX << "," << PosY << ")" << endl;

	//f0.setScale(CleanBeamScale);
	f0.setScale(getDefaultScale<T>());
	f0.setScale(0.0);
	f0.setAmp((Peak));
	f0.setXPos(PosX);
	f0.setYPos(PosY);

	//	f0.setAmp(Peak/fabs(Peak));

	m = PARAMS_PER_FLUXON*Pi;
	gsl_vector_set(x, m,   f0.getAmp());
	gsl_vector_set(x, m+1, f0.getXPos());
	gsl_vector_set(x, m+2, f0.getYPos());
	gsl_vector_set(x, m+3, f0.getScale());

	gsl_vector_set(tx, 0, f0.getAmp());
	gsl_vector_set(tx, 1, f0.getXPos());
	gsl_vector_set(tx, 2, f0.getYPos());
	gsl_vector_set(tx, 3, f0.getScale());
	//
	// Get an initial esitmate of the local scale and amplitude using the
	// CleanBeam and add the comp. to the complist.  Make the current 
	// model image
	//
	MI += f0;
	PIndex[Pi]=1;
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;
	PSF.resetSize();
	PSF.setSize(PSFSize);

	MImg.assign(0.0);

	ModelDI.resize(0);
	ModelDI = MI.convolve(PSF);
	MImg   += ModelDI;
	//	MImg += MI;
	//	Convo.convolve(MImg);

	//
	// Use only the main lobe of the PSF
	//
	/*
	cerr << "###Get initial guess..." << endl;
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;
	for (int i=0;i<PIndex.size();i++) if (PIndex[i] == 1) PIndex[i]=-1;
	PIndex[Pi]=1;
	setupSolver(&s1,
		    gsl_multimin_fdfminimizer_vector_bfgs,
		    tx,
		    &one_fluxon,
		    1,
		    par, 
		    &PIndex, 
		    &ResImg, 
		    &DImg, 
		    &MImg, 
		    &ModelDI, 
		    &MI, 
		    &PSF);
	PSF.setSize(1); 
	cerr << "###Using the first " << PSF.size() << " components of the PSF" << endl;
	InitStep = gsl_blas_dnrm2 (tx);
	gsl_multimin_fdfminimizer_set(s1, &one_fluxon, tx, InitStep, TOL);
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;
	findComponent(NIter,s1,tx,&one_fluxon,MI,Pi,Pi,1,50);
	for (int i=0;i<PIndex.size();i++) if (PIndex[i] == -1) PIndex[i]=1;
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;

	m = PARAMS_PER_FLUXON*Pi;
	x0 = gsl_multimin_fdfminimizer_x(s1);
	gsl_vector_set(x, m,   gsl_vector_get(x0,0));
	gsl_vector_set(x, m+1, gsl_vector_get(x0,1));
	gsl_vector_set(x, m+2, gsl_vector_get(x0,2));
	gsl_vector_set(x, m+3, gsl_vector_get(x0,3));

	PSF.setSize(PSFSize); // Reset the use of entire PSF
	cerr << "###Using the first " << PSF.size() << " components of the PSF" << endl;
	*/
	if (tx) gsl_vector_free(tx);
      }
      //
      // Find the Pi th. component.  This has the minimization loop
      //
      //      for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;

      setupSolver(&s,
		  gsl_multimin_fdfminimizer_vector_bfgs,
		  x,
		  &my_func,
		  Pi+1,
		  par, 
		  &PIndex, 
		  &ResImg, 
		  &DImg, 
		  &MImg, 
		  &ModelDI, 
		  &MI, 
		  &PSF);
      //      PSF.resetSize();  
      InitStep = gsl_blas_dnrm2 (x);
      gsl_multimin_fdfminimizer_set(s, &my_func, x, InitStep, TOL);
      findComponent(NIter,s,x,&my_func,MI,0,Pi,1,10);

      //      MI[Pi].setAmp(MI[Pi].getAmp()*GAIN);
      //
      // While computing the residual image, use the full PSF
      //
      PSF.resetSize();  
      /*
      ModelDI.resize(0);
      ModelDI = MI.convolve(PSF);
      MImg.assign(0.0);
      MImg   += ModelDI;
      */
      cerr << "###Info: Doing major cycle..." << MI << endl;
      MImg.assign(0.0);
      MImg += MI;
      Convo.convolve(MImg);

      for (int i=0;i<(NX);i++) 
	for (int j=0;j<(NY);j++) 
	  ResImg.setVal(i,j,(DImg(i,j) - GAIN*MImg(i,j)));
	  //	  ResImg.setVal(i,j,(DImg(i,j) - MImg(i,j)));
    }
  //
  // Having found the approx. Fluxon decomposition, go through one pass with the
  // full PSF...
  //
  if (N0==N1)
    {
      int m;
      NP--;

      for(int i=0;i<=N0;i++)
	if (MI.size() > i)
	  {
	    m = PARAMS_PER_FLUXON*i;
	    gsl_vector_set(x, m,   MI[i].getAmp());
	    gsl_vector_set(x, m+1, MI[i].getXPos());
	    gsl_vector_set(x, m+2, MI[i].getYPos());
	    gsl_vector_set(x, m+3, MI[i].getScale());
	  }
      cerr << "###Solving the full problem..." << endl;
      /*
      PSF.setSize(PSFSize);
      ModelDI.resize(0);
      ModelDI = MI.convolve(PSF);
      MImg   += ModelDI;
      */
      MImg.assign(0.0);
      MImg += MI;
      cerr << endl << MI << endl;

      Convo.convolve(MImg);

      for (int i=0;i<(NX);i++) 
	for (int j=0;j<(NX);j++) 
	  ResImg.setVal(i,j,(DImg(i,j) - MImg(i,j)));

      //      x0 = gsl_multimin_fdfminimizer_x(s);
      InitStep = gsl_blas_dnrm2 (x);
      setupSolver(&s,
		  gsl_multimin_fdfminimizer_vector_bfgs,
		  x,
		  &my_func,
		  N0+1,
		  par, 
		  &PIndex, 
		  &ResImg, 
		  &DImg, 
		  &MImg, 
		  &ModelDI, 
		  &MI, 
		  &PSF);
      gsl_multimin_fdfminimizer_set(s, &my_func, x, InitStep, TOL);
      findComponent(NIter,s,x,&my_func,MI,0,NP,1,50);
    }
  if (x)  gsl_vector_free(x);
}
//
//===================================================================
//
void readVal(Client &c, GlishEvent *e, int &N)
{
  N=-1;
  Value *v = e->Val(), *Reply;
  if (v->Type() == TYPE_INT)
    N = v->IntVal();
}
//
//===================================================================
//
void readVal(Client &c, GlishEvent *e, FTYPE &N)
{
  N=-1;
  Value *v = e->Val(), *Reply;
  if ((v->Type() == TYPE_DOUBLE) | (v->Type() == TYPE_FLOAT) | (v->Type() == TYPE_FLOAT))
    N = v->DoubleVal();
}
//
//===================================================================
//
template<class T> int readComps(Client &c, GlishEvent *e, FluxonCompList<T>& CompList)
{
  Value *v = e->Val();
  if (v->Type() == TYPE_STRING)
    {
      std::ifstream ccf(v->StringVal());
      //      std::ifstream ccf("test.fluxon.psf");
      ccf >> CompList;
    }
  return CompList.size();
}
//
//===================================================================
//
template<class T> int writeComps(Client &c, GlishEvent *e, FluxonCompList<T>& CompList)
{
  Value *v = e->Val();
  if (v->Type() == TYPE_STRING)
    {
      std::ofstream ccf(v->StringVal());
      //      std::ifstream ccf("test.fluxon.psf");
      ccf << CompList.size() << endl;
      ccf << CompList;
    }
  return CompList.size();
}
//
//===================================================================
//

template<class T> int makeModel(FluxonCompList<T>& PSF,
				FluxonCompList<T>& CompList,
				IMAGETYPE<T>& DImg)
{
  FluxonCompList<T> DI;

  DImg.assign(0);
  //  DI = CompList.convolve(PSF);
  DI = CompList;
  DImg += DI;
  return DImg.size();
}
//
//===================================================================
//
template<class T> int sendImage(Client &c, GlishEvent *e,
				 IMAGETYPE<T>& Img)
{
  T *ptr;
  ptr = Img.getStorage();
  int Nx, Ny;
  Img.size(Nx,Ny);

  Value *Reply = new Value(ptr, Nx*Ny, COPY_ARRAY);

  c.Reply(Reply);
  Unref(Reply);
  return Img.size();
}
//
//===================================================================
//

template <class T> int readImage(Client &c, GlishEvent *e, IMAGETYPE<T>& Img)
{
  int Nx=0,Ny=0;
  Value *v = e->Val();
  FILE *di;
  T *ptr;

  cerr << "###Got array of size " << v->Length() << endl;
  Img.size(Nx,Ny);
  cerr << "Imsize = " << Nx << "x" << Ny << endl;
  if (Nx*Ny < v->Length())
    {
      cerr << "###Error: Imsize < PSF length" << endl;
      return -1;
    }
  
  ptr =  v->DoublePtr(0);
  for(int i=0;i<Nx;i++)
    for(int j=0;j<Ny;j++)
      Img.setVal(i,j,ptr[i*Ny+j]);

  return Nx;
}
//
//===================================================================
//===================================================================
//
int main(int argc, char **argv)
{
  int NComp=0,N=(int)5E6, ImSize=0, NIter=100;
  IMAGETYPE<FTYPE> TruePSF, DirtyImage, MImg;
  FluxonCompList<FTYPE> CCList,PSF;
  Exp<FTYPE> ExpTab;
  Client c(argc,argv);
  GlishEvent *e;


  PSF.resize(0);
  CCList.resize(0);

  

  while (e=c.NextEvent())
    {
      if     (!strcmp(e->Name(),"ncomp"))   
	{
	  readVal(c, e, NComp);
	  if (NComp < CCList.size())
	    {
	      cerr << "###Error: NComp < CCList.size()!" << endl;
	      NComp = CCList.size();
	    }
	  //	  CCList.resize(NComp);
	}
      else if (!strcmp(e->Name(),"gain"))   // Set the max. no. of iterations
	{
	  readVal(c, e, GAIN);
	}
      else if (!strcmp(e->Name(),"niter"))   // Set the max. no. of iterations
	readVal(c, e, NIter);                
      else if (!strcmp(e->Name(),"sigma"))   // Set the significance level beyond which
	{                                    // the value of individual Fluxons are is 
	                                     // considered to be insignificant (zero).
	  readVal(c, e, DefaultSigma);
	  cerr << "###Info: Setting sigma = " << DefaultSigma << endl;
	  GlobalParams::NSigma=DefaultSigma;
	}
      else if (!strcmp(e->Name(),"scale"))   
	{
	  readVal(c, e, DEFAULT_SCALE);
	}
      else if (!strcmp(e->Name(),"init"))   
	{
	  //	  CCList.resize(0);
	  if ((NComp = readComps(c, e, CCList)))
	    cerr << "###Info: Read " << NComp << " components" << endl;
	}
      else if (!strcmp(e->Name(),"imsize"))  
	{
	  readVal(c, e, ImSize);
	  //
	  // Setup the global namespace parameters.  These are used by
	  // class Fluxon<T>
	  //
	  ExpTab.Build(N,(FTYPE)ImSize/N);           // Build the tabulated exp() function
	  GlobalParams::NSigma=DefaultSigma;         // Defines the support size after which the pixel 
                                                     // model not evaluated
	  GlobalParams::TableOfExponential = ExpTab; // The default table for exp()
	  GlobalParams::XCenter = ImSize/2.0+1;      // The image center pixel.  Required to get the 
	  GlobalParams::YCenter = ImSize/2.0+1;      // analytical form of convolution right.

	  DirtyImage.resize(ImSize,ImSize);  DirtyImage.assign(0.0);
	  MImg.resize(ImSize, ImSize);       MImg.assign(0.0);
	  TruePSF.resize(ImSize, ImSize);    TruePSF.assign(0.0);
	  
	}
      else if (!strcmp(e->Name(),"wpsf"))  
	{
	  cerr << "###No. of Comps. written " << writeComps(c, e, CCList) << endl;
	}
      else if (!strcmp(e->Name(),"psff"))     // Read the Fluxon model PSF from a file.
	{
	  cerr << "###No. of PSF Comps read " << readComps(c, e, PSF) << endl;
	  for (int i=0;i<PSF.size();i++) PSF[i].setAmp(PSF[i].getAmp()/0.379);
	  //	  DEFAULT_SCALE=PSF[0].getScale();
	}
      else if (!strcmp(e->Name(),"truepsf"))     // Read the Fluxon model PSF from a file.
	{
	  readImage(c, e, TruePSF);
	  Convo.makeXFR(TruePSF);
	  //	  DEFAULT_SCALE=PSF[0].getScale();
	}
      //      else if (!strcmp(e->Name(),"rpsf"))  
      //	{
      //	  readImage(c, e, PSFImage);
      //	}
      else if (!strcmp(e->Name(),"mod"))    // Read the pixelated Dirty Image
	{ 
	  readImage(c, e, DirtyImage);
	  CCList.resize(0);
	}
      else if (!strcmp(e->Name(),"conv"))  
	{
	  IMAGETYPE<FTYPE> Img;
	  FluxonCompList<FTYPE> tmp,res;
	  int PSFSize;
	  cerr << "###No. of PSF Comps read " << readComps(c, e, tmp) << endl;

	  //	  PSFSize = PSF.size();
	  //	  PSF.setSize(1);
	  res = PSF.convolve(tmp);
	  cerr << res << endl;
	  Img.resize(ImSize,ImSize);
	  Img.assign(0.0);
	  Img += res;
	  sendImage(c,e,Img);
	  //	  PSF.resetSize();
	}
      else if (!strcmp(e->Name(),"deconv"))  
	{
	  int Restart=0,PSFSize;
	  
	  {int tmp; readVal(c, e, tmp); if (tmp > 0)  PSFSize = tmp;}
	  
	  if (ImSize <= 0)
	    cerr << "###Error: Imsize not set!" << endl;
	  else if (NComp <= 0)
	    cerr << "###Error: NComp not set!" << endl;
	  else
	    {
	      int N0,N1;
	      Value *v = e->Val();
	      if (v->Length() > 0) Restart = v->IntVal();
	      N0=0;N1=NComp;

	      if (Restart)
		{
		  N0=CCList.size();
		  N1=NComp;
		  cerr << "  Restarting at " << N0 << " " << N1 << endl;
		}
	      //	      template <class T> void ssdeconvolve(int NP, 
	      //						   int NIter, 
	      //						   FluxonCompList<FTYPE> &PSF, 
	      //						   IMAGETYPE<FTYPE>      &DImg, 
	      //						   FluxonCompList<FTYPE> &MI,
	      //						   int N0, int N1,int PSFSize)
	      ssdeconvolve<FTYPE>(NComp, NIter, TruePSF, PSF, DirtyImage, CCList, N0, N1,PSFSize);
	      //
	      //...and finally, compute the model Dirty Image
	      //
	      {
		IMAGETYPE<FTYPE> PSFImg;
		FluxonCompList<FTYPE> ModelDI;
		
		PSFImg.resize(ImSize,ImSize);		PSFImg.assign(0.0);
		MImg.resize(ImSize,ImSize);		MImg.assign(0.0);

		PSF.resetSize();  
		ModelDI = CCList.convolve(PSF);
		MImg += ModelDI;
		PSFImg += PSF;
	      }
	      cerr << "...done!" << endl;
	    }
	}
      else if (!strcmp(e->Name(),"mi"))    // Return the current model dirty image
	{
	  IMAGETYPE<FTYPE> tmp;

	  tmp.resize(ImSize,ImSize);
	  //	  tmp = PSF.convolve(CCList);
	  tmp.assign(0);
	  tmp += CCList;
	  Convo.convolve(tmp);

	  sendImage(c,e,tmp);
	}
      else if (!strcmp(e->Name(),"res"))
	{
	  IMAGETYPE<FTYPE> tmp;
	  tmp.resize(ImSize,ImSize);
	  tmp.assign(0);
	  tmp += CCList;
	  Convo.convolve(tmp);
	  for(int i=0;i<ImSize;i++)
	    for(int j=0;j<ImSize;j++)
	      tmp.setVal(i,j,(DirtyImage(i,j)-tmp(i,j)));
	  sendImage(c,e,tmp);
	}
      else if (!strcmp(e->Name(),"ci"))    // Return the current Clean Component image.
	{
	  IMAGETYPE<FTYPE> Img;
	  Img.resize(ImSize,ImSize);	  Img.assign(0.0);
	  Img += CCList;
	  sendImage(c,e,Img);
	}
      else if (!strcmp(e->Name(),"di"))    // Return the true dirty image.
	{
	  sendImage(c,e,DirtyImage);
	}
      else if (!strcmp(e->Name(),"psfcomp"))   
	{
	  writeComps(c, e, CCList);
	}
      else if (!strcmp(e->Name(),"modpsf"))   // Return the Fluxon approixmation of the PSF
	{
	  //	  std::ofstream fd("modpsf.dat");
	  IMAGETYPE<FTYPE> PSFImg;

	  PSFImg.resize(ImSize,ImSize);
	  PSFImg.assign(0.0);
	  PSFImg += PSF;
	  sendImage(c,e,PSFImg);
	  //	  fd << PSFImg;
	}
      else if (!strcmp(e->Name(),"makedi"))   
	{
	  FluxonCompList<FTYPE> tCC,tt;
	  IMAGETYPE<FTYPE> tDI;
	  cerr << "###No. of PSF Comps read " << readComps(c, e, tCC) << endl;
	  tDI.resize(ImSize,ImSize);
	  tDI.assign(0.0);
	  tt = PSF.convolve(tCC);
	  tDI += tt;
	  sendImage(c,e,tDI);
	}
      else if (!strcmp(e->Name(),"done")) {cerr << "..bye" << endl;return 0;}
      else if (!strcmp(e->Name(),"smooth")) 
	{
	  IMAGETYPE<FTYPE> tmpr;
	  {
	    tmpr.resize(ImSize,ImSize);
	    tmpr.assign(0);
	    tmpr += CCList;
	    Convo.convolve(tmpr);
	    for(int i=0;i<ImSize;i++)
	      for(int j=0;j<ImSize;j++)
		tmpr.setVal(i,j,(DirtyImage(i,j)-tmpr(i,j)));
	    cerr << "###Info: Residuals made!" << endl;
	  }

	  IMAGETYPE<FTYPE> tmp, tSm;
	  tmp.resize(ImSize,ImSize);
	  tSm.resize(ImSize,ImSize);
	  Fluxon2D<FTYPE> tf;
	  tf = PSF[0];
	  tf.setScale(PSF[0].getScale()/2.0);
	  tf.setAmp(tf.getScale()/tf.getAmp());
	  cerr << "###Info: Smoothing residuals to " << tf.getScale() << " scale" << endl;
	  tmp.assign(0);
	  tmp += tf;
	  Convolver<FTYPE> tConvo(tmp);
	  tSm = tmpr;
	  tConvo.convolve(tSm);
	  sendImage(c,e,tSm);
	}
      else c.Unrecognized();
    }

  cerr << "Exiting..." << endl;
  return 0;
}
