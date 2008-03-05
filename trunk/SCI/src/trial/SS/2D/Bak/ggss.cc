#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Glish/Client.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <FluxonCompList.h>
#include <vector.h>
#include <Array2D.h>
#include <Exp.h>
#include <ErrorObj.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_blas.h>
#include <Params.h>

#define TOL 1E-6
#define PTOL 1E-02
#define FIRST_STEP 0.1
#define EDGE 100
#define GAIN 1.0

FTYPE DEFAULT_SCALE=0.01;

template <class T> T getDefaultScale() {return DEFAULT_SCALE;};

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
  MImg.assign(0.0);
  MImg   += ModelDI;
  //  Normalize(MImg,ModelDI);
  N = MImg.size();
  
  for(int i=EDGE;i<(N-EDGE);i++) 
    for(int j=EDGE;j<(N-EDGE);j++) 
      ResImg.setVal(i,j,(DImg(i,j) - MImg(i,j)));
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static T my_f (const gsl_vector *v, void *params)
{
  T ChiSq=0,A,P,S;
  int N, k;
  IMAGETYPE<FTYPE> *ResImg, *DImg, *MImg;
  vector<int> *Pj;
  FluxonCompList<FTYPE> *ModelDI, *MComp, *PSF;
  gsl_multimin_fdfminimizer *s;
  
  //
  // Extract the params from the void pointer...
  //
  Params<FTYPE> MyP(params);
  ResImg  = MyP.ResImg();
  MComp   = MyP.ModComp();
  s       = MyP.Minimizer();
  Pj      = MyP.CompList();
  DImg    = MyP.DImg();
  MImg    = MyP.MImg();
  ModelDI = MyP.ModelDI();
  PSF     = MyP.PSF();
  //
  // Make the model and residual image at the current location on the
  // Chisq surface
  //
  N = (*Pj).size();
  //  cerr << "#### " << (*PSF).size() << endl;

  k=0;
  for(int j=0;j<N;j++)
    if ((*Pj)[j] == 1)
    {
      int m;

      //      k=j;
      m=3*k;
      A = gsl_vector_get(v,m);
      P = gsl_vector_get(v,m+1);
      S = gsl_vector_get(v,m+2);
      (*MComp)[k].setAmp(A);
      (*MComp)[k].setPos(P);
      (*MComp)[k].setScale(S);
      k++;
    }

  (*ModelDI) = (*MComp).convolve((*PSF));

  MakeImages(*ModelDI, *DImg, *MImg, *ResImg);

  N = (*MImg).size();

  for (int i=EDGE;i<(N-EDGE);i++) 
    {
      ChiSq += (*ResImg)[i]*(*ResImg)[i];
    }      
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
      m=3*k;

      cerr << "my_f: " << k << " " << (*MComp)[k] << " " << ChiSq << " " 
	   << fabs(ChiSq-gsl_multimin_fdfminimizer_minimum(s)) << " "
	   << gsl_vector_get(dp0,m) << " " << gsl_vector_get(dp0,m+1) << " " 
	   << gsl_vector_get(dp0,m+2) << endl;
    }
  */
  return ChiSq;
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static void my_df (const gsl_vector *v, void *params, gsl_vector *g)
{
  T dA, dP, dS, A, P, S, tmp;
  int N,NP,k;
  IMAGETYPE<FTYPE> *ResImg;
  vector<int> *Pj;
  FluxonCompList<FTYPE> *MComp;
  //
  // Recover the pointers to the parameters... :-| (Phew!)
  //
  Params<FTYPE> MyP(params);
  ResImg = MyP.ResImg();
  MComp  = MyP.ModComp();
  Pj     = MyP.CompList();

  N = ResImg->size();

  NP = (*Pj).size();
  k=0;
  for(int j=0;j<NP;j++)
    if ((*Pj)[j] == 1)
      {
	int m;
	//	k=j;
	
	A = (*MComp)[k].getAmp();
	P = (*MComp)[k].getPos();
	S = fabs((*MComp)[k].getScale());

	dA = dP = dS = 0.0;
	for (int i=0;i<N;i++)
	  {
	    tmp = (*ResImg)[i]*(*MComp)[k](i);
	    dA += tmp; // Sum{P_k . ResImg}
	    tmp *= (i-P);
	    dP += tmp;                  // Sum{(x-x_k) . P_k . ResImg}
	    dS += (i-P)*tmp;            // Sum{(x-x_k)^2 . P_k . ResImg}
	    /*
	    dA += (tmp = (*ResImg)[i] * (*MComp)[k](i)); // Sum{P_k . ResImg}
	    dP += (tmp *= (i-P));                        // Sum{(x-x_k) . P_k . ResImg}
	    dS += (i-P)*tmp;                             // Sum{(x-x_k)^2 . P_k . ResImg}
	    */
	  }

	dA = -2.0*dA/A;
	dP = -2.0*dP*S*S*10;
	dS =  2.0*dS*S;

	m=3*k;
	gsl_vector_set(g,m  ,dA);
	gsl_vector_set(g,m+1,dP);
	gsl_vector_set(g,m+2,dS);
	k++;
	//	cerr << "G: " << dA << " " << dP << " " << dS << " " << (*MComp)[k] << endl;
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
template<class T> vector<T> getPeak(IMAGETYPE<FTYPE>& Img, vector<int>& PosX,vector<int>& PosX)
{
  int nx,ny, NPeaks;
  T P=EDGE;
  NPeaks = 0;
  vector<T> Peaks;

  Img.size(nx,ny);
  Peaks.resize(NPeaks+1);
  PosX.resize(NPeaks+1);
  PosX.resize(NPeaks+1);
  PosX[0]=PosY[0]=EDGE;

  //  for(int i=EDGE;i<(n-EDGE);i++) if (fabs(Img[i]) > fabs(Img[Pos[0]]))  
  for(int i=nx/4;i<3*nx/4;i++) 
    for(int j=ny/4;j<3*ny/4;j++) 
      if (fabs(Img(i,j)) > fabs(Img(PosX[0],PosY[0])))  
	{PosX[NPeaks] = i; PosY[NPeaks] = j;}
  /*
  for(int i=EDGE;i<(n-EDGE);i++) if (fabs(Img[i] - Img[Pos[0]]) < PTOL) 
    {
      cerr << "###No. of peaks = " << Img[i] << " " << i << endl;
      NPeaks++;
      Peaks.resize(NPeaks+1);
      Pos.resize(NPeaks+1);
      Pos[NPeaks] = i;
    }
  */
  for(int i=0;i<Pos.size();i++)
    Peaks[i] = (Img[Pos[i]]);

  return Peaks;
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> void setupSolver(gsl_multimin_fdfminimizer **Solver, 
				   const gsl_multimin_fdfminimizer_type *MinimizerType,
				   gsl_vector *x,
				   gsl_multimin_function_fdf *my_func,
				   int NP,
				   void *par[],
				   vector<int> *PIndex,
				   vector<T> *ResImg,
				   vector<T> *DImg,
				   vector<T> *MImg,
				   FluxonCompList<T> *ModelDI,
				   FluxonCompList<T> *MI,
				   FluxonCompList<T> *PSF
				   )
{
  int NParams;

  NParams = 3*(NP+1);

  //
  // Tell GSL to use only first NParams out of 
  // 3*NP to solve the current problem
  //
  x->size = NParams;

  if (*Solver) gsl_multimin_fdfminimizer_free(*Solver);
  *Solver = gsl_multimin_fdfminimizer_alloc(MinimizerType, NParams);

  //  (*PIndex)[NP] = NP;

  par[0]         = (void *)ResImg;
  par[1]         = (void *)MI;
  par[2]         = (void *)*Solver;
  par[3]         = (void *)PIndex;
  par[4]         = (void *)DImg;
  par[5]         = (void *)MImg;
  par[6]         = (void *)ModelDI;
  par[7]         = (void *)PSF;

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
  T InitStep;
  T A,P,S;
      
  FTYPE Min0=0, Min=0;
  do
    {
      if ((iter % RestartAt) == 0)
	{
	  cerr << "Restarting..." << endl;
	  
	  x0 = gsl_multimin_fdfminimizer_x(s);
	  InitStep = gsl_blas_dnrm2 (x0);
	  gsl_multimin_fdfminimizer_set(s, func, x0, InitStep, TOL);

	  gsl_multimin_fdfminimizer_restart(s);
	}

      //
      // Get the current position and gradients...
      //
      x0 = gsl_multimin_fdfminimizer_x(s);
      p0 = gsl_multimin_fdfminimizer_gradient(s);

      if (DoInit)
	for (int j=P0;j<=Pi;j++)
	  {
	    m=3*(j-P0);
	    m=3*(j);
	    A = gsl_vector_get(x0,m);
	    P = gsl_vector_get(x0,m+1);
	    S = gsl_vector_get(x0,m+2);
	    MI[j].setAmp(A);	  
	    MI[j].setPos(P);	  
	    MI[j].setScale(S);
	  }
      
      Min0 = Min;	  Min = gsl_multimin_fdfminimizer_minimum(s);
      for (int j=P0;j<=Pi;j++)
	{
	  int mm;
	  mm=3*(j-P0);
	  cerr << iter                    << " "
	       << Min                     << " "
	       << gsl_blas_dnrm2(s->dx)   << " "
	       << j                       << " "
	       << MI[j]                   << " "
	       << gsl_vector_get(p0,mm)   << " "
	       << gsl_vector_get(p0,mm+1) << " "
	       << gsl_vector_get(p0,mm+2) << " "
	       << endl;
	}

      //
      // Make the move!
      //
      status = gsl_multimin_fdfminimizer_iterate(s);
      if (status)  
	{
	  cerr << "###Info: " << gsl_strerror(status) << endl;
	  //	  if (status == GSL_ENOPROG) gsl_multimin_fdfminimizer_restart(s);
	  break;
	}
      status = gsl_multimin_test_gradient(s->gradient, 1E-3);
      
      p0 = gsl_multimin_fdfminimizer_gradient(s);
      
      iter++;
    }
  while(status == GSL_CONTINUE && iter < NIter);// && fabs(Min-Min0) > TOL);
  cerr << "###Info: " << gsl_strerror(status) << endl;
  
  return status;
}
//
//---------------------------------------------------------------------------------------------
//
template <class T> void ssdeconvolve(int NP, int NIter, FluxonCompList<T>& PSF, 
				     IMAGETYPE<T>& DImg, FluxonCompList<T>& MI,
				     int N0, int N1,int PSFSize)
{
  int N=1, ImSize;
  FTYPE Normalization, CleanBeamScale, InitStep;
  vector<FTYPE> Peak,PSFImg,MImg,ResImg;
  vector<int> Pos;
  FluxonCompList<FTYPE> ModelDI;
  Fluxon<FTYPE> f0;
  vector<int> PIndex;

  ImSize = DImg.size();

  PSFImg.resize(ImSize);
  MImg.resize(ImSize);  
  ResImg.resize(ImSize);
  PSFImg.assign(PSFImg.size(),0.0);
  MImg.assign(PSFImg.size(),0.0);
  
  PIndex.resize(NP);
  PIndex.assign(PIndex.size(),-10);

  if (MI.size() > 0)
    {
      cerr << "Computing model data..." << endl;
      ModelDI = MI.convolve(PSF);
      MImg   += ModelDI;
      
      for (int i=0;i<MI.size();i++) PIndex[i]=1;
    }
  else
    MI.resize(0);

  CleanBeamScale = PSF[0].getScale();

  //
  // Convert all component images to normal images
  //
  PSFImg += PSF;

  cerr << "Computing residual..." << endl;
  for (int i=0;i<ImSize;i++) ResImg[i] = DImg[i] - MImg[i];

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
  x = gsl_vector_alloc(3*NP);
  gsl_vector_set_zero(x);
  
  gsl_vector *x0;
  static void* par[8];
  int status,m,NParams;
  //  const gsl_multimin_fdfminimizer_type *MT;
  static gsl_multimin_fdfminimizer *s=NULL,*s1=NULL;
  static gsl_multimin_function_fdf my_func,one_fluxon;


  Normalization  = getDefaultScale<T>()*getDefaultScale<T>();
  Normalization  = sqrt(2*M_PI/(CleanBeamScale*CleanBeamScale + Normalization));
        
  for (int Pi=N0; Pi<N1; Pi++)
    {
      size_t iter = 0;

      NParams = 3*(Pi+1);

      // T = gsl_multimin_fdfminimizer_conjugate_fr;
      // T = gsl_multimin_fdfminimizer_conjugate_pr;
      // T = gsl_multimin_fdfminimizer_steepest_descent;
      // T = gsl_multimin_fdfminimizer_vector_bfgs;
      setupSolver(&s, gsl_multimin_fdfminimizer_vector_bfgs,
		  x,&my_func,Pi,par, &PIndex, &ResImg, &DImg, 
		  &MImg, &ModelDI, &MI, &PSF);
      cerr << "Using " << gsl_multimin_fdfminimizer_name(s) << " minimization." << endl;
      //
      // Set the initial guess
      //
      for(int i=0;i<=Pi;i++)
	if (MI.size() > i)
	  {
	    m = 3*i;
	    gsl_vector_set(x, m,   MI[i].getAmp());
	    gsl_vector_set(x, m+1, MI[i].getPos());
	    gsl_vector_set(x, m+2, MI[i].getScale());
	  }
      //
      // Get the peak and it's position from the residual image.
      // Setup a Fluxon with these values and add to the model
      // component image (MI).
      //
      //      if (!Restart)
      {
	gsl_vector *tx=NULL;
	
	
	tx = gsl_vector_alloc(3);

	Peak = getPeak(ResImg, Pos);
	for (int i=0;i<Peak.size();i++) Peak[i] /= PSF[0].getAmp()*Normalization;
	cerr << "Peak = " << Peak[0] << " " << '@' << Pos[0] << endl;

	//f0.setScale(CleanBeamScale);
	f0.setScale(getDefaultScale<T>());
	f0.setAmp((Peak[0]));
	f0.setPos(Pos[0]);

	m = 3*Pi;
	gsl_vector_set(x, m,   f0.getAmp());
	gsl_vector_set(x, m+1, f0.getPos());
	gsl_vector_set(x, m+2, f0.getScale());

	gsl_vector_set(tx, 0, f0.getAmp());
	gsl_vector_set(tx, 1, f0.getPos());
	gsl_vector_set(tx, 2, f0.getScale());
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
	ModelDI.resize(0);
	ModelDI = MI.convolve(PSF);
	MImg.assign(MImg.size(),0.0);
	MImg   += ModelDI;
	//
	// Use only the main lobe of the PSF
	//
	/*
	cerr << "Get initial guess..." << endl;
	setupSolver(&s1,gsl_multimin_fdfminimizer_vector_bfgs,
		    tx,&one_fluxon,0,par, &PIndex, &ResImg, &DImg, 
		    &MImg, &ModelDI, &MI, &PSF);
	PSF.setSize(100);
	InitStep = gsl_blas_dnrm2 (tx);
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;
	for (int i=0;i<PIndex.size();i++) if (PIndex[i] == 1) PIndex[i]=-1;
	PIndex[Pi]=1;
	gsl_multimin_fdfminimizer_set(s1, &one_fluxon, tx, InitStep, TOL);
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;
	findComponent(NIter,s1,tx,&one_fluxon,MI,Pi,Pi,1,50);
	for (int i=0;i<PIndex.size();i++) if (PIndex[i] == -1) PIndex[i]=1;
	//	for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;

	m = 3*Pi;
	x0 = gsl_multimin_fdfminimizer_x(s1);
	gsl_vector_set(x, m,   gsl_vector_get(x0,0));
	gsl_vector_set(x, m+1, gsl_vector_get(x0,1));
	gsl_vector_set(x, m+2, gsl_vector_get(x0,2));
	*/
	if (tx) gsl_vector_free(tx);
      }
      //
      // Find the Pi th. component.  This has the minimization loop
      //
      for (int i=0;i<PIndex.size();i++) cerr << PIndex[i] << " ";cerr <<endl;
      setupSolver(&s,gsl_multimin_fdfminimizer_vector_bfgs,
		  x,&my_func,Pi,par, &PIndex, &ResImg, &DImg, 
		  &MImg, &ModelDI, &MI, &PSF);
      //      PSF.resetSize();  
      InitStep = gsl_blas_dnrm2 (x);
      gsl_multimin_fdfminimizer_set(s, &my_func, x, InitStep, TOL);
      findComponent(NIter,s,x,&my_func,MI,0,Pi,1,50);
      //
      // While computing the residual image, use the full PSF
      //
      PSF.resetSize();  
      ModelDI.resize(0);
      ModelDI = MI.convolve(PSF);
      MImg.assign(MImg.size(),0.0);
      MImg   += ModelDI;
      for (int i=EDGE;i<(ImSize-EDGE);i++) ResImg[i] = DImg[i] - GAIN*MImg[i];
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
	    m = 3*i;
	    gsl_vector_set(x, m, MI[i].getAmp());
	    gsl_vector_set(x, m+1, MI[i].getPos());
	    gsl_vector_set(x, m+2, MI[i].getScale());
	  }
      cerr << "Solving the full problem..." << endl;
      PSF.setSize(PSFSize);
      ModelDI.resize(0);
      ModelDI = MI.convolve(PSF);
      MImg.assign(MImg.size(),0.0);
      MImg   += ModelDI;
      for (int i=EDGE;i<(ImSize-EDGE);i++) ResImg[i] = DImg[i] - MImg[i];

      x0 = gsl_multimin_fdfminimizer_x(s);
      InitStep = gsl_blas_dnrm2 (x0);
      gsl_multimin_fdfminimizer_set(s, &my_func, x0, InitStep, TOL);
      findComponent(NIter,s,x0,&my_func,MI,0,NP,1,50);
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
  if (v->Type() == TYPE_DOUBLE)
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
      ifstream ccf(v->StringVal());
      ccf >> CompList;
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

  DImg.assign(DImg.size(),0);
  DI = CompList.convolve(PSF);
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
  ptr = &Img[0];

  Value *Reply = new Value(ptr, Img.size(), COPY_ARRAY);

  c.Reply(Reply);
  Unref(Reply);
  return Img.size();
}
//
//===================================================================
//

template <class T> int readImage(Client &c, GlishEvent *e, IMAGETYPE<T>& Img)
{
  int N=0;
  Value *v = e->Val();
  if (v->Type() == TYPE_STRING)
    {
      ifstream di(v->StringVal());
      di >> N;
      Img.resize(N);
      for (int i=0;i<N;i++)
	di >> Img[i];
    }
  return N;
}
//
//===================================================================
//===================================================================
//
int main(int argc, char **argv)
{
  int NComp=0,N=5000000, ImSize=0, NIter=100,PSFSize=100;
  IMAGETYPE<FTYPE> DirtyImage, MImg;
  FluxonCompList<FTYPE> PSF, CompList;
  Exp<FTYPE> ExpTab;
  Client c(argc,argv);
  GlishEvent *e;


  PSF.resize(0);
  CompList.resize(0);

  while (e=c.NextEvent())
    {
      if     (!strcmp(e->Name(),"ncomp"))   
	{
	  readVal(c, e, NComp);
	  if (NComp < CompList.size())
	    {
	      cerr << "###Error: NComp < CompList.size()!" << endl;
	      NComp = CompList.size();
	    }
	}
      else if (!strcmp(e->Name(),"niter"))   // Set the max. no. of iterations
	readVal(c, e, NIter);                
      else if (!strcmp(e->Name(),"psfsize")) // Set the max. no. PSF components to be used  
	readVal(c, e, PSFSize);
      else if (!strcmp(e->Name(),"sigma"))   // Set the significance level beyond which
	{                                    // the value of individual Fluxons are is 
	  FTYPE Sigma;                       // considered to be insignificant (zero).
	  readVal(c, e, Sigma);
	  GlobalParams::NSigma=Sigma;
	}
      else if (!strcmp(e->Name(),"init"))    // Reset the Clean Comp. list.
	{
	  CompList.resize(0);
	}
      else if (!strcmp(e->Name(),"imsize"))  // Set the image size.
	{
	  readVal(c, e, ImSize);
	  //
	  // Setup the global namespace parameters.  These are used by
	  // class Fluxon<T>
	  //
	  ExpTab.Build(N,(FTYPE)ImSize/N);           // Build the tabulated exp() function
	  GlobalParams::NSigma=7.0;                  // Defines the support size after which the pixel 
                                                     // model not evaluated
	  GlobalParams::TableOfExponential = ExpTab; // The default table for exp()
	  GlobalParams::Center = ImSize/2.0;         // The image center pixel.  Required to get the 
	                                             // analytical form of convolution right.
	  DirtyImage.resize(ImSize);
	  MImg.resize(ImSize);
	  DirtyImage.assign(ImSize,0.0);
	  MImg.assign(ImSize,0.0);
	}
      else if (!strcmp(e->Name(),"scale"))    // Set the default scale to be used.
	readVal(c, e, DEFAULT_SCALE);
      else if (!strcmp(e->Name(),"psff"))     // Read the pixelated PSF from a file.
	{
	  cerr << "No. of PSF Comps read " << readComps(c, e, PSF) << endl;
	  DEFAULT_SCALE=PSF[0].getScale();
	}
      else if (!strcmp(e->Name(),"modf"))     // Read the model from a file and make a
	{                                     // Dirty image.
	  cerr << "No. of model comps read " << readComps(c, e, CompList) << endl;
	  if (PSF.size() <= 0)
	    cerr << "###Error: PSF not loaded" << endl;
	  else
	    {
	      makeModel(PSF,CompList,DirtyImage);
	      CompList.resize(0);
	    }
	}
      else if (!strcmp(e->Name(),"rddif"))   // Read the pixelated Dirty image from a file.
	{
	  readImage(c,e,DirtyImage);
	}
      else if (!strcmp(e->Name(),"deconv"))  // Do the deconvolution
	{
	  if (ImSize <= 0)
	    cerr << "###Error: Imsize not set!" << endl;
	  else if (NComp <= 0)
	    cerr << "###Error: NComp not set!" << endl;
	  else
	    {
	      int Restart=0,N0,N1;
	      Value *v = e->Val();
	      if (v->Length() > 0)
		Restart = v->IntVal();
	      N0=0;N1=NComp;
	      if (Restart)
		{
		  N0=CompList.size();
		  N1=NComp;
		  cerr << "  Restarting at " << N0 << " " << N1 << endl;
		}
	      ssdeconvolve(NComp, NIter, PSF, DirtyImage, CompList,N0,N1,PSFSize);
	      //
	      //...and finally, compute the model Dirty Image
	      //
	      {
		IMAGETYPE<FTYPE> PSFImg;
		FluxonCompList<FTYPE> ModelDI;
		
		PSFImg.resize(ImSize);
		MImg.resize(ImSize);
		
		PSFImg.assign(ImSize,0.0);
		MImg.assign(ImSize,0.0);
		
		PSF.resetSize();  
		ModelDI = CompList.convolve(PSF);
		MImg += ModelDI;
		PSFImg += PSF;
	      }
	      cerr << "...done!" << endl;
	    }
	}
      else if (!strcmp(e->Name(),"mi"))    // Return the current model dirty image
	{
	  sendImage(c,e,MImg);
	}
      else if (!strcmp(e->Name(),"ci"))    // Return the current Clean Component image.
	{
	  IMAGETYPE<FTYPE> Img;
	  Img.resize(ImSize);
	  Img.assign(ImSize,0.0);
	  Img += CompList;
	  sendImage(c,e,Img);
	}
      else if (!strcmp(e->Name(),"di"))    // Return the true dirty image.
	{
	  sendImage(c,e,DirtyImage);
	}
      else if (!strcmp(e->Name(),"psf"))   // Return the PSF in use.
	{
	  IMAGETYPE<FTYPE> PSFImg;
	  PSFImg.resize(ImSize);
	  PSFImg.assign(ImSize,0.0);
	  PSFImg += PSF;
	  sendImage(c,e,PSFImg);
	}
      /*
	else if (!strcmp(e->Name(),"ccomp"))   
	sendCC(c,e,CompList);
      */
      else if (!strcmp(e->Name(),"done")) return 0;
      else c.Unrecognized();
    }

  cerr << "Exiting..." << endl;
  return 0;
}
