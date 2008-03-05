#include <stdio.h>
#include <string.h>
#include <math.h>
#include "Glish/Client.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <FluxonCompList.h>
#include <vector.h>
#include <Exp.h>
//#include <ErrorObj.h>
#include <Convolver.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_blas.h>
#include <lmder_Params.h>

#define TOL 1E-4
#define PTOL 1E-02
#define FIRST_STEP 0.1
#define EDGE 20
#define GAIN 1.0

int PARAMS_PER_FLUXON;
FTYPE DEFAULT_SCALE=0.1;

template<class T> T getPeak(IMAGETYPE<T>& Img, int& PosX,int &PosY);

void err_handler(const char * reason, const char * file, int line, int gsl_errno)
{
  cerr << "###Error: " << reason << " " << " in file " << file << " at line no. " << line << endl;
}

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
  N = ModelDI.size();

  //
  // ModelDI is only the latest Fluxon being fitted.
  //
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
template <class T> static int my_f (const gsl_vector *v, void *params, gsl_vector *f)
{
  //
  // 'v' are the parameters we are looking for (Amp, Location, Scale, etc.)
  // 'params' are, for some reason, a pointer to a structure which will hold
  // various other data which is needed to compute the chisq (like the data itself!).
  // 'f' is the residual computed at each point!
  //
  T ChiSq=0,A,Px,Py,S;
  int N, k;
  IMAGETYPE<T> *ResImg, *DImg, *MImg;
  vector<int> *Pj;
  FluxonCompList<T> *ModelDI, *MComp, *PSF;
  gsl_multifit_fdfsolver *s;
  
  //
  // Extract the params from the void pointer...
  //
  Params<FTYPE> MyP(params);
  ResImg  = MyP.ResImg();
  MComp   = MyP.ModComp(); // OneComp
  s       = MyP.Minimizer();
  Pj      = MyP.CompList();
  DImg    = MyP.DImg(); // PSFImage
  MImg    = MyP.MImg(); // ModelPSFImg
  ModelDI = MyP.ModelDI();
  PSF     = MyP.PSF();
  //
  // Make the model and residual image at the current location on the
  // Chisq surface
  //
  N = (*Pj).size();

  k=0;
  for(int j=0;j<N;j++)
    if ((*Pj)[j] == 1)
    {
      int m;

      //      k=j;
      m=PARAMS_PER_FLUXON*k;
      A  = gsl_vector_get(v,m);
      Px = gsl_vector_get(v,m+1);
      Py = gsl_vector_get(v,m+2);
      S  = gsl_vector_get(v,m+3);

      (*MComp)[k].setAmp(A);
      (*MComp)[k].setXPos(Px);
      (*MComp)[k].setYPos(Py);
      (*MComp)[k].setScale(S);
      //      cerr << "Params in my_f: " << (*MComp)[k] << endl;
      k++;
    }

  N = (*MImg).size();
  //  (*MImg).assign(0.0);
  //  (*MImg) += (*MComp);
  MakeImages(*MComp, *DImg, *MImg, *ResImg);


  //
  // Set the the value ResImg[i,j] in a linear gsl_vector 'f'.
  // The mapping of the co-ordinate (i,j) to the linear index
  // of f is given by ResImg.mapNdx(i,n).
  //
  for (int i=EDGE;i<(N-EDGE);i++) 
    for (int j=EDGE;j<(N-EDGE);j++) 
      gsl_vector_set(f,(*ResImg).mapNdx(i,j),(*ResImg)(i,j));

  //
  // Debugging code...
  //
  /*
  gsl_vector *dp0;
  dp0 = gsl_multifit_fdfsolver_gradient(s);

  N = (*Pj).size();
  for(int j=0;j<N;j++)
    if ((k = (*Pj)[j]) > -1)
    {
      int m;
      m=PARAMS_PER_FLUXON*k;

      cerr << "my_f: " << k << " " << (*MComp)[k] << " " << ChiSq << " " 
	   << fabs(ChiSq-gsl_multimin_fdfsolver_minimum(s)) << " "
	   << gsl_vector_get(dp0,m) << " " << gsl_vector_get(dp0,m+1) << " " 
	   << gsl_vector_get(dp0,m+2) << endl;
    }
  */
  //  cerr << "ChiSq: " << ChiSq << endl;

  return GSL_SUCCESS;
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static int my_df (const gsl_vector *v, void *params, gsl_matrix *J)
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
  NP=(*MComp).size();
  //  cerr << "Params in my_df: " << (*MComp)[k] << " " << NP << endl;

  for(int j=0;j<NP;j++)
    //    if ((*Pj)[j] == 1)
      {
	int m,XRange0,XRange1,YRange0,YRange1;
	Fluxon2D<T> tF;
	//	k=j;
	
	//	int junk;
	//	T tt;
	//	junk=0;
	AGAIN: dA = dPx = dPy = dS = 0.0;
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

		/*
		if (junk==1) 
		  {
		    tt=tF(i,j);
		    if (tt != 0) cerr << i << " " << j << " " << tt << endl;
		  }
		*/
		tF.evalDerivatives(i,j);
	
		//		tmp = (*ResImg)(i,j);
		dA  = tF.dF(tF.DA);
		dPx = tF.dF(tF.DXP);
		dPy = tF.dF(tF.DYP);
		dS  = tF.dF(tF.DS);
		gsl_matrix_set(J, (*ResImg).mapNdx(i,j), 0, dA);
		gsl_matrix_set(J, (*ResImg).mapNdx(i,j), 1, dPx);
		gsl_matrix_set(J, (*ResImg).mapNdx(i,j), 2, dPy);
		gsl_matrix_set(J, (*ResImg).mapNdx(i,j), 3, dS);
	      }
	  }

	//	if (dA==0) {junk=1;goto AGAIN;}
	m=PARAMS_PER_FLUXON*k;
	/*
	gsl_vector_set(g,m  ,dA);
	gsl_vector_set(g,m+1,dPx);
	gsl_vector_set(g,m+2,dPy);
	gsl_vector_set(g,m+3,dS);
	*/
	//	cerr << "G: " << dA << " " << dPx << " " << dPy << " " << dS << " " << (*MComp)[k] << endl;
	k++;
      }
  return GSL_SUCCESS;
}
//
//---------------------------------------------------------------------------------------
//
template <class T> static int my_fdf (const gsl_vector *v, void *params, gsl_vector *f, gsl_matrix *J)
{
  my_f<T>(v,params,f);
  my_df<T>(v,params,J);
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> T getPeak(IMAGETYPE<T>& Img, int& PosX, int &PosY)
{
  int n=Img.size();
  T P=EDGE;
  PosX=PosY=EDGE;

  for(int i=EDGE;i<(n-EDGE);i++) 
    for(int j=EDGE;j<(n-EDGE);j++) 
      if (fabs(Img(i,j)) > fabs(Img(PosX,PosY)))  
	{PosX = i; PosY=j;}
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

  return Img(PosX,PosY);
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> void setupSolver(gsl_multifit_fdfsolver **Solver, 
				   const gsl_multifit_fdfsolver_type *MinimizerType,
				   gsl_vector *x,
				   gsl_multifit_function_fdf *my_func,
				   int NPixons,
				   void *par[],
				   vector<int> *PIndex,
				   IMAGETYPE<T> *ResImg,
				   IMAGETYPE<T> *DImg, //PSFImage
				   IMAGETYPE<T> *MImg, //ModelPSFImg
				   FluxonCompList<T> *ModelDI, // NULL
				   FluxonCompList<T> *FCList,  // OneComp
				   FluxonCompList<T> *PSFList  // NULL
				   )
{
  size_t NParams, NData;

  NParams = PARAMS_PER_FLUXON*(NPixons);
  NData   = (*ResImg).size(); NData *= NData;

  //
  // Tell GSL to use only first NParams out of 
  // 3*NP to solve the current problem
  //
  x->size = NParams;

  if (*Solver) gsl_multifit_fdfsolver_free(*Solver);
  //  NData = 100;
  *Solver = gsl_multifit_fdfsolver_alloc(MinimizerType, NData, NParams);
  //  *Solver = gsl_multifit_fdfsolver_alloc(gsl_multifit_fdfsolver_lmder,NData, NParams);

  //  cerr << "Using " << gsl_multifit_fdfsolver_name(*Solver) << " minimization" << " for " << NParams << " params" << endl;

  //  (*PIndex)[NP] = NP;

  par[0]         = (void *)ResImg;
  par[1]         = (void *)FCList;
  par[2]         = (void *)*Solver;
  par[3]         = (void *)PIndex;
  par[4]         = (void *)DImg;
  par[5]         = (void *)MImg;
  //  par[6]         = (void *)ModelDI;
  //  par[7]         = (void *)PSFList;

  my_func->f      = &my_f<FTYPE>;
  my_func->df     = &my_df<FTYPE>;
  my_func->fdf    = &my_fdf<FTYPE>;
  my_func->n      = NData;
  my_func->p      = NParams;
  my_func->params = par;
}
//
//---------------------------------------------------------------------------------------------
//
template<class T> int findComponent(int NIter, gsl_multifit_fdfsolver *s, 
				    gsl_vector *x, 
				    gsl_multifit_function_fdf *func, 
				    FluxonCompList<T> &MI, 
				    int P0,int Pi, int DoInit=1,int RestartAt=300)
{
  int iter = 0,m,status;
  gsl_vector *x0;
  gsl_vector *p0;
  T A,P,S;

  FTYPE Min0=0, Min=0;
  T InitStep;
  do
    {
      gsl_vector_set(x,0,MI[0].getAmp());
      gsl_vector_set(x,1,MI[0].getXPos());
      gsl_vector_set(x,2,MI[0].getYPos());
      gsl_vector_set(x,3,MI[0].getScale());

      if ((iter % RestartAt) == 0)
	{
	  gsl_multifit_fdfsolver_set(s, func, x);

	  //	  gsl_multifit_fdfsolver_restart(s);
	}

      x0 = gsl_multifit_fdfsolver_position(s);

      //      Min0 = Min;	  
      //      Min = gsl_multifit_fdfsolver_minimum(s);
      //
      // Make the move!
      //
      //      status = gsl_multifit_test_delta(s->dx, s->x, 1E-3, 1E-3);
      //      p0 = gsl_multifit_fdfsolver_gradient(s);
      //      if (status==GSL_SUCCESS) return status;

      status = gsl_multifit_fdfsolver_iterate(s);
      //	  if (status == GSL_ENOPROG) gsl_multifit_fdfsolver_restart(s);
      /*
      for (int i=0;i<4;i++) 
	cerr << gsl_vector_get(s->dx,0) << " "
	     << gsl_vector_get(s->dx,1) << " "
	     << gsl_vector_get(s->dx,2) << " "
	     << gsl_vector_get(s->dx,3) << endl;
      */
      if (status)  
	{
	  cerr << "###Info: " << gsl_strerror(status) << endl;
	  break;
	}

      status = gsl_multifit_test_delta(s->dx, s->x, 1E-3, 1E-3);
      if (status==GSL_SUCCESS) return status;

      //     p0 = gsl_multifit_fdfsolver_gradient(s);
      
      //      float Norm;

      //      Norm=gsl_blas_dnrm2(s->dx);
      //      cerr << 1/Norm << endl;
      /*
      if (Norm > 2.0)
	{
	  cerr << "####Phew: " 
	       << gsl_vector_get(p0,0) << " " 
	       << gsl_vector_get(p0,1) << " "
	       << gsl_vector_get(p0,2) << " "
	       << gsl_vector_get(p0,3) << " "
	       << status << " "
	       << InitStep << " "
	       << endl;
	  exit(0);
	}
      */

      /*
      if (iter%20 == 0)
	cerr << "Intermediate results " 
	     << iter                    << " "
	     << Min                     << " "
	     << gsl_blas_dnrm2(s->dx)   << " "
	     << Norm << " "
	     << endl;
      */
      iter++;
    }
  while(status == GSL_CONTINUE && iter < NIter);
  //  cerr << "###Info: " << gsl_strerror(status) << endl;

  MI[0].setAmp(gsl_vector_get(s->x,0));
  MI[0].setXPos(gsl_vector_get(s->x,1));
  MI[0].setYPos(gsl_vector_get(s->x,2));
  MI[0].setScale(gsl_vector_get(s->x,3));

  return status;
}
//
//---------------------------------------------------------------------------------------------
//
template <class T> void psfdecomp(int NP, int NIter, FluxonCompList<T>& FPSF, 
				  IMAGETYPE<T>& PSFImage, int N0, int N1)
{
  int N=1, ImSize;
  FTYPE InitStep;
  FTYPE Peak;
  IMAGETYPE<FTYPE> ResImg,ModelPSFImg, SmoothPSF;
  int PosX,PosY, Smoothed=0, Nx, Ny;
  Fluxon2D<FTYPE> f0;
  vector<int> PIndex;
  FluxonCompList<T> OneComp;
  
  PARAMS_PER_FLUXON=f0.getNoOfParams();

  PSFImage.size(ImSize,ImSize);  ModelPSFImg.resize(ImSize,ImSize);  ResImg.resize(ImSize,ImSize);
  ResImg.assign(0.0);            ModelPSFImg.assign(0.0);
  
  PIndex.resize(1);  
  PIndex.assign(PIndex.size(),1);
  /*
  if (FPSF.size() > 0)
    {
      cerr << "Computing model data..." << endl;
      ModelPSFImg   += FPSF;
      //      for (int i=0;i<FPSF.size();i++) PIndex[i]=1;
    }
  else
    FPSF.resize(0);
  */
  cerr << "Computing residual..." << endl;
  for (int i=0;i<ImSize;i++) 
    for (int j=0;j<ImSize;j++) 
      ResImg.setVal(i,j,(PSFImage(i,j) - ModelPSFImg(i,j)));

  gsl_vector *x=NULL;
  x = gsl_vector_alloc(PARAMS_PER_FLUXON*NP);
  gsl_vector_set_zero(x);
  
  gsl_vector *x0;
  static void* par[8];
  int status,m,NParams, Pi;
  //  const gsl_multifit_fdfsolver_type *MT;
  static gsl_multifit_fdfsolver *s=NULL,*s1=NULL;
  static gsl_multifit_function_fdf my_func,one_fluxon;

  size_t iter = 0;

  Pi = FPSF.size();
  //  PIndex.resize(Pi);
  NParams = PARAMS_PER_FLUXON*1;

  OneComp.resize(1);

  //
  // Set the initial guess
  //
  cerr << "Fitting " << Pi << " fluxons" << endl;
  SmoothPSF=PSFImage;

  for(int i=0;i<Pi;i++)
    {
      T Peak;
      int PeakX, PeakY;

      //      m = PARAMS_PER_FLUXON*i;
      
      //      Peak = getPeak(ResImg, PeakX, PeakY);
      Peak = getPeak(SmoothPSF, PeakX, PeakY);
      cerr << "###Initial guess: \t\t" << Peak << " " << ResImg(PeakX,PeakY) 
	   << " " << PeakX << " " << PeakY << " " << DEFAULT_SCALE << endl;
      Peak = ResImg(PeakX,PeakY)/1.5;
      //      PIndex[i]=1;
      OneComp[0].setAmp(Peak);
      OneComp[0].setPos(PeakX,PeakY);
      OneComp[0].setScale(DEFAULT_SCALE);

      setupSolver(&s,
		  gsl_multifit_fdfsolver_lmsder,
		  x,
		  &my_func,
		  1,//Pi,
		  par, 
		  &PIndex, 
		  &ResImg, 
		  &PSFImage,
		  &ModelPSFImg, 
		  (FluxonCompList<T> *)NULL,
		  &OneComp,//	      &FPSF,
		  (FluxonCompList<T> *)NULL);

      //      FPSF[0] = OneComp[0];

      gsl_vector_set(x, 0, OneComp[0].getAmp());
      gsl_vector_set(x, 1, OneComp[0].getXPos());
      gsl_vector_set(x, 2, OneComp[0].getYPos());
      gsl_vector_set(x, 3, OneComp[0].getScale());
      //
      // Find the Pi th. component.  findComponent has the
      // minimization loop
      //
      gsl_multifit_fdfsolver_set(s, &my_func, x);
      findComponent(NIter,s,x,&my_func,OneComp,0,0,1,20);
      cerr << "###Final fit: \t\t" << i << " " << OneComp[0] << endl;

      //
      // Add the latest component (OneComp) to the component list (FPSF).
      // Make a pixelated image out of OneComp (ModelPSFImg).
      // Make Residual image (ResImg = PSFImage - ModelPSFImg).
      //

      //
      // If the fitted component is not to weak, do the usual
      // subtraction
      //
      FTYPE Threshold=1E-4;

      {
	ModelPSFImg.size(Nx,Ny);
	FPSF[i] = OneComp[0];
	ModelPSFImg.assign(0.0);
	ModelPSFImg   += OneComp;
	/*
	  if (i > 0)
	    {
	      float XPos, YPos;
	      XPos = OneComp[0].getXPos()-Nx/2;
	      YPos = OneComp[0].getYPos()-Ny/2;
	      OneComp[0].setPos(Nx/2-XPos,Ny/2-YPos);

	      ModelPSFImg   += OneComp;
	    }
	*/
	if (i==0)  f0 = FPSF[0];

	//
	// Subtract only the latest Fluxon from PSFImage
	//
	/*
	for (int i=0;i<ImSize;i++) 
	  for (int j=0;j<ImSize;j++) 
	    {
	      T val;
	      val = (PSFImage(i,j) - GAIN*ModelPSFImg(i,j));
	      PSFImage.setVal(i,j,val);
	    }
	*/
	PSFImage -= OneComp;
	//
	// If the PSF was smoothed, then the current component,
	// convolved with the smoothing function, must also be
	// subtracted from the Smoothed image.
	//
	if (Smoothed)
	  {
	    Fluxon2D<T> tmpf;
	    tmpf = (OneComp.convolve(f0))[0];
	    cerr << "!!!!!!!!!!!!!!!! " << tmpf << endl;
	    SmoothPSF -= tmpf;
	  }
	//
	// If no smoothing was done on the Residual image,
	// keep using it as-is.
	//
	else SmoothPSF = PSFImage;
      }
      //
      // If the fitted component is too weak, it's probably trying
      // to fit to "noise" peak.  In which case, smooth the residual
      // by the main lobe, and here after search in this smoothed
      // residual image.  However for the rest of fitting machinary,
      // continue using the normal Residual Image.  Except that,
      // since the search for peak will be done in the smoothed
      // residual, keep removing the component convolved with the
      // main lobe from this smoothed image as well.
      //
      if ((fabs(OneComp[0].getAmp()) < Threshold))
	{
	  Convolver<T> Convo;
	  IMAGETYPE<T> tmp;
	  
	  tmp = PSFImage;  // Just to set the size of tmp!
	  tmp.assign(0);
	  
	  f0.setScale(f0.getScale()/1.4);
	  tmp += f0;
	  Convo.makeXFR(tmp);
	  SmoothPSF = PSFImage;
	  Convo.convolve(SmoothPSF);
	  //	  SmoothPSF.normalize((T)(Nx*Ny));
	  
	  Smoothed = 1;
	  Threshold /= 10;
	  cerr << "###Info: Smoothing the PSF by scale " << f0.getScale() << endl;
	  
	  i--;
	}
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

  cerr << "Got array of size " << v->Length() << endl;
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
  IMAGETYPE<FTYPE> PSFImage;
  FluxonCompList<FTYPE> CompList;
  Exp<FTYPE> ExpTab;
  Client c(argc,argv);
  GlishEvent *e;


  gsl_set_error_handler_off();
  PSFImage.resize(0,0);
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
	  CompList.resize(NComp);
	}
      else if (!strcmp(e->Name(),"scale"))   
	{
	  readVal(c, e, DEFAULT_SCALE);
	}
      else if (!strcmp(e->Name(),"init"))   
	{
	  CompList.resize(0);
	}
      else if (!strcmp(e->Name(),"imsize"))  
	{
	  readVal(c, e, ImSize);
	  //
	  // Setup the global namespace parameters.  These are used by
	  // class Fluxon<T>
	  //
	  ExpTab.Build(N,(FTYPE)ImSize/N);           // Build the tabulated exp() function
	  GlobalParams::NSigma=5.0;                 // Defines the support size after which the pixel 
                                                     // model not evaluated
	  GlobalParams::TableOfExponential = ExpTab; // The default table for exp()
	  GlobalParams::XCenter = ImSize/2.0+1;      // The image center pixel.  Required to get the 
	  GlobalParams::YCenter = ImSize/2.0+1;      // analytical form of convolution right.
	  PSFImage.resize(ImSize,ImSize);
	  cerr << "Image size set to " << ImSize << "x" << ImSize << endl;
	}
      else if (!strcmp(e->Name(),"wpsf"))  
	{
	  cerr << "No. of PSF Comps written " << writeComps(c, e, CompList) << endl;
	}
      else if (!strcmp(e->Name(),"rpsf"))  
	{
	  readImage(c, e, PSFImage);
	  /*
	  int NX,NY;
	  readVal(c, e, NX);
	  readVal(c, e, NY);
	  cerr << NX << " " << NY << endl;
	  FTYPE *Storage=PSFImage.getStorage();
	  for (int i=0;i<NX;i++)
	    for (int j=0;j<NY;j++)
	      readVal(c, e, Storage[i*NY+j]);
	  */
	}
      else if (!strcmp(e->Name(),"decomp"))  
	{
	  {int tmp; readVal(c, e, tmp); if (tmp > 0) NIter = tmp;}
	  
	  if (ImSize <= 0)
	    cerr << "###Error: Imsize not set!" << endl;
	  else if (NComp <= 0)
	    cerr << "###Error: NComp not set!" << endl;
	  else
	    {
	      int Restart=0,N0,N1;
	      Value *v = e->Val();
	      if (v->Length() > 0) Restart = v->IntVal();
	      N0=0;N1=CompList.size();

	      //	      gsl_set_error_handler(&err_handler);

	      if (Restart)
		{
		  N0=CompList.size();
		  N1=NComp;
		  cerr << "  Restarting at " << N0 << " " << N1 << endl;
		}
	      /*
	      ssdeconvolve(int NP, int NIter, FluxonCompList<T>& PSF, 
			   IMAGETYPE<T>& DImg, FluxonCompList<T>& MI,
			   int N0, int N1)
	      */
	      //	      ssdeconvolve(PSF.size(), NIter, PSF, DirtyImage, PSF, N0,N1);
	      psfdecomp(100, NIter, CompList, PSFImage, N0,N1);
	      cerr << CompList << endl;
	      cerr << "...done!" << endl;
	    }
	}
      else if (!strcmp(e->Name(),"psfcomp"))   
	{
	  writeComps(c, e, CompList);
	}
      else if (!strcmp(e->Name(),"respsf"))   
	{
	  std::ofstream fd("respsf.dat");
	  fd << PSFImage;
	  sendImage(c,e,PSFImage);
	}
      else if (!strcmp(e->Name(),"modpsf"))   
	{
	  std::ofstream fd("modpsf.dat");
	  FluxonCompList<FTYPE> tmp;
	  IMAGETYPE<FTYPE> PSFImg;

	  PSFImg.resize(ImSize,ImSize);
	  PSFImg.assign(0.0);
	  PSFImg += CompList;

	  tmp = CompList;
	  for (int i=1;i<tmp.size();i++)
	    tmp[i].setPos(-tmp[i].getXPos(),-tmp[i].getYPos());
	  //	  PSFImg += tmp;


	  sendImage(c,e,PSFImg);
	  fd << PSFImg;
	}
      /*
	else if (!strcmp(e->Name(),"ccomp"))   
	sendCC(c,e,CompList);
      */
      else if (!strcmp(e->Name(),"done")) {cerr << "..bye" << endl;return 0;}
      else c.Unrecognized();
    }

  cerr << "Exiting..." << endl;
  return 0;
}
