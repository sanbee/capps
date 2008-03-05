#include <casa/aips.h>
#include <images/Images/ImageInterface.h>
#include <images/Images/PagedImage.h>
#include <images/Images/TempImage.h>
#include <coordinates/Coordinates/CoordinateSystem.h>
#include <coordinates/Coordinates/LinearCoordinate.h>
#include <coordinates/Coordinates/DirectionCoordinate.h>
#include <coordinates/Coordinates/SpectralCoordinate.h>
#include <coordinates/Coordinates/StokesCoordinate.h>
#include <coordinates/Coordinates/Projection.h>
#include <synthesis/MeasurementComponents/Utils.h>

#include <cl.h>
#include <clinteract.h>

#include <LWAs.h>


#include <Coordinate_conversion.h>

#include <iostream>
#include <fstream>

using namespace std;
using namespace casa;
#define RestartUI(Label)  {if(clIsInteractive()) {goto Label;}}

void UI(Bool restart, int argc, char **argv, 
	string& imageName, Float& Dec, Float& HA, Float &LMax)
{
  if (!restart)
    {
      BeginCL(argc,argv);
      char TBuf[FILENAME_MAX];
      clgetConfigFile(TBuf,argv[0]);strcat(TBuf,".config");
      clloadConfig(TBuf);
      clInteractive(0);
    }
  else
   clRetry();
  try
    {
      int i;
      {
	i=1;clgetSValp("image",imageName, i);
	i=1;clgetFValp("dec",Dec, i);
	i=1;clgetFValp("ha",HA, i);
	i=1;clgetFValp("lmax",LMax, i);
      }
      EndCL();
    }
  catch (clError x)
    {
      x << x << endl;
      clRetry();
    }
}

void makePBImage(ImageInterface<Float>& pb, Int Nx, Int Ny, Float LMax,
		 Double Nu, Double dNu)
{
  IPosition pbShape(3,(Int)Nx, (Int)Ny, 1);
  pb.resize(pbShape);
  CoordinateSystem pbCoords;
  LinearCoordinate lc(2);
  Vector<Double> val(2);
  val(0) = val(1) = LMax/Nx;     lc.setIncrement(val);
  val(0) = Nx/2; val(1) = Ny/2;  lc.setReferencePixel(val);
  val(0) = val(1) = 0.0;         lc.setReferenceValue(val);
  pbCoords.addCoordinate(lc);

  Double restfreq = Nu;
  Double crpix = 1.0;
  Double crval = Nu;
  Double cdelt = dNu;
  SpectralCoordinate sc(MFrequency::TOPO, crval, cdelt, crpix, restfreq);
  pbCoords.addCoordinate(sc);
  pb.setCoordinateInfo(pbCoords);
  pb.set(0.0);
}

int main(int argc, char **argv)
{

  Bool restart=False;
  Float Dec=0, HA=0,LMax=50;
  string imageName;
  UI(restart,argc,argv,imageName,Dec,HA,LMax);
  /****************Setting start*******************************************************/

  //observing Frequency, Width, Ch, Dipole Primary Beam, Noise Setting
  double Freq2 = 20.0e6;
  double FW2   = 8.0e6;
  unsigned int ch2 = 1;
  bool DPB2 = 1;
  bool NS2  = 0;

 
  //Observatory1(EL) setting using constractor2
  double LWAObs1_Lati = 90.0*M_PI/180.0;
  double LWAObs1_Long = -105.16*M_PI/180.0;
  //double LWAObs1_Lati = 32.89*M_PI/180.0;
  //double LWAObs1_Long = -105.16*M_PI/180.0;
  
  //Observatory2(SM) setting using constractor2
  double LWAObs2_Lati = 90.0*M_PI/180.0;
  double LWAObs2_Long = -105.16*M_PI/180.0;
  //double LWAObs2_Lati = 33.48*M_PI/180.0;
  //double LWAObs2_Long = -108.93*M_PI/180.0;


  //Beam Center (Greenwich standard) longtitude setting using constractor2
  //  double delta_H    = 178.0;
  double delta_H = HA;
  double BeamCenter = (LWAObs1_Long + LWAObs2_Long)/2.0 + (delta_H*M_PI/180.0);
  //  double dec2      = ((40.0 + 44.0/60.0 + 0.189/3600.0)-0.0)*M_PI/180.0;
  double dec2      = Dec*M_PI/180.0;


  //l,m,lm step
  //  double lm_max2 = 50.0;
  double lm_max2 = LMax;


  //Dipole Primary Beam file, Dipole Configuration file 
  string DPBfile2  = "bbl_3mx3mGS_20MHzbp.txt.cln";
  string Dconfile2 = "LWA_strawman_station.txt";

/****************Setting end*******************************************************/




  Convert_HDtoAE check1_El(LWAObs1_Lati, dec2, delta_H);
  Convert_HDtoAE check2_El(LWAObs2_Lati, dec2, delta_H);
  
  if(check1_El.get_El() < 0.0 || check2_El.get_El() < 0.0){ 
    cout << "Beam Center is below the horizon!";
    return(1);
  } 
  
  cout << check1_El.get_El()*180.0/M_PI << endl;
  cout << check1_El.get_z_zenith();

  //------------------------------------------------------------------
  //Make construct a CASA image and fill it with the computed Power
  //Pattern values.
  //
  // Prof. Masaya: The pixels array below needs to be filled with the
  // power pattern values.  Can you add the code to fill this array?
  //
  TempImage<Float> pb;
  makePBImage(pb, (Int)lm_max2, (Int)lm_max2, 1.0, Freq2, FW2);



  String name(imageName.c_str());
  storeImg(name, pb);


  //H1:station1, H2:station2
  double H1    =  BeamCenter - LWAObs1_Long;
  double H2    =  BeamCenter - LWAObs2_Long;


  //LWA station1, LWA station2: constractor2
  LWAs LWA1(Freq2,FW2,ch2,DPB2,NS2,lm_max2,LWAObs1_Lati,dec2,H1,
           DPBfile2,Dconfile2);

  LWAs LWA2(Freq2,FW2,ch2,DPB2,NS2,lm_max2,LWAObs2_Lati,dec2,H2,
           DPBfile2,Dconfile2);

  
  Array<Float> pixels;

  Int nx,ny;
  nx=ny=(Int)lm_max2*2+1;

  pb.get(pixels);

  IPosition ndx(3,0,0,0)

  Int N =0;
  for(Int i=0;i<nx;i++)
    for(Int j=0;j<ny;j++)
      {
	ndx(0)=i; 
	ndx(1)=j;
	pixels(ndx)=sqrt(powl(LWA1.get_LWAs_Ereal(N), 2.0)+powl(LWA1.get_LWAs_Eimag(N), 2.0))* 
	            sqrt(powl(LWA2.get_LWAs_Ereal(N), 2.0)+powl(LWA2.get_LWAs_Eimag(N), 2.0));
        N += 1;
      }
      
      pb.put(pixels);
      
  }


    



  //cout << "ch = " << LWA1c2.get_Ch() << endl;
  //cout << "Psize = "<< LWA1c2.get_LWAs_Powersize() << endl;
  //cout << "Esize = "<< LWA1c2.get_LWAs_Erealsize() << endl;

  cout << "***LWA Station1 constractor2***" << endl;
  cout << "Obs1Lati = " << LWA1.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  cout << "Obs1Long = " << LWAObs1_Long*180.0/M_PI << endl;
  cout << "Dec  = "     << LWA1.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  cout << "Hour  = "    << LWA1.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  cout << "Hour_h  = "  << LWA1.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  cout << "Hour_m  = "  << LWA1.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  cout << "Hour_s  = "  << LWA1.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  cout << "Az  = "      << LWA1.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  cout << "El  = "      << LWA1.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  cout << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"             << endl;
  cout << "X_south = "  << LWA1.Convert_HDtoAE::get_x_south()             << endl;
  cout << "Y_east = "   << LWA1.Convert_HDtoAE::get_y_east()              << endl;
  cout << "Z_zenith = "   << LWA1.Convert_HDtoAE::get_z_zenith()          << endl;
  cout << " " << endl;

  cout << "***LWA Station2 constractor2***" << endl;
  cout << "Obs2Lati = " << LWA2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  cout << "Obs2Long = " << LWAObs2_Long*180.0/M_PI << endl;
  cout << "Dec  = "     << LWA2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  cout << "Hour  = "    << LWA2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  cout << "Hour_h  = "  << LWA2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  cout << "Hour_m  = "  << LWA2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  cout << "Hour_s  = "  << LWA2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  cout << "Az  = "      << LWA2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  cout << "El  = "      << LWA2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  cout << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"             << endl;
  cout << "X_south = "  << LWA2.Convert_HDtoAE::get_x_south()             << endl;
  cout << "Y_east = "   << LWA2.Convert_HDtoAE::get_y_east()              << endl;
  cout << "Z_zenith = "   << LWA2.Convert_HDtoAE::get_z_zenith()          << endl;
  cout << " " << endl;



  ofstream fo7("logfile.txt");

  fo7 << "Beam Center (Hour angle) = " << BeamCenter*180.0/M_PI << "(deg)" <<endl;
  fo7 << "Angle from Beam Center = " << delta_H << "(deg)" << endl;
  fo7 << "Dec = " << dec2*180.0/M_PI << endl;

  fo7 << "Freqency = " << Freq2 << endl;
  fo7 << "Freqency Width = " << FW2 << endl; 
  fo7 << "ch = " << LWA1.get_Ch() << endl;
  fo7 << "Dipole Primary Beam = " << DPBfile2.c_str() << endl;
  fo7 << "Dipole Configuration = " << Dconfile2.c_str() << endl;
  fo7 << endl; 

  fo7 << "***LWA Station1 constractor2***" << endl;
  fo7 << "Obs1Lati = " << LWA1.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  fo7 << "Obs1Long = " << LWAObs1_Long*180.0/M_PI << endl;
  fo7 << "Dec  = "     << LWA1.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  fo7 << "Hour  = "    << LWA1.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  fo7 << "Hour_h  = "  << LWA1.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  fo7 << "Hour_m  = "  << LWA1.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  fo7 << "Hour_s  = "  << LWA1.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  fo7 << "Az  = "      << LWA1.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "El  = "      << LWA1.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"           << endl;
  fo7 << "X_south = "  << LWA1.Convert_HDtoAE::get_x_south()             << endl;
  fo7 << "Y_east = "   << LWA1.Convert_HDtoAE::get_y_east()              << endl;
  fo7 << "Z_zenith = "   << LWA1.Convert_HDtoAE::get_z_zenith()          << endl;
  fo7 << " " << endl;

  fo7 << "***LWA Station2 constractor2***" << endl;
  fo7 << "Obs2Lati = " << LWA2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  fo7 << "Obs2Long = " << LWAObs2_Long*180.0/M_PI << endl;
  fo7 << "Dec  = "     << LWA2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  fo7 << "Hour  = "    << LWA2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  fo7 << "Hour_h  = "  << LWA2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  fo7 << "Hour_m  = "  << LWA2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  fo7 << "Hour_s  = "  << LWA2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  fo7 << "Az  = "      << LWA2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "El  = "      << LWA2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"           << endl;
  fo7 << "X_south = "  << LWA2.Convert_HDtoAE::get_x_south()             << endl;
  fo7 << "Y_east = "   << LWA2.Convert_HDtoAE::get_y_east()              << endl;
  fo7 << "Z_zenith = "   << LWA2.Convert_HDtoAE::get_z_zenith()          << endl;
  fo7 << " " << endl;


  //File out (Station1:Power)
  ofstream fo1("LWAs1_Power.txt");
  cerr << "Generating LWAs1 power..." << endl;
   for(long int i = 0; i < LWA1.get_LWAs_Powersize(); i++){
    fo1 << LWA1.get_LWAs_L(i) << ' ' << LWA1.get_LWAs_M(i) << ' '
        << LWA1.get_LWAs_Power(i) << endl;
  } 
  

  
  //File out (Station2:Power)
  ofstream fo2("LWAs2_Power.txt");
   for(long int i = 0; i < LWA2.get_LWAs_Powersize(); i++){
    fo2 << LWA2.get_LWAs_L(i) << ' ' << LWA2.get_LWAs_M(i) << ' '
        << LWA2.get_LWAs_Power(i) << endl;
  }
  



  ofstream fo6("LWAs12_Power.txt");
  ofstream fo3("LWAs1_Ereal.txt");
  ofstream fo4("LWAs1_Eimag.txt");
  ofstream fo5("LWAs2_Ereal.txt");
  ofstream fo8("LWAs2_Eimag.txt");

  for(long int i = 0; i < LWA1.get_LWAs_Powersize(); i++){
    
    double Station12_Power = 0.0;  
    double Station1_Real = 0.0;
    double Station1_Imag = 0.0;
    double Station2_Real = 0.0;
    double Station2_Imag = 0.0;
    
    for(int k = i*LWA1.get_Ch(); k < (i+1)*LWA1.get_Ch(); k++){    
      
      Station12_Power += sqrt(powl(LWA1.get_LWAs_Ereal(k), 2.0)+powl(LWA1.get_LWAs_Eimag(k), 2.0))
	               * sqrt(powl(LWA2.get_LWAs_Ereal(k), 2.0)+powl(LWA2.get_LWAs_Eimag(k), 2.0));

      Station1_Real += LWA1.get_LWAs_Ereal(k);
      Station1_Imag += LWA1.get_LWAs_Eimag(k);        
      
      Station2_Real += LWA2.get_LWAs_Ereal(k);
      Station2_Imag += LWA2.get_LWAs_Eimag(k);

    }
 
    


    fo6 << LWA1.get_LWAs_L(i) << ' ' << LWA1.get_LWAs_M(i) << ' '
        << Station12_Power  << endl;

    fo3 << LWA1.get_LWAs_L(i) << ' ' << LWA1.get_LWAs_M(i) << ' '
        << Station1_Real << endl;

    fo4 << LWA1.get_LWAs_L(i) << ' ' << LWA1.get_LWAs_M(i) << ' '
        << Station1_Imag << endl;

    fo5 << LWA2.get_LWAs_L(i) << ' ' << LWA2.get_LWAs_M(i) << ' '
        << Station2_Real << endl;
    
    fo8 << LWA2.get_LWAs_L(i) << ' ' << LWA2.get_LWAs_M(i) << ' '
        << Station2_Imag << endl;

  }


  fo1.close();
  fo2.close();
  fo6.close();
  fo7.close();
  fo3.close();
  fo4.close();
  fo5.close();
  fo8.close();



  
  return(0);
}
