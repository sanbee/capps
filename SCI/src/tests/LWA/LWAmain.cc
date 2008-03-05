#include "LWAStation.h"

#include <iostream>
#include <fstream>

using namespace std;


int main()
{
  
  
  //Beam Center (Greenwich standard)
//  int source_h = 0;
//  int source_m = 0;
//  double source_s = 0;

//  double dec      = (40.0 + 44.0/60.0 + 0.189/3600.0)*M_PI/180.0;
  
  //Observatory1(El) setting
//  double Obs1_Lati = 32.89*M_PI/180.0;
//  int Obs1_Long_h = 0;
//  int Obs1_Long_m = 0;
//  double Obs1_Long_s = 0.0;
//  int H1_h    = Obs1_Long_h - source_h;
//  int H1_m    = Obs1_Long_m - source_m;
//  double H1_s = Obs1_Long_s - source_s;

  //Observatory2(ST) setting
//  double Obs2_Lati = 31.81*M_PI/180.0;
//  int Obs2_Long_h = 1;
//  int Obs2_Long_m = 0;
//  double Obs2_Long_s = 0.0;
//  int H2_h    = Obs2_Long_h - source_h;
//  int H2_m    = Obs2_Long_m - source_m;
//  double H2_s = Obs2_Long_s - source_s;

 
  //observing Frequency, Width, Ch, Dipole Primary Beam, Noise Setting
//  double Freq= 20.0 * powl(10.0, 6.0);
//  double FW  = 4.0  * powl(10.0, 6.0);
//  unsigned int ch = 2;
//  bool DPB = 1;
//  bool NS  = 0;

  
  //l,m,lm step
//  const double lm_max = 50;
 


  //Dipole Primary Beam file, Dipole Configuration file 
//  string DPBfile  = "bbl_3mx3mGS_30MHzbp.txt.cln";
//  string Dconfile = "LWA_strawman_station_ellipse.txt";

 

  //LWA station1, LWA station2: constractor1
//  LWAs LWA1(Freq,FW,ch,DPB,NS,lm_max,Obs1_Lati,dec,H1_h,H1_m,H1_s,
//           DPBfile,Dconfile);
  
//  LWAs LWA2(Freq,FW,ch,DPB,NS,lm_max,Obs2_Lati,dec,H2_h,H2_m,H2_s,
//           DPBfile,Dconfile);

  


  ///////////////////////////////////////////////////////////////

  ofstream fo7("logfile.txt");
  

  //Observatory1(EL) setting constractor2
  double LWAObs1_Lati = 32.89*M_PI/180.0;
  double LWAObs1_Long = -105.16*M_PI/180.0;
  
  //Observatory2(SM) setting constractor2
  double LWAObs2_Lati = 32.89*M_PI/180.0;
  double LWAObs2_Long = -105.16*M_PI/180.0;
  //double LWAObs2_Lati = 33.48*M_PI/180.0;
  //double LWAObs2_Long = -108.93*M_PI/180.0;

  //constractor2
  //Beam Center (Greenwich standard) longtitude
  double delta_H    = 90.0;
  double BeamCenter = (LWAObs1_Long + LWAObs2_Long)/2.0 + (delta_H*M_PI/180.0);
  double dec2      = ((40.0 + 44.0/60.0 + 0.189/3600.0)-0.0)*M_PI/180.0;
  

  fo7 << "Beam Center (Hour angle) = " << BeamCenter*180.0/M_PI << "(deg)" <<endl;
  fo7 << "Angle from Beam Center = " << delta_H << "(deg)" << endl;
  fo7 << "Dec = " << dec2*180.0/M_PI << endl;


  //H1:station1, H2:station2
  double H1    =  BeamCenter - LWAObs1_Long;
  double H2    =  BeamCenter - LWAObs2_Long;


  //observing Frequency, Width, Ch, Dipole Primary Beam, Noise Setting
  double Freq2 = 20.0 * powl(10.0, 6.0);
  double FW2   = 4.0  * powl(10.0, 6.0);
  unsigned int ch2 = 40;
  bool DPB2 = 1;
  bool NS2  = 0;
  

  //l,m,lm step
  double lm_max2 = 50.0;
 


   //Dipole Primary Beam file, Dipole Configuration file 
  string DPBfile2  = "bbl_3mx3mGS_20MHzbp.txt.cln";
  string Dconfile2 = "LWA_strawman_station.txt";


  //LWA station1, LWA station2: constractor2
  LWAs LWA1c2(Freq2,FW2,ch2,DPB2,NS2,lm_max2,LWAObs1_Lati,dec2,H1,
           DPBfile2,Dconfile2);

  LWAs LWA2c2(Freq2,FW2,ch2,DPB2,NS2,lm_max2,LWAObs2_Lati,dec2,H2,
           DPBfile2,Dconfile2);


  cout << "ch = " << LWA1c2.get_Ch() << endl;
  cout << "Psize = "<< LWA1c2.get_LWAs_Powersize() << endl;
  cout << "Esize = "<< LWA1c2.get_LWAs_Erealsize() << endl;

  fo7 << "Freqency = " << Freq2 << endl;
  fo7 << "Freqency Width = " << FW2 << endl; 
  fo7 << "ch = " << LWA1c2.get_Ch() << endl;
  fo7 << "Dipole Primary Beam = " << DPBfile2.c_str() << endl;
  fo7 << "Dipole Configuration = " << Dconfile2.c_str() << endl;
  fo7 << endl; 


  /*
  cout << "***LWA Station1 constractor1***" << endl;
  cout << "Obs1Lati = " << LWA1.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  cout << "Dec  = "     << LWA1.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  cout << "Hour  = "    << LWA1.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  cout << "Hour_h  = "  << LWA1.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  cout << "Hour_m  = "  << LWA1.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  cout << "Hour_s  = "  << LWA1.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  cout << "Az  = "      << LWA1.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  cout << "El  = "      << LWA1.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  cout << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"           << endl;
  cout << "X_south = "  << LWA1.Convert_HDtoAE::get_x_south()             << endl;
  cout << "Y_east = "   << LWA1.Convert_HDtoAE::get_y_east()              << endl;
  cout << "Z_zenith = "   << LWA1.Convert_HDtoAE::get_z_zenith()          << endl;
  cout << " " << endl;
  */


  cout << "***LWA Station1 constractor2***" << endl;
  cout << "Obs1Lati = " << LWA1c2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  cout << "Obs1Long = " << LWAObs1_Long*180.0/M_PI << endl;
  cout << "Dec  = "     << LWA1c2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  cout << "Hour  = "    << LWA1c2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  cout << "Hour_h  = "  << LWA1c2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  cout << "Hour_m  = "  << LWA1c2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  cout << "Hour_s  = "  << LWA1c2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  cout << "Az  = "      << LWA1c2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  cout << "El  = "      << LWA1c2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  cout << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"             << endl;
  cout << "X_south = "  << LWA1c2.Convert_HDtoAE::get_x_south()             << endl;
  cout << "Y_east = "   << LWA1c2.Convert_HDtoAE::get_y_east()              << endl;
  cout << "Z_zenith = "   << LWA1c2.Convert_HDtoAE::get_z_zenith()          << endl;
  cout << " " << endl;


  fo7 << "***LWA Station1 constractor2***" << endl;
  fo7 << "Obs1Lati = " << LWA1c2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  fo7 << "Obs1Long = " << LWAObs1_Long*180.0/M_PI << endl;
  fo7 << "Dec  = "     << LWA1c2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  fo7 << "Hour  = "    << LWA1c2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  fo7 << "Hour_h  = "  << LWA1c2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  fo7 << "Hour_m  = "  << LWA1c2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  fo7 << "Hour_s  = "  << LWA1c2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  fo7 << "Az  = "      << LWA1c2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "El  = "      << LWA1c2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"             << endl;
  fo7 << "X_south = "  << LWA1c2.Convert_HDtoAE::get_x_south()             << endl;
  fo7 << "Y_east = "   << LWA1c2.Convert_HDtoAE::get_y_east()              << endl;
  fo7 << "Z_zenith = "   << LWA1c2.Convert_HDtoAE::get_z_zenith()          << endl;
  fo7 << " " << endl;


  
  /*
  cout << "***LWA Station2 constractor1***" << endl;
  cout << "Obs2Lati = " << LWA2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  cout << "Dec  = "     << LWA2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  cout << "Hour  = "    << LWA2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  cout << "Hour_h  = "  << LWA2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  cout << "Hour_m  = "  << LWA2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  cout << "Hour_s  = "  << LWA2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  cout << "Az  = "      << LWA2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  cout << "El  = "      << LWA2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  cout << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"           << endl;
  cout << "X_south = "  << LWA2.Convert_HDtoAE::get_x_south()             << endl;
  cout << "Y_east = "   << LWA2.Convert_HDtoAE::get_y_east()              << endl;
  cout << "Z_zenith = "   << LWA2.Convert_HDtoAE::get_z_zenith()          << endl;
  cout << " " << endl;
  */

  cout << "***LWA Station2 constractor2***" << endl;
  cout << "Obs2Lati = " << LWA2c2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  cout << "Obs2Long = " << LWAObs2_Long*180.0/M_PI << endl;
  cout << "Dec  = "     << LWA2c2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  cout << "Hour  = "    << LWA2c2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  cout << "Hour_h  = "  << LWA2c2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  cout << "Hour_m  = "  << LWA2c2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  cout << "Hour_s  = "  << LWA2c2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  cout << "Az  = "      << LWA2c2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  cout << "El  = "      << LWA2c2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  cout << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"             << endl;
  cout << "X_south = "  << LWA2c2.Convert_HDtoAE::get_x_south()             << endl;
  cout << "Y_east = "   << LWA2c2.Convert_HDtoAE::get_y_east()              << endl;
  cout << "Z_zenith = "   << LWA2c2.Convert_HDtoAE::get_z_zenith()          << endl;
  cout << " " << endl;

  fo7 << "***LWA Station2 constractor2***" << endl;
  fo7 << "Obs2Lati = " << LWA2c2.Convert_HDtoAE::get_Obs_Lati()*180.0/M_PI << "(deg)" << endl;
  fo7 << "Obs2Long = " << LWAObs2_Long*180.0/M_PI << endl;
  fo7 << "Dec  = "     << LWA2c2.Convert_HDtoAE::get_Dec()*180.0/M_PI      << "(deg)" << endl;
  fo7 << "Hour  = "    << LWA2c2.Convert_HDtoAE::get_Hour()*180.0/M_PI     << "(deg)" << endl;
  fo7 << "Hour_h  = "  << LWA2c2.Convert_HDtoAE::get_Hour_h()              << "(h)"   << endl;
  fo7 << "Hour_m  = "  << LWA2c2.Convert_HDtoAE::get_Hour_m()              << "(m)"   << endl;
  fo7 << "Hour_s  = "  << LWA2c2.Convert_HDtoAE::get_Hour_s()              << "(s)"   << endl;
  fo7 << "Az  = "      << LWA2c2.Convert_HDtoAE::get_Az()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "El  = "      << LWA2c2.Convert_HDtoAE::get_El()*180.0/M_PI       << "(deg)" << endl;
  fo7 << "Beam-Center Direction Cos (X_south,Y_east,Z_zenith)"             << endl;
  fo7 << "X_south = "  << LWA2c2.Convert_HDtoAE::get_x_south()             << endl;
  fo7 << "Y_east = "   << LWA2c2.Convert_HDtoAE::get_y_east()              << endl;
  fo7 << "Z_zenith = "   << LWA2c2.Convert_HDtoAE::get_z_zenith()          << endl;
  fo7 << " " << endl;



  /*
  cout << "***LWAs1 Dipole Configuration***" << endl;
  cout << "Filename = " << LWA1.LWAs_Dipole_Conf::get_fname() << endl;
  cout << "Number = " << LWA1.LWAs_Dipole_Conf::get_number(255) << endl;
  cout << "East = " << LWA1.LWAs_Dipole_Conf::get_East(255) << endl;
  cout << "North = " << LWA1.LWAs_Dipole_Conf::get_North(255) << endl;
  cout << "Zenith = " << LWA1.LWAs_Dipole_Conf::get_Zenith() << endl;
  cout << LWA1.LWAs_Dipole_Conf::get_size() << endl;
  cout << " " << endl;

  cout << "***LWAs2 Dipole Configuration***" << endl;
  cout << "Filename = " << LWA2.LWAs_Dipole_Conf::get_fname() << endl;
  cout << "Number = " << LWA2.LWAs_Dipole_Conf::get_number(255) << endl;
  cout << "East = " << LWA2.LWAs_Dipole_Conf::get_East(255) << endl;
  cout << "North = " << LWA2.LWAs_Dipole_Conf::get_North(255) << endl;
  cout << "Zenith = " << LWA2.LWAs_Dipole_Conf::get_Zenith() << endl;
  cout << LWA2.LWAs_Dipole_Conf::get_size() << endl;
  cout << " " << endl;

  cout << "***LWAs1 Dipole PB***"<< endl;
  cout << LWA1.LWAs_Dipole_PB::get_elevation(255) << endl;
  cout << LWA1.LWAs_Dipole_PB::get_azimuth(255) << endl;
  cout << LWA1.LWAs_Dipole_PB::get_power(255) << endl;
  cout << LWA1.LWAs_Dipole_PB::get_size() << endl;
  cout << " " << endl;
  
  cout << "***LWAs2 Dipole PB***"<< endl;
  cout << LWA2.LWAs_Dipole_PB::get_elevation(255) << endl;
  cout << LWA2.LWAs_Dipole_PB::get_azimuth(255) << endl;
  cout << LWA2.LWAs_Dipole_PB::get_power(255) << endl;
  cout << LWA2.LWAs_Dipole_PB::get_size() << endl;
  cout << " " << endl;
  */


  /*
  cout << "Frequency = "       << LWA1.get_Freq()      << endl;
  cout << "Frequency Width = " << LWA1.get_FreqW()     << endl;
  cout << "light velocity = "  << LWA1.get_C()         << endl;
  cout << "Delta_Frequency = " << LWA1.get_DeltaFreq() << endl;
  cout << "Channel = "         << LWA1.get_Ch()        << endl;
  */

  
  /*
  cout << "rotation_W_size = " << LWA.get_rotation_W_size() << endl;
  for(int i = 0; i < LWA.get_rotation_W_size(); i++ ){
    cout << i << ' ' << LWA.get_rotation_W(i) << endl;
  }
  */


  //File out (Station1:Power)
  ofstream fo1;
  fo1.open("test_LWAs1_Power.txt");
   for(long int i = 0; i < LWA1c2.get_LWAs_Powersize(); i++){
    fo1 << LWA1c2.get_LWAs_L(i) << ' ' << LWA1c2.get_LWAs_M(i) << ' '
        << LWA1c2.get_LWAs_Power(i) << endl;
  } 
  fo1.close();

  
  //File out (Station2:Power)
  ofstream fo2;
  fo2.open("test_LWAs2_Power.txt");
   for(long int i = 0; i < LWA2c2.get_LWAs_Powersize(); i++){
    fo2 << LWA2c2.get_LWAs_L(i) << ' ' << LWA2c2.get_LWAs_M(i) << ' '
        << LWA2c2.get_LWAs_Power(i) << endl;
  }
  fo2.close();

/*  
  //File out (Station1:Ereal, Eimag)
  ofstream fo3;
  fo3.open("test_LWAs1_Ereal_Eimag.txt");
  for(long int i = 0; i < LWA1.get_LWAs_Erealsize(); i++){
    fo3 << LWA1.get_LWAs_L(i) << ' ' << LWA1.get_LWAs_M(i) << ' '
        << LWA1.get_LWAs_Ereal(i) << ' ' << LWA1.get_LWAs_Eimag(i) <<endl;
  }
  fo3.close();


  //File out (Station2:Ereal, Eimag)
  ofstream fo4;
  fo4.open("test_LWAs2_Ereal_Eimag.txt");
  for(long int i = 0; i < LWA2.get_LWAs_Erealsize(); i++){
    fo4 << LWA2.get_LWAs_L(i) << ' ' << LWA2.get_LWAs_M(i) << ' '
        << LWA2.get_LWAs_Ereal(i) << ' ' << LWA2.get_LWAs_Eimag(i) <<endl;
  }
  fo4.close();

  //File out (Station12:Ereal1*Ereal2, Eimag2*Eimag2)
  ofstream fo5;
  fo5.open("test_LWAs12_Ereal12_Eimag12.txt");
  for(long int i = 0; i < LWA1.get_LWAs_Powersize(); i++){
    fo5 << LWA1.get_LWAs_L(i) << ' ' << LWA1.get_LWAs_M(i) << ' '
        << LWA1.get_LWAs_Ereal(i)*LWA2.get_LWAs_Ereal(i)
          +LWA1.get_LWAs_Eimag(i)*LWA2.get_LWAs_Eimag(i) <<endl;
  }
  fo5.close();
  
*/

  //constractor2
  //File out (Station12:Ereal1*Ereal2, Eimag2*Eimag2)
  ofstream fo6("test_LWAs12c2_Ereal12_Eimag12.txt");
  for(long int i = 0; i < LWA1c2.get_LWAs_Powersize(); i++){
    
    double Station12_Power = 0.0;
    
    for(int k = i*LWA1c2.get_Ch(); k < (i+1)*LWA1c2.get_Ch(); k++){    
      //Station12_Power += LWA1c2.get_LWAs_Ereal(k) * LWA2c2.get_LWAs_Ereal(k)
      //                +LWA1c2.get_LWAs_Eimag(k) * LWA2c2.get_LWAs_Eimag(k);

      Station12_Power += sqrt(powl(LWA1c2.get_LWAs_Ereal(k), 2.0)+powl(LWA1c2.get_LWAs_Eimag(k), 2.0))
	               * sqrt(powl(LWA2c2.get_LWAs_Ereal(k), 2.0)+powl(LWA2c2.get_LWAs_Eimag(k), 2.0));


    }
 

    fo6 << LWA1c2.get_LWAs_L(i) << ' ' << LWA1c2.get_LWAs_M(i) << ' '
        << Station12_Power  << endl;

  }

  fo6.close();
    

  fo7.close();



  //lati->VL, dec->CygA
  //long double latitude = (34.07)*M_PI/180.0;
  //long double dec      = (40.0 + 44.0/60.0 + 0.189/3600.0)*M_PI/180.0;
  //int H_h = 0;
  //int H_m = 0;
  //long double H_s = 0.0;

  //Convert_HDtoAE a(latitude, dec, H_h, H_m, H_s);

  //Convert_HDtoAE a2(latitude, dec, M_PI/2.0);
  

  //cout << "Latitude = " << a2.get_Lati()*180.0/M_PI << "deg" << endl;
  //cout << "Hour = " << a2.get_Hour()*180.0/M_PI << "deg" << endl;
  //cout << "Hour_h = " << a2.get_Hour_h() << "h"   << endl;
  //cout << "Hour_m = " << a2.get_Hour_m() << "m"   << endl;
  //cout << "Hour_s = " << a2.get_Hour_s() << "s"   << endl;

  //cout << "Dec = " << a2.get_Dec()*180.0/M_PI << "deg" << endl;
  //cout << "Az = " << a2.get_Az()*180.0/M_PI << "deg" << endl;
  //cout << "El = " << a2.get_El()*180.0/M_PI << "deg" << endl;
  
  ///////////////////////////////////

//  string fname = "LWA_strawman_station.txt";
//  fstream fi(fname.c_str(), ios::in);
  
//  if(fi.fail()){
//    cerr << "Could not open file" << endl;
//    return 1;
//  }
  
//  LWAs_Dipole_Conf b(fi, fname);
  
 
  
 // cout << "Filename = " << b.get_fname()<< endl;
 // cout << "Size = " << b.get_size() << endl;

 // for(int i = 0; i < b.get_size(); i++){
 //   cout << "Dipole number = " << b.get_number(i) << endl;
 //   cout << "East = " << b.get_East(i) << endl;
 //   cout << "North = " << b.get_North(i) << endl;
 //   cout << endl;
 // }
  
 // fi.close();

  ///////////////////////////////////
  
//  string fname2 = "lgBlade_EZNEC_1deg_Bill_EG_30MHz.dat.cln";
//  fstream fi2(fname2.c_str(), ios::in);
  
//  if(fi2.fail()){
//    cerr << "Could not open file" << endl;
//    return 1;
//  }
  
//  LWAs_Dipole_PB c(fi2, fname2);
  
 
  
//  cout << "Filename = " << c.get_fname()<< endl;
//  cout << "Size = " << c.get_size() << endl;

//  for(int i = 0; i < c.get_size(); i++){
//    cout << "Elevation = " << c.get_elevation(i) << endl;
//    cout << "Azimuth = " << c.get_azimuth(i) << endl;
//    cout << "Power = " << c.get_power(i) << endl;
//    cout << endl;
//  }
  
//  fi2.close();



  return(0);

}
