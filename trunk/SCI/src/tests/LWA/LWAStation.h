#ifndef LWAs_H
#define LWAs_H

#include "Coordinate_conversion.h"
#include "LWAs_Dipole_Configuration.h"
#include "LWAs_Dipole_PB.h"

#include "LWAStation.cc"

#include <iostream>

using namespace std;

class LWAs : public Convert_HDtoAE, public LWAs_Dipole_PB, public LWAs_Dipole_Conf
{
    private:
      
      double Frequency;
      double Frequency_Width;
      double C;
      
      double Delta_Frequency;
      unsigned int Channel;
      
          
      bool Diople_PB;
      bool Noise_Switch;

      double lm_Max;
          

      vector<long double> LWAs_Ereal;
      vector<long double> LWAs_Eimag;
      vector<long double> LWAs_Power;

      vector<double> LWAs_L;
      vector<double> LWAs_M; 
      

      //LWA station Primary Beam Center(rad),(x,y,z)
      //double LWAs_PBCenter;
      //double LWAs_PBCenter_Xsouth;
      //double LWAs_PBCenter_Yeast;
      //double LWAs_PBCenter_Zzenith;


      vector<double> rotation_W;

      //frequency devided by nchan
      vector<double> f;

    public:
    
      //constructor
      LWAs();
      
      LWAs(const double& F, const double& FW, const unsigned int& Ch, 
           bool DPB, bool NS, const double& lm_max, 
           double& obs_lati, const double& Dec, 
           const int& H_h, const int& H_m, const double& H_s,
	   string& DPBfile, string& Dconfile);
      
      LWAs(const double& F, const double& FW, const unsigned int& Ch, 
           bool DPB, bool NS, const double& lm_max,
           double& obs_lati, const double& Dec, 
           const double& H,
	   string& DPBfile, string& Dconfile);

      //destructor
      ~LWAs();


      //set
      void set_Freq(double& F)       {Frequency = F;}
      void set_FreqW(double& FW)     {Frequency_Width = FW;} 
      void set_Ch(const int& ch)     {Channel = ch;}

      //get
      double get_Freq()       {return(Frequency);}
      double get_FreqW()      {return(Frequency_Width);} 
      double get_C()          {return(C);}
      
      double get_DeltaFreq()  {return(Delta_Frequency);}
      unsigned int get_Ch()         {return(Channel);}

      long double get_rotation_W(int& i) {return(rotation_W[i]);}
      int get_rotation_W_size() {return(rotation_W.size());}

      long double get_LWAs_Ereal(long int i) {return(LWAs_Ereal[i]);}
      long double get_LWAs_Eimag(long int i) {return(LWAs_Eimag[i]);}
      long double get_LWAs_Power(long int i) {return(LWAs_Power[i]);}
      
      double get_LWAs_L(long int i) {return(LWAs_L[i]);}
      double get_LWAs_M(long int i) {return(LWAs_M[i]);}
      
      long int get_LWAs_Erealsize(){return(LWAs_Ereal.size());}
      long int get_LWAs_Eimagsize(){return(LWAs_Eimag.size());}
      long int get_LWAs_Powersize(){return(LWAs_Power.size());}
      
      long int get_LWAs_Lsize(){return(LWAs_L.size());}
      long int get_LWAs_Msize(){return(LWAs_M.size());}

      double get_lm_Max()  {return(lm_Max);}
      //double get_lm_Step() {return(lm_Step);}

      //bool
      void D_PB(bool pb)             {Diople_PB = pb;}
      void N_SW(bool n)              {Noise_Switch = n;}

      //calculation
      void calc(); 
      
      //Gometrical delay on each dipole from the direction of beam center
      void calc_GeometricalDelay();

};

//constractor
LWAs::LWAs():Convert_HDtoAE(), LWAs_Dipole_PB(), LWAs_Dipole_Conf()
{
  Frequency = 0.0;
  Frequency_Width = 0.0;
  C=2.99792458*powl(10.0, 8.0);
  
  Channel = 1;    
  Delta_Frequency = Frequency_Width/Channel;
      
  Diople_PB = 1;
  Noise_Switch = 0;


  
  //LWA VL station(rad)
  //Obs_Lati = 34.07*M_PI/180.0;

 

}

//constractor1
LWAs::LWAs(const double& F, const double& FW, const unsigned int& Ch, 
           bool DPB, bool NS, const double& lm_max,
           double& obs_lati, const double& Dec, 
           const int& H_h, const int& H_m, const double& H_s,
	   string& DPBfile, string& Dconfile)
           : Convert_HDtoAE(obs_lati, Dec, H_h, H_m, H_s), 
             LWAs_Dipole_PB(DPBfile), LWAs_Dipole_Conf(Dconfile)

{
  Frequency = F;
  Frequency_Width = FW;
  C = 2.99792458*powl(10.0, 8.0);
  
  Channel = Ch;    
  Delta_Frequency = Frequency_Width/Channel;
      
  Diople_PB = DPB;
  Noise_Switch = NS; 

  lm_Max = lm_max;
  //lm_Step = lm_step;

//Obs_Lati = obs_lati;

  calc_GeometricalDelay();

  for(unsigned int i = 0; i < Channel; i++){
    f.push_back(F-FW/2.0+Delta_Frequency*i+Delta_Frequency/2.0);
  }


  calc();

}

//constractor2
LWAs::LWAs(const double& F, const double& FW, const unsigned int& Ch, 
           bool DPB, bool NS, const double& lm_max,
           double& obs_lati, const double& Dec, 
           const double& H,
	   string& DPBfile, string& Dconfile)
           : Convert_HDtoAE(obs_lati, Dec, H), 
             LWAs_Dipole_PB(DPBfile), LWAs_Dipole_Conf(Dconfile)

{
  Frequency = F;
  Frequency_Width = FW;
  C = 2.99792458*powl(10.0, 8.0);
  
  Channel = Ch;    
  Delta_Frequency = Frequency_Width/Channel;
      
  Diople_PB = DPB;
  Noise_Switch = NS; 

  lm_Max = lm_max;
  //lm_Step = lm_step;

//Obs_Lati = obs_lati;

  calc_GeometricalDelay();

  for(unsigned int i = 0; i < Channel; i++){
    f.push_back(F-FW/2.0+Delta_Frequency*i+Delta_Frequency/2.0);
  }


  calc();

}



//destructor
LWAs::~LWAs()
{
}



#endif

      
