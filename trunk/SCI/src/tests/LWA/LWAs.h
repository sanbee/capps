#ifndef LWAs_H
#define LWAs_H

#include "Coordinate_conversion.h"
#include "LWAs_Dipole_Configuration.h"
#include "LWAs_Dipole_PB.h"

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
      //      double lm_Step;
      


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

void LWAs::calc_GeometricalDelay()
{
  //***Direction cosine***
  long double x_south  = Convert_HDtoAE::get_x_south();
  long double y_east   = Convert_HDtoAE::get_y_east();
  long double z_zenith = Convert_HDtoAE::get_z_zenith();

 
  for(long int i = 0; i < LWAs_Dipole_Conf::get_size(); i++){
    rotation_W.push_back(-LWAs_Dipole_Conf::get_North(i)*x_south+
                          LWAs_Dipole_Conf::get_East(i)*y_east+
			  LWAs_Dipole_Conf::get_Zenith()*z_zenith);
  }

}





//Calculation
void LWAs::calc()
{

 
  int search_direc = 5;
  


  for(double l=-lm_Max; l <= lm_Max; l+=1){
    double power;
    double E_real;
    double E_imag;

    double l2 = l/lm_Max;

cout << l << endl;

    for(double m=-lm_Max; m <=lm_Max; m+=1){
          
      double m2 = m/lm_Max;

      double check = 1-powl(l2,2.0)-powl(m2,2.0); 
    
      if(check < 0.0){
        power = 0.0;
        E_real = 0.0;
        E_imag = 0.0;

        LWAs_L.push_back(l2);
        LWAs_M.push_back(m2);
        LWAs_Power.push_back(power);
        
        for(unsigned int i = 0; i < Channel; i++){
          LWAs_Ereal.push_back(E_real);
          LWAs_Eimag.push_back(E_imag);
        }
       
	
     
      }else{
      
        double  n = sqrtl(check);

        double L = Convert_HDtoAE::get_Obs_Lati();
        double H = Convert_HDtoAE::get_Hour();
        double D = Convert_HDtoAE::get_Dec();


        double x = l2*sin(L)*sin(H) + m2*(-sin(L)*cos(H)*sin(D)-cos(L)*cos(D)) + n*(sin(L)*cos(H)*cos(D)-cos(L)*sin(D));
        double y = l2*cos(H)+m2*sin(H)*sin(D)-n*sin(H)*cos(D);
        double z = l2*cos(L)*sin(H) + m2*(-cos(L)*cos(H)*sin(D)+sin(L)*cos(D)) + n*(cos(L)*cos(H)*cos(D)+sin(L)*sin(D));




/*
        //double x = cosl(Convert_HDtoAE::get_Az())*(m2*sinl(Convert_HDtoAE::get_El())
	//	 -n*cosl(Convert_HDtoAE::get_El()))-l2*sinl(Convert_HDtoAE::get_Az());
               
        double x = sinl(Convert_HDtoAE::get_Hour())*l2-cosl(Convert_HDtoAE::get_Hour())*
	  (sinl(Convert_HDtoAE::get_Dec())*m2-cosl(Convert_HDtoAE::get_Dec())*n);

        double y = (l2-x*sinl(Convert_HDtoAE::get_Hour()))/cosl(Convert_HDtoAE::get_Hour()); 

        //if(cosl(Convert_HDtoAE::get_Az()) == 0.0){
        //  y = (x*cosl(Convert_HDtoAE::get_Az())-m2*sinl(Convert_HDtoAE::get_El())
	//       +n*cosl(Convert_HDtoAE::get_El()))/sinl(Convert_HDtoAE::get_Az());
        //}else{
        //  y = (-l2-x*sinl(Convert_HDtoAE::get_Az()))/cosl(Convert_HDtoAE::get_Az());
        //} 

        double z = sqrtl(1.0-powl(x,2.0)-powl(y,2.0));

        x = sinl(Convert_HDtoAE::get_Obs_Lati())*x -z*cosl(Convert_HDtoAE::get_Obs_Lati());
        z = x*cosl(Convert_HDtoAE::get_Obs_Lati()) + z*sinl(Convert_HDtoAE::get_Obs_Lati());
*/



 
        vector<double> W;
       
        for(long int i = 0; i < LWAs_Dipole_Conf::get_size(); i++){
	  W.push_back(-1*LWAs_Dipole_Conf::get_North(i)*x + LWAs_Dipole_Conf::get_East(i)*y + LWAs_Dipole_Conf::get_Zenith()*z);
        }


        power = 0.0;
        E_real = 0.0;
        E_imag = 0.0;
               
      
        for(unsigned int k = 0; k < Channel; k++){
                
          for(unsigned int i = 0; i < LWAs_Dipole_Conf::get_size(); i++){
            
            double A = 2.0*M_PI*(W[i])/C;
            double B = 2.0*M_PI*(rotation_W[i])/C; 
            
            //if (l==0.0 && m==0.0){    
	    if(fabs(l2) < pow(10.0,-15) && fabs(m2) < pow(10.0,-15)){  
              
              E_real += Delta_Frequency*cosl((A-B)*f[k]);
              E_imag += Delta_Frequency*sinl((A-B)*f[k]);
              
              //if (Noise_Switch == 1){
              //  E_real += sqrtl(Delta_Frequency)*random.gauss(0,Noise_sig)
              //  E_imag += sqrtl(Delta_Frequency)*random.gauss(0,Noise_sig)
              //}  
             
            }else{
             
              E_real += Delta_Frequency*cosl((A-B)*f[k])*sinl((A-B)*(Delta_Frequency/2.0))/((A-B)*(Delta_Frequency/2.0));
              E_imag += Delta_Frequency*sinl((A-B)*f[k])*sinl((A-B)*(Delta_Frequency/2.0))/((A-B)*(Delta_Frequency/2.0));
	      
		//if (Noise_Switch == 1){
                //E_real += sqrtl(Delta_Frequency)*random.gauss(0,Noise_sig)
                //E_imag += sqrtl(Delta_Frequency)*random.gauss(0,Noise_sig)
	        //}
            
	    }
	  }
          
	  

          if (Diople_PB == 1){

            long double min = pow(10.0,10.0);

	    double angle = M_PI - atan2(y,x);
                                       
	    double del_angle = LWAs_Dipole_PB::get_size()/(2.0*M_PI);
                    
	    double DPB = 0.0;
	    unsigned int n  = 0;

	    //LWAs_Dipole_PB::get_gap_angle() = 1.0deg  search_direc = 5lines
	    int start = int(angle*del_angle-search_direc*(M_PI/180.0*LWAs_Dipole_PB::get_gap_angle())*del_angle);
            int stop  = int(0.5 + (angle*del_angle+search_direc*(M_PI/180.0*LWAs_Dipole_PB::get_gap_angle())*del_angle));
	    


////////
	    if (z < 0.0 ){
	      
	      DPB = 0.0;
	    
	    }else if (start < 0){
              
              int start1 = 0;
              int stop1 = stop;
    
              for(int i = start1;i < stop1; i++ ){
                double cs   = cosl(M_PI/180.0*(LWAs_Dipole_PB::get_elevation(i)));
                long double rms = sqrtl(powl(x-cs*cosl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0)
		                 +powl(y-cs*sinl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0));
	        
                if (rms < min){
                  min = rms;
                  DPB = powl(10.0, LWAs_Dipole_PB::get_power(i)/10.0);
		  n = i;
                }
	      }


              start1 = int(LWAs_Dipole_PB::get_size()-1 - search_direc*(M_PI/180.0*LWAs_Dipole_PB::get_gap_angle())*del_angle);
              stop1  = int(LWAs_Dipole_PB::get_size()-1);
              for(int i = start1;i < stop1; i++ ){
                double cs   = cosl(M_PI/180.0*(LWAs_Dipole_PB::get_elevation(i)));
                long double rms = sqrtl(powl(x-cs*cosl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0)
		                +powl(y-cs*sinl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0));  
                
                if (rms < min){
                  min = rms;
                  DPB = powl(10.0, LWAs_Dipole_PB::get_power(i)/10.0);
		  n = i;
                }
              }

              
/////////

	    }else if(stop > LWAs_Dipole_PB::get_size()-1){
              int start2 = start;
              int stop2 = LWAs_Dipole_PB::get_size()-1;

              for(int i = start2;i < stop2; i++ ){
                double cs   = cosl(M_PI/180.0*(LWAs_Dipole_PB::get_elevation(i)));
                long double rms = sqrtl(powl(x-cs*cosl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0)
		                 +powl(y-cs*sinl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0));
	        
                if (rms < min){
                  min = rms;
                  DPB = powl(10.0, LWAs_Dipole_PB::get_power(i)/10.0);
		  n = i;
                }
	      }


              start2 = 0;
              stop2  = int(search_direc*(M_PI/180.0*LWAs_Dipole_PB::get_gap_angle())*del_angle);
              for(int i = start2;i < stop2; i++ ){
                double cs   = cosl(M_PI/180.0*(LWAs_Dipole_PB::get_elevation(i)));
                long double rms = sqrtl(powl(x-cs*cosl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0)
	  	                 +powl(y-cs*sinl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0));  
                
                if (rms < min){
                  min = rms;
                  DPB = powl(10.0, LWAs_Dipole_PB::get_power(i)/10.0);
		  n = i;
                }
              }

            }else{
             
              for(int i = start;i < stop; i++ ){
                double cs   = cosl(M_PI/180.0*(LWAs_Dipole_PB::get_elevation(i)));
                long double rms = sqrtl(powl(x-cs*cosl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0)
		                 +powl(y-cs*sinl(M_PI-M_PI/180.0*(LWAs_Dipole_PB::get_azimuth(i))),2.0));
	        
                if (rms < min){
                  min = rms;
                  DPB = powl(10.0, LWAs_Dipole_PB::get_power(i)/10.0);
		  n = i;
                }
	      }                

            }
	
    
//////////


            LWAs_Ereal.push_back(E_real*sqrtl(DPB));
            LWAs_Eimag.push_back(E_imag*sqrtl(DPB));
            power += powl(E_real*sqrtl(DPB),2.0)+powl(E_imag*sqrtl(DPB),2.0);  

	  }else{
	    
            LWAs_Ereal.push_back(E_real);
            LWAs_Eimag.push_back(E_imag);
            power += powl(E_real,2.0)+powl(E_imag,2.0);

          }//if (Diople_PB == 1)

        }//for(unsigned int k = 0; k < Channel; k++)
       
      
    
        LWAs_L.push_back(l2);
        LWAs_M.push_back(m2);
        LWAs_Power.push_back(power);
        

      }//if(check <0.0)

    }//for(double m=-lm_max; m <=lm_max; m+=lm_Step)
  
  }//for(double l=-lm_max; l <= lm_max; l+=lm_Step)  

      

}

#endif

      
