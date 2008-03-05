
#ifndef LWAs2_H
#define LWAs2_H

#include "LWAs.h"


using namespace std;

class LWAs2 : public LWAs
{
    private:
      

      //LWA station Longtitude, rad,h,m,s
      double Obs_Long;
      int Obs_Long_h;
      int Obs_Long_m;
      double Obs_Long_s;
      
      

    public:

      //constractor
      LWAs2();
      LWAs2(const double& obslong,const double& F, const double& FW, const unsigned int& Ch, 
           bool DPB, bool NS, const double& lm_max,  
           double& obs_lati, const double& Dec, 
           const int& H_h, const int& H_m, const double& H_s,
	   string& DPBfile, string& Dconfile);

      LWAs2(const int& obslong_h,const int& obslong_m,const double& obslong_s,
           const double& F, const double& FW, const unsigned int& Ch, 
           bool DPB, bool NS, const double& lm_max,  
           double& obs_lati, const double& Dec, 
           const int& H_h, const int& H_m, const double& H_s,
	   string& DPBfile, string& Dconfile);



      //set
      void set_ObsLong(double i)   {Obs_Long = i;}
      void set_ObsLong_h(int i)    {Obs_Long_h = i;}
      void set_ObsLong_m(int i)    {Obs_Long_m = i;}
      void set_ObsLong_s(double i) {Obs_Long_s = i;}

      //get
      double get_Obs_Long()       {return(Obs_Long);}
      int get_Obs_Long_h()        {return(Obs_Long_h);}
      int get_Obs_Long_m()        {return(Obs_Long_m);}
      double get_Obs_Long_s()     {return(Obs_Long_s);}

};

//constractor
LWAs2::LWAs2():LWAs()
{
  //rad
  Obs_Long = 0.0;
  
  Obs_Long_h     = 0;
  Obs_Long_m     = 0;
  Obs_Long_s     = 0.0;
}

LWAs2::LWAs2(const double& obslong,const double& F, const double& FW, const unsigned int& Ch, 
             bool DPB, bool NS, const double& lm_max, 
             double& obs_lati, const double& Dec, 
             const int& H_h, const int& H_m, const double& H_s,
	     string& DPBfile, string& Dconfile)
             : LWAs(F,FW,Ch,DPB,NS,lm_max,obs_lati,Dec,H_h,H_m,H_s,DPBfile,Dconfile)
{

  //rad
  Obs_Long = obslong;
  
  //h,m,s
  Obs_Long_h     = int(obslong*12.0/M_PI);
  Obs_Long_m     = int(((obslong*12.0/M_PI)-Obs_Long_h)*60.0);
  Obs_Long_s     = ((((obslong*12.0/M_PI)-Obs_Long_h)*60.0)-Obs_Long_m)*60.0;

}

LWAs2::LWAs2(const int& obslong_h,const int& obslong_m,const double& obslong_s,
             const double& F, const double& FW, const unsigned int& Ch, 
             bool DPB, bool NS, const double& lm_max,  
             double& obs_lati, const double& Dec, 
             const int& H_h, const int& H_m, const double& H_s,
	     string& DPBfile, string& Dconfile)
             : LWAs(F,FW,Ch,DPB,NS,lm_max,obs_lati,Dec,H_h,H_m,H_s,DPBfile,Dconfile)
{
  //h,m,s
  Obs_Long_h     = obslong_h;
  Obs_Long_m     = obslong_m;
  Obs_Long_s     = obslong_s;

  //rad
  Obs_Long = (obslong_h + obslong_m/60.0 + obslong_s/3600.0)*M_PI/12.0;

}



#endif
