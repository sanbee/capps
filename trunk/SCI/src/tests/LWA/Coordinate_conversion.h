//***This class is the one to convert a (Hour,Dec,Latitude) coordinate to a (Az,El) coordinate.***

#ifndef Convert_HDtoAE_H
#define Convert_HDtoAE_H


#include <cmath>
//#include <math.h>

using namespace std;

class Convert_HDtoAE
{
    private:
      //Observatory latitude (rad) pi/2->north pole
      long double Obs_Lati;     
      
      //radio source Dec(rad),H(rad)
      long double Dec;
      long double Hour;
      
      //radio source (h,m,s) 
      int Hour_h;
      int Hour_m;
      long double Hour_s;
      
      //rad
      long double Az;
      long double El;
      
      //direction cos
      long double x_south;
      long double y_east;
      long double z_zenith;
  
      //set
      void set1(const long double&, const long double&, const int&, const int&, const long double&);
      void set2(const long double&, const long double&, const long double&);


      //calculate 
      long double calc_HourtoRad(const int&, const int&, const long double&);

 
    public:

      //constructor
      Convert_HDtoAE();
      Convert_HDtoAE(const long double&, const long double&,
                     const int&, const int&, const long double&);
      Convert_HDtoAE(const long double&, const long double&, const long double&);


      //destructor
      ~Convert_HDtoAE();


      //set
      void set_Obs_Lati(const long double& l)  {Obs_Lati = l;}
      void set_Hour(const double& h)           {Hour = h;}
      void set_Hour_h(const int& h)            {Hour_h = h;}
      void set_Hour_m(const int& m)            {Hour_m = m;}
      void set_Hour_s(const long double& s)    {Hour_s = s;}
      void set_Dec(const long double& dc)      {Dec  = dc;}
      void set_Az(const long double& az)       {Az   = az;}
      void set_El(const long double& el)       {El   = el;}


      //get
      long double get_Obs_Lati()       {return(Obs_Lati);}
      long double get_Dec()        {return(Dec);}
      long double get_Hour()       {return(Hour);}
      int get_Hour_h()             {return(Hour_h);}
      int get_Hour_m()             {return(Hour_m);}
      long double get_Hour_s()     {return(Hour_s);}
      long double get_Az()         {return(Az);}
      long double get_El()         {return(El);}      
      long double get_x_south()    {return(x_south);}
      long double get_y_east()     {return(y_east);}
      long double get_z_zenith()   {return(z_zenith);}
      
      //calculate 
      void calc();

};

//constructor
Convert_HDtoAE::Convert_HDtoAE()
{
  Obs_Lati     = M_PI/2.0;
  Dec      = 0.0;
  Hour     = 0.0;
  Hour_h   = 0;
  Hour_m   = 0;
  Hour_s   = 0.0;
  Az       = M_PI;
  El       = 0.0;

  x_south  = 0.0;
  y_east   = 0.0;
  z_zenith = 0.0;
}


//constructor
Convert_HDtoAE::Convert_HDtoAE(const long double& l, const long double& d,
                               const int& h, const int& m, const long double& s)
{
  set1(l, d, h, m, s);
  calc(); 
}


//constructor
Convert_HDtoAE::Convert_HDtoAE(const long double& l, const long double& d, const long double& h)
{  
  set2(l, d, h); 
  calc(); 
}


//destructor
Convert_HDtoAE::~Convert_HDtoAE()
{
}


//set
void Convert_HDtoAE::set1(const long double& l, const long double& d, const int& h, const int& m, const long double& s)
{
  Obs_Lati = l;
  Dec      = d;
  Hour_h   = h;
  Hour_m   = m;
  Hour_s   = s;
  Hour     = calc_HourtoRad(Hour_h, Hour_m, Hour_s);
}


//set
void Convert_HDtoAE::set2(const long double& l, const long double& d, const long double& h)
{
  Obs_Lati = l;
  Dec      = d;
  Hour     = h;

  Hour_h   = int(Hour*12.0/M_PI);
  Hour_m   = int(((Hour*12.0/M_PI)-Hour_h)*60.0);
  Hour_s   = ((((Hour*12.0/M_PI)-Hour_h)*60.0)-Hour_m)*60.0;

  /*
  if( Hour_s == 60.0 ){
    Hour_s = 0.0;
    Hour_m += 1;
  }

  if( Hour_m == 60 ){
    Hour_m = 0;
    Hour_h += 1;
  }
  
  for(;;){
    if( Hour_h >= 24 ){
      Hour_h = Hour_h - 24;
    }else{
      break;
    }
  }
  */

}


//calculate
void Convert_HDtoAE::calc()
{
  x_south = sinl(Obs_Lati)*cosl(Dec)*cosl(Hour)-cosl(Obs_Lati)*sinl(Dec);
  y_east = -cosl(Dec)*sinl(Hour);
  z_zenith = cosl(Obs_Lati)*cosl(Dec)*cosl(Hour)+sinl(Obs_Lati)*sinl(Dec);
  
  Az = M_PI - atan2l(y_east,x_south);
  
  if( Az == 2*M_PI ){
    Az = 0.0;
  }
  
  El = atan2l(z_zenith,sqrtl(powl(x_south,2.0)+powl(y_east,2.0)));
  
}


//calculate
long double Convert_HDtoAE::calc_HourtoRad(const int& h, const int& m , const long double& s)
{
  return (h + m/60.0 + s/3600.0)*(2.0*M_PI)/24.0;
} 


#endif


