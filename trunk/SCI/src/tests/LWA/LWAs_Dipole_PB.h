#ifndef LWAs_Dipole_PB_H
#define LWAs_Dipole_PB_H

#include <vector>
#include <fstream>
#include <string>

using namespace std;

class LWAs_Dipole_PB
{
    private:
      
      //deg
      vector<double> Elevation;
      vector<double> Azimuth;
      vector<double> Power;
    
      string filename;
      
      double gap_angle;       

    public:
         
      //constructor
      LWAs_Dipole_PB();
      LWAs_Dipole_PB(string&);
      
      //destructor
      ~LWAs_Dipole_PB(); 

      //input file
      void Input_file(string&);

      //get
      string get_fname()           {return(filename);}
      double get_elevation(int i)  {return(Elevation[i]);}
      double get_azimuth(int i)    {return(Azimuth[i]);}
      double get_power(int i)      {return(Power[i]);}
      unsigned long int get_size() {return(Elevation.size());}

      double get_gap_angle() {return(gap_angle);}
};
 
//constructor
LWAs_Dipole_PB::LWAs_Dipole_PB()
{
  filename = "aaa.txt";
  //Input_file(filename);

  gap_angle = 1.0;
}


//constructor
LWAs_Dipole_PB::LWAs_Dipole_PB(string& fname)
{
  filename = fname;
  Input_file(fname);
  
  gap_angle = 1.0;
}

//destructor
LWAs_Dipole_PB::~LWAs_Dipole_PB()
{
}


//input file
void LWAs_Dipole_PB::Input_file(string& fname)
{
  fstream fi;
  fi.open(fname.c_str());

  double el;
  double az;
  double power;
  
  while(fi >> el){
    fi >> az >> power;
    Elevation.push_back(el);
    Azimuth.push_back(az);
    Power.push_back(power);
  }

  fi.close();
}




#endif

