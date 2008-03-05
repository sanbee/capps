#ifndef LWAs_Dipole_Conf_H
#define LWAs_Dipole_Conf_H

#include <vector>
#include <fstream>
#include <string>

using namespace std;

class LWAs_Dipole_Conf
{
    private:
      
      vector<long int> D_number;
      vector<double> East;
      vector<double> North;
    
      string filename;
      

    public:
      
      //constructor
      LWAs_Dipole_Conf();
      LWAs_Dipole_Conf(string&);

      //destructor
      ~LWAs_Dipole_Conf(); 

      //input file
      void Input_file(string&);

      //get
      string get_fname()      {return(filename);}
      unsigned long int get_number(int i)   {return(D_number[i]);}
      double get_East(int i)  {return(East[i]);}
      double get_North(int i) {return(North[i]);}
      double get_Zenith() {return(0.0);}
      unsigned long int get_size() {return(D_number.size());}

      
};
 
//constructor
LWAs_Dipole_Conf::LWAs_Dipole_Conf()
{
  filename = "LWA_strawman_station.txt";
  //Input_file(filename);
}

//constructor
LWAs_Dipole_Conf::LWAs_Dipole_Conf(string& fname)
{
  filename = fname;
  Input_file(fname);
      
}

//destructor
LWAs_Dipole_Conf::~LWAs_Dipole_Conf()
{
}


//input file
void LWAs_Dipole_Conf::Input_file(string& fname)
{
  fstream fi;
  fi.open(fname.c_str());
  
  long int number;
  double east;
  double north;
  
  while(fi >> number){
    fi >> east >> north;
    D_number.push_back(number);
    East.push_back(east);
    North.push_back(north);
  }
 
  fi.close();

  
}

#endif

