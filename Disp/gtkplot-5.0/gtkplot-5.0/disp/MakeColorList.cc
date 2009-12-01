#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <namespace.h>

int MakeColorList(vector<char *> &ColorList)
{
  ifstream infile;
  string Name;
  char *t=getenv("MPCOLOURS");
  int i=0;

  if (t) infile.open(t);
  else return -1;

  if (!infile.is_open()) {return -1;}

  while (!infile.eof())
    {
      infile >> Name;
      if ((Name.size() > 0) && (Name[0] != '#'))
	{
	  ColorList.resize(i+1);
	  ColorList[i]=new char [Name.size()+1];
	  strcpy(ColorList[i],Name.c_str());
	  i++;
	}
    }
  ColorList.resize(i-1);
  return ColorList.size();
}
