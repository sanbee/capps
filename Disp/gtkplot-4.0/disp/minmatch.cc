#include <iostream.h>
#include <vector.h>
#include <string.h>
#include <string>

/*-----------------------------------------------------------------------------*/

int MinimumMatch(vector<string>& KeyWord, string& Msg)
{
/*
   Compares the string "Msg" with the strings in vector string
   "KeyWord". Does a mimimum match, starting from the 0th elements of
   the strings, and returns the number of matches found. If exactly 1
   match is found, replaces "Msg" with the element of KeyWord, which
   matched.
*/
  int i, j = 0, count = 0;
  for (i = 0; i < (int)KeyWord.size(); i++)
  {
    if ( KeyWord[i].substr(0,Msg.size()) == Msg)
    {
      j = i;
      count++;
    }
  }

  if (count == 1)
    Msg = KeyWord[j];

  return count;
}

/*-----------------------------------------------------------------------------*/

int GetKeyWords(vector<string>& KeyWord)
{
/*
   Supplies back a vector string "KeyWord" and returns the number of
   elements in it.  
*/
  int nkeys = 25;

  KeyWord.resize(nkeys); 
  KeyWord[0]  = "ascii";
  KeyWord[1]  = "autoscale";
  KeyWord[2]  = "binary";
  KeyWord[3]  = "clear";
  KeyWord[4]  = "colour";  // =
  KeyWord[5]  = "columns"; // =
  KeyWord[6]  = "exit";
  KeyWord[7]  = "in";      // =
  KeyWord[8]  = "linestyle"; // =
  KeyWord[9]  = "load";    // =
  KeyWord[10] = "markerstyle"; // =
  KeyWord[11] = "plot";
  KeyWord[12] = "replot";
  KeyWord[13] = "scroll";  // =
  KeyWord[14] = "title[yindex]";   // =
  KeyWord[15] = "xaxis";   // =
  KeyWord[16] = "xlabel[xindex]";  // =
  KeyWord[17] = "xpanels"; // =
  KeyWord[18] = "xrange";  // =
  KeyWord[19] = "xticks";  // =
  KeyWord[20] = "ylabel[yindex]";  // =
  KeyWord[21] = "ypanels"; // =
  KeyWord[22] = "yrange";  // =
  KeyWord[23] = "yticks";  // =

  return KeyWord.size();
}

/*-----------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------*/

int GetTok(string& Msg, vector<string>& word)
{
/*
   Splits up the ' ' sperated substrings in string "Msg" into
   induvidual strings, which are supplied as a vector string, "word".
   The way this algorithm works is that it reads "Msg" on charcter by
   character till a ' ' is hit. The part of "Msg" from the begining of
   "Msg" or the from the last ' ', as applicable, is then copied into
   "word[i]". It requires therfore that the last character in "Msg"
   should be a ' '.  Handles more than one consecutive ' 's
   sloppyly. Does not make much difference though.  
*/
  int i, j = 0, present = 0, prev = 0;

  EatExtraBlanks(Msg);
  Msg += " ";
  for (i = 0; i < (int)Msg.size(); i++)
  {
    if (Msg[i] == ' ') 
    {
      present = i;
      word.resize(j+1);
      word[j] = Msg.substr(prev, present-prev); 
      prev = present + 1;
      j++;
    }
  }
  return j;
}

/*-------------------------------------------------------------------------*/

main(void)
{
  int i, count, nwords, nkeys;
  string Msg = "aut 7   pl : ex rep xt";
  vector<string> KeyWord, word;

  nkeys = GetKeyWords(KeyWord);
  nwords = GetTok(Msg, word);

  cout << Msg << endl;

  Msg = "";
  for (i = 0; i < (int)word.size(); i++)
  {
    if (isalpha(word[i][0]))
    { 
      count = MinimumMatch(KeyWord, word[i]);
      if (count == 0)
      {
//  Throw an exception?        
        cerr << "No match " << "(" << word[i] << ")" << endl;
      }
      else if (count > 1)
      {
//  Throw an exception?        
        cerr << "No unambigous match " << "(" << word[i] << ")" << endl;
      }
    }
    Msg += word[i] + " ";
  }
  cout << Msg << endl;

/*
  if (Msg == "columns")
  {
  }
  else if (Msg == "xaxis")
  {
  }
  else if (Msg == "xpanels")
  {
  }
  else if (Msg == "in")
  {
  }
  else if (Msg == "plot")
  {
  }
*/
  columns = 7;
  xaxis = 1;
  xpanels = 2;
  in = "tst.dat";

  for (i = 0; i < 100; i++)
  {
    data[0] = i;
    for (j = 1; j < columns; j++)
      data[j] = i+j*i;

    for (j = 0; j < xpanels; j++)
    {
      MP[j].AddXData(&data[0]);
      MP[j].AddYData(&data[1+j*3], 3);
    }
  }
  
}

/*
if (strncmp(Msg, "autoscale", strlen("autoscale")) == 0)
{
}
else if (strncmp(Msg, "clear", strlen("clear")) == 0)
{
}
else if (strncmp(Msg, "exit", strlen("exit")) == 0)
{
  exit(0);
}
else if (strncmp(Msg, "plot", strlen("plot")) == 0)
{
 
}
else if (strncmp(Msg, "replot", strlen("replot")) == 0)
{
}
else if (strncmp(Msg, "xlabel", strlen("xlabel")) == 0)
{
}
else if (strncmp(Msg, "ylabel", strlen("ylabel")) == 0)
{
}
else if (strncmp(Msg, "title", strlen("title")) == 0)
{
}
else if (strncmp(Msg, "xrange", strlen("xrange")) == 0)
{
}
else if (strncmp(Msg, "yrange", strlen("yrange")) == 0)
{
}
*/

