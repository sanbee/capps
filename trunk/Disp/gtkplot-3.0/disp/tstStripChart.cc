#include <iostream.h>
#include <stdio.h>
#include "StripChart.h"
#include <forms.h>

main(int argc, char **argv)
{
  FL_FORM *Form;
  char str[10];
  Display *TheDisplay;
  StripChart SC[10];

  TheDisplay = fl_initialize(&argc, argv, "StripChart", 0,0);

  Form = fl_bgn_form(FL_NO_BOX,480,310);
  {
    fl_add_box(FL_UP_BOX,0,0,480,310,""); 
    SC[0].Init(Form,410,308);
  }
  fl_end_form();

  fl_show_form(Form,FL_PLACE_FREE,FL_FULLBORDER,"StripChart");
  fl_do_forms();


  scanf("%s",str);
}
