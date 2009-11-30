/* gtkplotfont - Fonts handling for gtkplot
 * Copyright 1999  Adrian E. Feiguin <feiguin@ifir.edu.ar>
 *
 * Some code borrowed from
 * DiaCanvas -- a technical canvas widget
 * Copyright (C) 1999 Arjan Molenaar
 * Dia -- an diagram creation/manipulation program
 * Copyright (C) 1998 Alexander Larsson
 *
 * and Xfig
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <gtk/gtk.h>
#include <stdio.h>
#include <string.h>
#include "gtkplotfont.h"

#define FONTCACHE_SIZE 17
#define NUM_X11_FONTS 2

static GtkPlotFont font_data[] = 
{
  { "Times-Roman",
    "Times-Roman",
    { "-adobe-times-medium-r-normal",
      NULL
    }
  }, 
  { "Times-Italic",
    "Times-Italic",
    { "-adobe-times-medium-i-normal",
      NULL
    }
  }, 
  { "Times-Bold",
    "Times-Bold",
    { "-adobe-times-bold-r-normal",
      NULL
    }
  }, 
  { "Times-BoldItalic",
    "Times-BoldItalic",
    { "-adobe-times-bold-i-normal",
      NULL
    }
  }, 
  { "AvantGarde-Book",
    "AvantGarde-Book",
    { "-adobe-avantgarde-book-r-normal",
      "-schumacher-clean-medium-r-normal"
    }
  },
  { "AvantGarde-BookOblique",
    "AvantGarde-BookOblique",
    { "-adobe-avantgarde-book-o-normal",
      "-schumacher-clean-medium-i-normal"
    }
  },
  { "AvantGarde-Demi",
    "AvantGarde-Demi",
    { "-adobe-avantgarde-demibold-r-normal",
      "-schumacher-clean-bold-r-normal"
    }
  },
  { "AvantGarde-DemiOblique",
    "AvantGarde-DemiOblique",
    { "-adobe-avantgarde-demibold-o-normal",
      "-schumacher-clean-bold-i-normal"
    }
  },
  { "Bookman-Light",
    "Bookman-Light",
    { "-adobe-bookman-light-r-normal",
      "-adobe-times-medium-r-normal"
    }
  },
  { "Bookman-LightItalic",
    "Bookman-LightItalic",
    { "-adobe-bookman-light-i-normal",
      "-adobe-times-medium-i-normal"
    }
  },
  { "Bookman-Demi",
    "Bookman-Demi",
    { "-adobe-bookman-demibold-r-normal",
      "-adobe-times-bold-r-normal"
    }
  },
  { "Bookman-DemiItalic",
    "Bookman-DemiItalic",
    { "-adobe-bookman-demibold-i-normal",
      "-adobe-times-bold-i-normal"
    }
  },
  { "Courier",
    "Courier",
    { "-adobe-courier-medium-r-normal",
      NULL
    }
  },
  { "Courier-Oblique",
    "Courier-Oblique",
    { "-adobe-courier-medium-o-normal",
      NULL
    }
  },
  { "Courier-Bold",
    "Courier-Bold",
    { "-adobe-courier-bold-r-normal",
      NULL
    }
  },
  { "Courier-BoldOblique",
    "Courier-BoldOblique",
    { "-adobe-courier-bold-o-normal",
      NULL
    }
  },
  { "Helvetica",
    "Helvetica",
    { "-adobe-helvetica-medium-r-normal",
      NULL
    }
  },
  { "Helvetica-Oblique",
    "Helvetica-Oblique",
    { "-adobe-helvetica-medium-o-normal",
      NULL
    }
  },
  { "Helvetica-Bold",
    "Helvetica-Bold",
    { "-adobe-helvetica-bold-r-normal",
      NULL
    }
  },
  { "Helvetica-BoldOblique",
    "Helvetica-BoldOblique",
    { "-adobe-helvetica-bold-o-normal",
      NULL
    }
  },
  { "Helvetica-Narrow",
    "Helvetica-Narrow",
    { "-adobe-helvetica-medium-r-normal",
      NULL
    }
  },
  { "Helvetica-Narrow-Oblique",
    "Helvetica-Narrow-Oblique",
    { "-adobe-helvetica-medium-o-normal",
      NULL
    }
  },
  { "Helvetica-Narrow-Bold",
    "Helvetica-Narrow-Bold",
    { "-adobe-helvetica-bold-r-normal",
      NULL
    }
  },
  { "Helvetica-Narrow-BoldOblique",
    "Helvetica-Narrow-BoldOblique",
    { "-adobe-helvetica-bold-o-normal",
      NULL
    }
  },
  { "NewCenturySchoolbook-Roman",
    "NewCenturySchlbk-Roman",
    { "-adobe-new century schoolbook-medium-r-normal",
      NULL
    }
  },
  { "NewCenturySchoolbook-Italic",
    "NewCenturySchlbk-Italic",
    { "-adobe-new century schoolbook-medium-i-normal",
      NULL
    }
  },
  { "NewCenturySchoolbook-Bold",
    "NewCenturySchlbk-Bold",
    { "-adobe-new century schoolbook-bold-r-normal",
      NULL
    }
  },
  { "NewCenturySchoolbook-BoldItalic",
    "NewCenturySchlbk-BoldItalic",
    { "-adobe-new century schoolbook-bold-i-normal",
      NULL
    }
  },
  { "Palatino-Roman",
    "Palatino-Roman",
    { "-adobe-palatino-medium-r-normal",
      "-*-lucidabright-medium-r-normal"
    }
  },
  { "Palatino-Italic",
    "Palatino-Italic",
    { "-adobe-palatino-medium-i-normal",
      "-*-lucidabright-medium-i-normal"
    }
  },
  { "Palatino-Bold",
    "Palatino-Bold",
    { "-adobe-palatino-bold-r-normal",
      "-*-lucidabright-demibold-r-normal"
    }
  },
  { "Palatino-BoldItalic",
    "Palatino-BoldItalic",
    { "-adobe-palatino-bold-i-normal",
      "-*-lucidabright-demibold-i-normal"
    }
  },
  { "Symbol",
    "Symbol",
    {
      "-adobe-symbol-medium-r-normal",
      "-*-symbol-medium-r-normal"
    }
  },
  { "ZapfChancery-MediumItalic",
    "ZapfChancery-MediumItalic",
    { "-adobe-zapf chancery-medium-i-normal",
      "-*-itc zapf chancery-medium-i-normal"
    }
  },
  { "ZapfDingbats",
    "ZapfDingbats",
    { "-adobe-zapf dingbats-medium-r-normal",
      "-*-itc zapf dingbats-*-*-*"
    }
  },
};

#define NUM_FONTS (sizeof(font_data)/sizeof(GtkPlotFont))

gchar *last_resort_fonts[] = {
  "-adobe-courier-medium-r-normal",
  "fixed" /* Must be last. This is guaranteed to exist on an X11 system. */
};

#define NUM_LAST_RESORT_FONTS (sizeof(last_resort_fonts)/sizeof(GtkPlotFont))

static GList *user_fonts;

static GtkPlotFont *	find_psfont		(gchar *name);

GtkPlotFont *
gtk_plot_font_getfont(gchar *name)
{
  GtkPlotFont *font;


  font = find_psfont(name);

  if (font == NULL) {
    font = find_psfont("Courier");
    if (font == NULL) {
      g_warning ("Error, couldn't locate font. Shouldn't happend.\n");
    } else {
      g_message ("Font %s not found, using Courier instead.\n", name);
    }
  }

  return (GtkPlotFont *)font;
}

GdkFont *
gtk_plot_font_get_gdkfont(gchar *name, gint height)
{
  GtkPlotFont *fontdata;
  GdkFont *gdk_font = NULL;
  gchar *x11_font;
  gint bufsize;
  gchar *buffer;
  gint i;
  gint auxheight;

  if (height <= 0) height = 1;
 
  fontdata = gtk_plot_font_getfont(name);
 
  for (i=0;i<NUM_X11_FONTS;i++) {
    x11_font = fontdata->xfont[i];
    if (x11_font != NULL) {
     bufsize = strlen(x11_font)+25;  /* Should be enought*/
     buffer = (gchar *)g_malloc(bufsize);

     for(auxheight = height; auxheight <= 2*height; auxheight++){
      sprintf(buffer, "%s-*-%d-*-*-*-*-*-*-*", x11_font, auxheight);
    
      gdk_font = gdk_font_load(buffer);
      if (gdk_font != NULL) {
         g_free(buffer);
         break;
      }
     }

     if(gdk_font != NULL) break;
    }

    g_free(buffer);
  }

  if (gdk_font == NULL) {
    for (i=0;i<NUM_LAST_RESORT_FONTS;i++) {
      x11_font = last_resort_fonts[i];
      bufsize = strlen(x11_font)+25;  /* Should be enought*/
      buffer = (char *)g_malloc(bufsize);
      
      for(auxheight = height; auxheight <= 2*height; auxheight++){
       sprintf(buffer, "%s-*-%d-*-*-*-*-*-*-*", x11_font, auxheight);
    
       gdk_font = gdk_font_load(buffer);
        if (gdk_font != NULL) {
          g_free(buffer);
          break;
       }
	g_free(buffer); //SanB

      }

      if (gdk_font != NULL) {
	g_warning("Couldn't find X Font for %s, using %s instead.\n",
		  name, x11_font);
	break;
      }
    }
  }

  if (gdk_font == NULL) 
	g_warning("Couldn't find X Font for %s", name);
	
  return gdk_font;
}


gchar *
gtk_plot_font_get_psfontname(gchar *fontname)
{
  GtkPlotFont *font = NULL;
 
  font = find_psfont(fontname); 
  if(!font) 
     font = find_psfont("Courier");  

  return font->psname;
}

void
gtk_plot_font_add_font (gchar *fontname, gchar *psname, gchar *x_string[])
{
  GtkPlotFont *font;

  font = g_new(GtkPlotFont, 1);

  font->fontname = g_strdup(fontname); 
  font->psname = g_strdup(psname); 
  font->xfont[0] = g_strdup(x_string[0]);
  font->xfont[1] = g_strdup(x_string[1]);

  user_fonts = g_list_append(user_fonts, font);
}

static GtkPlotFont *
find_psfont(gchar *name)
{
  GtkPlotFont *fontdata = NULL;
  GtkPlotFont *data = NULL;
  GList *fonts;
  gint i;

  for(i = 0; i < NUM_FONTS; i++){
    if(strcmp(name, font_data[i].fontname) == 0) { 
       fontdata = &font_data[i];
       break;
    }
    if(strcmp(name, font_data[i].psname) == 0) { 
       fontdata = &font_data[i];
       break;
    }
  }


  if(font_data == NULL) {
    fonts = user_fonts;
    while(fonts){
      data = (GtkPlotFont *) fonts->data;
      if(strcmp(name, data->fontname) == 0) {
         fontdata = data;
         break;
      }
      if(strcmp(name, data->psname) == 0) {
         fontdata = data;
         break;
      }
      fonts = fonts->next;
    }
  }


  return fontdata;
}
