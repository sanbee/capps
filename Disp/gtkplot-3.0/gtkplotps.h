/* gtkplotps - gtkplot PostScript driver
 * Copyright 1999  Adrian E. Feiguin <feiguin@ifir.edu.ar>
 *
 * Some few lines of code borrowed from
 * DiaCanvas -- a technical canvas widget
 * Copyright (C) 1999 Arjan Molenaar
 * Dia -- an diagram creation/manipulation program
 * Copyright (C) 1998 Alexander Larsson
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

#ifndef __GTK_PLOT_PS_H__
#define __GTK_PLOT_PS_H__


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Page size */

enum{
     GTK_PLOT_LETTER,
     GTK_PLOT_LEGAL,
     GTK_PLOT_A4,
     GTK_PLOT_EXECUTIVE,
     GTK_PLOT_CUSTOM,
};

#define GTK_PLOT_LETTER_W 	612   /* Width and Height in points */
#define GTK_PLOT_LETTER_H 	792

#define GTK_PLOT_LEGAL_W	612
#define GTK_PLOT_LEGAL_H	1008

#define GTK_PLOT_A4_W		595
#define GTK_PLOT_A4_H		842

#define GTK_PLOT_EXECUTIVE_W	540
#define GTK_PLOT_EXECUTIVE_H	720


/* Page orientation */
enum{
     GTK_PLOT_PORTRAIT,
     GTK_PLOT_LANDSCAPE,
};

/* Size units */
enum{
     GTK_PLOT_INCHES,
     GTK_PLOT_MM,
     GTK_PLOT_CM,
     GTK_PLOT_PSPOINTS,
};

typedef struct _GtkPlotEPS GtkPlotEPS;

struct _GtkPlotEPS
{
   FILE *psfile;
   gchar *psname;
   gint orientation;
   gint epsflag;

   /* measure units for page size */
   gint units;  
   gfloat width;
   gfloat height;

   /* page size in points (depends on orientation) */
   gint page_width;
   gint page_height;
};

GtkPlotEPS *gtk_plot_eps_new				(gchar *psname,
							 gint orientation,
							 gint epsflag,
							 gint page_size);
							 
GtkPlotEPS *gtk_plot_eps_new_with_size			(gchar *psname,
							 gint orientation,
							 gint epsflag,
							 gint units,
							 gfloat width,
							 gfloat height);

void gtk_plot_eps_set_size			        (GtkPlotEPS *eps,
							 gint units,
							 gfloat width,
							 gfloat height);


#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __GTK_PLOT_PS_H__ */

