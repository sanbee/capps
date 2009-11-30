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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <gtk/gtk.h>

#include "gtkplot.h"
#include "gtkplotfont.h"
#include "gtkplotlayout.h"
#include "gtkplotcanvas.h"

static void gtk_plot_export			(GtkPlot *plot,
						 GtkPlotEPS *eps);
static void gtk_plot_layout_export		(GtkPlotLayout *layout,
						 GtkPlotEPS *eps);
static gint gtk_plot_ps_init			(GtkPlotEPS *eps, 
						 gfloat scalex, 
						 gfloat scaley);
static void gtk_plot_ps_leave			(GtkPlotEPS *eps);
static void gtk_plot_drawps			(GtkPlot *plot);
static void psdrawaxis				(GtkPlot *plot, 
						 GtkPlotAxis axis, 
						 int x, int y);
static void psdrawgrids				(GtkPlot *plot);
static void psdrawlabels			(GtkPlot *plot,
            					 GtkPlotAxis axis,
            					 gint x, gint y);
static void psdrawerrbars			(GtkPlot *plot, 
						 GtkPlotData *dataset, 
						 GtkPlotPoint point);
static void psconnectpoints			(GtkPlot *plot, 
						 GtkPlotData *dataset);
static void psdrawxy 				(GtkPlot *plot,
                  				 GtkPlotData *dataset,
                  				 GtkPlotPoint point);
static void psdrawlegends 			(GtkPlot *plot);
static void psdrawlines				(GtkPlotEPS *eps,
						 GdkPoint *points, 
						 gint numpoints);
static void psdrawline				(GtkPlotEPS *eps,
						 gint x0, gint y0, 
						 gint xf, gint yf);
static void psdrawpolygon			(GtkPlotEPS *eps,
						 GdkPoint *points, 
						 gint numpoints, 
						 gint filled);
static void psgetpoint				(GtkPlot *plot, 
						 gdouble px, gdouble py, 
						 gint *x, gint *y);
static void pssetpoint				(GtkPlot *plot, 
						 gint x, gint y, 
						 GtkPlotPoint *point);
static void psdrawdataset			(GtkPlot *plot, 
						 GtkPlotData *dataset);
static void psdrawpoint                 	(GtkPlot *plot,
						 gboolean clip,
                                                 GtkPlotPoint point,
                                                 GtkPlotSymbolType symbol,
                                                 GdkColor color,
                                                 GtkPlotSymbolStyle symbol_style,
                                                 gint symbol_size,
                                                 gint line_width);
static void psdrawsymbol                	(GtkPlot *plot,
                                                 GtkPlotPoint point,
                                                 GtkPlotSymbolType symbol,
                                                 GdkColor color,
                                                 gint filled,
                                                 gint symbol_size,
                                                 gint line_width);
static void psdrawrectangle			(GtkPlotEPS *eps,
                                                 gint x, gint y, 
						 gint x1, gint y1, 
						 gint filled);
static void psdrawsquare                        (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size,
                                                 gint filled);
static void psdrawdowntriangle                  (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size,
                                                 gint filled);
static void psdrawuptriangle                    (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size,
                                                 gint filled);
static void psdrawdiamond                       (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size,
                                                 gint filled);
static void psdrawplus                          (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size);
static void psdrawcross                         (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size);
static void psdrawstar                          (GtkPlotEPS *eps,
                                                 gint x, gint y,
                                                 gint size);
static void psdrawcircle			(GtkPlotEPS *eps,
                                                 gint x, gint y, 
						 gint size, gint filled);

static void pssetcolor				(GtkPlotEPS *eps, 
						 GdkColor color); 
static void pssetlinewidth			(GtkPlotEPS *eps, 
						 gint width);
static void pssetlinestyle			(GtkPlotEPS *eps, 
						 GtkPlotLine line);

static void psdrawstring			(GtkPlotEPS *eps,
             					 gint x, gint y,
             					 gint justification,
             					 gint angle,
             					 gchar *text);
static void psdrawtext				(GtkWidget *widget, 
						 GtkPlotEPS *eps,
						 GtkPlotText text);
static void pssetfont				(GtkPlotEPS *eps, 
						 gchar *font, 
						 gint height);

static gint transform_y                         (GtkPlot *plot, gdouble y);
static gint transform_x                         (GtkPlot *plot, gdouble x);
static gint transform_dy                        (GtkPlot *plot, gdouble dy);
static gint transform_dx                        (GtkPlot *plot, gdouble dx);
static gdouble inverse_y                        (GtkPlot *plot, gint y);
static gdouble inverse_x                        (GtkPlot *plot, gint x);
static gdouble inverse_dy                       (GtkPlot *plot, gint dy);
static gdouble inverse_dx                       (GtkPlot *plot, gint dx);
static gint roundint                             (gdouble x);
static void parse_label                         (gdouble val,
                                                 gint precision,
                                                 gint style,
                                                 gchar *label);
static void spline_solve 			(int n, 
						 GtkPlotPoint *points, 
						 gfloat *y2);
static gfloat spline_eval 			(int n, 
						 GtkPlotPoint *points, 
						 gfloat *y2, gfloat val);

static void pssetcolor(GtkPlotEPS *eps, GdkColor color)
{
    FILE *psout = eps->psfile;

    fprintf(psout, "%f %f %f setrgbcolor\n",
	    (gdouble) color.red / 65535.0,
	    (gdouble) color.green / 65535.0,
	    (gdouble) color.blue / 65535.0);
}

static void pssetlinewidth(GtkPlotEPS *eps, gint width)
{
    FILE *psout = eps->psfile;

    fprintf(psout, "%d slw\n", (int)width);
}

static void pssetlinestyle(GtkPlotEPS *eps, GtkPlotLine line)
{
    gint style = line.line_style;
    gint line_width = line.line_width;
    FILE *psout = eps->psfile;

    if(line.line_style == GTK_PLOT_LINE_NONE) return;

    pssetcolor(eps, line.color);
    pssetlinewidth(eps, line_width);

    switch (style) {
      case GTK_PLOT_LINE_SOLID:				/* solid */
	fprintf(psout, "[] 0 sd\n");
	break;
      case GTK_PLOT_LINE_DOTTED:			/* dotted */
	fprintf(psout, "[2 3] 0 sd\n");
	break;
      case GTK_PLOT_LINE_DASHED:			/* long dash */
	fprintf(psout, "[6 4] 0 sd\n");
	break;
      case GTK_PLOT_LINE_DOT_DASH:			/* dot-dashed */
	fprintf(psout, "[6 4 2 4] 0 sd\n");
	break;
    }
  
    fprintf(psout,"0 slc\n");
}

static void 
psdrawaxis(GtkPlot *plot, GtkPlotAxis axis, int x, int y)
{
  gdouble x_tick, y_tick;
  gint xx, yy;
  gint line_width;
  gint xp, yp, width, height;
  gdouble min;
  gint ticks_length;
  FILE *psout = plot->eps->psfile;
  GtkPlotEPS *eps = plot->eps;

  xp = roundint(plot->x * (gdouble)GTK_WIDGET(plot)->allocation.width);
  yp = roundint(plot->y * (gdouble)GTK_WIDGET(plot)->allocation.height);
  width = roundint(plot->width * (gdouble)GTK_WIDGET(plot)->allocation.width);
  height = roundint(plot->height * (gdouble)GTK_WIDGET(plot)->allocation.height);

  line_width = axis.line.line_width;
  ticks_length = axis.ticks_length;

  min = (gdouble)((gint)axis.min/axis.major_ticks)*axis.major_ticks;
  switch(axis.orientation){
     case GTK_ORIENTATION_HORIZONTAL:
         if(axis.line.line_style == GTK_PLOT_LINE_NONE) break;
         pssetlinestyle(eps, axis.line);
         fprintf(psout,"2 slc\n");
         psdrawline(eps, x, y, x+width, y);
         pssetlinestyle(eps, axis.line);
         pssetlinewidth(eps, axis.ticks_width);
         for(x_tick=min; x_tick<=axis.max; x_tick+=axis.major_ticks){
           xx = transform_x(plot, x_tick);
           if(axis.ticks_mask & GTK_PLOT_TICKS_UP)
              psdrawline(eps, x+xx, y, x+xx, y-ticks_length);
           if(axis.ticks_mask & GTK_PLOT_TICKS_DOWN)
              psdrawline(eps, x+xx, y, x+xx, y+ticks_length);
         }
         for(x_tick=min; x_tick<=axis.max; x_tick+=axis.minor_ticks){
           xx = transform_x(plot, x_tick);
           if(axis.ticks_mask & GTK_PLOT_TICKS_UP)
              psdrawline(eps, x+xx, y, x+xx, y-ticks_length/2-1);
           if(axis.ticks_mask & GTK_PLOT_TICKS_DOWN)
              psdrawline(eps, x+xx, y, x+xx, y+ticks_length/2+1);
         }
         break;
     case GTK_ORIENTATION_VERTICAL:
         if(axis.line.line_style == GTK_PLOT_LINE_NONE) break;
         y = y + height;
         pssetlinestyle(eps, axis.line);
         fprintf(psout,"2 slc\n");
         psdrawline(eps, x, y, x, y-height);
         pssetlinestyle(eps, axis.line);
         pssetlinewidth(eps, axis.ticks_width);
         for(y_tick=min; y_tick<=axis.max; y_tick+=axis.major_ticks){
           yy = transform_y(plot, y_tick);
           if(axis.ticks_mask & GTK_PLOT_TICKS_RIGHT)
              psdrawline(eps, x, y-yy, x+ticks_length, y-yy);
           if(axis.ticks_mask & GTK_PLOT_TICKS_LEFT)
              psdrawline(eps, x, y-yy, x-ticks_length, y-yy);
         }
         for(y_tick=min; y_tick<=axis.max; y_tick+=axis.minor_ticks){
           yy = transform_y(plot, y_tick);
           if(axis.ticks_mask & GTK_PLOT_TICKS_RIGHT)
              psdrawline(eps, x, y-yy, x+ticks_length/2+1, y-yy);
           if(axis.ticks_mask & GTK_PLOT_TICKS_LEFT)
              psdrawline(eps, x, y-yy, x-ticks_length/2-1, y-yy);
         }
         break;
  }

}

static void
psdrawgrids(GtkPlot *plot)
{
  GtkWidget *widget;
  gdouble x, y;
  gint ix, iy;
  gint width, height;
  gint xp, yp;
  GtkPlotEPS *eps = plot->eps;

  widget = GTK_WIDGET(plot);

  xp = roundint(plot->x * (gdouble)widget->allocation.width);
  yp = roundint(plot->y * (gdouble)widget->allocation.height);
  width = roundint(plot->width * (gdouble)widget->allocation.width);
  height = roundint(plot->height * (gdouble)widget->allocation.height);

  if(GTK_PLOT_SHOW_X0(plot) && plot->x0_line.line_style != GTK_PLOT_LINE_NONE)
    {
          pssetlinestyle(eps, plot->x0_line);
          if(plot->xmin <= 0. && plot->xmax >= 0.)
            {
              ix = transform_x(plot, 0.);
              ix += xp;
              psdrawline(eps, ix, yp+1, ix, yp+height);
            }
    }

  if(GTK_PLOT_SHOW_Y0(plot) && plot->y0_line.line_style != GTK_PLOT_LINE_NONE)
    {
          pssetlinestyle(eps, plot->y0_line);
          if(plot->ymin <= 0. && plot->ymax >= 0.)
            {
              iy = transform_y(plot, 0.);
              iy = height+yp-iy;
              psdrawline(eps, xp, iy, xp + width, iy);
            }
    }
  if(GTK_PLOT_SHOW_V_GRID(plot))
    {
        if(plot->major_vgrid.line_style != GTK_PLOT_LINE_NONE){
          pssetlinestyle(eps, plot->major_vgrid);
          for(x=plot->xmin; x<=plot->xmax; x+=plot->bottom.major_ticks){
           ix = transform_x(plot, x);
           ix += xp;
           psdrawline(eps, ix, yp+1, ix, yp+height);
          }
        }
        if(plot->minor_vgrid.line_style != GTK_PLOT_LINE_NONE){
          pssetlinestyle(eps, plot->minor_vgrid);
          for(x=plot->xmin; x<=plot->xmax; x+=plot->bottom.minor_ticks){
           ix = transform_x(plot, x);
           ix += xp;
           psdrawline(eps, ix, yp+1, ix, yp+height);
          }
        }
    }
  if(GTK_PLOT_SHOW_H_GRID(plot))
    {
        if(plot->major_hgrid.line_style != GTK_PLOT_LINE_NONE){
          pssetlinestyle(eps, plot->major_hgrid);
          for(y=plot->ymin; y<=plot->ymax; y+=plot->left.major_ticks){
           iy = transform_y(plot, y);
           iy = height+yp-iy;
           psdrawline(eps, xp, iy, xp + width, iy);
          }
        }
        if(plot->minor_hgrid.line_style != GTK_PLOT_LINE_NONE){
          pssetlinestyle(eps, plot->minor_hgrid);
          for(y=plot->ymin; y<=plot->ymax; y+=plot->left.minor_ticks){
           iy = transform_y(plot, y);
           iy = height+yp-iy;
           psdrawline(eps, xp, iy, xp + width, iy);
          }
        }
    }
}

static void
psdrawlabels(GtkPlot *plot,
             GtkPlotAxis axis,
             gint x, gint y)
{
  GtkWidget *widget;
  GdkFont *font;
  gchar *psfont;
  GtkPlotText title;
  gchar label[100];
  gdouble x_tick, y_tick;
  gint xx, yy;
  gint text_height;
  gint xp, yp, width, height;
  gdouble min;
  GtkPlotEPS *eps = plot->eps;

  widget = GTK_WIDGET(plot);
  xp = roundint(plot->x * widget->allocation.width);
  yp = roundint(plot->y * widget->allocation.height);
  width = roundint(plot->width * widget->allocation.width);
  height = roundint(plot->height * widget->allocation.height);

  pssetcolor (eps, axis.label_attr.fg);

  min = (gdouble)((gint)axis.min/axis.major_ticks)*axis.major_ticks;
  font = gtk_plot_font_get_gdkfont(axis.label_attr.font, axis.label_attr.height);
  psfont = gtk_plot_font_get_psfontname(axis.label_attr.font);
  pssetfont(eps, psfont, axis.label_attr.height);
  text_height = axis.label_attr.height;

  switch(axis.orientation){
     case GTK_ORIENTATION_VERTICAL:
       y += height;
       for(y_tick=min; y_tick<=axis.max; y_tick+=axis.major_ticks){
           yy = transform_y(plot, y_tick);
           parse_label(y_tick, axis.label_precision, axis.label_style, label);
           if(axis.label_mask & GTK_PLOT_LABEL_LEFT)
              psdrawstring(eps,
                           x-axis.labels_offset, 
                           y-yy+font->descent, 
                           GTK_JUSTIFY_RIGHT, 0,
                           label);
           if(axis.label_mask & GTK_PLOT_LABEL_RIGHT)
              psdrawstring(eps,
                           x+axis.labels_offset, 
                           y-yy+font->descent, 
                           GTK_JUSTIFY_LEFT, 0,
                           label);
       }
       if(axis.title_visible && axis.title.text)
            {
              title = axis.title;
              psdrawtext(GTK_WIDGET(plot), plot->eps, title);
            }
       break;
     case GTK_ORIENTATION_HORIZONTAL:
       for(x_tick=min; x_tick<=axis.max; x_tick+=axis.major_ticks){
           xx = transform_x(plot, x_tick);
           parse_label(x_tick, axis.label_precision, axis.label_style, label);
           if(axis.label_mask & GTK_PLOT_LABEL_TOP)
              psdrawstring(eps,
                           x+xx, 
                           y-2*font->descent-axis.labels_offset, 
                           GTK_JUSTIFY_CENTER, 0,
                           label);

           if(axis.label_mask & GTK_PLOT_LABEL_BOTTOM)
              psdrawstring(eps,
                           x+xx, 
                           y+text_height+2*font->descent+axis.labels_offset, 
                           GTK_JUSTIFY_CENTER, 0,
                           label);
       }
       if(axis.title_visible && axis.title.text)
            {
              title = axis.title;
              psdrawtext(GTK_WIDGET(plot), plot->eps, title);
            }
       break;
   }
}

static void gtk_plot_ps_leave(GtkPlotEPS *eps)
{
    fprintf(eps->psfile, "showpage\n");
    fprintf(eps->psfile, "%%%%Trailer\n");
    fclose(eps->psfile);
}

static gint gtk_plot_ps_init			(GtkPlotEPS *eps,
                                                 gfloat scalex,
                                                 gfloat scaley)
{
    time_t now;
    FILE *psout;

    now = time(NULL);

    if ((psout = fopen(eps->psname, "w")) == NULL){
       g_warning("ERROR: Cannot open file: %s", eps->psname); 
       return -1;
    }

    eps->psfile = psout;

    if(eps->epsflag)
       fprintf (psout, "%%!PS-Adobe-2.0 EPSF-2.0\n");
    else
       fprintf (psout, "%%!PS-Adobe-2.0\n");

    fprintf (psout,
             "%%%%Title: %s\n"
             "%%%%Creator: %s v%s Copyright (c) 1999 Adrian E. Feiguin\n"
             "%%%%CreationDate: %s"
             "%%%%Magnification: 1.0000\n",
             eps->psname,
             "GtkPlot", "3.x",
             ctime (&now));

    if(eps->orientation == GTK_PLOT_PORTRAIT)
             fprintf(psout,"%%%%Orientation: Portrait\n");
    else
             fprintf(psout,"%%%%Orientation: Landscape\n");

    if(eps->epsflag)
          fprintf (psout,
                   "%%%%BoundingBox: 0 0 %d %d\n"
                   "%%%%Pages: 1\n"
                   "%%%%EndComments\n",
                   eps->page_width,
                   eps->page_height);


    fprintf (psout,
             "/cp {closepath} bind def\n"
             "/c {curveto} bind def\n"
             "/f {fill} bind def\n"
             "/a {arc} bind def\n"
             "/ef {eofill} bind def\n"
             "/ex {exch} bind def\n"
             "/gr {grestore} bind def\n"
             "/gs {gsave} bind def\n"
             "/sa {save} bind def\n"
             "/rs {restore} bind def\n"
             "/l {lineto} bind def\n"
             "/m {moveto} bind def\n"
             "/rm {rmoveto} bind def\n"
             "/n {newpath} bind def\n"
             "/s {stroke} bind def\n"
             "/sh {show} bind def\n"
             "/slc {setlinecap} bind def\n"
             "/slj {setlinejoin} bind def\n"
             "/slw {setlinewidth} bind def\n"
             "/srgb {setrgbcolor} bind def\n"
             "/rot {rotate} bind def\n"
             "/sc {scale} bind def\n"
             "/sd {setdash} bind def\n"
             "/ff {findfont} bind def\n"
             "/sf {setfont} bind def\n"
             "/scf {scalefont} bind def\n"
             "/sw {stringwidth pop} bind def\n"
             "/tr {translate} bind def\n"
  
             "\n/ellipsedict 8 dict def\n"
             "ellipsedict /mtrx matrix put\n"
             "/ellipse\n"
             "{ ellipsedict begin\n"
             "   /endangle exch def\n"
             "   /startangle exch def\n"
             "   /yrad exch def\n"
             "   /xrad exch def\n"
             "   /y exch def\n"
             "   /x exch def"
             "   /savematrix mtrx currentmatrix def\n"
             "   x y tr xrad yrad sc\n"
             "   0 0 1 startangle endangle arc\n"
             "   savematrix setmatrix\n"
             "   end\n"
             "} def\n\n"
    ); 
    
    if(eps->orientation == GTK_PLOT_PORTRAIT)
             fprintf(psout, "%d %d translate\n"
                            "%f %f scale\n",
                            0, eps->page_height,
                            scalex, -scaley);

    if(eps->orientation == GTK_PLOT_LANDSCAPE)
             fprintf(psout, "%f %f scale\n"
                            "-90 rotate \n",
                            scalex, -scaley);

             
    fprintf(psout,"%%%%EndProlog\n\n\n");

    return 0;
}


static gint
transform_y(GtkPlot *plot, gdouble y)
{
    gdouble height;

    height = (gdouble)GTK_WIDGET(plot)->allocation.height * plot->height;
    return(roundint(height*(y-plot->ymin)/(plot->ymax-plot->ymin)));
}

static gint
transform_x(GtkPlot *plot, gdouble x)
{
    gdouble width;

    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(roundint(width*(x-plot->xmin)/(plot->xmax-plot->xmin)));
}


static gint
transform_dx(GtkPlot *plot, gdouble dx)
{
    gdouble width;

    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(roundint(width*dx/(plot->xmax-plot->xmin)));
}

static gint
transform_dy(GtkPlot *plot, gdouble dy)
{
    gdouble height;

    height = (gdouble)GTK_WIDGET(plot)->allocation.height * plot->height;
    return(roundint(height*dy/(plot->ymax-plot->ymin)));
}

static gdouble
inverse_y(GtkPlot *plot, gint y)
{
    return(inverse_dy(plot, y)+plot->ymin);
}

static gdouble
inverse_x(GtkPlot *plot, gint x)
{
    return(inverse_dx(plot, x)+plot->xmin);
}


static gdouble
inverse_dy(GtkPlot *plot, gint dy)
{
    gdouble height;

    height = (gdouble)GTK_WIDGET(plot)->allocation.height * plot->height;
    return(((gdouble)dy)*(plot->ymax-plot->ymin)/height);
}

static gdouble
inverse_dx(GtkPlot *plot, gint dx)
{
    gdouble width;

    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(((gdouble)dx)*(plot->xmax-plot->xmin)/width);
}


static gint
roundint (gdouble x)
{
 return (x+.50999999471);
}

void
gtk_plot_eps_set_size 				(GtkPlotEPS *eps,
						 gint units,
						 gfloat width,
						 gfloat height)
{
  eps->units = units;
  eps->width = width;
  eps->height = height;

  switch(units){
   case GTK_PLOT_MM:
   case GTK_PLOT_CM:
   case GTK_PLOT_INCHES:
        eps->page_width = width * 72;
        eps->page_height = height * 72;
        break;
   case GTK_PLOT_PSPOINTS:
   default:
        eps->page_width = width;
        eps->page_height = height;
   }

}

GtkPlotEPS *
gtk_plot_eps_new				(gchar *psname,
						 gint orientation,
						 gint epsflag,
                                                 gint page_size)
{
  GtkPlotEPS *eps;
  gint width, height;

  eps = g_new(GtkPlotEPS, 1);

  eps->psname = g_strdup(psname);
  eps->orientation = orientation;
  eps->epsflag = epsflag;
  eps->units = GTK_PLOT_PSPOINTS;

  switch (page_size){
   case GTK_PLOT_LEGAL:
        width = GTK_PLOT_LEGAL_W;
        height = GTK_PLOT_LEGAL_H;
        break;
   case GTK_PLOT_A4:
        width = GTK_PLOT_A4_W;
        height = GTK_PLOT_A4_H;
        break;
   case GTK_PLOT_EXECUTIVE:
        width = GTK_PLOT_EXECUTIVE_W;
        height = GTK_PLOT_EXECUTIVE_H;
        break;
   case GTK_PLOT_LETTER:
   default:
        width = GTK_PLOT_LETTER_W;
        height = GTK_PLOT_LETTER_H;
  }

        
  gtk_plot_eps_set_size(eps, GTK_PLOT_PSPOINTS, width, height);

  return eps;
}

GtkPlotEPS *
gtk_plot_eps_new_with_size			(gchar *psname,
						 gint orientation,
						 gint epsflag,
                                                 gint units,
						 gfloat width, gfloat height)
{
  GtkPlotEPS *eps;

  eps = gtk_plot_eps_new(psname, orientation, epsflag, GTK_PLOT_CUSTOM);

  gtk_plot_eps_set_size(eps, units, width, height);

  return eps;
}

void 
gtk_plot_export_ps			        (GtkPlot *plot, 
					 	 char *psname, 
						 int orient, 
						 int epsflag, 
						 gint page_size)
{
  GtkPlotEPS *eps;

  eps = gtk_plot_eps_new(psname, orient, epsflag, page_size);

  gtk_plot_export(plot, eps);

  g_free(eps);
}

void 
gtk_plot_export_ps_with_size			(GtkPlot *plot, 
					 	 char *psname, 
						 gint orient, 
						 gint epsflag, 
						 gint units,
						 gint width,
                                                 gint height)
{
  GtkPlotEPS *eps;

  eps = gtk_plot_eps_new_with_size(psname, 
				   orient, 
                                   epsflag, 
                                   units, width, height);

  gtk_plot_export(plot, eps);

  g_free(eps);
}

static void
gtk_plot_export(GtkPlot *plot, GtkPlotEPS *eps)
{
  GtkPlotEPS *plot_eps;
  gfloat scalex, scaley;

  if(eps->orientation == GTK_PLOT_PORTRAIT){
    scalex = (gfloat)eps->page_width /
                                (gfloat)GTK_WIDGET(plot)->allocation.width;
    scaley = (gfloat)eps->page_height / 
                                (gfloat)GTK_WIDGET(plot)->allocation.height;
  }else{
    scalex = (gfloat)eps->page_width /
                                (gfloat)GTK_WIDGET(plot)->allocation.height;
    scaley = (gfloat)eps->page_height / 
                                (gfloat)GTK_WIDGET(plot)->allocation.width;
  }

  gtk_plot_ps_init(eps, scalex, scaley);

  plot_eps = plot->eps;
  plot->eps = eps;
  gtk_plot_drawps(plot);
  plot->eps = plot_eps;

  gtk_plot_ps_leave(eps);

}

void 
gtk_plot_layout_export_ps			(GtkPlotLayout *layout, 
					 	 char *psname, 
						 int orient, 
						 int epsflag, 
						 gint page_size)
{
  GtkPlotEPS *eps;

  eps = gtk_plot_eps_new(psname, orient, epsflag, page_size);

  gtk_plot_layout_export(layout, eps);

  g_free(eps);
}


void 
gtk_plot_layout_export_ps_with_size		(GtkPlotLayout *layout, 
					 	 char *psname, 
						 gint orient, 
						 gint epsflag, 
						 gint units,
						 gint width, 
						 gint height)
{
  GtkPlotEPS *eps;

  eps = gtk_plot_eps_new_with_size(psname, 
                                   orient, 
                                   epsflag, 
                                   units, width, height);
  
  gtk_plot_layout_export(layout, eps);

  g_free(eps);
}

static void
gtk_plot_layout_export(GtkPlotLayout *layout, GtkPlotEPS *eps)
{
  GList *plots;
  GtkPlot *plot;
  GtkPlotEPS *plot_eps;
  GList *text;
  GtkPlotText *child_text;
  gfloat scalex, scaley;
  GtkAllocation allocation;

  if(eps->orientation == GTK_PLOT_PORTRAIT){
    scalex = (gfloat)eps->page_width / (gfloat)layout->width;
    scaley = (gfloat)eps->page_height / (gfloat)layout->height;
  }else{
    scalex = (gfloat)eps->page_width / (gfloat)layout->height;
    scaley = (gfloat)eps->page_height / (gfloat)layout->width;
  }

  gtk_plot_ps_init(eps, scalex, scaley);

  pssetcolor(eps, layout->background);
  psdrawrectangle(eps, 
                  0, 0,
                  layout->width, layout->height,
                  TRUE);

  plots = layout->plots;
  while(plots)
   {
     plot = GTK_PLOT(plots->data);
     plot_eps = plot->eps;
     plot->eps = eps;
     gtk_plot_drawps(plot);
     plot->eps = plot_eps;

     plots = plots->next;
   }

  allocation = GTK_WIDGET(layout)->allocation;
  GTK_WIDGET(layout)->allocation.width = layout->width;
  GTK_WIDGET(layout)->allocation.height = layout->height;

  text = layout->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     psdrawtext(GTK_WIDGET(layout), eps, *child_text);
     text = text->next;
   }

  GTK_WIDGET(layout)->allocation = allocation;

  gtk_plot_ps_leave(eps);

}

static void
gtk_plot_drawps(GtkPlot *plot)
{
  GtkWidget *widget;
  GList *dataset;
  GList *text;
  GtkPlotText *child_text;
  gint xoffset, yoffset;
  gint width, height;

  widget = GTK_WIDGET(plot);
  xoffset = widget->allocation.x + 
            roundint(plot->x * widget->allocation.width);
  yoffset = widget->allocation.y +
            roundint(plot->y * widget->allocation.height);
  width = roundint(plot->width * widget->allocation.width);
  height = roundint(plot->height * widget->allocation.height);

  pssetcolor(plot->eps, plot->background);

  if(!GTK_PLOT_TRANSPARENT(plot))
    psdrawrectangle (plot->eps,
                     xoffset, yoffset,
                     xoffset + width , yoffset + height,
                     TRUE);


  psdrawgrids(plot);

  if(GTK_PLOT_SHOW_BOTTOM_AXIS(plot))
    {
      psdrawaxis(plot, plot->bottom,
                 xoffset,
                 yoffset+height);
      psdrawlabels(plot, plot->bottom,
                   xoffset,
                   yoffset+height);
    }

  if(GTK_PLOT_SHOW_TOP_AXIS(plot))
    {
      psdrawaxis(plot, plot->top,
                 xoffset,
                 yoffset);
      psdrawlabels(plot, plot->top,
                   xoffset,
                   yoffset);
    }

  if(GTK_PLOT_SHOW_LEFT_AXIS(plot))
    {
      psdrawaxis(plot, plot->left,
                 xoffset,
                 yoffset);
      psdrawlabels(plot, plot->left,
                   xoffset,
                   yoffset);
    }

  if(GTK_PLOT_SHOW_RIGHT_AXIS(plot))
    {
      psdrawaxis(plot, plot->right,
                 xoffset+width,
                 yoffset);
      psdrawlabels(plot, plot->right,
                   xoffset+width,
                   yoffset);
    }

  dataset = plot->data_sets;
  while(dataset)
   {
     psdrawdataset(plot, (GtkPlotData *)dataset->data);
     dataset = dataset->next;
   }

  text = plot->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     psdrawtext(GTK_WIDGET(plot), plot->eps, *child_text);
     text = text->next;
   }

  psdrawlegends(plot);

}

static void
psdrawdataset(GtkPlot *plot, GtkPlotData *dataset)
{
  GtkPlotPoint point;
  GtkPlotData function;
  GtkPlotPoint *function_points;
  gint n;
  gdouble x, y;
  gboolean error;

  if(!dataset->is_visible) return;

  if(!dataset->is_function)
    {
       psconnectpoints (plot, dataset);
       for(n=0; n<=dataset->num_points-1; n++)
         {
           point = dataset->points[n];
           psdrawxy(plot, dataset, point);
           psdrawerrbars(plot, dataset, point);
         }

       for(n=0; n<=dataset->num_points-1; n++)
         {
           point = dataset->points[n];
           psdrawpoint(plot,
		       FALSE,
                       point,
                       dataset->symbol.symbol_type,
                       dataset->symbol.color,
                       dataset->symbol.symbol_style,
                       dataset->symbol.size,
                       dataset->symbol.line_width);
         }
    }
  else
    {
       function = *dataset;
       function_points = NULL;
       function.num_points = 0;
       for(x=plot->xmin; x<=plot->xmax+inverse_dx(plot, function.x_step);
                         x+=inverse_dx(plot, function.x_step)) {
            function.num_points++;
            function_points = (GtkPlotPoint *)g_realloc(function_points,
                                                        function.num_points*
                                                        sizeof(GtkPlotPoint));
            y = function.function (x, &error);
            if(error)
              {
                 function.points = function_points;
                 function.num_points--;
                 if(function.num_points > 1)
                       psconnectpoints (plot, &function);
                 function.num_points = 0;
              }
            else
              {
                if(function.num_points >= 2)
                  {
                     if(y > plot->ymax &&
                        function_points[function.num_points-2].y <= plot->ymin)
                       {
                          function.points = function_points;
                          function.num_points--;
                          psconnectpoints(plot, &function);
                          function.num_points = 1;
                       }
                      if(y < plot->ymin &&
                        function_points[function.num_points-2].y >= plot->ymax)
                       {
                          function.points = function_points;
                          function.num_points--;
                          psconnectpoints(plot, &function);
                          function.num_points = 1;
                       }
                   }
                function_points[function.num_points-1].x = x;
                function_points[function.num_points-1].y = y;
              }
         }
       if(function.num_points > 1 )
         {
            function.points = function_points;
            psconnectpoints (plot, &function);
         }
       g_free(function_points);
    }
}

static void
psdrawerrbars(GtkPlot *plot, GtkPlotData *dataset, GtkPlotPoint point)
{
  GdkPoint errbar[6];
  gint x, y;
  gint ex, ey;
  FILE *psout = plot->eps->psfile;

  if(point.x < plot->xmin || point.x > plot->xmax ||
     point.y < plot->ymin || point.y > plot->ymax) return;

  pssetcolor(plot->eps, dataset->symbol.color);

  fprintf(psout,"0 slc\n");
  fprintf(psout,"%d slw\n", dataset->symbol.line_width/2);

  psgetpoint(plot, point.x, point.y, &x, &y);

  ex = transform_dx(plot, point.xerr);
  ey = transform_dy(plot, point.yerr);

  if(dataset->show_xerrbars)
    {
      errbar[0].x = x-ex;
      errbar[0].y = y-dataset->xerrbar_length/2;
      errbar[1].x = x-ex;
      errbar[1].y = y+dataset->xerrbar_length/2;
      errbar[2].x = x-ex;
      errbar[2].y = y;
      errbar[3].x = x+ex;
      errbar[3].y = y;
      errbar[4].x = x+ex;
      errbar[4].y = y-dataset->xerrbar_length/2;
      errbar[5].x = x+ex;
      errbar[5].y = y+dataset->xerrbar_length/2;
      psdrawlines(plot->eps, errbar, 6);
    }

  if(dataset->show_yerrbars)
    {
      errbar[0].x = x-dataset->yerrbar_length/2;
      errbar[0].y = y-ey;
      errbar[1].x = x+dataset->yerrbar_length/2;
      errbar[1].y = y-ey;
      errbar[2].x = x;
      errbar[2].y = y-ey;
      errbar[3].x = x;
      errbar[3].y = y+ey;
      errbar[4].x = x-dataset->yerrbar_length/2;
      errbar[4].y = y+ey;
      errbar[5].x = x+dataset->yerrbar_length/2;
      errbar[5].y = y+ey;
      psdrawlines(plot->eps, errbar, 6);
    }

}


static void
psconnectpoints(GtkPlot *plot, GtkPlotData *dataset)
{
  GdkPoint points[2*dataset->num_points];
  GtkPlotData spline;
  GdkPoint *spline_points;
  GtkPlotPoint point;
  gfloat spline_coef[dataset->num_points];
  gfloat spline_x, spline_y;
  gint n;
  gint x, y;
  gint x1, y1;
  gint num_points = dataset->num_points;
  GdkRectangle clip;
  FILE *psout = plot->eps->psfile;

  if(dataset->line.line_style == GTK_PLOT_LINE_NONE) return;

  clip.x = roundint(plot->x * (gdouble)GTK_WIDGET(plot)->allocation.width);
  clip.y = roundint(plot->y * (gdouble)GTK_WIDGET(plot)->allocation.height);
  clip.width = roundint(plot->width * 
                       (gdouble)GTK_WIDGET(plot)->allocation.width);
  clip.height = roundint(plot->height * 
                        (gdouble)GTK_WIDGET(plot)->allocation.height);

  fprintf(psout,"gs \n");
/*  fprintf(psout,"n \n");
  fprintf(psout,"%d %d m \n",clip.x, clip.y);
  fprintf(psout,"%d %d l \n",clip.x+clip.width, clip.y);
  fprintf(psout,"%d %d l \n",clip.x+clip.width, clip.y+clip.height);
  fprintf(psout,"%d %d l \n",clip.x, clip.y+clip.height);
  fprintf(psout,"clip \n");
*/

  fprintf(psout,"%d %d %d %d rectclip\n",clip.x,clip.y,clip.width,clip.height);
  pssetlinestyle(plot->eps, dataset->line);

  switch(dataset->line_connector){
   case GTK_PLOT_CONNECT_STRAIGHT:
      if(dataset->num_points == 1) break;
      for(n=0; n<dataset->num_points; n++)
        {
          psgetpoint(plot, dataset->points[n].x, dataset->points[n].y, &x, &y);
          points[n].x = x;
          points[n].y = y;
        }
      break;
   case GTK_PLOT_CONNECT_HV_STEP:
       if(dataset->num_points == 1) break;
       num_points=0;
       for(n=0; n < dataset->num_points; n++)
        {
          psgetpoint(plot, dataset->points[n].x, dataset->points[n].y, &x, &y);
          points[num_points].x = x;
          points[num_points].y = y;
          num_points++;
          if(n < dataset->num_points-1)
            {
              psgetpoint(plot, 
                         dataset->points[n+1].x, 
                         dataset->points[n+1].y,
                         &x, &y);
              points[num_points].x = x;
              points[num_points].y = points[num_points-1].y;
              num_points++;
            }
        }
       break;
    case GTK_PLOT_CONNECT_VH_STEP:
       if(dataset->num_points == 1) break;
       num_points=0;
       for(n=0; n < dataset->num_points; n++)
        {
          psgetpoint(plot, 
                     dataset->points[n].x,
                     dataset->points[n].y,
                     &x, &y);
          points[num_points].x = x;
          points[num_points].y = y;
          num_points++;
          if(n < dataset->num_points-1)
            {
              psgetpoint(plot, 
                         dataset->points[n+1].x,
                         dataset->points[n+1].y,
                         &x, &y);
              points[num_points].x = points[num_points-1].x;
              points[num_points].y = y;
              num_points++;
            }
        }
       break;
     case GTK_PLOT_CONNECT_MIDDLE_STEP:
       if(dataset->num_points == 1) break;
       num_points=1;
       for(n=1; n < dataset->num_points; n++)
        {
          psgetpoint(plot, 
                     dataset->points[n].x,
                     dataset->points[n].y,
                     &x, &y);
          psgetpoint(plot, 
                     dataset->points[n-1].x,
                     dataset->points[n-1].y,
                     &x1, &y1);
          points[num_points].x = (x+x1)/2;
          points[num_points].y = y1;
          num_points++;
          points[num_points].x = points[num_points-1].x;
          points[num_points].y = y;
          num_points++;
        }
        psgetpoint(plot, 
                   dataset->points[0].x,
                   dataset->points[0].y,
                   &x, &y);
        points[0].x = x;
        points[0].y = y;
        psgetpoint(plot, 
                   dataset->points[dataset->num_points-1].x,
                   dataset->points[dataset->num_points-1].y,
                   &x, &y);
        points[num_points].x = x;
        points[num_points].y = y;
        num_points++;
        break;
     case GTK_PLOT_CONNECT_SPLINE:
        spline = *dataset;
        spline_points = NULL;
        spline.num_points = 0;
        spline_solve(dataset->num_points, dataset->points, spline_coef);
        for(spline_x=dataset->points[0].x;
            spline_x<=dataset->points[dataset->num_points-1].x;
            spline_x+=inverse_dx(plot, spline.x_step)) {
              spline.num_points++;
              spline_points = (GdkPoint *)g_realloc(spline_points,
                                                    spline.num_points*
                                                    sizeof(GdkPoint));
              spline_y = spline_eval(dataset->num_points, dataset->points,
                                     spline_coef, spline_x);
              point.x = spline_x;
              point.y = spline_y;
              psgetpoint(plot, point.x, point.y, &x, &y);
              spline_points[spline.num_points-1].x = x;
              spline_points[spline.num_points-1].y = y;
         }
        psdrawlines(plot->eps, spline_points, spline.num_points);
        g_free(spline_points);
        fprintf(psout,"gr \n");
        return;
     case GTK_PLOT_CONNECT_NONE:
     default:
        fprintf(psout,"gr \n");
        return;
    }


  psdrawlines(plot->eps, points, num_points);
  fprintf(psout,"gr \n");

}

static void
psdrawlines(GtkPlotEPS *eps, GdkPoint *points, gint numpoints)
{
  gint i;
  FILE *psout = eps->psfile;
  
  fprintf(psout,"n\n");
  fprintf(psout,"%d %d m\n", points[0].x, points[0].y);
  for(i = 1; i < numpoints; i++)
      fprintf(psout,"%d %d l\n", points[i].x, points[i].y);

  fprintf(psout,"s\n");
}

static void
psdrawpolygon(GtkPlotEPS *eps, GdkPoint *points, gint numpoints, gint filled)
{
  gint i;
  FILE *psout = eps->psfile;

  fprintf(psout,"n\n");
  fprintf(psout,"%d %d m\n", points[0].x, points[0].y);
  for(i = 1; i < numpoints; i++)
      fprintf(psout,"%d %d l\n", points[i].x, points[i].y);

  if(filled)
     fprintf(psout,"f\n");
  else
     fprintf(psout,"cp\n");

  fprintf(psout,"s\n");
}



static void
psgetpoint(GtkPlot *plot, gdouble px, gdouble py, gint *x, gint *y)
{
    gint xp, yp, width, height;

    xp = roundint(plot->x * GTK_WIDGET(plot)->allocation.width);
    yp = roundint(plot->y * GTK_WIDGET(plot)->allocation.height);
    width = roundint(plot->width * GTK_WIDGET(plot)->allocation.width);
    height = roundint(plot->height * GTK_WIDGET(plot)->allocation.height);

    *y = transform_y(plot, py);
    *x = transform_x(plot, px);

    *x = GTK_WIDGET(plot)->allocation.x + *x + xp;
    *y = GTK_WIDGET(plot)->allocation.y + yp + height - *y;

}

static void
pssetpoint(GtkPlot *plot, gint x, gint y, GtkPlotPoint *point)
{
    GtkWidget *widget;
    gint xx, yy;
    gint xp, yp, width, height;

    widget = GTK_WIDGET(plot);
    xp = roundint(plot->x * widget->allocation.width);
    yp = roundint(plot->y * widget->allocation.height);
    width = roundint(plot->width * widget->allocation.width);
    height = roundint(plot->height * widget->allocation.height);

    xx = x - widget->allocation.x - xp;
    yy = widget->allocation.y + height + yp - y;

    point->x = inverse_x(plot, xx);
    point->y = inverse_y(plot, yy);
}

/* Solve the tridiagonal equation system that determines the second
   derivatives for the interpolation points.  (Based on Numerical
   Recipies 2nd Edition.) */
static void
spline_solve (int n, GtkPlotPoint *points, gfloat y2[])
{
  gfloat p, sig, *u;
  gfloat x[n], y[n];
  gint i, k;

  for (i=0; i<n ; i++)
      {
        x[i] = points[i].x;
        y[i] = points[i].y;
      }

  u = g_malloc ((n - 1) * sizeof (u[0]));

  y2[0] = u[0] = 0.0;   /* set lower boundary condition to "natural" */

  for (i = 1; i < n - 1; ++i)
    {
      sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      p = sig * y2[i - 1] + 2.0;
      y2[i] = (sig - 1.0) / p;
      u[i] = ((y[i + 1] - y[i])
              / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]));
      u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }

  y2[n - 1] = 0.0;
  for (k = n - 2; k >= 0; --k)
    y2[k] = y2[k] * y2[k + 1] + u[k];

  g_free (u);
}

static gfloat
spline_eval (int n, GtkPlotPoint *points, gfloat y2[], gfloat val)
{
  gint k_lo, k_hi, k;
  gfloat h, b, a;
  gfloat x[n], y[n];

  for (k=0; k<n ; k++)
      {
        x[k] = points[k].x;
        y[k] = points[k].y;
      }

  /* do a binary search for the right interval: */
  k_lo = 0; k_hi = n - 1;
  while (k_hi - k_lo > 1)
    {
      k = (k_hi + k_lo) / 2;
      if (x[k] > val)
        k_hi = k;
      else
        k_lo = k;
    }

  h = x[k_hi] - x[k_lo];
  g_assert (h > 0.0);

  a = (x[k_hi] - val) / h;
  b = (val - x[k_lo]) / h;
  return a*y[k_lo] + b*y[k_hi] +
    ((a*a*a - a)*y2[k_lo] + (b*b*b - b)*y2[k_hi]) * (h*h)/6.0;
}

static void
psdrawpoint (GtkPlot *plot,
	     gboolean clip,
             GtkPlotPoint point,
             GtkPlotSymbolType symbol,
             GdkColor color,
             GtkPlotSymbolStyle symbol_style,
             gint size,
             gint line_width)
{
  gint fill = FALSE;

  if(clip)
    if(point.x < plot->xmin || point.x > plot->xmax ||
       point.y < plot->ymin || point.y > plot->ymax) return;

  if(symbol_style == GTK_PLOT_SYMBOL_OPAQUE && symbol < GTK_PLOT_SYMBOL_PLUS)
     psdrawsymbol (plot, point, symbol,
                   plot->background,
                   TRUE,
                   size,
                   line_width);

  if(symbol_style == GTK_PLOT_SYMBOL_FILLED) fill = TRUE;

  psdrawsymbol (plot, point, symbol,
                color,
                fill,
                size,
                line_width);
}


static void
psdrawsymbol(GtkPlot *plot,
             GtkPlotPoint point,
             GtkPlotSymbolType symbol,
             GdkColor color,
             gint filled,
             gint size,
             gint line_width)
{
    gint x,y;
    gint x0, y0;
    FILE *psout = plot->eps->psfile;

    pssetcolor(plot->eps, color);
    fprintf(psout, "0 slc\n");
    fprintf(psout, "%d slw\n", line_width);
    
    psgetpoint(plot, point.x, point.y, &x, &y);

    switch(symbol) {
       case GTK_PLOT_SYMBOL_NONE:
              break;
       case GTK_PLOT_SYMBOL_SQUARE:
              psdrawsquare (plot->eps, x, y, size, filled);
              break;
       case GTK_PLOT_SYMBOL_CIRCLE:
              psdrawcircle (plot->eps, x, y, size/2, filled);
              break;
       case GTK_PLOT_SYMBOL_UP_TRIANGLE:
              psdrawuptriangle (plot->eps, x, y, size, filled);          
              break;
       case GTK_PLOT_SYMBOL_DOWN_TRIANGLE:
              psdrawdowntriangle (plot->eps, x, y, size, filled);        
              break;
       case GTK_PLOT_SYMBOL_DIAMOND:
              psdrawdiamond (plot->eps, x, y, size, filled);              
              break;
       case GTK_PLOT_SYMBOL_PLUS:
              psdrawplus (plot->eps, x, y, size);
              break;
       case GTK_PLOT_SYMBOL_CROSS:
              psdrawcross (plot->eps, x, y, size);
              break;
       case GTK_PLOT_SYMBOL_STAR:
              psdrawstar (plot->eps, x, y, size);
              break;
       case GTK_PLOT_SYMBOL_BAR:
              psgetpoint(plot, point.x, 0., &x0, &y0);
              psdrawrectangle (plot->eps, x, MIN(y0,y),
                               transform_dx(plot, point.deltax)+1, abs(y-y0), 
                               filled);
              break;
       case GTK_PLOT_SYMBOL_IMPULSE:
              psgetpoint(plot, point.x, 0., &x0, &y0);
              psdrawline(plot->eps, x, MIN(y0,y), x, MAX(y0,y));
              break;

    }
}


static void
psdrawxy (GtkPlot *plot,
          GtkPlotData *dataset,
          GtkPlotPoint point)
{
  gint x,y;
  gint x0,y0;


  if(point.x < plot->xmin || point.x > plot->xmax ||
     point.y < plot->ymin || point.y > plot->ymax) return;

  psgetpoint(plot, point.x, point.y, &x, &y);

  if(dataset->x.line_style != GTK_PLOT_LINE_NONE){
     psgetpoint(plot, point.x, 0., &x0, &y0);
     pssetlinestyle(plot->eps, dataset->x);
     psdrawline(plot->eps, x, y, x, y0);
  }

  if(dataset->x.line_style != GTK_PLOT_LINE_NONE){
    psgetpoint(plot, 0., point.y, &x0, &y0);
    pssetlinestyle(plot->eps, dataset->y);
    psdrawline(plot->eps, x, y, x0, y);
  }
}

static void psdrawline(GtkPlotEPS *eps, gint x0, gint y0, gint xf, gint yf)
{
  FILE *psout = eps->psfile;

  fprintf(psout, "%d %d m\n", x0, y0);
  fprintf(psout, "%d %d l\n", xf, yf);
  fprintf(psout, "s\n");
}

static void
psdrawrectangle(GtkPlotEPS *eps, gint x, gint y, gint x1, gint y1, gint filled)
{
  GdkPoint point[4];

  point[0].x = x ;
  point[0].y = y ;

  point[1].x = x1;
  point[1].y = y;

  point[2].x = x1;
  point[2].y = y1;

  point[3].x = x;
  point[3].y = y1;
 
  psdrawpolygon(eps, point, 4, filled);

}

static void
psdrawsquare(GtkPlotEPS *eps, gint x, gint y, gint size, gint filled)
{
  GdkPoint point[4];

  point[0].x = x - size/2;
  point[0].y = y - size/2;

  point[1].x = x + size/2;
  point[1].y = y - size/2.;

  point[2].x = x + size/2;
  point[2].y = y + size/2;

  point[3].x = x - size/2;
  point[3].y = y + size/2.;
 
  psdrawpolygon(eps, point, 4, filled);

}


static void
psdrawdowntriangle(GtkPlotEPS *eps, gint x, gint y, gint size, gint filled)
{
  GdkPoint point[3];
  gdouble pi = acos(-1.);

  point[0].x = x - size*cos(pi/6.);
  point[0].y = y - size*sin(pi/6.);

  point[1].x = x + size*cos(pi/6.);
  point[1].y = y - size*sin(pi/6.);

  point[2].x = x;
  point[2].y = y + size;

  psdrawpolygon(eps, point, 3, filled);

}

static void
psdrawuptriangle(GtkPlotEPS *eps, gint x, gint y, gint size, gint filled)
{
  GdkPoint point[3];
  gdouble pi = acos(-1.);

  point[0].x = x - size*cos(pi/6.);
  point[0].y = y + size*sin(pi/6.);

  point[1].x = x + size*cos(pi/6.);
  point[1].y = y + size*sin(pi/6.);

  point[2].x = x;
  point[2].y = y - size;

  psdrawpolygon (eps, point, 3, filled);

}

static void
psdrawdiamond(GtkPlotEPS *eps, gint x, gint y, gint size, gint filled)
{
  GdkPoint point[4];

  point[0].x = x - size/2;
  point[0].y = y;

  point[1].x = x;
  point[1].y = y - size/2.;

  point[2].x = x + size/2;
  point[2].y = y;

  point[3].x = x;
  point[3].y = y + size/2.;
 
  psdrawpolygon(eps, point, 4, filled);

}

static void
psdrawplus(GtkPlotEPS *eps, gint x, gint y, gint size)
{
  psdrawline(eps, x-size/2, y, x+size/2, y);

  psdrawline(eps, x, y-size/2, x, y+size/2);
}

static void
psdrawcross(GtkPlotEPS *eps, gint x, gint y, gint size)
{
  psdrawline(eps, x-size/2, y-size/2, x+size/2, y+size/2);

  psdrawline(eps, x-size/2, y+size/2, x+size/2, y-size/2);
}

static void
psdrawstar(GtkPlotEPS *eps, gint x, gint y, gint size)
{
  gdouble s2 = size*sqrt(2.)/4.;

  psdrawline(eps, x-size/2, y, x+size/2, y);

  psdrawline(eps, x, y-size/2, x, y+size/2);

  psdrawline(eps, x-s2, y-s2, x+s2, y+s2);

  psdrawline(eps, x-s2, y+s2, x+s2, y-s2);
}

static void
psdrawcircle(GtkPlotEPS *eps, gint x, gint y, gint size, gint filled)
{
  FILE *psout = eps->psfile;

  fprintf(psout,"n %f %f %f %f 0 360 ellipse\n", 
          (gdouble)x, (gdouble)y, (gdouble)size, (gdouble)size);

  if(filled)
     fprintf(psout,"f\n");
  else
     fprintf(psout,"cp\n");

  fprintf(psout,"s\n");
}

static void
psdrawstring(GtkPlotEPS *eps,
             gint x, gint y,
             gint justification, 
             gint angle,
             gchar *text)
{
  gchar *buffer;
  gchar *str;
  gint len;
  FILE *psout = eps->psfile;

  /* TODO: Use latin-1 encoding */

  /* Escape all '(' and ')':  */
  buffer = g_malloc(2*strlen(text)+1);
  *buffer = 0;
  str = text;
  while (*str != 0) {
    len = strcspn(str,"()\\");
    strncat(buffer, str, len);
    str += len;
    if (*str != 0) {
      strcat(buffer,"\\");
      strncat(buffer, str, 1);
      str++;
    }
  }
  fprintf(psout, "(%s) ", buffer);
  g_free(buffer);

  switch (justification) {
  case GTK_JUSTIFY_LEFT:
    fprintf(psout, "%f %f m", (gfloat)x, (gfloat)y);
    break;
  case GTK_JUSTIFY_CENTER:
    fprintf(psout, "dup sw 2 div %f ex sub %f m", (gfloat)x, (gfloat)y);
    break;
  case GTK_JUSTIFY_RIGHT:
  default:
    fprintf(psout, "dup sw %f ex sub %f m", (gfloat)x, (gfloat)y);
    break;
  }

  fprintf(psout, " gs %f rotate 1 -1 sc sh gr\n", -(gfloat)angle);
}

static void
psdrawtext(GtkWidget *widget, GtkPlotEPS *eps, GtkPlotText text)
{
  gint x, y;

  x = widget->allocation.x + text.x * widget->allocation.width;
  y = widget->allocation.y + text.y * widget->allocation.height;

  if(text.angle == 0 || text.angle == 180)
    y = y + (1 + pow(-1.0 ,text.angle/90))/2 * text.height;
  else
    x = x + pow(-1.0 ,text.angle/90+1) * text.height;

  pssetcolor(eps, text.fg);
  pssetfont(eps, gtk_plot_font_get_psfontname(text.font), text.height);
 
  psdrawstring(eps, x, y, GTK_JUSTIFY_LEFT, text.angle, text.text); 

}

static void
pssetfont(GtkPlotEPS *eps, gchar *font, gint height)
{
  FILE *psout = eps->psfile;

  fprintf(psout, "/%s ff %f scf sf\n", font, (double)height);
}

static void
parse_label(gdouble val, gint precision, gint style, gchar *label)
{
  gdouble auxval;
  gint intspace = 0;

  auxval = abs(val);

  if(auxval > 1)
    intspace = (gint)log10(auxval);

  switch(style){
    case GTK_PLOT_LABEL_EXP:
      sprintf (label, "%*.*E", 1, precision, val);
      break;
    case GTK_PLOT_LABEL_FIXED:
    default:
      sprintf (label, "%*.*f", intspace, precision, val);
  }

}

static void
psdrawlegends (GtkPlot *plot)
{
  GtkPlotEPS *eps;
  GList *datasets;
  GtkPlotData *dataset;
  GtkPlotPoint point;
  GdkRectangle legend_area;
  GtkAllocation allocation;
  GdkFont *font;
  gchar *psfont;
  gint x0, y0;
  gint x, y;
  gint width = 0;
  gint height;

  if(!plot->show_legends) return;

  eps = plot->eps;
  font = gtk_plot_font_get_gdkfont(plot->legends_attr.font,
				   plot->legends_attr.height);
  psfont = gtk_plot_font_get_psfontname(plot->legends_attr.font);

/* first draw the rectangle for the background */
  allocation = GTK_WIDGET(plot)->allocation;
  x0 = allocation.x + plot->x * allocation.width +
       plot->legends_x * plot->width * allocation.width;
  y0 = allocation.y + plot->y * allocation.height +
       plot->legends_y * plot->height * allocation.height;

  height = 8;

  datasets = g_list_first(plot->data_sets);
  while(datasets)
   {
     dataset = (GtkPlotData *)datasets->data;

     if(dataset->is_visible)
       {
         height += font->ascent + font->descent;
         if(dataset->legend)
            width = MAX(width,
                    gdk_string_width(font, dataset->legend));
       }

     datasets = datasets->next;
   }

  legend_area.x = x0;
  legend_area.y = y0;
  legend_area.width = width + plot->legends_line_width + 12;
  legend_area.height = height;

  if(!plot->legends_attr.transparent){
     pssetcolor(eps, plot->legends_attr.bg);
     psdrawrectangle(eps, 
                     legend_area.x, legend_area.y,
                     legend_area.x+legend_area.width, 
                     legend_area.y+legend_area.height,
                     TRUE);
  }

  plot->legends_width = legend_area.width;
  plot->legends_height = legend_area.height;

/* now draw the legends */

  height = 4;

  datasets = plot->data_sets;
  while(datasets)
   {
     dataset = (GtkPlotData *)datasets->data;

     if(dataset->is_visible)
       {
         x = x0 + 4;
         height += font->ascent + font->descent;
         y = y0 + height;

         pssetlinestyle(eps, dataset->line);
         psdrawline(eps,
                    x, y - font->ascent / 2,
                    x + plot->legends_line_width, y - font->ascent / 2);

         pssetcolor(eps, plot->legends_attr.fg);

         pssetpoint(plot, x + plot->legends_line_width / 2,
                    y - font->ascent / 2,
                    &point);

         if(dataset->symbol.symbol_type != GTK_PLOT_SYMBOL_BAR &&
            dataset->symbol.symbol_type != GTK_PLOT_SYMBOL_IMPULSE)
                      psdrawpoint (plot,
                                   FALSE,
                                   point,
                                   dataset->symbol.symbol_type,
                                   dataset->symbol.color,
                                   dataset->symbol.symbol_style,
                                   dataset->symbol.size,
                                   dataset->symbol.line_width);

         if(dataset->legend){
                pssetcolor(eps, plot->legends_attr.fg);
                pssetfont(eps, psfont, plot->legends_attr.height);
                psdrawstring (eps,
                              x + plot->legends_line_width + 4, y,
                              GTK_JUSTIFY_LEFT, 0,
                              dataset->legend);
         }
       }
     datasets=datasets->next;
   }

   pssetlinewidth(eps, plot->legends_border_width);
   pssetcolor(eps, plot->legends_attr.fg);

   if(plot->show_legends_border)
      {
        psdrawrectangle(eps,
                        legend_area.x, legend_area.y,
                        legend_area.x + legend_area.width, 
                        legend_area.y + legend_area.height, 
                        FALSE);
      }

   pssetlinewidth(eps, 0);
   if(plot->show_legends_border && plot->show_legends_shadow)
      {
        psdrawrectangle(eps,
                        legend_area.x + plot->legends_shadow_width,
                        legend_area.y + legend_area.height,
                        legend_area.x + plot->legends_shadow_width +
                        legend_area.width, 
                        legend_area.y + legend_area.height +
                        plot->legends_shadow_width,
                        TRUE);
        psdrawrectangle(eps,
                        legend_area.x + legend_area.width,
                        legend_area.y + plot->legends_shadow_width,
                        legend_area.x + legend_area.width +
                        plot->legends_shadow_width , 
                        legend_area.y + plot->legends_shadow_width +
                        legend_area.height,
			TRUE);
      }
}

