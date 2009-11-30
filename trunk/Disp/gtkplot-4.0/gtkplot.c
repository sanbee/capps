/* gtkplot - 2d scientific plots widget for gtk+
 * Copyright 1999  Adrian E. Feiguin <feiguin@ifir.edu.ar>
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gtk/gtk.h>
#include "gtkplot.h"
#include "gtkplotfont.h"
#include "gtkplotlayout.h"

#define DEFAULT_WIDTH 420
#define DEFAULT_HEIGHT 340
#define DEFAULT_FONT_HEIGHT 12

#define roundint(N) ((N)+0.5)
gchar DEFAULT_FONT[] = "Helvetica";

/* Signals */

enum
{
  CHANGED,
  MOVED,
  RESIZED,
  LAST_SIGNAL,
};

static void gtk_plot_class_init 		(GtkPlotClass *class);
static void gtk_plot_init 			(GtkPlot *plot);
static void gtk_plot_size_request 		(GtkWidget *widget, 
                                                 GtkRequisition *requisition);
void gtk_plot_draw 			(GtkWidget *widget, 
						 GdkRectangle *area);
static void gtk_plot_calc_ticks			(GtkPlot *plot, 
						 gint orientation);
static void gtk_plot_paint 			(GtkWidget *widget, 
						 GdkRectangle *area);
static void gtk_plot_draw_grids                 (GtkPlot *plot, 
						 GdkRectangle area);
static void gtk_plot_draw_axis			(GtkPlot *plot, 
					 	 GtkPlotAxis axis, 
					 	 gint x, gint y);
static void gtk_plot_draw_labels		(GtkPlot *plot, 
 						 GdkRectangle area, 
						 GtkPlotAxis axis, 
						 gint x, gint y);
static void gtk_plot_real_draw_dataset		(GtkPlot *plot, 
						 GdkRectangle *area, 
						 GdkGC *gc,
						 GtkPlotData *dataset);
static void gtk_plot_draw_point			(GtkPlot *plot, 
						 GdkRectangle area,
						 GdkGC *gc,
						 gint x, gint y, gdouble dx, 
						 GtkPlotSymbolType symbol,
        					 GdkColor color,
						 GtkPlotSymbolStyle symbol_style,
                   				 gint symbol_size, 
					 	 gint line_width);
static void gtk_plot_draw_symbol		(GtkPlot *plot, 
						 GdkRectangle area,
						 GdkGC *gc,
						 gint x, gint y, gdouble dx, 
						 GtkPlotSymbolType symbol,
        					 GdkColor color,
						 gint filled,
                   				 gint symbol_size, 
						 gint line_width);
static void gtk_plot_draw_xy 			(GtkPlot *plot, 
						 GdkRectangle area,
						 GdkGC *gc,
						 GtkPlotData *dataset);
static void gtk_plot_draw_errbars		(GtkPlot *plot, 
						 GdkRectangle area,
						 GdkGC *gc,
						 GtkPlotData *dataset); 
static void gtk_plot_draw_down_triangle		(GtkPlot *plot, 
						 GdkGC *gc, 
                            		  	 gint x,
 						 gint y, 
						 gint size, 
						 gint filled);
static void gtk_plot_draw_up_triangle		(GtkPlot *plot, 
						 GdkGC *gc, 
                            		  	 gint x,
 						 gint y, 
						 gint size, 
						 gint filled);
static void gtk_plot_draw_diamond		(GtkPlot *plot, 
						 GdkGC *gc, 
                            		  	 gint x,
 						 gint y, 
						 gint size, 
						 gint filled);
static void gtk_plot_draw_plus			(GtkPlot *plot, 
						 GdkGC *gc, 
                            		  	 gint x,
 						 gint y, 
						 gint size); 
static void gtk_plot_draw_cross			(GtkPlot *plot, 
						 GdkGC *gc, 
                            		  	 gint x,
 						 gint y, 
						 gint size); 
static void gtk_plot_draw_star			(GtkPlot *plot, 
						 GdkGC *gc, 
                            		  	 gint x,
 						 gint y, 
						 gint size); 

static void gtk_plot_connect_points		(GtkPlot *plot, 
						 GdkRectangle area,
						 GdkGC *gc,
						 GtkPlotData *data);

static void gtk_plot_draw_line			(GtkPlot *plot, 
						 GdkGC *gc, 
                   				 GtkPlotLine line,
                   				 gint x1, gint y1, 
						 gint x2, gint y2);
static void gtk_plot_set_line_attributes	(GtkPlot *plot, 
                             			 GtkPlotLine line,
                             			 GdkGC *gc);
static void gtk_plot_draw_legends		(GtkPlot *plot,
						 GdkRectangle area);
static void gtk_plot_draw_text             	(GtkPlot *plot, 
						 GdkRectangle area,
                                                 GtkPlotText text); 
static GdkPixmap *rotate_text			(GtkPlot *plot, 
						 GtkPlotText text, 
                                                 gint *width, gint *height);
static void gtk_plot_get_real_pixel		(GtkPlot *plot, 
						 gdouble xx, gdouble yy , 
						 gint *x, gint *y, 
						 GdkRectangle area);
static void gtk_plot_pixel_get_real_point	(GtkPlot *plot, 
						 gint x, gint y, 
					 	 gdouble *px, gdouble *py,
			 			 GdkRectangle area);
static void gtk_plot_set_wcstransform           (GtkPlot *plot);
static gint transform_y				(GtkPlot *plot, gdouble y);
static gint transform_x				(GtkPlot *plot, gdouble x);
static gint transform_dy			(GtkPlot *plot, gdouble dy);
static gint transform_dx			(GtkPlot *plot, gdouble dx);
static gdouble inverse_y			(GtkPlot *plot, gint y);
static gdouble inverse_x			(GtkPlot *plot, gint x);
static gdouble inverse_dy			(GtkPlot *plot, gint dy);
static gdouble inverse_dx			(GtkPlot *plot, gint dx);
/*static gint roundint				(gdouble x);*/
static void parse_label			        (gdouble val, 
						 gint precision, 
						 gint style,
                                                 gchar *label);
static void spline_solve 			(int n, 
                                                 gdouble x[], gdouble y[], 
						 gdouble y2[]);
static gdouble spline_eval 			(int n, 
                                                 gdouble x[], 
                                                 gdouble y[], 
						 gdouble y2[], gdouble val);

typedef gboolean (*GtkPlotSignal) (GtkObject *object,
                                   gpointer arg1,
                                   gpointer arg2,
                                   gpointer user_data);

static void
gtk_plot_marshal_BOOL__POINTER_POINTER   (GtkObject *object,
                                          GtkSignalFunc func,
                                          gpointer func_data,
                                          GtkArg * args);

static GtkWidgetClass *parent_class = NULL;
static guint plot_signals[LAST_SIGNAL] = {0};


guint
gtk_plot_get_type (void)
{
  static GtkType plot_type = 0;

  if (!plot_type)
    {
      GtkTypeInfo plot_info =
      {
	"GtkPlot",
	sizeof (GtkPlot),
	sizeof (GtkPlotClass),
	(GtkClassInitFunc) gtk_plot_class_init,
	(GtkObjectInitFunc) gtk_plot_init,
	/* reserved 1*/ NULL,
        /* reserved 2 */ NULL,
        (GtkClassInitFunc) NULL,
      };

      plot_type = gtk_type_unique (GTK_TYPE_MISC, &plot_info);
    }
  return plot_type;
}

static void
gtk_plot_class_init (GtkPlotClass *class)
{
  GtkObjectClass *object_class;
  GtkWidgetClass *widget_class;

  parent_class = gtk_type_class (gtk_widget_get_type ());

  object_class = (GtkObjectClass *) class;
  widget_class = (GtkWidgetClass *) class;

  widget_class->draw = gtk_plot_draw;
  widget_class->size_request = gtk_plot_size_request;

  plot_signals[CHANGED] = 
    gtk_signal_new("changed",
                   GTK_RUN_LAST,
                   object_class->type,
                   GTK_SIGNAL_OFFSET (GtkPlotClass, changed),
                   gtk_marshal_NONE__NONE,
                   GTK_TYPE_NONE, 0); 

  plot_signals[MOVED] = 
    gtk_signal_new("moved",
                   GTK_RUN_LAST,
                   object_class->type,
                   GTK_SIGNAL_OFFSET (GtkPlotClass, moved),
                   gtk_plot_marshal_BOOL__POINTER_POINTER,
                   GTK_TYPE_NONE, 2, GTK_TYPE_POINTER, GTK_TYPE_POINTER); 

  plot_signals[RESIZED] = 
    gtk_signal_new("resized",
                   GTK_RUN_LAST,
                   object_class->type,
                   GTK_SIGNAL_OFFSET (GtkPlotClass, resized),
                   gtk_plot_marshal_BOOL__POINTER_POINTER,
                   GTK_TYPE_NONE, 2, GTK_TYPE_POINTER, GTK_TYPE_POINTER); 

  gtk_object_class_add_signals (object_class, plot_signals, LAST_SIGNAL);

  class->changed = NULL;
  class->moved = NULL;
  class->resized = NULL;

}

static void
gtk_plot_marshal_BOOL__POINTER_POINTER         (GtkObject *object,
                                                GtkSignalFunc func,
                                                gpointer func_data,
                                                GtkArg * args)
{
  GtkPlotSignal rfunc;
  gboolean *veto;
  veto = GTK_RETLOC_BOOL (args[2]);

  rfunc = (GtkPlotSignal) func;

  *veto = (*rfunc) (object,
                    GTK_VALUE_POINTER (args[0]),
                    GTK_VALUE_POINTER (args[1]),
                    func_data);
}


static void
gtk_plot_init (GtkPlot *plot)
{
  GtkWidget *widget;
  GTK_WIDGET_SET_FLAGS(plot, GTK_NO_WINDOW);
  GTK_WIDGET_UNSET_FLAGS(plot,GTK_PLOT_FREEZE);

  widget = GTK_WIDGET(plot);
  gdk_color_black(gtk_widget_get_colormap(widget), &widget->style->black);
  gdk_color_white(gtk_widget_get_colormap(widget), &widget->style->white);

  plot->xmin = 0.;
  plot->xmax = 1.000000;
  plot->ymin = 0.;
  plot->ymax = 1.000000;
  plot->xwcs2pix = 0.0;
  plot->ywcs2pix = 0.0;

  plot->xmajor.nticks = 0;
  plot->xmajor.ticks = NULL;
  plot->xmajor.value = NULL;
  plot->xmajor.set_limits = FALSE;
  plot->xmajor.begin = 0;
  plot->xmajor.end = 0;
  plot->xmajor.step = .100000000;

  plot->xminor.nticks = 0;
  plot->xminor.ticks = NULL;
  plot->xminor.value = NULL;
  plot->xminor.set_limits = FALSE;
  plot->xminor.begin = 0;
  plot->xminor.end = 0;
  plot->xminor.step = .0500000000;

  plot->ymajor.nticks = 0;
  plot->ymajor.ticks = NULL;
  plot->ymajor.value = NULL;
  plot->ymajor.set_limits = FALSE;
  plot->ymajor.begin = 0;
  plot->ymajor.end = 0;
  plot->ymajor.step = .100000000;

  plot->yminor.nticks = 0;
  plot->yminor.ticks = NULL;
  plot->yminor.value = NULL;
  plot->yminor.set_limits = FALSE;
  plot->yminor.begin = 0;
  plot->yminor.end = 0;
  plot->yminor.step = .0500000000;


  plot->bottom.min = 0.0;
  plot->bottom.max = 1.0;
  plot->bottom.labels_offset = 0;
  plot->bottom.ticks_mask = GTK_PLOT_TICKS_UP;
  plot->bottom.ticks_length = 8;
  plot->bottom.ticks_width = 1;
  plot->bottom.orientation = GTK_ORIENTATION_HORIZONTAL;
  plot->bottom.scale_type = GTK_PLOT_SCALE_LINEAR;
  plot->top.min = 0.0;
  plot->top.max = 1.0;
  plot->top.ticks_mask = GTK_PLOT_TICKS_DOWN;
  plot->top.ticks_length = 8;
  plot->top.ticks_width = 1;
  plot->top.labels_offset = 0;
  plot->top.orientation = GTK_ORIENTATION_HORIZONTAL;
  plot->top.scale_type = GTK_PLOT_SCALE_LINEAR;
  plot->left.min = 0.0;
  plot->left.max = 1.0;
  plot->left.ticks_mask = GTK_PLOT_TICKS_RIGHT;
  plot->left.ticks_length = 8;
  plot->left.ticks_width = 1;
  plot->left.labels_offset = 10;
  plot->left.orientation = GTK_ORIENTATION_VERTICAL;
  plot->left.scale_type = GTK_PLOT_SCALE_LINEAR;
  plot->right.min = 0.0;
  plot->right.max = 1.0;
  plot->right.ticks_mask = GTK_PLOT_TICKS_LEFT;
  plot->right.ticks_length = 8;
  plot->right.ticks_width = 1;
  plot->right.labels_offset = 10;
  plot->right.orientation = GTK_ORIENTATION_VERTICAL;
  plot->right.scale_type = GTK_PLOT_SCALE_LINEAR;


  plot->left.line.line_style = GTK_PLOT_LINE_SOLID;
  plot->left.line.line_width = 2;
  plot->left.line.color = widget->style->black; 
  plot->left.label_attr.font = g_strdup(DEFAULT_FONT);
  plot->left.label_attr.height = DEFAULT_FONT_HEIGHT;
  plot->left.label_attr.fg = widget->style->black;
  plot->left.label_attr.bg = widget->style->white;
  plot->left.label_attr.transparent = TRUE;
  plot->left.label_mask = GTK_PLOT_LABEL_LEFT;
  plot->left.label_style = GTK_PLOT_LABEL_FLOAT;
  plot->left.label_precision = 1;
  plot->left.title.angle = 90;
  plot->left.title.justification = GTK_JUSTIFY_CENTER;
  plot->left.title.font = g_strdup(DEFAULT_FONT);
  plot->left.title.height = DEFAULT_FONT_HEIGHT;
  plot->left.title.fg = widget->style->black;
  plot->left.title.bg = widget->style->white;
  plot->left.title.transparent = TRUE;
  plot->left.title.text = g_strdup("Y Title");
  plot->left.title_visible = TRUE;

  plot->right.line.line_style = GTK_PLOT_LINE_SOLID;
  plot->right.line.line_width = 2;
  plot->right.line.color = widget->style->black; 
  plot->right.label_attr.font = g_strdup(DEFAULT_FONT);
  plot->right.label_attr.height = DEFAULT_FONT_HEIGHT;
  plot->right.label_attr.fg = widget->style->black;
  plot->right.label_attr.bg = widget->style->white;
  plot->right.label_attr.transparent = TRUE;
  plot->right.label_mask = GTK_PLOT_LABEL_RIGHT;
  plot->right.label_style = GTK_PLOT_LABEL_FLOAT;
  plot->right.label_precision = 1;
  plot->right.title.angle = 270;
  plot->right.title.justification = GTK_JUSTIFY_CENTER;
  plot->right.title.font = g_strdup(DEFAULT_FONT);
  plot->right.title.height = DEFAULT_FONT_HEIGHT;
  plot->right.title.fg = widget->style->black;
  plot->right.title.bg = widget->style->white;
  plot->right.title.transparent = TRUE;
  plot->right.title.text = g_strdup("Y Title");
  plot->right.title_visible = TRUE;

  plot->bottom.line.line_style = GTK_PLOT_LINE_SOLID;
  plot->bottom.line.line_width = 2;
  plot->bottom.line.color = widget->style->black; 
  plot->bottom.label_attr.font = g_strdup(DEFAULT_FONT);
  plot->bottom.label_attr.height = DEFAULT_FONT_HEIGHT;
  plot->bottom.label_attr.fg = widget->style->black;
  plot->bottom.label_attr.bg = widget->style->white;
  plot->bottom.label_attr.transparent = TRUE;
  plot->bottom.label_mask = GTK_PLOT_LABEL_BOTTOM;
  plot->bottom.label_style = GTK_PLOT_LABEL_FLOAT;
  plot->bottom.label_precision = 1;
  plot->bottom.title.angle = 0;
  plot->bottom.title.justification = GTK_JUSTIFY_CENTER;
  plot->bottom.title.font = g_strdup(DEFAULT_FONT);
  plot->bottom.title.height = DEFAULT_FONT_HEIGHT;
  plot->bottom.title.fg = widget->style->black;
  plot->bottom.title.bg = widget->style->white;
  plot->bottom.title.transparent = TRUE;
  plot->bottom.title.text = g_strdup("X Title");
  plot->bottom.title_visible = TRUE;

  plot->top.line.line_style = GTK_PLOT_LINE_SOLID;
  plot->top.line.line_width = 2;
  plot->top.line.color = widget->style->black; 
  plot->top.label_attr.font = g_strdup(DEFAULT_FONT);
  plot->top.label_attr.height = DEFAULT_FONT_HEIGHT;
  plot->top.label_attr.fg = widget->style->black;
  plot->top.label_attr.bg = widget->style->white;
  plot->top.label_attr.transparent = TRUE;
  plot->top.label_mask = GTK_PLOT_LABEL_TOP;
  plot->top.label_style = GTK_PLOT_LABEL_FLOAT;
  plot->top.label_precision = 1;
  plot->top.title.angle = 0;
  plot->top.title.justification = GTK_JUSTIFY_CENTER;
  plot->top.title.font = g_strdup(DEFAULT_FONT);
  plot->top.title.height = DEFAULT_FONT_HEIGHT;
  plot->top.title.fg = widget->style->black;
  plot->top.title.bg = widget->style->white;
  plot->top.title.transparent = TRUE;
  plot->top.title.text = g_strdup("X Title");
  plot->top.title_visible = TRUE;

  plot->x0_line.line_style = GTK_PLOT_LINE_SOLID;
  plot->x0_line.line_width = 0;
  plot->x0_line.color = widget->style->black; 

  plot->y0_line.line_style = GTK_PLOT_LINE_SOLID;
  plot->y0_line.line_width = 0;
  plot->y0_line.color = widget->style->black; 

  plot->major_vgrid.line_style = GTK_PLOT_LINE_SOLID;
  plot->major_vgrid.line_width = 0;
  plot->major_vgrid.color = widget->style->black; 

  plot->minor_vgrid.line_style = GTK_PLOT_LINE_DOTTED;
  plot->minor_vgrid.line_width = 0;
  plot->minor_vgrid.color = widget->style->black;

  plot->major_hgrid.line_style = GTK_PLOT_LINE_SOLID;
  plot->major_hgrid.line_width = 0;
  plot->major_hgrid.color = widget->style->black; 

  plot->minor_hgrid.line_style = GTK_PLOT_LINE_DOTTED;
  plot->minor_hgrid.line_width = 0;
  plot->minor_hgrid.color = widget->style->black;

  plot->legends_x = .6;
  plot->legends_y = .1;
  plot->legends_width = 0;
  plot->legends_height = 0;
  plot->legends_line_width = 30;
  plot->legends_border_width = 1;
  plot->legends_shadow_width = 3;
  plot->show_legends =  TRUE;
  plot->show_legends_border =  TRUE;
  plot->show_legends_shadow =  FALSE;
  plot->legends_attr.font = g_strdup(DEFAULT_FONT);
  plot->legends_attr.height = DEFAULT_FONT_HEIGHT;
  plot->legends_attr.fg = widget->style->black;
  plot->legends_attr.bg = widget->style->white;
  plot->legends_attr.transparent = TRUE;
  
  plot->background = widget->style->white;

  gtk_plot_set_wcstransform(plot);
  gtk_plot_calc_ticks(plot, GTK_ORIENTATION_HORIZONTAL);
  gtk_plot_calc_ticks(plot, GTK_ORIENTATION_VERTICAL);
}


void
gtk_plot_draw (GtkWidget *widget, GdkRectangle *area)
{
  gtk_plot_paint(widget, area);
}

static void
gtk_plot_paint (GtkWidget *widget, GdkRectangle *drawing_area)
{
  GtkPlot *plot;
  GtkPlotText *child_text;
  GtkStyle *style;
  GdkPixmap *pixmap;
  GdkGC *gc;
  GList *dataset;
  GList *text;
  GdkRectangle area;
  gint width, height;
  gint xoffset, yoffset ;

  if(!GTK_WIDGET_DRAWABLE(widget)) return;
  plot = GTK_PLOT(widget);

  gtk_plot_set_wcstransform(plot);

  if(!plot->drawable) return;

  if(drawing_area == NULL){
     area.x = widget->allocation.x;
     area.y = widget->allocation.y;
     area.width = widget->allocation.width;
     area.height = widget->allocation.height;
  } else {
     area = *drawing_area;
  }

  xoffset = area.x + roundint(plot->x * widget->allocation.width);
  yoffset = area.y + roundint(plot->y * widget->allocation.height);
  width = roundint(plot->width * widget->allocation.width);
  height = roundint(plot->height * widget->allocation.height);

  style = gtk_widget_get_style(widget);

  pixmap = plot->drawable;

  gc = gdk_gc_new(pixmap);
  gdk_gc_set_foreground(gc, &plot->background);
  /*  gdk_gc_set_clip_rectangle(gc, &area);*/


  if(!GTK_PLOT_TRANSPARENT(plot))
    gdk_draw_rectangle (pixmap, gc, TRUE,
  		        xoffset, yoffset,
		        width , height);

  /* draw frame to guide the eyes*/
/*  gdk_draw_rectangle (pixmap, gc, FALSE,
		      xoffset, yoffset,
		      width , height);
*/

  /* draw the tips & grid lines */

  gtk_plot_calc_ticks(plot, GTK_ORIENTATION_HORIZONTAL);
  gtk_plot_calc_ticks(plot, GTK_ORIENTATION_VERTICAL);

  gtk_plot_draw_grids(plot, area);

  if(GTK_PLOT_SHOW_BOTTOM_AXIS(plot))
    {
      /*SanB
	If labels are to be shown, clear the previous labels.
	The {x,y}offsets and width,height in gdk_draw_rectangle
	is computed very crudely and will not always be the right thing.  
	This must be computed with the knowledge of the height and width
	of the fonts used
      */
      if (gtk_plot_get_axis (plot, GTK_PLOT_AXIS_BOTTOM)->label_mask != 
	  GTK_PLOT_LABEL_NONE)
	{
	  gdk_draw_rectangle (pixmap, gc, TRUE,
			      0, yoffset+height,
			      width+xoffset+10, 40);
	  gtk_plot_draw_axis(plot, plot->bottom, 
			     xoffset,
			     yoffset+height);
	  gtk_plot_draw_labels(plot, area, plot->bottom, 
			       xoffset,
			       yoffset+height);
	}
    }

  if(GTK_PLOT_SHOW_TOP_AXIS(plot))
    {
      gtk_plot_draw_axis(plot, plot->top,
                         xoffset,
                         yoffset);
      gtk_plot_draw_labels(plot, area, plot->top,
                           xoffset,
                           yoffset);
    }

  if(GTK_PLOT_SHOW_LEFT_AXIS(plot))
    {
      /*SanB
	If labels are to be shown, clear the previous labels.
	The {x,y}offsets and width,height in gdk_draw_rectangle
	is computed very crudely and will not always be the right thing.  
	This must be computed with the knowledge of the height and width
	of the fonts used
      */
      if (gtk_plot_get_axis (plot, GTK_PLOT_AXIS_LEFT)->label_mask != 
	  GTK_PLOT_LABEL_NONE)
	{
	  gdk_draw_rectangle (pixmap, gc, TRUE,
			      0, yoffset,
			      xoffset-1, height);
	  
	}
      gtk_plot_draw_axis(plot, plot->left,
                         xoffset,
                         yoffset);
      gtk_plot_draw_labels(plot, area, plot->left,
                           xoffset,
                           yoffset);

    }

  if(GTK_PLOT_SHOW_RIGHT_AXIS(plot))
    {
      gtk_plot_draw_axis(plot, plot->right,
                         xoffset+width,
                         yoffset);
      gtk_plot_draw_labels(plot, area, plot->right,
                           xoffset+width,
                           yoffset);
    }

  dataset = plot->data_sets;
  while(dataset)
   {
     gtk_plot_real_draw_dataset(plot, &area, gc, (GtkPlotData *)dataset->data);
     dataset = dataset->next;
   }

  text = plot->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;  
     gtk_plot_draw_text(plot, area, *child_text);
     text = text->next;
   }

  gtk_plot_draw_legends(plot, area);

  //gtk_plot_refresh (plot, &area);
  gdk_gc_unref(gc);
}

void 
gtk_plot_refresh (GtkPlot *plot, GdkRectangle *drawing_area)
{
  GtkWidget *widget;
  GdkPixmap *pixmap;
  GdkRectangle area;

  widget = GTK_WIDGET(plot);
  if(!GTK_WIDGET_DRAWABLE(widget)) return;

  if(!plot->drawable) return;
  pixmap = plot->drawable;

  if(drawing_area == NULL){
     area.x = widget->allocation.x;
     area.y = widget->allocation.y;
     area.width = widget->allocation.width;
     area.height = widget->allocation.height;
  } else {
     area = *drawing_area;
  }


  //  if (!(GTK_PLOT_FLAGS(plot) & GTK_PLOT_FREEZE))
  gdk_draw_pixmap(widget->window,
                  widget->style->fg_gc[GTK_STATE_NORMAL],
                  pixmap,
                  area.x, 
                  area.y, 
                  widget->allocation.x, 
                  widget->allocation.y, 
                  widget->allocation.width, 
                  widget->allocation.height);  
}

static void
gtk_plot_size_request (GtkWidget *widget, GtkRequisition *requisition)
{
  GtkPlot *plot;

  plot = GTK_PLOT(widget);

  requisition->width =  DEFAULT_WIDTH;
  requisition->height =  DEFAULT_HEIGHT;
}

GtkWidget*
gtk_plot_new (GdkPixmap *pixmap)
{
  GtkPlot *plot;

  plot = gtk_type_new (gtk_plot_get_type ());
  plot->drawable = pixmap;

  GTK_PLOT_SET_FLAGS(plot, GTK_PLOT_SHOW_BOTTOM_AXIS);
  GTK_PLOT_SET_FLAGS(plot, GTK_PLOT_SHOW_LEFT_AXIS);

  plot->x = .15;
  plot->y = .1;
  plot->width = .6;
  plot->height = .6;

  plot->left.title.x = plot->x;  
  plot->left.title.y = plot->y + plot->height / 2.;
  plot->right.title.x = plot->x + plot->width;  
  plot->right.title.y = plot->y + plot->height / 2.;
  plot->top.title.x = plot->x + plot->width / 2.;  
  plot->top.title.y = plot->y;
  plot->bottom.title.x = plot->x + plot->width / 2.;  
  plot->bottom.title.y = plot->y + plot->height;

  plot->left.title.x -= 50. / (gdouble)DEFAULT_WIDTH;  
  plot->right.title.x += 30. / (gdouble)DEFAULT_WIDTH;  
  plot->top.title.y -= 35. / (gdouble)DEFAULT_HEIGHT;
  plot->bottom.title.y += 25. / (gdouble)DEFAULT_HEIGHT;

  return GTK_WIDGET (plot);
}

GtkWidget*
gtk_plot_new_with_size (GdkPixmap *pixmap, gdouble width, gdouble height)
{
  GtkWidget *plot; 

  plot = gtk_plot_new(pixmap);

  gtk_plot_resize (GTK_PLOT(plot), width, height);

  return(plot);
}

void
gtk_plot_set_drawable (GtkPlot *plot, GdkDrawable *drawable)
{
  plot->drawable = drawable;
}

GdkDrawable *
gtk_plot_get_drawable (GtkPlot *plot)
{
  return(plot->drawable);
}

static void
gtk_plot_draw_grids(GtkPlot *plot, GdkRectangle area)
{
  GtkWidget *widget;
  gint ix, iy;
  gint width, height;
  gint xp, yp;
  gint ntick;

  widget = GTK_WIDGET(plot);

  xp = roundint(plot->x * (gdouble)widget->allocation.width);
  yp = roundint(plot->y * (gdouble)widget->allocation.height);
  width = roundint(plot->width * (gdouble)widget->allocation.width);
  height = roundint(plot->height * (gdouble)widget->allocation.height);
 
  if(GTK_PLOT_SHOW_X0(plot))
    {
          if(plot->xmin <= 0. && plot->xmax >= 0.)
            {
              ix = transform_x(plot, 0.);
              ix += area.x+xp;
              gtk_plot_draw_line(plot, NULL, plot->x0_line,
                                 ix, 
                                 area.y+yp+1,
                                 ix, 
                                 area.y+yp+height);
            }
    }

  if(GTK_PLOT_SHOW_Y0(plot))
    {
          if(plot->ymin <= 0. && plot->ymax >= 0.)
            {
              iy = transform_y(plot, 0.);
              iy = area.y+height+yp-iy;
              gtk_plot_draw_line(plot, NULL, plot->y0_line,
                                 area.x+xp, 
                                 iy,
                                 area.x + xp + width, 
                                 iy);
            }
    }

  if(GTK_PLOT_SHOW_V_GRID(plot))
    {
          for(ntick = 0; ntick < plot->xmajor.nticks; ntick++){
           ix = area.x+xp+plot->xmajor.ticks[ntick];
           gtk_plot_draw_line(plot, NULL, plot->major_vgrid,
                              ix, 
                              area.y+yp+1,
                              ix, 
                              area.y+yp+height);
          }
          for(ntick = 0; ntick < plot->xminor.nticks; ntick++){
           ix = area.x+xp+plot->xminor.ticks[ntick];
           gtk_plot_draw_line(plot, NULL, plot->minor_vgrid,
                              ix, 
                              area.y+yp+1,
                              ix, 
                              area.y+yp+height);
          }
    }
  if(GTK_PLOT_SHOW_H_GRID(plot))
    {
          for(ntick = 0; ntick < plot->ymajor.nticks; ntick++){
           iy = area.y+height+yp-plot->ymajor.ticks[ntick];
           gtk_plot_draw_line(plot, NULL, plot->major_hgrid,
                              area.x+xp, 
                              iy,
                              area.x + xp + width, 
                              iy);
          }
          for(ntick = 0; ntick < plot->yminor.nticks; ntick++){
           iy = area.y+height+yp-plot->yminor.ticks[ntick];
           gtk_plot_draw_line(plot, NULL, plot->minor_hgrid,
                              area.x+xp, 
                              iy,
                              area.x + xp + width, 
                              iy);
          }
    }
}

static void
gtk_plot_draw_axis(GtkPlot *plot, GtkPlotAxis axis, gint x, gint y)
{
  GtkWidget *widget;
  GdkGC *gc;
  gint xx, yy;
  gint line_width;
  gint xp, yp, width, height;
  gint ntick;

  widget = GTK_WIDGET(plot); 
  xp = roundint(plot->x * (gdouble)widget->allocation.width);
  yp = roundint(plot->y * (gdouble)widget->allocation.height);
  width = roundint(plot->width * (gdouble)widget->allocation.width);
  height = roundint(plot->height * (gdouble)widget->allocation.height);

  gc = gdk_gc_new(plot->drawable);

  line_width = axis.line.line_width;

  switch(axis.orientation){
     case GTK_ORIENTATION_HORIZONTAL:
         gdk_gc_set_line_attributes(gc, axis.line.line_width, 0, 3, 0);
         gdk_draw_line(plot->drawable,
                       gc,
                       x, 
                       y,
                       x+width, 
                       y);
         gdk_gc_set_line_attributes(gc, axis.ticks_width, 0, 1, 0);
	 if ((!plot->xmajor.ticks) || (!plot->xminor.ticks)) break;
         for(ntick = 0; ntick < plot->xmajor.nticks; ntick++){
             xx = plot->xmajor.ticks[ntick];
             if(axis.ticks_mask & GTK_PLOT_TICKS_UP)
                gdk_draw_line(plot->drawable,
                              gc,
                              x+xx, 
                              y,
                              x+xx, 
                              y-axis.ticks_length);
             if(axis.ticks_mask & GTK_PLOT_TICKS_DOWN)
                gdk_draw_line(plot->drawable,
                              gc,
                              x+xx, 
                              y+axis.ticks_length,
                              x+xx, 
                              y);
         }     
         for(ntick = 0; ntick < plot->xminor.nticks; ntick++){
             xx = plot->xminor.ticks[ntick];
             if(axis.ticks_mask & GTK_PLOT_TICKS_UP)
                gdk_draw_line(plot->drawable,
                              gc,
                              x+xx, 
                              y,
                              x+xx, 
                              y-axis.ticks_length/2-1);
               
             if(axis.ticks_mask & GTK_PLOT_TICKS_DOWN)
                gdk_draw_line(plot->drawable,
                              gc,
                              x+xx, 
                              y+axis.ticks_length/2+1,
                              x+xx, 
                              y);
         }     
         break;  
     case GTK_ORIENTATION_VERTICAL:
         y = y + height;
         gdk_gc_set_line_attributes(gc, axis.line.line_width, 0, 3, 0);
         gdk_draw_line(plot->drawable,
                       gc,
                       x, 
                       y-height,
                       x, 
                       y);
         gdk_gc_set_line_attributes(gc, axis.ticks_width, 0, 1, 0);
	 if ((!plot->ymajor.ticks) || (!plot->yminor.ticks)) break;
         for(ntick = 0; ntick < plot->ymajor.nticks; ntick++){
             yy = plot->ymajor.ticks[ntick];
             if(axis.ticks_mask & GTK_PLOT_TICKS_RIGHT)
                gdk_draw_line(plot->drawable,
                              gc,
                              x,
                              y-yy, 
                              x+axis.ticks_length,
                              y-yy); 
               
             if(axis.ticks_mask & GTK_PLOT_TICKS_LEFT)
                gdk_draw_line(plot->drawable,
                              gc,
                              x-axis.ticks_length,
                              y-yy, 
                              x,
                              y-yy); 
         }     
         for(ntick = 0; ntick < plot->yminor.nticks; ntick++){
             yy = plot->yminor.ticks[ntick];
             if(axis.ticks_mask & GTK_PLOT_TICKS_RIGHT)
                gdk_draw_line(plot->drawable,
                              gc,
                              x,
                              y-yy, 
                              x+axis.ticks_length/2+1,
                              y-yy); 
               
             if(axis.ticks_mask & GTK_PLOT_TICKS_LEFT)
                gdk_draw_line(plot->drawable,
                              gc,
                              x-axis.ticks_length/2-1,
                              y-yy, 
                              x,
                              y-yy); 
         }     
         break;  
  }

  gdk_gc_unref(gc);

}


static void
gtk_plot_draw_labels(GtkPlot *plot, 
		     GdkRectangle area,
                     GtkPlotAxis axis, 
	             gint x, gint y)
{
  GtkWidget *widget;
  GdkGC *gc;
  static GdkFont *font=NULL;
  GtkPlotText title;
  gchar label[100];
  gdouble x_tick, y_tick;
  gint xx, yy;
  gint text_height, text_width;
  gint xp, yp, width, height;
  gint ntick;

  widget = GTK_WIDGET(plot); 
  xp = roundint(plot->x * widget->allocation.width);
  yp = roundint(plot->y * widget->allocation.height);
  width = roundint(plot->width * widget->allocation.width);
  height = roundint(plot->height * widget->allocation.height);

  gc = gdk_gc_new (plot->drawable);

  gdk_gc_set_foreground (gc, &axis.label_attr.fg);

  if (!font)
  font = gtk_plot_font_get_gdkfont(axis.label_attr.font, axis.label_attr.height);
  text_height = font->ascent + font->descent;

  switch(axis.orientation){
     case GTK_ORIENTATION_VERTICAL:
       if (!(plot->ymajor.ticks)) break;
       y += height;
       for(ntick = 0; ntick < plot->ymajor.nticks; ntick++){
           yy = plot->ymajor.ticks[ntick];
           y_tick = plot->ymajor.value[ntick];
           parse_label(y_tick, axis.label_precision, axis.label_style, label);
           text_width = gdk_string_width(font, label);

           if(axis.label_mask & GTK_PLOT_LABEL_LEFT)
              gdk_draw_string   (plot->drawable,
                                 font, gc, 
                                 x-axis.labels_offset-text_width, 
                                 y-yy+font->descent,
                                 label);

           if(axis.label_mask & GTK_PLOT_LABEL_RIGHT)
              gdk_draw_string   (plot->drawable,
                                 font, gc, 
                                 x+axis.labels_offset, 
                                 y-yy+font->descent,
                                 label);
       }
        if(axis.title_visible && axis.title.text)
            {
              title = axis.title;
              gtk_plot_draw_text(plot, area, title); 
            }
       break;
     case GTK_ORIENTATION_HORIZONTAL:
       if (!(plot->xmajor.ticks)) break;
       for(ntick = 0; ntick < plot->xmajor.nticks; ntick++){
           xx = plot->xmajor.ticks[ntick];
           x_tick = plot->xmajor.value[ntick];
           parse_label(x_tick, axis.label_precision, axis.label_style, label);
           text_width = gdk_string_width(font, label);
           xx=xx-text_width/2;
           if(axis.label_mask & GTK_PLOT_LABEL_TOP)
              gdk_draw_string   (plot->drawable,
                                 font, gc, 
                                 x+xx, y-3*font->descent-axis.labels_offset,
                                 label);
            
           if(axis.label_mask & GTK_PLOT_LABEL_BOTTOM)
              gdk_draw_string   (plot->drawable, 
                                 font, gc, 
                                 x+xx, 
                                 y+text_height+font->descent+axis.labels_offset,
                                 label);
       }
       if(axis.title_visible && axis.title.text)
            {
              title = axis.title;
              gtk_plot_draw_text(plot, area, title); 
            }
       break;
   } 
  /*  gtk_signal_emit(GTK_OBJECT(plot),plot_signals[CHANGED]);*/
  //  gdk_font_unref(font);
  gdk_gc_unref(gc);
}

static void
gtk_plot_real_draw_dataset(GtkPlot *plot,  
                           GdkRectangle *drawing_area, 
		  	   GdkGC *gc,
                           GtkPlotData *dataset)
{
  GtkWidget *widget;
  GtkPlotData function;
  GdkRectangle area;
  gdouble x, y, dx;
  gint n;
  gdouble *fx;
  gdouble *fy;
  gint px, py;
  gboolean error;

  widget = GTK_WIDGET(plot);

  if(!GTK_WIDGET_DRAWABLE(widget)) return;
  if(!dataset->is_visible) return;

  if(drawing_area == NULL){
     area.x = widget->allocation.x;
     area.y = widget->allocation.y;
     area.width = widget->allocation.width;
     area.height = widget->allocation.height;
  } else {
     area = *drawing_area;
  }

  if(!dataset->is_function)
    {
       gtk_plot_connect_points (plot, area, gc, dataset);
       gtk_plot_draw_xy(plot, area, gc, dataset);
       gtk_plot_draw_errbars(plot, area, gc, dataset); 
       for(n=0; n<=dataset->num_points-1; n++)
         {
           x = dataset->x[n];
           y = dataset->y[n];
           if(dataset->dx) dx = dataset->dx[n];
           if(x >= plot->xmin && x <= plot->xmax &&
              y >= plot->ymin && y <= plot->ymax){ 
               gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
               gtk_plot_draw_point(plot,
	   	   	           area,
                                   gc,
                                   px, py, dx,
	     	                   dataset->symbol.symbol_type, 
                                   dataset->symbol.color,
                                   dataset->symbol.symbol_style,
                                   dataset->symbol.size,
                                   dataset->symbol.line_width);
           }
         } 
    } 
  else
    {
       function = *dataset;
       fx = NULL;
       fy = NULL;
       function.num_points = 0;
       for(x=plot->xmin; x<=plot->xmax+inverse_dx(plot, function.x_step); 
                         x+=inverse_dx(plot, function.x_step)) {
            function.num_points++;
            fx = (gdouble *)g_realloc(fx, function.num_points*sizeof(gdouble));
            fy = (gdouble *)g_realloc(fy, function.num_points*sizeof(gdouble)); 
            y = function.function (x, &error);

            if(error)
              {
                 function.x = fx;
                 function.y = fy;
		 function.num_points--;
                 if(function.num_points > 1)
                       gtk_plot_connect_points (plot, area, gc, &function);
                 function.num_points = 0;
              }
            else
              {
                if(function.num_points >= 2)
                  {
                     if(y > plot->ymax && 
                        fy[function.num_points-2] <= plot->ymin)
                       {
                          function.x = fx;
                          function.y = fy;
			  function.num_points--;
                          gtk_plot_connect_points(plot, area, gc, &function);
                          function.num_points = 1;
                       }
                      if(y < plot->ymin && 
                        fy[function.num_points-2] >= plot->ymax)
                       {
                          function.x = fx;
                          function.y = fy;
			  function.num_points--;
                          gtk_plot_connect_points(plot, area, gc, &function);
                          function.num_points = 1;
                       }
                   }
                fx[function.num_points-1] = x;
                fy[function.num_points-1] = y;
              }
         }
       if(function.num_points > 1 ) 
         {
            function.x = fx;
            function.y = fy;
            gtk_plot_connect_points (plot, area, gc, &function);
         }
       g_free(fx);
       g_free(fy);
    }
}

static void
gtk_plot_draw_point (GtkPlot *plot, 
		     GdkRectangle area,
                     GdkGC *gc,
                     gint x, gint y, gdouble dx, 
                     GtkPlotSymbolType symbol,
                     GdkColor color,
                     GtkPlotSymbolStyle symbol_style,
                     gint size, 
                     gint line_width)
{
  gint fill = FALSE;


  if(symbol_style == GTK_PLOT_SYMBOL_OPAQUE && symbol < GTK_PLOT_SYMBOL_PLUS)
     gtk_plot_draw_symbol (plot, area, gc, x, y, dx, symbol, 
                           plot->background,
                           TRUE,
                           size,
                           line_width);  

  if(symbol_style == GTK_PLOT_SYMBOL_FILLED) fill = TRUE;

  gtk_plot_draw_symbol (plot, area, gc, x, y, dx, symbol, 
                        color,
                        fill,
                        size,
                        line_width);  
}

static void
gtk_plot_draw_symbol(GtkPlot *plot, 
		     GdkRectangle area, 
                     GdkGC *gc,
                     gint x, gint y, gdouble dx, 
                     GtkPlotSymbolType symbol,
                     GdkColor color,
                     gint filled,
                     gint size, 
                     gint line_width)
{
    GdkGCValues values;
    gdouble x0, y0;
    gint px0, py0; 

    gdk_gc_get_values(gc, &values);
    if(values.function != GDK_XOR && values.function != GDK_INVERT)
            gdk_gc_set_foreground (gc, &color);

    gdk_gc_set_line_attributes (gc, line_width, 0, 0, 0);

    switch(symbol) {
       case GTK_PLOT_SYMBOL_NONE:
              break;
       case GTK_PLOT_SYMBOL_SQUARE:
              gdk_draw_rectangle (plot->drawable,
                                  gc,
                                  filled,
                                  x-size/2, y-size/2,
                                  size/2*2, size/2*2);
	      break;
       case GTK_PLOT_SYMBOL_CIRCLE:
              gdk_draw_arc (plot->drawable, gc,
                            filled, 
                            x-size/2, y-size/2, 
                            size, size, 0, 25000);
	      break;
       case GTK_PLOT_SYMBOL_UP_TRIANGLE:
              gtk_plot_draw_up_triangle (plot, gc, x, y, size, filled);                  
	      break;
       case GTK_PLOT_SYMBOL_DOWN_TRIANGLE:
              gtk_plot_draw_down_triangle (plot, gc, x, y, size, filled);                  
	      break;
       case GTK_PLOT_SYMBOL_DIAMOND:
              gtk_plot_draw_diamond (plot, gc, x, y, size, filled);                  
	      break;
       case GTK_PLOT_SYMBOL_PLUS:
              gtk_plot_draw_plus (plot, gc, x, y, size);                  
	      break;
       case GTK_PLOT_SYMBOL_CROSS:
              gtk_plot_draw_cross (plot, gc, x, y, size);                  
	      break;
       case GTK_PLOT_SYMBOL_STAR:
              gtk_plot_draw_star (plot, gc, x, y, size);                  
	      break;
       case GTK_PLOT_SYMBOL_BAR:
              x0 = x;
              y0 = 0.;
              gtk_plot_get_real_pixel(plot, x0, y0, &px0, &py0, area);
              gdk_draw_rectangle (plot->drawable,
                                  gc,
                                  filled,
                                  x, MIN(py0,y), 
                                  transform_dx(plot, dx)+1,
                                  abs(y-py0));
              break;
       case GTK_PLOT_SYMBOL_IMPULSE:
              x0 = x;
              y0 = 0.;
              gtk_plot_get_real_pixel(plot, x0, y0, &px0, &py0, area);
              gdk_draw_line(plot->drawable, gc, 
                            x, MIN(py0,y), 
                            x,
                            MAX(py0,y));
              break;
    }
}

static void
gtk_plot_draw_xy (GtkPlot *plot, 
		  GdkRectangle area, 
                  GdkGC *gc,
                  GtkPlotData *dataset)
{ 
  gint n;
  gdouble x, y;
  gint px, py;
  gint x0, y0;

  if(!dataset->x || !dataset->y) return;

  for(n=0; n<=dataset->num_points-1; n++)
    {
      x = dataset->x[n];
      y = dataset->y[n];

      if(x >= plot->xmin && x <= plot->xmax &&
         y >= plot->ymin && y <= plot->ymax) {

            gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
            gtk_plot_get_real_pixel(plot, x, 0., &x0, &y0, area);

            gtk_plot_draw_line(plot, gc , dataset->x_line,
                               px,
                               py,
                               px, 
                               y0);

            gtk_plot_get_real_pixel(plot, 0., y, &x0, &y0, area);
            gtk_plot_draw_line(plot, gc, dataset->y_line,
                               px, 
                               py, 
                               x0, 
                               py);
      }
    }
}

static void
gtk_plot_draw_errbars(GtkPlot *plot, 
		      GdkRectangle area, 
                      GdkGC *gc,
                      GtkPlotData *dataset) 
{
  GdkGCValues values;
  GdkPoint errbar[6];
  gdouble x, y;
  gint px, py;
  gint ex, ey;
  gint n;

  if(!dataset->x || !dataset->y) return;
  if(!dataset->dx || !dataset->dy) return;
  
  gdk_gc_get_values(gc, &values);
  if(values.function != GDK_XOR && values.function != GDK_INVERT)
           gdk_gc_set_foreground (gc, &dataset->symbol.color);

  gdk_gc_set_line_attributes (gc, dataset->symbol.line_width/2, 0, 0, 0);

  for(n=0; n<=dataset->num_points-1; n++)
    {
      x = dataset->x[n];
      y = dataset->y[n];

      if(x >= plot->xmin && x <= plot->xmax &&
         y >= plot->ymin && y <= plot->ymax) {

                gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);

                ex = transform_dx(plot, dataset->dx[n]);
                ey = transform_dy(plot, dataset->dy[n]);

                if(dataset->show_xerrbars)
                   {
                      errbar[0].x = px-ex; 
                      errbar[0].y = py-dataset->xerrbar_length/2; 
                      errbar[1].x = px-ex; 
                      errbar[1].y = py+dataset->xerrbar_length/2; 
                      errbar[2].x = px-ex; 
                      errbar[2].y = py; 
                      errbar[3].x = px+ex; 
                      errbar[3].y = py; 
                      errbar[4].x = px+ex; 
                      errbar[4].y = py-dataset->xerrbar_length/2; 
                      errbar[5].x = px+ex; 
                      errbar[5].y = py+dataset->xerrbar_length/2; 
                      gdk_draw_lines(plot->drawable, gc, errbar, 6);
                   }

                if(dataset->show_yerrbars)
                  {
                      errbar[0].x = px-dataset->yerrbar_length/2; 
                      errbar[0].y = py-ey; 
                      errbar[1].x = px+dataset->yerrbar_length/2; 
                      errbar[1].y = py-ey; 
                      errbar[2].x = px; 
                      errbar[2].y = py-ey; 
                      errbar[3].x = px; 
                      errbar[3].y = py+ey; 
                      errbar[4].x = px-dataset->yerrbar_length/2; 
                      errbar[4].y = py+ey; 
                      errbar[5].x = px+dataset->yerrbar_length/2; 
                      errbar[5].y = py+ey; 
                      gdk_draw_lines(plot->drawable, gc, errbar, 6);
                  }
      }
    }
}
 
static void
gtk_plot_draw_down_triangle(GtkPlot *plot, GdkGC *gc, 
                            gint x, gint y, gint size, gint filled)
{
  GdkPoint point[3];
  gdouble pi = acos(-1.);

  point[0].x = x - size*cos(pi/6.);
  point[0].y = y - size*sin(pi/6.);

  point[1].x = x + size*cos(pi/6.);
  point[1].y = y - size*sin(pi/6.);

  point[2].x = x;
  point[2].y = y + size;

  gdk_draw_polygon (plot->drawable,
                    gc,
                    filled,
                    point,
                    3); 
  
}

static void
gtk_plot_draw_up_triangle(GtkPlot *plot, GdkGC *gc, 
                          gint x, gint y, gint size, gint filled)
{
  GdkPoint point[3];
  gdouble pi = acos(-1.);

  point[0].x = x - size*cos(pi/6.);
  point[0].y = y + size*sin(pi/6.);

  point[1].x = x + size*cos(pi/6.);
  point[1].y = y + size*sin(pi/6.);

  point[2].x = x;
  point[2].y = y - size;

  gdk_draw_polygon (plot->drawable,
                    gc,
                    filled,
                    point,
                    3); 
  
}

static void
gtk_plot_draw_diamond(GtkPlot *plot, GdkGC *gc, 
                      gint x, gint y, gint size, gint filled)
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

  gdk_draw_polygon (plot->drawable,
                    gc,
                    filled,
                    point,
                    4); 
  
}

static void
gtk_plot_draw_plus(GtkPlot *plot, GdkGC *gc, 
                   gint x, gint y, gint size)
{
  gdk_draw_line (plot->drawable,
                 gc,
                 x-size/2, y, x+size/2, y);
  
  gdk_draw_line (plot->drawable,
                 gc,
                 x, y-size/2, x, y+size/2);
}

static void
gtk_plot_draw_cross(GtkPlot *plot, GdkGC *gc, 
                    gint x, gint y, gint size)
{
  gdk_draw_line (plot->drawable,
                 gc,
                 x-size/2, y-size/2, x+size/2, y+size/2);
  
  gdk_draw_line (plot->drawable,
                 gc,
                 x-size/2, y+size/2, x+size/2, y-size/2);
}

static void
gtk_plot_draw_star(GtkPlot *plot, GdkGC *gc, 
                       gint x, gint y, gint size)
{
  gdouble s2 = size*sqrt(2.)/4.;

  gdk_draw_line (plot->drawable,
                 gc,
                 x-size/2, y, x+size/2, y);
  
  gdk_draw_line (plot->drawable,
                 gc,
                 x, y-size/2, x, y+size/2);

  gdk_draw_line (plot->drawable,
                 gc,
                 x-s2, y-s2, x+s2, y+s2);
  
  gdk_draw_line (plot->drawable,
                 gc,
                 x-s2, y+s2, x+s2, y-s2);
}

static void
gtk_plot_connect_points(GtkPlot *plot, 
                        GdkRectangle area, 
                        GdkGC *gc,
                        GtkPlotData *dataset)
{
  GtkWidget *widget;
  GdkRectangle clip_area;
  GdkPoint points[2*dataset->num_points];
  GtkPlotData spline;
  GdkPoint *spline_points;
  gdouble spline_coef[dataset->num_points];
  gdouble x, y;
  gint n;
  gint px, py;
  gint x1, y1;
  gint num_points = dataset->num_points;

  widget = GTK_WIDGET(plot);
  clip_area.x = area.x + roundint(plot->x * widget->allocation.width);
  clip_area.y = area.y + roundint(plot->y * widget->allocation.height);
  clip_area.width = roundint(plot->width * widget->allocation.width);
  clip_area.height = roundint(plot->height * widget->allocation.height);


  if(dataset->line.line_style == GTK_PLOT_LINE_NONE) return;

  gdk_gc_set_clip_rectangle(gc, &clip_area);

  gtk_plot_set_line_attributes(plot, dataset->line, gc);

  switch(dataset->line_connector){
   case GTK_PLOT_CONNECT_STRAIGHT:
      if(dataset->num_points == 1) break;
      for(n=0; n<dataset->num_points; n++)
        {
          x = dataset->x[n];
          y = dataset->y[n];
          gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
          points[n].x = px;
          points[n].y = py;
        }
      break;
   case GTK_PLOT_CONNECT_HV_STEP:
       if(dataset->num_points == 1) break;
       num_points=0;
       for(n=0; n < dataset->num_points; n++)
        {
          x = dataset->x[n];
          y = dataset->y[n];
          gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
          points[num_points].x = px;
          points[num_points].y = py;
          num_points++;
          if(n < dataset->num_points-1)
            {
              gtk_plot_get_real_pixel(plot, 
                                 dataset->x[n+1], 
                                 dataset->y[n+1], 
                                 &px, &py, area);
              points[num_points].x = px;
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
          x = dataset->x[n];
          y = dataset->y[n];
          gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
          points[num_points].x = px;
          points[num_points].y = py;
          num_points++;
          if(n < dataset->num_points-1)
            {
              gtk_plot_get_real_pixel(plot, 
                                 dataset->x[n+1], 
                                 dataset->y[n+1], 
                                 &px, &py, area);
              points[num_points].x = points[num_points-1].x;
              points[num_points].y = py;
              num_points++;
            }
        }
       break;
     case GTK_PLOT_CONNECT_MIDDLE_STEP:
       if(dataset->num_points == 1) break;
       num_points=1;
       for(n=1; n < dataset->num_points; n++)
        {
          x = dataset->x[n];
          y = dataset->y[n];
          gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
          x = dataset->x[n-1];
          y = dataset->y[n-1];
          gtk_plot_get_real_pixel(plot, x, y, &x1, &y1, area);
          points[num_points].x = (px+x1)/2;
          points[num_points].y = y1;
          num_points++;
          points[num_points].x = points[num_points-1].x;
          points[num_points].y = py;
          num_points++;
        }
        x = dataset->x[0];
        y = dataset->y[0];
        gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
        points[0].x = px;
        points[0].y = py;
        x = dataset->x[dataset->num_points-1];
        y = dataset->y[dataset->num_points-1];
        gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
        points[num_points].x = px;
        points[num_points].y = py;
        num_points++;
        break;
     case GTK_PLOT_CONNECT_SPLINE:
        spline = *dataset;
        spline_points = NULL;
        spline.num_points = 0;
        spline_solve(dataset->num_points, dataset->x, dataset->y, spline_coef);
        for(x=dataset->x[0]; x<=dataset->x[dataset->num_points-1]; 
            x+=inverse_dx(plot, spline.x_step)) {
              spline.num_points++;
              spline_points = (GdkPoint *)g_realloc(spline_points,
                                                    spline.num_points*
                                                    sizeof(GdkPoint)); 
              y = spline_eval(dataset->num_points, dataset->x, dataset->y, 
                              spline_coef, x);
              gtk_plot_get_real_pixel(plot, x, y, &px, &py, area);
              spline_points[spline.num_points-1].x = px;
              spline_points[spline.num_points-1].y = py;
         }
        gdk_draw_lines(plot->drawable, gc, spline_points, spline.num_points);
        g_free(spline_points);
        gdk_gc_set_clip_rectangle(gc, NULL);
        return;
     case GTK_PLOT_CONNECT_NONE:
     default:
        gdk_gc_set_clip_rectangle(gc, NULL);
        return;
    }

 
  gdk_draw_lines(plot->drawable, gc, points, num_points);
  gdk_gc_set_clip_rectangle(gc, NULL);
   
}

static void
gtk_plot_draw_line(GtkPlot *plot, 
                   GdkGC *gc,
                   GtkPlotLine line,
                   gint x1, gint y1, gint x2, gint y2)
{
  gboolean new_gc = FALSE;

  if(line.line_style == GTK_PLOT_LINE_NONE) return;

  if(gc == NULL){
     new_gc = TRUE; 
     gc = gdk_gc_new(plot->drawable);
  }

  gtk_plot_set_line_attributes(plot, line, gc);

  gdk_draw_line(plot->drawable, gc, x1, y1, x2, y2);

  if(new_gc) gdk_gc_unref(gc);
}

static void
gtk_plot_set_line_attributes(GtkPlot *plot, 
                             GtkPlotLine line,
                             GdkGC *gc)
{
  GdkGCValues values;

  gdk_gc_get_values(gc, &values);
  if(values.function != GDK_XOR && values.function != GDK_INVERT)
           gdk_gc_set_foreground (gc, &line.color);

  switch(line.line_style){
   case GTK_PLOT_LINE_SOLID:
        gdk_gc_set_line_attributes(gc, line.line_width, 0, 0, 0);
        break;
   case GTK_PLOT_LINE_DOTTED:
        gdk_gc_set_dashes(gc, 0,"\2\3", 2);
        gdk_gc_set_line_attributes(gc, line.line_width, 
                                   GDK_LINE_ON_OFF_DASH, 0, 0);
        break;
   case GTK_PLOT_LINE_DASHED:
        gdk_gc_set_dashes(gc, 0,"\6\4", 2);
        gdk_gc_set_line_attributes(gc, line.line_width, 
                                   GDK_LINE_ON_OFF_DASH, 0, 0);
        break;
   case GTK_PLOT_LINE_DOT_DASH:
        gdk_gc_set_dashes(gc, 0,"\6\4\2\4", 4);
        gdk_gc_set_line_attributes(gc, line.line_width, 
                                   GDK_LINE_ON_OFF_DASH, 0, 0);
        break;
   case GTK_PLOT_LINE_DOT_DOT_DASH:
        gdk_gc_set_dashes(gc, 0,"\6\4\2\4\2\4", 6);
        gdk_gc_set_line_attributes(gc, line.line_width, 
                                   GDK_LINE_ON_OFF_DASH, 0, 0);
        break;
   case GTK_PLOT_LINE_DOT_DASH_DASH:
        gdk_gc_set_dashes(gc, 0,"\6\4\6\4\2\4", 6);
        gdk_gc_set_line_attributes(gc, line.line_width, 
                                   GDK_LINE_ON_OFF_DASH, 0, 0);
        break;
   case GTK_PLOT_LINE_NONE:
   default:
        break;
  }  
}

static void
gtk_plot_draw_legends (GtkPlot *plot, GdkRectangle area)
{
  GdkGC *gc;
  GList *datasets; 
  GtkPlotData *dataset;
  GdkRectangle legend_area;
  GtkAllocation allocation;
  static GdkFont *font=NULL;
  gint x0, y0;
  gint x, y;
  gint width = 0;
  gint height;

  if(!plot->show_legends) return;

  gc = gdk_gc_new(plot->drawable);
  if (!font)
  font = gtk_plot_font_get_gdkfont(plot->legends_attr.font,
                                   plot->legends_attr.height);

/* first draw the white rectangle for the background */
  allocation = GTK_WIDGET(plot)->allocation;
  x0 = area.x + plot->x * allocation.width + 
       plot->legends_x * plot->width * allocation.width;
  y0 = area.y + plot->y * allocation.height +
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
     gdk_gc_set_foreground(gc, &plot->legends_attr.bg);
     gdk_draw_rectangle(plot->drawable,
                        gc,
                        TRUE,
                        legend_area.x, legend_area.y,
                        legend_area.width, legend_area.height);
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

         gdk_gc_set_foreground(gc, &dataset->line.color);
         if(dataset->line_connector != GTK_PLOT_CONNECT_NONE ||
            dataset->symbol.symbol_type == GTK_PLOT_SYMBOL_IMPULSE)
              gtk_plot_draw_line(plot, NULL, dataset->line, 
                                 x, 
                                 y - font->ascent / 2, 
                                 x + plot->legends_line_width, 
                                 y - font->ascent / 2);

         gdk_gc_set_foreground(gc, &plot->legends_attr.fg);

         if(dataset->symbol.symbol_type != GTK_PLOT_SYMBOL_BAR &&
            dataset->symbol.symbol_type != GTK_PLOT_SYMBOL_IMPULSE)
              gtk_plot_draw_point (plot, 
				   area,
				   gc,
                                   area.x + x + plot->legends_line_width / 2, 
                                   area.y + y - font->ascent / 2,
                                   0., 
                                   dataset->symbol.symbol_type,
                                   dataset->symbol.color,
                                   dataset->symbol.symbol_style,
                                   dataset->symbol.size, 
                                   dataset->symbol.line_width);

         if(dataset->legend){
                gdk_gc_set_foreground(gc, &plot->legends_attr.fg);
                gdk_draw_string (plot->drawable,
                                 font,
                                 gc,
                                 x + plot->legends_line_width + 4, y,
                                 dataset->legend);
         }
       }
     datasets=datasets->next;
   }
 
   gdk_gc_set_line_attributes(gc, plot->legends_border_width, 0, 0, 0);
   gdk_gc_set_foreground(gc, &plot->legends_attr.fg);

   if(plot->show_legends_border)
      {
        gdk_draw_rectangle(plot->drawable,
                           gc,
                           FALSE,
                           legend_area.x, legend_area.y,
                           legend_area.width, legend_area.height);
      }

   gdk_gc_set_line_attributes(gc, 0, 0, 0, 0);
   if(plot->show_legends_border && plot->show_legends_shadow)
      {
        gdk_draw_rectangle(plot->drawable,
			   gc,
                           TRUE,
                           legend_area.x + plot->legends_shadow_width, 
                           legend_area.y + legend_area.height,
                           legend_area.width, plot->legends_shadow_width);
        gdk_draw_rectangle(plot->drawable,
                           gc,
                           TRUE,
                           legend_area.x + legend_area.width, 
                           legend_area.y + plot->legends_shadow_width,
                           plot->legends_shadow_width , legend_area.height);
      }
   //   gdk_font_unref(font);
  gdk_gc_unref(gc);
}

static void gtk_plot_set_wcstransform(GtkPlot *plot)
{
  gdouble dwidth;

  plot->ywcs2pix = (GTK_WIDGET(plot)->allocation.height*plot->height)/
    (plot->ymax-plot->ymin);

  switch (plot->xscale)
    {
    case GTK_PLOT_SCALE_LOG10:
      if(plot->xmin <= 0. || plot->xmax <= 0.){
	dwidth=0.0;
      }else{
	dwidth = log10(plot->xmax)-log10(plot->xmin);
      }
      break;
    case GTK_PLOT_SCALE_LINEAR:
    default:
      dwidth = plot->xmax - plot->xmin;
    }
     
  if (dwidth==0.0) plot->xwcs2pix = 0.0;
  else 
    plot->xwcs2pix = (GTK_WIDGET(plot)->allocation.width*plot->width)/dwidth;
}

static gint
transform_y(GtkPlot *plot, gdouble y)
{
    return (roundint(plot->ywcs2pix*(y-plot->ymin)));
    /*

    gdouble height;

    height = (gdouble)GTK_WIDGET(plot)->allocation.height * plot->height;
    return(roundint(height*(y-plot->ymin)/(plot->ymax-plot->ymin)));
    */
}

static gint
transform_x(GtkPlot *plot, gdouble x)
{
  //    gdouble width;
    gdouble dx;
    //    gdouble dwidth;

    switch(plot->xscale){
       case GTK_PLOT_SCALE_LOG10:
        if(x <= 0. || plot->xmin <= 0. || plot->xmax <= 0.){
          return 0;
        }else{
          dx = log10(x) - log10(plot->xmin);
	  //	  dwidth = log10(plot->xmax)-log10(plot->xmin);
        }
	break;
       case GTK_PLOT_SCALE_LINEAR:
       default:
         dx = x - plot->xmin;
	 //	 dwidth = plot->xmax - plot->xmin;
    }

    /*
    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(roundint(width*dx/dwidth));
    */
    return (roundint(plot->xwcs2pix*dx));
}

static gint
transform_dy(GtkPlot *plot, gdouble dy)
{
  return (roundint(plot->ywcs2pix*dy));
  /*
    gdouble height;
 
    height = (gdouble)GTK_WIDGET(plot)->allocation.height * plot->height;
    return(roundint(height*dy/(plot->ymax-plot->ymin)));
  */
}

static gint
transform_dx(GtkPlot *plot, gdouble dx)
{
  return (roundint(plot->xwcs2pix*dx));
  /*
    gdouble width;

    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(roundint(width*dx/(plot->xmax-plot->xmin)));
  */
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
  return ((gdouble)dy/plot->ywcs2pix);
  /*
    gdouble height;
 
    height = (gdouble)GTK_WIDGET(plot)->allocation.height * plot->height;
    return(((gdouble)dy)*(plot->ymax-plot->ymin)/height);
  */
}

static gdouble
inverse_dx(GtkPlot *plot, gint dx)
{
  return ((gdouble)dx/plot->xwcs2pix);
  /*
    gdouble width;
 
    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(((gdouble)dx)*(plot->xmax-plot->xmin)/width);
  */
}
/*
static gint
roundint (gdouble x)
{
 gint sign = 1;

* if(x <= 0.) sign = -1; 
*
 return (x+sign*.50999999471);
}
*/
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
    case GTK_PLOT_LABEL_FLOAT:
    default:
      sprintf (label, "%*.*f", intspace, precision, val);
  }

}

static void
gtk_plot_draw_text(GtkPlot *plot, 
		   GdkRectangle area, 
                   GtkPlotText text) 
{
  GdkPixmap *text_pixmap;
  GdkImage *image;
  GdkColor color;
  GdkGC *gc;
  GdkColormap *colormap;
  GdkColorContext *cc;
  GdkVisual *visual;
  gint x, y;
  gint xp, yp;
  gint width, height;

  if(plot->drawable == NULL) return;

  x = text.x * GTK_WIDGET(plot)->allocation.width;
  y = text.y * GTK_WIDGET(plot)->allocation.height;

  text_pixmap = rotate_text(plot, text, &width, &height);

  switch(text.justification){
    case GTK_JUSTIFY_LEFT:
      break;
    case GTK_JUSTIFY_RIGHT:
      if(text.angle == 0 || text.angle == 180)
            x -= width;
      else
            y -= height;
      break;
    case GTK_JUSTIFY_CENTER:
    default:
      if(text.angle == 0 || text.angle == 180)
            x -= width/2;
      else
            y -= height/2;
  }

  colormap = gtk_widget_get_colormap (GTK_WIDGET(plot));
  visual = gtk_widget_get_visual (GTK_WIDGET(plot));
  cc = gdk_color_context_new(visual, colormap);
  image = gdk_image_get(text_pixmap, 0, 0, width, height);
  gc = gdk_gc_new(plot->drawable);

  for(yp = 0; yp < height; yp++)
    for(xp = 0; xp < width; xp++)
       {
          color.pixel = gdk_image_get_pixel(image, xp, yp);
          gdk_color_context_query_color(cc, &color);

          gdk_gc_set_foreground(gc, &color);
          if(!text.transparent || gdk_color_equal(&color, &text.fg))
             gdk_draw_point(plot->drawable, gc, 
                            area.x + x + xp, area.y + y + yp);

       }

  gdk_color_context_free(cc);
  gdk_image_destroy(image);
  gdk_pixmap_unref(text_pixmap);
  gdk_gc_unref(gc);
}

static GdkPixmap *
rotate_text(GtkPlot *plot, 
	    GtkPlotText text, 
            gint *width, gint *height)
{
  GdkWindow *window;
  GdkPixmap *old_pixmap; 
  GdkPixmap *new_pixmap; 
  GdkImage *image;
  GdkGC *gc;
  GdkColormap *colormap;
  GdkColorContext *cc;
  GdkVisual *visual;
  GdkColor color;
  static GdkFont *font=NULL;
  gint x, y;
  gint old_width, old_height;

  window = GTK_WIDGET(plot)->window;
  colormap = gtk_widget_get_colormap (GTK_WIDGET(plot));
  visual = gtk_widget_get_visual (GTK_WIDGET(plot));
  cc = gdk_color_context_new(visual, colormap);
  gc = gdk_gc_new (plot->drawable);
  if (!font)
  font = gtk_plot_font_get_gdkfont(text.font, text.height);

  old_width = gdk_string_width (font, text.text);
  old_height = font->ascent + font->descent;

  gtk_plot_text_get_size(text, width, height);

  old_pixmap = gdk_pixmap_new(window, old_width, old_height, -1); 
  new_pixmap = gdk_pixmap_new(window, *width, *height, -1); 

  gdk_gc_set_foreground(gc, &text.bg); 
  gdk_draw_rectangle(old_pixmap, gc, TRUE, 
                     0, 0, -1, -1); 
  gdk_draw_rectangle(new_pixmap, gc, TRUE, 
                     0, 0, -1, -1); 

  gdk_gc_set_foreground(gc, &text.fg);
  gdk_draw_string (old_pixmap, font, gc, 0, font->ascent, text.text); 

  image = gdk_image_get(old_pixmap, 0, 0, old_width, old_height);  

  for(y=0; y<old_height; y++)
    for(x=0; x<old_width; x++)
       {
          color.pixel = gdk_image_get_pixel(image, x, y);
          gdk_color_context_query_color(cc, &color);
          gdk_gc_set_foreground(gc, &color);
          if(text.angle == 0)
             gdk_draw_point(new_pixmap, gc, x, y);
          if(text.angle == 90)
             gdk_draw_point(new_pixmap, gc, y, old_width - x);
          if(text.angle == 180)
             gdk_draw_point(new_pixmap, gc, old_width - x, old_height - y);
          if(text.angle == 270)
             gdk_draw_point(new_pixmap, gc, old_height - y, x);
       }

  gdk_image_destroy(image);
  gdk_color_context_free(cc);
  gdk_pixmap_unref(old_pixmap);
  //  gdk_font_unref(font);
  gdk_gc_unref(gc);
  return (new_pixmap);

}

void
gtk_plot_text_get_size(GtkPlotText text, gint *width, gint *height)
{
  static GdkFont *font=NULL;
  gint old_width, old_height;

  if (!font)
  font = gtk_plot_font_get_gdkfont(text.font, text.height);
  old_width = gdk_string_width (font, text.text);
  old_height = font->ascent + font->descent;

  *width = old_width;
  *height = old_height;
  if(text.angle == 90 || text.angle == 270)
    {
      *width = old_height;
      *height = old_width;
    }
  //  gdk_font_unref(font);

}

/******************************************
 *	gtk_plot_get_position
 *	gtk_plot_get_size
 *      gtk_plot_get_internal_allocation 
 *	gtk_plot_set_background
 *	gtk_plot_move
 *	gtk_plot_resize
 *	gtk_plot_move_resize
 *	gtk_plot_get_pixel
 *	gtk_plot_get_point
 *	gtk_plot_get_real_pixel
 *      gtk_plot_set_xscale
 *      gtk_plot_set_yscale
 *      gtk_plot_draw_text
 ******************************************/
void
gtk_plot_get_position (GtkPlot *plot, gdouble *x, gdouble *y)
{
  *x = plot->x;
  *y = plot->y;
}

void
gtk_plot_get_size (GtkPlot *plot, gdouble *width, gdouble *height)
{
  *width = plot->width;
  *height = plot->height;
}

GtkAllocation 
gtk_plot_get_internal_allocation (GtkPlot *plot)
{
  GtkAllocation allocation;
  GtkWidget *widget;

  widget = GTK_WIDGET(plot);

  allocation.x = widget->allocation.x + plot->x * widget->allocation.width;
  allocation.y = widget->allocation.y + plot->y * widget->allocation.height;
  allocation.width = plot->width * widget->allocation.width;
  allocation.height = plot->height * widget->allocation.height;

  return(allocation);
}

void
gtk_plot_set_background(GtkPlot *plot, GdkColor color)
{
  plot->background = color;
  gtk_widget_queue_draw(GTK_WIDGET(plot));
}

void
gtk_plot_move (GtkPlot *plot, gdouble x, gdouble y)
{
  gboolean veto = TRUE;
  GList *Text;
  GtkPlotText *data;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[MOVED],
                   &x, &y, &veto);

  if(!veto) return;

  plot->left.title.x += (x - plot->x);  
  plot->left.title.y += (y - plot->y);
  plot->right.title.x += (x - plot->x);  
  plot->right.title.y += (y - plot->y);
  plot->top.title.x += (x - plot->x);  
  plot->top.title.y += (y - plot->y);
  plot->bottom.title.x += (x - plot->x);  
  plot->bottom.title.y += (y - plot->y);

  /* Carry the inset text around when the plot is moved */
  Text = plot->text;
  while(Text)
    {
      data = (GtkPlotText *)(Text->data);
      data->x += (x - plot->x);
      data->y += (y - plot->y);
      Text = Text->next;
    }
  plot->x = x;
  plot->y = y;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_resize (GtkPlot *plot, gdouble width, gdouble height)
{
  gboolean veto = TRUE;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[RESIZED],
                   &width, &height, &veto);

  if(!veto) return;

  plot->left.title.y += (height - plot->height) / 2.;
  plot->right.title.x += (width - plot->width);  
  plot->right.title.y += (height - plot->height) / 2.;
  plot->top.title.x += (width - plot->width) / 2.;  
  plot->bottom.title.x += (width - plot->width) / 2.;  
  plot->bottom.title.y += (height - plot->height);

  plot->width = width;
  plot->height = height;

  gtk_plot_set_wcstransform(plot);
  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_move_resize (GtkPlot *plot, 
	              gdouble x, gdouble y,
                      gdouble width, gdouble height)
{
  gtk_plot_move(plot, x, y);
  gtk_plot_resize(plot, width, height);

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_get_pixel(GtkPlot *plot, gdouble xx, gdouble yy, gint *x, gint *y)
{
    GdkRectangle area;

    area.x = GTK_WIDGET(plot)->allocation.x;
    area.y = GTK_WIDGET(plot)->allocation.y;
    area.width = GTK_WIDGET(plot)->allocation.width;
    area.height = GTK_WIDGET(plot)->allocation.height;
    gtk_plot_get_real_pixel (plot, xx, yy, x, y, area);
}

void 
gtk_plot_get_point(GtkPlot *plot, gint x, gint y, gdouble *xx, gdouble *yy)
{
    GdkRectangle area;

    area.x = GTK_WIDGET(plot)->allocation.x;
    area.y = GTK_WIDGET(plot)->allocation.y;
    area.width = GTK_WIDGET(plot)->allocation.width;
    area.height = GTK_WIDGET(plot)->allocation.height;
    gtk_plot_pixel_get_real_point (plot, x, y, xx, yy, area);

}


static void
gtk_plot_get_real_pixel(GtkPlot *plot, gdouble xx, gdouble yy , 
			gint *x, gint *y, GdkRectangle area)
{
    GtkWidget *widget;
    gint xp, yp, width, height;

    widget = GTK_WIDGET(plot); 
    xp = roundint(plot->x * widget->allocation.width);
    yp = roundint(plot->y * widget->allocation.height);
    width = roundint(plot->width * widget->allocation.width);
    height = roundint(plot->height * widget->allocation.height);

    *y = transform_y(plot, yy);
    *x = transform_x(plot, xx);

    *x = *x + area.x + xp;
    *y = area.y + yp + height - *y;
}

static void 
gtk_plot_pixel_get_real_point(GtkPlot *plot, 
                              gint x, gint y, gdouble *px, gdouble *py,
			      GdkRectangle area)
{
    GtkWidget *widget;
    gint xx, yy;
    gint xp, yp, width, height;

    widget = GTK_WIDGET(plot); 
    xp = roundint(plot->x * widget->allocation.width);
    yp = roundint(plot->y * widget->allocation.height);
    width = roundint(plot->width * widget->allocation.width);
    height = roundint(plot->height * widget->allocation.height);

    xx = x - area.x - xp;
    yy = area.y + height + yp - y;

    *px = inverse_x(plot, xx);
    *py = inverse_y(plot, yy);
}


void
gtk_plot_set_range (GtkPlot *plot, 
                    gdouble xmin, gdouble xmax, 
                    gdouble ymin, gdouble ymax)
{
  plot->xmin = xmin;
  plot->xmax = xmax;
  plot->ymin = ymin;
  plot->ymax = ymax;

  gtk_plot_set_wcstransform(plot);

  plot->bottom.min = xmin;
  plot->bottom.max = xmax;
  plot->top.min = xmin;
  plot->top.max = xmax;
  plot->left.min = ymin;
  plot->left.max = ymax;
  plot->right.min = ymin;
  plot->right.max = ymax;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_set_xscale (GtkPlot *plot, GtkPlotScale scale_type)
{
  plot->xscale = scale_type;
  plot->bottom.scale_type = scale_type;
  plot->top.scale_type = scale_type;

  gtk_plot_set_wcstransform(plot);

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_set_yscale (GtkPlot *plot, GtkPlotScale scale_type)
{
  plot->yscale = scale_type;
  plot->left.scale_type = scale_type;
  plot->right.scale_type = scale_type;

  gtk_plot_set_wcstransform(plot);

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_put_text (GtkPlot *plot, gdouble x, gdouble y, gint angle,
                   gchar *font, gint height,
                   GdkColor *fg, GdkColor *bg, 
		   gint justification,
	           gchar *text)
{
  GtkWidget *widget;
  GtkPlotText *text_attr;
  GdkRectangle area;

  widget = GTK_WIDGET(plot);

  text_attr = g_new(GtkPlotText, 1);

  area.x = widget->allocation.x;
  area.y = widget->allocation.y;

  text_attr->x = x;
  text_attr->y = y;
  text_attr->angle = angle;
  text_attr->justification = justification;
  if(!font) {
    text_attr->font = g_strdup(DEFAULT_FONT);
    text_attr->height = DEFAULT_FONT_HEIGHT;
  } else {
    text_attr->font = g_strdup(font);
    text_attr->height = height;
  }
  text_attr->text = NULL;
  if(text) text_attr->text = g_strdup(text);

  if(fg != NULL)
    text_attr->fg = *fg;

  text_attr->transparent = TRUE;
  if(bg != NULL){
    text_attr->bg = *bg;
    text_attr->transparent = FALSE;
  }

  plot->text = g_list_append(plot->text, text_attr);
  gtk_plot_draw_text(plot, area, *text_attr);

}

gint
gtk_plot_remove_text(GtkPlot *plot, char *text)
{
  GList *TextList;
  GtkPlotText *data;
  
  TextList = plot->text;

  while(TextList)
   {
     data = (GtkPlotText *)(TextList->data);
     
     if(!strcmp(data->text,text))
       {
	 g_list_remove_link(plot->text, TextList);
	 if (TextList->data==NULL) 
	   {
	     g_list_free_1(TextList);
	     TextList=NULL;
	   }
	 return TRUE;
       }
     TextList = TextList->next;
   }

   return FALSE;
}

/******************************************
 *      gtk_plot_get_axis
 *	gtk_plot_axis_set_title
 *	gtk_plot_axis_show_title
 *	gtk_plot_axis_hide_title
 *	gtk_plot_axis_move_title
 *	gtk_plot_axis_justify_title
 *	gtk_plot_axis_show_labels
 *	gtk_plot_axis_set_attributes
 *	gtk_plot_axis_set_ticks
 *	gtk_plot_axis_set_ticks_length
 *	gtk_plot_axis_show_ticks
 *	gtk_plot_axis_set_ticks_limits
 *	gtk_plot_axis_unset_ticks_limits
 *	gtk_plot_axis_labels_set_attributes
 *	gtk_plot_axis_labels_set_numbers
 ******************************************/

GtkPlotAxis *
gtk_plot_get_axis (GtkPlot *plot, gint axis)
{
  GtkPlotAxis *aux;

  switch(axis){
    case GTK_PLOT_AXIS_LEFT:
         aux = &plot->left;
         break;
    case GTK_PLOT_AXIS_RIGHT:
         aux = &plot->right;
         break;
    case GTK_PLOT_AXIS_TOP:
         aux = &plot->top;
         break;
    case GTK_PLOT_AXIS_BOTTOM:
         aux = &plot->bottom;
         break;
  }
  return aux; 
}
void
gtk_plot_axis_set_title (GtkPlot *plot, gint axis, gchar *title)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);

  if(aux->title.text)
     g_free(aux->title.text);

  aux->title.text = g_strdup(title);

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_show_title (GtkPlot *plot, gint axis)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->title_visible = TRUE;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_hide_title (GtkPlot *plot, gint axis)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->title_visible = FALSE;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_move_title (GtkPlot *plot, gint axis, gint angle, gint x, gint y)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->title.angle += angle;
  aux->title.x = x;
  aux->title.y = y;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_justify_title (GtkPlot *plot, gint axis, gint justification)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->title.justification = justification;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void 
gtk_plot_axis_set_attributes (GtkPlot *plot, gint axis, 
			      gint width, GdkColor color)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->line.line_width = width;
  aux->line.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}


void
gtk_plot_axis_set_ticks (GtkPlot *plot,
                         GtkOrientation orientation,
		         gdouble major_ticks,
		         gdouble minor_ticks)
{
  if(orientation == GTK_ORIENTATION_HORIZONTAL){
  	plot->xmajor.step = major_ticks;
  	plot->xminor.step = minor_ticks;
  }else{
  	plot->ymajor.step = major_ticks;
  	plot->yminor.step = minor_ticks;
  }

  gtk_plot_calc_ticks(plot, orientation);

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_set_ticks_length (GtkPlot *plot, gint axis, gint length)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->ticks_length = length;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_set_ticks_width (GtkPlot *plot, gint axis, gint width)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->ticks_width = width;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_show_ticks (GtkPlot *plot,	
                          gint axis,
			  gint ticks_mask)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->ticks_mask = ticks_mask;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_set_ticks_limits (GtkPlot *plot,	
                                GtkOrientation orientation,
                          	gdouble begin, gdouble end)
{
  if(end < begin) return;

  if(orientation == GTK_ORIENTATION_HORIZONTAL){
  	plot->xmajor.begin = begin;
  	plot->xminor.end = end;
        plot->xmajor.set_limits = TRUE;
        plot->xminor.set_limits = TRUE;
  }else{
  	plot->ymajor.begin = begin;
  	plot->yminor.end = end;
        plot->ymajor.set_limits = TRUE;
        plot->yminor.set_limits = TRUE;
  }

  gtk_plot_calc_ticks(plot, orientation);
  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_unset_ticks_limits (GtkPlot *plot,	
                                  GtkOrientation orientation)
{
  if(orientation == GTK_ORIENTATION_HORIZONTAL){
        plot->xmajor.set_limits = FALSE;
        plot->xminor.set_limits = FALSE;
  }else{
        plot->ymajor.set_limits = FALSE;
        plot->yminor.set_limits = FALSE;
  }

  gtk_plot_calc_ticks(plot, orientation);
  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_show_labels (GtkPlot *plot, gint axis, gint mask)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->label_mask = mask;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}


void
gtk_plot_axis_labels_set_attributes (GtkPlot *plot,	
              		             gint axis,
				     gchar *font,
                                     gint height,
			             GdkColor fg,
			             GdkColor bg)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);

  if(!font){
   /* Use previous font */
/*    aux->title.font = g_strdup(DEFAULT_FONT);
    aux->title.height = DEFAULT_FONT_HEIGHT;
*/
  } else {
    
    if (aux->title.font) g_free(aux->title.font); //SanB
    aux->title.font = g_strdup(font);
    aux->title.height = height;
  }
  aux->title.fg = fg;
  aux->title.bg = bg;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_axis_labels_set_numbers (GtkPlot *plot,	
              		          gint axis,
              		          gint style,
              		          gint precision)
{
  GtkPlotAxis *aux;

  aux = gtk_plot_get_axis (plot, axis);
  aux->label_style = style;
  aux->label_precision = precision;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}


/******************************************
 *      gtk_plot_set_x0line_attributes
 *      gtk_plot_set_y0line_attributes
 *      gtk_plot_set_major_vgrid_attributes
 *      gtk_plot_set_minor_vgrid_attributes
 *      gtk_plot_set_major_hgrid_attributes
 *      gtk_plot_set_minor_hgrid_attributes
 ******************************************/

void 
gtk_plot_set_x0line_attributes(GtkPlot *plot, 
                               GtkPlotLineStyle line_style,
                               gint width,
                               GdkColor color)
{
  plot->x0_line.line_style = line_style;
  plot->x0_line.line_width = width;
  plot->x0_line.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void 
gtk_plot_set_y0line_attributes(GtkPlot *plot, 
                               GtkPlotLineStyle line_style,
                               gint width,
                               GdkColor color)
{
  plot->y0_line.line_style = line_style;
  plot->y0_line.line_width = width;
  plot->y0_line.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void 
gtk_plot_set_major_vgrid_attributes(GtkPlot *plot, 
                                    GtkPlotLineStyle line_style,
                                    gint width,
                                    GdkColor color)
{
  plot->major_vgrid.line_style = line_style;
  plot->major_vgrid.line_width = width;
  plot->major_vgrid.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void 
gtk_plot_set_minor_vgrid_attributes(GtkPlot *plot, 
                                    GtkPlotLineStyle line_style,
                                    gint width,
                                    GdkColor color)
{
  plot->minor_vgrid.line_style = line_style;
  plot->minor_vgrid.line_width = width;
  plot->minor_vgrid.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void 
gtk_plot_set_major_hgrid_attributes(GtkPlot *plot, 
                                    GtkPlotLineStyle line_style,
                                    gint width,
                                    GdkColor color)
{
  plot->major_hgrid.line_style = line_style;
  plot->major_hgrid.line_width = width;
  plot->major_hgrid.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void 
gtk_plot_set_minor_hgrid_attributes(GtkPlot *plot, 
                                    GtkPlotLineStyle line_style,
                                    gint width,
                                    GdkColor color)
{
  plot->minor_hgrid.line_style = line_style;
  plot->minor_hgrid.line_width = width;
  plot->minor_hgrid.color = color;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

/******************************************
 * gtk_plot_show_legends
 * gtk_plot_hide_legends
 * gtk_plot_show_legends_border
 * gtk_plot_hide_legends_border
 * gtk_plot_legends_move
 * gtk_plot_legends_get_position
 * gtk_plot_legends_get_allocation
 * gtk_plot_set_legends_attributes
 ******************************************/

void
gtk_plot_show_legends(GtkPlot *plot)
{
  plot->show_legends = TRUE;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}


void
gtk_plot_hide_legends(GtkPlot *plot)
{
  plot->show_legends = FALSE;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_show_legends_border(GtkPlot *plot, 
                             gboolean show_shadow, 
                             gint shadow_width)
{
  plot->show_legends_border = TRUE;
  plot->show_legends_shadow = show_shadow;
  plot->legends_shadow_width = shadow_width;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_hide_legends_border(GtkPlot *plot)
{
  plot->show_legends_border = FALSE;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_legends_move(GtkPlot *plot, gdouble x, gdouble y)
{
  plot->legends_x = x;
  plot->legends_y = y;

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_legends_get_position(GtkPlot *plot, gdouble *x, gdouble *y)
{
  *x = plot->legends_x;
  *y = plot->legends_y;
}

GtkAllocation
gtk_plot_legends_get_allocation(GtkPlot *plot)
{
  GtkAllocation allocation;
  GtkWidget *widget;

  widget = GTK_WIDGET(plot);

  allocation.x = widget->allocation.x + plot->x * widget->allocation.width +
                 plot->legends_x * plot->width * widget->allocation.width; 
  allocation.y = widget->allocation.y + plot->y * widget->allocation.height +
                 plot->legends_y * plot->height * widget->allocation.height; 
  allocation.width = plot->legends_width;
  allocation.height = plot->legends_height;

  return(allocation);
}

void
gtk_plot_legends_set_attributes(GtkPlot *plot, gchar *font, gint height, 
			        GdkColor *foreground, GdkColor *background)
{
  if (plot->legends_attr.font) g_free(plot->legends_attr.font); //SanB
  if(!font) {
    plot->legends_attr.font = g_strdup(DEFAULT_FONT);
    plot->legends_attr.height = DEFAULT_FONT_HEIGHT;
  } else {
    plot->legends_attr.font = g_strdup(font);
    plot->legends_attr.height = height;
  }
  plot->legends_attr.fg = GTK_WIDGET(plot)->style->black;
  plot->legends_attr.bg = GTK_WIDGET(plot)->style->white;

  if(foreground != NULL)
    plot->legends_attr.fg = *foreground;

  plot->legends_attr.transparent = TRUE;
  if(background != NULL){
    plot->legends_attr.bg = *background;
    plot->legends_attr.transparent = FALSE;
  }

  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

/******************************************
 * gtk_plot_dataset_new
 * gtk_plot_add_dataset
 * gtk_plot_add_function
 * gtk_plot_draw_dataset
 * gtk_plot_dataset_set_points
 * gtk_plot_dataset_get_points
 * gtk_plot_dataset_set_x
 * gtk_plot_dataset_set_y
 * gtk_plot_dataset_set_dx
 * gtk_plot_dataset_set_dy
 * gtk_plot_dataset_get_x
 * gtk_plot_dataset_get_y
 * gtk_plot_dataset_get_dx
 * gtk_plot_dataset_get_dy
 * gtk_plot_dataset_set_numpoints
 * gtk_plot_dataset_get_numpoints
 * gtk_plot_dataset_set_symbol
 * gtk_plot_dataset_get_symbol
 * gtk_plot_dataset_set_connector
 * gtk_plot_dataset_get_connector
 * gtk_plot_dataset_set_xy_attributes
 * gtk_plot_dataset_show_xerrbars
 * gtk_plot_dataset_show_yerrbars
 * gtk_plot_dataset_hide_xerrbars
 * gtk_plot_dataset_hide_yerrbars
 * gtk_plot_dataset_set_legend
 * gtk_plot_dataset_set_name
 * gtk_plot_show_dataset
 * gtk_plot_hide_dataset
 * gtk_plot_remove_dataset
 * gtk_plot_redraw
 ******************************************/

GtkPlotData *
gtk_plot_dataset_new (GtkPlot *plot)
{
  GtkPlotData  *dataset;

  dataset = g_new(GtkPlotData, 1);

  dataset->is_function = FALSE;
  dataset->is_visible = TRUE;
  dataset->function = NULL;
  dataset->num_points = 0;
  dataset->x = NULL;
  dataset->y = NULL;
  dataset->dx = NULL;
  dataset->dy = NULL;
  dataset->x_step = 2;
  dataset->line_connector = GTK_PLOT_CONNECT_STRAIGHT;

  dataset->line.line_style = GTK_PLOT_LINE_SOLID;
  dataset->line.line_width = 1;
  dataset->line.color = GTK_WIDGET(plot)->style->black; 

  dataset->x_line.line_style = GTK_PLOT_LINE_NONE;
  dataset->x_line.line_width = 1;
  dataset->x_line.color = GTK_WIDGET(plot)->style->black; 

  dataset->y_line.line_style = GTK_PLOT_LINE_NONE;
  dataset->y_line.line_width = 1;
  dataset->y_line.color = GTK_WIDGET(plot)->style->black; 

  dataset->symbol.symbol_type = GTK_PLOT_SYMBOL_NONE;
  dataset->symbol.symbol_style = GTK_PLOT_SYMBOL_EMPTY;
  dataset->symbol.size = 6;
  dataset->symbol.line_width = 1;
  dataset->symbol.color = GTK_WIDGET(plot)->style->black; 

  dataset->show_xerrbars = FALSE;
  dataset->show_yerrbars = FALSE;
  dataset->xerrbar_length = 8;
  dataset->yerrbar_length = 8;

  dataset->legend = NULL;
  dataset->name = NULL;

  return dataset;
}

void
gtk_plot_add_dataset(GtkPlot *plot, 
                     GtkPlotData *dataset) 
{

  plot->data_sets = g_list_append(plot->data_sets, dataset);

  gtk_widget_queue_draw(GTK_WIDGET(plot));
}

GtkPlotData *
gtk_plot_add_function(GtkPlot *plot, GtkPlotFunc function)
{
  GtkPlotData *dataset;

  dataset = gtk_plot_dataset_new(plot);
  dataset->is_function = TRUE;
  dataset->function = function;

  plot->data_sets = g_list_append(plot->data_sets, dataset);
  gtk_widget_queue_draw(GTK_WIDGET(plot));
  return (dataset);
}


void
gtk_plot_draw_dataset(GtkPlot *plot, GdkGC *gc, GtkPlotData *dataset)
{
  gboolean new_gc = FALSE;

  if(gc == NULL){
      new_gc = TRUE;
      gc = gdk_gc_new(GTK_WIDGET(plot)->window);
  } 

  gtk_plot_real_draw_dataset(plot, NULL, gc, dataset);

  if(new_gc) gdk_gc_unref(gc);
}

void
gtk_plot_dataset_set_points(GtkPlotData *data, 
                            gdouble *x, gdouble *y,
                            gdouble *dx, gdouble *dy,
                            gint num_points)
{
  data->x = x;
  data->y = y;
  data->dx = dx;
  data->dy = dy;
  data->num_points = num_points;
}

void
gtk_plot_dataset_get_points(GtkPlotData *data, 
                            gdouble *x, gdouble *y,
                            gdouble *dx, gdouble *dy,
                            gint *num_points)
{
  x = data->x;
  y = data->y;
  dx = data->dx;
  dy = data->dy;
  *num_points = data->num_points;
}

void
gtk_plot_dataset_set_x(GtkPlotData *data, 
                       gdouble *x) 
{
  data->x = x;
}


void
gtk_plot_dataset_set_y(GtkPlotData *data, 
                       gdouble *y) 
{
  data->y = y;
}

void
gtk_plot_dataset_set_dx(GtkPlotData *data, 
                        gdouble *dx) 
{
  data->dx = dx;
}

void
gtk_plot_dataset_set_dy(GtkPlotData *data, 
                        gdouble *dy) 
{
  data->dy = dy;
}

gdouble *
gtk_plot_dataset_get_x(GtkPlotData *dataset, gint *num_points)
{
  *num_points = dataset->num_points;
  return(dataset->x);
}

gdouble *
gtk_plot_dataset_get_y(GtkPlotData *dataset, gint *num_points)
{
  *num_points = dataset->num_points;
  return(dataset->y);
}

gdouble *
gtk_plot_dataset_get_dx(GtkPlotData *dataset, gint *num_points)
{
  *num_points = dataset->num_points;
  return(dataset->dx);
}

gdouble *
gtk_plot_dataset_get_dy(GtkPlotData *dataset, gint *num_points)
{
  *num_points = dataset->num_points;
  return(dataset->dy);
}

void
gtk_plot_dataset_set_numpoints(GtkPlotData *dataset, gint numpoints)
{
  dataset->num_points = numpoints;
}

gint
gtk_plot_dataset_get_numpoints(GtkPlotData *dataset)
{
  return(dataset->num_points);
}

void
gtk_plot_dataset_set_symbol (GtkPlotData *dataset,
		             GtkPlotSymbolType type,
		             GtkPlotSymbolStyle style,
                             gint size, gint line_width, GdkColor color)
{
  dataset->symbol.symbol_type = type;
  dataset->symbol.symbol_style = style;
  dataset->symbol.size = size;
  dataset->symbol.line_width = line_width;
  dataset->symbol.color = color;
}

void
gtk_plot_dataset_get_symbol (GtkPlotData *dataset,
		             gint *type,
		             gint *style,
                             gint *size, gint *line_width, GdkColor *color)
{
  *type = dataset->symbol.symbol_type;
  *style = dataset->symbol.symbol_style;
  *size = dataset->symbol.size;
  *line_width = dataset->symbol.line_width;
  *color = dataset->symbol.color;
}

void
gtk_plot_dataset_set_line_attributes (GtkPlotData *dataset, 
                                      GtkPlotLineStyle style,
                                      gint width,
                                      GdkColor color)
{
  dataset->line.line_style = style; 
  dataset->line.line_width = width; 
  dataset->line.color = color; 
}

void
gtk_plot_dataset_get_line_attributes (GtkPlotData *dataset, 
                                      gint * style,
                                      gint *width,
                                      GdkColor *color)
{
  *style = dataset->line.line_style; 
  *width = dataset->line.line_width; 
  *color = dataset->line.color; 
}


void
gtk_plot_dataset_set_connector (GtkPlotData *dataset,
		                GtkPlotConnector connector)
{
  dataset->line_connector = connector;
}

gint 
gtk_plot_dataset_get_connector (GtkPlotData *dataset)
{
  return (dataset->line_connector);
}

void
gtk_plot_dataset_set_xy_attributes (GtkPlotData *dataset, 
                            	    GtkPlotLineStyle style,
                            	    gint width,
                            	    GdkColor color)
{
  dataset->x_line.line_style = style; 
  dataset->x_line.line_width = width; 
  dataset->x_line.color = color; 

  dataset->y_line.line_style = style; 
  dataset->y_line.line_width = width; 
  dataset->y_line.color = color; 
}

void
gtk_plot_dataset_show_xerrbars(GtkPlotData *dataset) 
{
  dataset->show_xerrbars = TRUE;
}

void
gtk_plot_dataset_show_yerrbars(GtkPlotData *dataset) 
{
  dataset->show_yerrbars = TRUE;
}

void
gtk_plot_dataset_hide_xerrbars(GtkPlotData *dataset) 
{
  dataset->show_xerrbars = FALSE;
}

void
gtk_plot_dataset_hide_yerrbars(GtkPlotData *dataset) 
{
  dataset->show_yerrbars = FALSE;
}

void
gtk_plot_dataset_set_legend(GtkPlotData *dataset, gchar *legend)
{
  if(dataset->legend)
     g_free(dataset->legend);

  dataset->legend = g_strdup(legend);
}

void
gtk_plot_dataset_set_name(GtkPlotData *dataset, gchar *name)
{
  if(dataset->name)
     g_free(dataset->name);

  dataset->name = g_strdup(name);
}

void
gtk_plot_redraw(GtkPlot *plot)
{
  gtk_signal_emit (GTK_OBJECT(plot), plot_signals[CHANGED]);
}

void
gtk_plot_show_dataset(GtkPlotData *dataset)
{
  dataset->is_visible = TRUE;
}

void
gtk_plot_hide_dataset(GtkPlotData *dataset)
{
  dataset->is_visible = FALSE;
}

gint
gtk_plot_remove_dataset(GtkPlot *plot, GtkPlotData *dataset)
{
  GList *datasets;
  gpointer data;

  datasets = plot->data_sets;

  while(datasets)
   {
     data = datasets->data;
     
     if((GtkPlotData *)data == dataset){
              g_list_remove_link(plot->data_sets, datasets);
              g_list_free_1(datasets);
	      return TRUE;
     }
     datasets = datasets->next;
   }

   return FALSE;
}

/* Solve the tridiagonal equation system that determines the second
   derivatives for the interpolation points.  (Based on Numerical
   Recipies 2nd Edition.) */
static void
spline_solve (int n, gdouble x[], gdouble y[], gdouble y2[])
{
  gdouble p, sig, *u;
  gint i, k;

  u = g_malloc ((n - 1) * sizeof (u[0]));

  y2[0] = u[0] = 0.0;	/* set lower boundary condition to "natural" */

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

static gdouble
spline_eval (int n, gdouble x[], gdouble y[], gdouble y2[], gdouble val)
{
  gint k_lo, k_hi, k;
  gdouble h, b, a;

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
gtk_plot_calc_ticks(GtkPlot *plot, gint orientation)
{
  GtkPlotTicks *major;
  GtkPlotTicks *minor;
  gdouble vmajor;
  gdouble vminor;
  gdouble min, max;
  gint pt;
  gdouble tick;
  gint nticks;
 
  switch(orientation){
    case GTK_ORIENTATION_HORIZONTAL:
       major = &plot->xmajor; 
       minor = &plot->xminor; 
       max = plot->xmax;
       min = plot->xmin;
       break;
    case GTK_ORIENTATION_VERTICAL:
       major = &plot->ymajor; 
       minor = &plot->yminor; 
       max = plot->ymax;
       min = plot->ymin;
       break;
  }
 
  vminor = (gdouble)(floor(min/minor->step))*minor->step;
  vmajor = (gdouble)(floor(min/major->step))*major->step;

  if(major->set_limits || minor->set_limits){
       max = MIN(vmajor, major->end);
       min = MAX(vminor, minor->begin);
  }

  if(major->ticks != NULL){
     g_free(major->ticks);
     g_free(major->value);
     major->ticks = NULL;
     major->value = NULL;
  }
  if(minor->ticks != NULL){
     g_free(minor->ticks);
     g_free(minor->value);
     minor->ticks = NULL;
     minor->value = NULL;
  }

  nticks = 0; 
  for(tick = vmajor; tick <= max + 2*major->step; tick += major->step){
     if(tick >= min-1.E-10 && tick <= max+1.E-10){
        if(orientation == GTK_ORIENTATION_HORIZONTAL)
                                          pt = transform_x(plot, tick);
        else
                                          pt = transform_y(plot, tick);
        nticks ++;
        major->ticks = (gint *)g_realloc(major->ticks, nticks*sizeof(gint)); 
        major->value = (gdouble *)g_realloc(major->value, nticks*sizeof(gdouble)); 
        major->ticks[nticks-1] = pt;
        major->value[nticks-1] = tick;
        major->nticks = nticks;
     }
  }

  nticks = 0; 
  for(tick = vminor; tick <= max + major->step; tick += minor->step){
     if(tick >= min-1.E-10 && tick <= max+1.E-10){
        if(orientation == GTK_ORIENTATION_HORIZONTAL)
                                          pt = transform_x(plot, tick);
        else
                                          pt = transform_y(plot, tick);
        nticks ++;
        minor->ticks = (gint *)g_realloc(minor->ticks, nticks*sizeof(gint)); 
        minor->value = (gdouble *)g_realloc(minor->value, nticks*sizeof(gdouble)); 
        minor->ticks[nticks-1] = pt;
        minor->value[nticks-1] = tick;
        minor->nticks = nticks;
     }
  }
}

void gtk_plot_freeze(GtkObject *obj) 
{
  /*  gtk_signal_handler_block(obj,plot_signals[CHANGED]);*/
}

void gtk_plot_unfreeze(GtkObject *obj) 
{
  /*  gtk_signal_handler_unblock(obj,plot_signals[CHANGED]);*/
}
