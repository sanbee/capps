/* gtkplotlayout - gtkplot layout widget for gtk+
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
#include <gtk/gtk.h>
#include "gtkplot.h"
#include "gtkplotlayout.h"
#include "gtkpsfont.h"
#include <math.h>

#define DEFAULT_OFFSET 25
#define DEFAULT_WIDTH 150
#define DEFAULT_HEIGHT 120

#define GRAPH_MASK    (GDK_EXPOSURE_MASK |		\
                       GDK_POINTER_MOTION_MASK |	\
                       GDK_POINTER_MOTION_HINT_MASK |	\
                       GDK_BUTTON_PRESS_MASK |		\
                       GDK_BUTTON_RELEASE_MASK)
                       

static void gtk_plot_layout_class_init 		(GtkPlotLayoutClass *class);
static void gtk_plot_layout_init 		(GtkPlotLayout *plot_layout);
static void gtk_plot_layout_remove		(GtkContainer *container, 
						 GtkWidget *child);
static void gtk_plot_layout_finalize 		(GtkObject *object);
static void gtk_plot_layout_map			(GtkWidget *widget);
static void gtk_plot_layout_size_request 	(GtkWidget *widget, 
                                                 GtkRequisition *requisition);
static void gtk_plot_layout_draw 		(GtkWidget *widget, 
						 GdkRectangle *area);
static void gtk_plot_layout_paint 		(GtkWidget *widget); 
static gint gtk_plot_layout_expose		(GtkWidget *widget, 
						 GdkEventExpose *event);
static void gtk_plot_layout_create_pixmap	(GtkWidget *widget, 
						 gint width, gint height);
static void gtk_plot_layout_set_plots_pixmap    (GtkPlotLayout *plot_layout);
static void gtk_plot_layout_draw_text		(GtkPlotLayout *layout,
                          			 GtkPlotText text);
static GdkPixmap * rotate_text			(GtkPlotLayout *layout,
            					 GtkPlotText text,
            					 gint *width, gint *height);
static GtkLayoutClass *parent_class = NULL;


guint
gtk_plot_layout_get_type (void)
{
  static GtkType plot_layout_type = 0;

  if (!plot_layout_type)
    {
      GtkTypeInfo plot_layout_info =
      {
	"GtkPlotLayout",
	sizeof (GtkPlotLayout),
	sizeof (GtkPlotLayoutClass),
	(GtkClassInitFunc) gtk_plot_layout_class_init,
	(GtkObjectInitFunc) gtk_plot_layout_init,
	/* reserved 1*/ NULL,
        /* reserved 2 */ NULL,
        (GtkClassInitFunc) NULL,
      };

      plot_layout_type = gtk_type_unique (GTK_TYPE_LAYOUT, &plot_layout_info);
    }
  return plot_layout_type;
}

static void
gtk_plot_layout_class_init (GtkPlotLayoutClass *class)
{
  GtkObjectClass *object_class;
  GtkWidgetClass *widget_class;
  GtkContainerClass *container_class;

  parent_class = gtk_type_class (gtk_layout_get_type ());

  object_class = (GtkObjectClass *) class;
  widget_class = (GtkWidgetClass *) class;
  container_class = (GtkContainerClass *) class;

  object_class->finalize = gtk_plot_layout_finalize;

  widget_class->draw = gtk_plot_layout_draw;
  widget_class->map = gtk_plot_layout_map;
  widget_class->expose_event = gtk_plot_layout_expose;

  container_class->remove = gtk_plot_layout_remove;

  widget_class->size_request = gtk_plot_layout_size_request;

}

static void
gtk_plot_layout_init (GtkPlotLayout *plot_layout)
{
  GtkWidget *widget;
  widget = GTK_WIDGET(plot_layout);

  gdk_color_black(gtk_widget_get_colormap(widget), &widget->style->black);
  gdk_color_white(gtk_widget_get_colormap(widget), &widget->style->white);

  gtk_widget_set_events (widget, gtk_widget_get_events(widget)|
                         GRAPH_MASK);

  plot_layout->num_plots = 0;
  plot_layout->background = widget->style->white;
}

static void
gtk_plot_layout_remove(GtkContainer *container, GtkWidget *child)
{
  GtkPlotLayout *layout;
  GList *list;

  layout = GTK_PLOT_LAYOUT(container);

  list = g_list_find(layout->plots, child);
  if(list) {
     layout->plots = g_list_remove(list, child);
     layout->num_plots --;
  }

  GTK_CONTAINER_CLASS(parent_class)->remove(container, child);
}

static void
gtk_plot_layout_draw (GtkWidget *widget, GdkRectangle *area)
{
  gtk_plot_layout_paint(widget);
}

static void
gtk_plot_layout_paint (GtkWidget *widget)
{
  GtkPlotLayout *plot_layout;
  GtkLayout *layout;
  GdkGC *gc;
  GList *text;
  GtkPlotText *child_text;

  plot_layout = GTK_PLOT_LAYOUT(widget);
  layout = GTK_LAYOUT(widget);

  if(!plot_layout->pixmap) return;

  gc = gdk_gc_new(plot_layout->pixmap);
  gdk_gc_set_foreground(gc, &plot_layout->background);

  gdk_draw_rectangle(plot_layout->pixmap,
                     gc,
                     TRUE,
                     0,0,plot_layout->width, plot_layout->height);

  gtk_plot_layout_set_plots_pixmap(plot_layout);

  gdk_draw_pixmap(GTK_LAYOUT(plot_layout)->bin_window,
                  widget->style->fg_gc[GTK_STATE_NORMAL],
                  GTK_PLOT_LAYOUT(widget)->pixmap,
                  layout->xoffset, layout->yoffset, 
                  0, 0, 
                  -1, -1);  

  text = plot_layout->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     gtk_plot_layout_draw_text(plot_layout, *child_text);
     text = text->next;
   }

  gdk_gc_unref(gc);
}

void
gtk_plot_layout_refresh(GtkPlotLayout *layout)
{
  GList *plots;
  GList *text;
  GtkPlot *plot;
  GdkRectangle area;
  GdkGC *gc;
  GtkPlotText *child_text;

  gc = gdk_gc_new(layout->pixmap);
  gdk_gc_set_foreground(gc, &layout->background);

  gdk_draw_rectangle(layout->pixmap,
                     gc,
                     TRUE,
                     0,0,layout->width, layout->height);

  plots = layout->plots; 
  while(plots) 
    {
      plot = GTK_PLOT(plots->data);
      gtk_plot_set_drawable(plot, layout->pixmap);
      area.x = GTK_WIDGET(plot)->allocation.x+GTK_LAYOUT(layout)->xoffset;
      area.y = GTK_WIDGET(plot)->allocation.y+GTK_LAYOUT(layout)->yoffset;
      area.width = GTK_WIDGET(plot)->allocation.width;	
      area.height = GTK_WIDGET(plot)->allocation.height;	
      gtk_plot_paint(GTK_WIDGET(plot), &area);
      plots = plots->next;
    }


  text = layout->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     gtk_plot_layout_draw_text(layout, *child_text);
     text = text->next;
   }


  gdk_draw_pixmap(GTK_LAYOUT(layout)->bin_window,
                  GTK_WIDGET(layout)->style->fg_gc[GTK_STATE_NORMAL],
                  layout->pixmap,
                  GTK_LAYOUT(layout)->xoffset, GTK_LAYOUT(layout)->yoffset,
                  0, 0,
                  -1, -1);

  gdk_gc_unref(gc);
}

void
gtk_plot_layout_refresh_chart(GtkPlotLayout *layout, GtkPlot *myplot)
{
  GList *plots;
  GList *text;
  GtkPlot *plot;
  GdkRectangle area;
  GdkGC *gc;
  GtkPlotText *child_text;

  printf("####%d %d\n",layout->width, layout->height);
  if (layout->pixmap->user_data == NULL) return;
  GdkPixmap *pixmap0=gdk_pixmap_new(GTK_WIDGET(layout),
				    layout->width, layout->height,-1);

  gc = gdk_gc_new(pixmap0);
  gdk_gc_set_foreground(gc, &layout->background);

  gdk_draw_rectangle(pixmap0,
                     gc,
                     TRUE,
                     0,0,layout->width, layout->height);

  if (myplot)
    {
      gtk_plot_set_drawable(myplot, pixmap0);
      area.x = GTK_WIDGET(myplot)->allocation.x+GTK_LAYOUT(layout)->xoffset;
      area.y = GTK_WIDGET(myplot)->allocation.y+GTK_LAYOUT(layout)->yoffset;
      area.width = GTK_WIDGET(myplot)->allocation.width;	
      area.height = GTK_WIDGET(myplot)->allocation.height;	
      gtk_plot_paint(GTK_WIDGET(myplot), &area);
    }
  else
    {
      plots = layout->plots; 
      while(plots) 
	{
	  plot = GTK_PLOT(plots->data);
	  gtk_plot_set_drawable(plot, layout->pixmap);
	  area.x = GTK_WIDGET(plot)->allocation.x+GTK_LAYOUT(layout)->xoffset;
	  area.y = GTK_WIDGET(plot)->allocation.y+GTK_LAYOUT(layout)->yoffset;
	  area.width = GTK_WIDGET(plot)->allocation.width;	
	  area.height = GTK_WIDGET(plot)->allocation.height;	
	  gtk_plot_paint(GTK_WIDGET(plot), &area);
	  plots = plots->next;
	}
    }


  text = layout->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     gtk_plot_layout_draw_text(layout, *child_text);
     text = text->next;
   }


  gdk_draw_pixmap(GTK_LAYOUT(layout)->bin_window,
                  GTK_WIDGET(layout)->style->fg_gc[GTK_STATE_NORMAL],
                  pixmap0,
                  GTK_LAYOUT(layout)->xoffset, GTK_LAYOUT(layout)->yoffset,
                  0, 0,
                  -1, -1);

  gdk_gc_unref(gc);
}

static void 
gtk_plot_layout_create_pixmap(GtkWidget *widget, gint width, gint height)
{
  GtkPlotLayout *plot_layout;
  GdkGC* gc;
  gint pixmap_width, pixmap_height;

  plot_layout=GTK_PLOT_LAYOUT(widget);

  if (!plot_layout->pixmap)
    plot_layout->pixmap = gdk_pixmap_new (widget->window,
				    width,
				    height, -1);
  else{
    gdk_window_get_size(plot_layout->pixmap, &pixmap_width, &pixmap_height);
    if(width != pixmap_width || height != pixmap_height)        
        gdk_pixmap_unref(plot_layout->pixmap);
        plot_layout->pixmap = gdk_pixmap_new (widget->window,
				        width,
				        height, -1);
  }

  gc = gdk_gc_new(plot_layout->pixmap);
  gdk_gc_set_foreground(gc, &plot_layout->background);

  gdk_draw_rectangle(plot_layout->pixmap,
                     gc,
                     TRUE,
                     0, 0, plot_layout->width, plot_layout->height);

  gtk_plot_layout_set_plots_pixmap(plot_layout);

  gdk_gc_unref(gc);
}


static void
gtk_plot_layout_map(GtkWidget *widget)
{
  GtkPlotLayout *plot_layout;
  GtkLayout *layout;

  plot_layout=GTK_PLOT_LAYOUT(widget);
  layout=GTK_LAYOUT(widget);

  GTK_WIDGET_CLASS(parent_class)->map(widget);

  if(!plot_layout->pixmap){
      gtk_plot_layout_create_pixmap(widget, plot_layout->width, plot_layout->height);
      gtk_plot_layout_paint(widget);
      return;
  }

  gtk_plot_layout_refresh(plot_layout);

}

static gint
gtk_plot_layout_expose(GtkWidget *widget, GdkEventExpose *event)
{
  GtkPlotLayout *plot_layout;
  GtkLayout *layout;
  GdkPixmap *pixmap;
  GList *text;
  GtkPlotText *child_text;

  if(!GTK_WIDGET_DRAWABLE(widget)) return FALSE;

  plot_layout=GTK_PLOT_LAYOUT(widget);
  layout=GTK_LAYOUT(widget);

  if(!plot_layout->pixmap){
      gtk_plot_layout_create_pixmap(widget, plot_layout->width, plot_layout->height);
      gtk_plot_layout_paint(widget);
      return FALSE;
  }

  pixmap = plot_layout->pixmap;
  gdk_draw_pixmap(GTK_LAYOUT(plot_layout)->bin_window,
                  widget->style->fg_gc[GTK_STATE_NORMAL],
                  pixmap,
                  layout->xoffset+event->area.x, layout->yoffset+event->area.y, 
                  event->area.x, event->area.y, 
                  event->area.width, event->area.height); 

  text = plot_layout->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     gtk_plot_layout_draw_text(plot_layout, *child_text);
     text = text->next;
   }

  return FALSE;
}


static void
gtk_plot_layout_size_request (GtkWidget *widget, GtkRequisition *requisition)
{
  GtkPlotLayout *plot_layout;

  plot_layout=GTK_PLOT_LAYOUT(widget);

  GTK_WIDGET_CLASS(parent_class)->size_request(widget, requisition);

  widget->requisition.width = MAX(plot_layout->width, requisition->width);
  widget->requisition.height = MAX(plot_layout->height, requisition->height);

}

GtkWidget*
gtk_plot_layout_new (gint width, gint height)
{
  GtkPlotLayout *plot_layout;

  plot_layout = gtk_type_new (gtk_plot_layout_get_type ());
  plot_layout->width = width;
  plot_layout->height = height;
  return GTK_WIDGET (plot_layout);
}


static void
gtk_plot_layout_finalize (GtkObject *object)
{
  GtkPlotLayout *plot_layout;

  g_return_if_fail (object != NULL);
  g_return_if_fail (GTK_IS_PLOT_LAYOUT (object));

  plot_layout = GTK_PLOT_LAYOUT (object);

  if (GTK_OBJECT_CLASS (parent_class)->finalize)
    (*GTK_OBJECT_CLASS (parent_class)->finalize) (object);
}

void
gtk_plot_layout_add_plot(GtkPlotLayout *plot_layout, GtkPlot *plot, gint x, gint y)
{
  plot_layout->num_plots += 1;

  plot_layout->plots = g_list_append(plot_layout->plots, plot);
 
  gtk_layout_put(GTK_LAYOUT(plot_layout), GTK_WIDGET(plot), x, y);

  gtk_plot_layout_set_plots_pixmap(plot_layout);
}

static void
gtk_plot_layout_set_plots_pixmap(GtkPlotLayout *plot_layout)
{
  GdkRectangle area;
  GList *plots;
  GtkPlot *plot;
  GList *text;
  GtkPlotText *child_text;

  if(!plot_layout->pixmap) return;
  plots = plot_layout->plots; 
  while(plots) 
    {
      plot = GTK_PLOT(plots->data);
      gtk_plot_set_drawable(plot, plot_layout->pixmap);
      area.x = GTK_WIDGET(plot)->allocation.x+GTK_LAYOUT(plot_layout)->xoffset;
      area.y = GTK_WIDGET(plot)->allocation.y+GTK_LAYOUT(plot_layout)->yoffset;
      area.width = GTK_WIDGET(plot)->allocation.width;	
      area.height = GTK_WIDGET(plot)->allocation.height;	
      gtk_widget_draw(GTK_WIDGET(plot), &area);
      plots = plots->next;
    }

  text = plot_layout->text;
  while(text)
   {
     child_text = (GtkPlotText *) text->data;
     gtk_plot_layout_draw_text(plot_layout, *child_text);
     text = text->next;
   }

}

void
gtk_plot_layout_set_background (GtkPlotLayout *layout, GdkColor color)
{
  
  g_return_if_fail (layout != NULL);
  g_return_if_fail (GTK_IS_PLOT_LAYOUT (layout));

  layout->background = color;

  if(GTK_WIDGET_REALIZED(GTK_WIDGET(layout)))
       gtk_widget_queue_draw(GTK_WIDGET(layout));
}

void
gtk_plot_layout_put_text (GtkPlotLayout *layout, 
                          gdouble x, gdouble y, gint angle,
                          gchar *font, gint height,
                          GdkColor *fg, GdkColor *bg, 
			  gint justification,
                          gchar *text)
{
  GtkWidget *widget;
  GtkPlotText *text_attr;

  widget = GTK_WIDGET(layout);

  text_attr = g_new(GtkPlotText, 1);

  text_attr->x = x;
  text_attr->y = y;
  text_attr->angle = angle;
  text_attr->font = font;
  text_attr->fg = widget->style->black;
  text_attr->bg = widget->style->white;
  text_attr->text = NULL;
  text_attr->font = g_strdup(font);
  text_attr->height = height;
  text_attr->justification = justification;
  if(text) text_attr->text = g_strdup(text);

  if(fg != NULL)
    text_attr->fg = *fg;

  text_attr->transparent = TRUE;
  if(bg != NULL){
    text_attr->bg = *bg;
    text_attr->transparent = FALSE;
  }

  layout->text = g_list_append(layout->text, text_attr);
  gtk_plot_layout_draw_text(layout, *text_attr);
}

static void
gtk_plot_layout_draw_text(GtkPlotLayout *plot_layout,
                          GtkPlotText text)
{
  GdkPixmap *text_pixmap;
  GdkPixmap *pixmap;
  GtkLayout *layout;
  GdkImage *image;
  GdkColor color;
  GdkGC *gc;
  GdkColormap *colormap;
  GdkColorContext *cc;
  GdkVisual *visual;
  gint x, y;
  gint xp, yp;
  gint width, height;

  if(plot_layout->pixmap == NULL) return;
  layout = GTK_LAYOUT(plot_layout);

  text_pixmap = rotate_text(plot_layout, text, &width, &height);

  colormap = gtk_widget_get_colormap (GTK_WIDGET(layout));
  visual = gtk_widget_get_visual (GTK_WIDGET(layout));
  cc = gdk_color_context_new(visual, colormap);
  image = gdk_image_get(text_pixmap, 0, 0, width, height);
  gc = gdk_gc_new(plot_layout->pixmap);

  x = text.x * plot_layout->width;
  y = text.y * plot_layout->height;

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
       if(text.angle == 0 || text.angle == 180)
          x -= width/2;
       else
          y -= height/2;
       break;
  }

  pixmap = gdk_pixmap_new(GTK_WIDGET(layout)->window, width, height, -1);
  gdk_draw_pixmap(pixmap,
		  GTK_WIDGET(layout)->style->fg_gc[0],
		  plot_layout->pixmap,
		  x, y,
		  0, 0,
		  width, height);

  for(yp = 0; yp < height; yp++)
    for(xp = 0; xp < width; xp++)
       {
          color.pixel = gdk_image_get_pixel(image, xp, yp);
          gdk_color_context_query_color(cc, &color);

          gdk_gc_set_foreground(gc, &color);
          if(!text.transparent || gdk_color_equal(&color, &text.fg))
             gdk_draw_point(plot_layout->pixmap, gc, x + xp, y + yp);
          
       }
 
  gdk_draw_pixmap(layout->bin_window,
                  GTK_WIDGET(layout)->style->fg_gc[GTK_STATE_NORMAL],
                  plot_layout->pixmap,
                  layout->xoffset + x, layout->yoffset + y, 
                  x, y, 
                  width, height); 

  gdk_pixmap_unref(pixmap);
  gdk_gc_unref(gc);
  gdk_image_destroy(image);
  gdk_color_context_free(cc);
}

static GdkPixmap *
rotate_text(GtkPlotLayout *layout,
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
  GdkFont *font;
  gint x, y;
  gint old_width, old_height;

  window = GTK_WIDGET(layout)->window;
  colormap = gtk_widget_get_colormap (GTK_WIDGET(layout));
  visual = gtk_widget_get_visual (GTK_WIDGET(layout));
  cc = gdk_color_context_new(visual, colormap);
  gc = gdk_gc_new (layout->pixmap);
  font = gtk_psfont_get_gdkfont(text.font, text.height);

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

  gdk_gc_unref(gc);
  gdk_image_destroy(image);
  gdk_pixmap_unref(old_pixmap);
  gdk_color_context_free(cc);
  return (new_pixmap);
}

void
gtk_plot_layout_get_pixel(GtkPlotLayout *plot_layout, gdouble px, gdouble py,
                          gint *x, gint *y)
{
  *x = plot_layout->width * px;
  *y = plot_layout->height * py;
}

void
gtk_plot_layout_get_position(GtkPlotLayout *plot_layout, gint x, gint y,
                             gdouble *px, gdouble *py)
{
  *px = (gdouble) x / (gdouble) plot_layout->width;
  *py = (gdouble) y / (gdouble) plot_layout->height;
}

