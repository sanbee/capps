#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include "gtkplot.h"
#include "gtkplotlayout.h"
#include "gtkplotcanvas.h"

GdkPixmap *pixmap=NULL;
GtkWidget **plots;
GtkPlotData *dataset;
gint nlayers = 0;
gint32 timer;
GtkWidget *layout;
GtkWidget *canvas;
gdouble *px;
gdouble *py;

void
quit ()
{
  gtk_timeout_remove(timer);
  gtk_main_quit();
}


GtkWidget *
new_layer(GtkWidget *layout)
{
 gchar label[10];

 nlayers++;

 plots = (GtkWidget **)g_realloc(plots, nlayers * sizeof(GtkWidget *));

 sprintf(label, "%d", nlayers);
 
 plots[nlayers-1] = gtk_plot_new_with_size(NULL, .65, .45);

 return plots[nlayers-1];
}

gint
update(gpointer data)
{
 GtkPlot *plot;
 gdouble x, y;
 gdouble xmin, xmax;

 plot = GTK_PLOT(data);

 px = (gdouble *)g_realloc(px, (dataset->num_points+1)*sizeof(gdouble));
 py = (gdouble *)g_realloc(py, (dataset->num_points+1)*sizeof(gdouble));

 y = rand()%10/10.;

 if(dataset->num_points == 0)
   x = 1;
 else
   x = px[dataset->num_points-1] + 1;

 px[dataset->num_points] = x;
 py[dataset->num_points] = y;

 gtk_plot_dataset_set_numpoints(dataset, dataset->num_points+1); 
 gtk_plot_dataset_set_x(dataset, px); 
 gtk_plot_dataset_set_y(dataset, py); 

 gtk_plot_get_xrange(plot, &xmin , &xmax);

 if(x > xmax)
   gtk_plot_set_xrange(plot, xmin + 5. , xmax + 5.);

 gtk_plot_layout_refresh(GTK_PLOT_LAYOUT(canvas));

 return TRUE;
}


int main(int argc, char *argv[]){

 GtkWidget *window1;
 GtkWidget *vbox1;
 GtkWidget *scrollw1;
 GtkWidget *active_plot;
 GdkColor color;
 gint page_width, page_height;
 gfloat scale = 1.;
 
 page_width = GTK_PLOT_LETTER_W * scale;
 page_height = GTK_PLOT_LETTER_H * scale;
 
 gtk_init(&argc,&argv);

 window1=gtk_window_new(GTK_WINDOW_TOPLEVEL);
 gtk_window_set_title(GTK_WINDOW(window1), "GtkPlot Real Time Demo");
 gtk_widget_set_usize(window1,550,600);
 gtk_container_border_width(GTK_CONTAINER(window1),0);

 gtk_signal_connect (GTK_OBJECT (window1), "destroy",
		     GTK_SIGNAL_FUNC (quit), NULL);

 vbox1=gtk_vbox_new(FALSE,0);
 gtk_container_add(GTK_CONTAINER(window1),vbox1);
 gtk_widget_show(vbox1);

 scrollw1=gtk_scrolled_window_new(NULL, NULL);
 gtk_container_border_width(GTK_CONTAINER(scrollw1),0);
 gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrollw1),
				GTK_POLICY_ALWAYS,GTK_POLICY_ALWAYS);
 gtk_box_pack_start(GTK_BOX(vbox1),scrollw1, TRUE, TRUE,0);
 gtk_widget_show(scrollw1);

 canvas = gtk_plot_canvas_new(page_width, page_height);
 GTK_PLOT_CANVAS_UNSET_FLAGS(GTK_PLOT_CANVAS(canvas), GTK_PLOT_CANVAS_DND_FLAGS);
 layout = canvas;
 gtk_container_add(GTK_CONTAINER(scrollw1),layout);
 gtk_layout_set_size(GTK_LAYOUT(layout), page_width, page_height);
 GTK_LAYOUT(layout)->hadjustment->step_increment = 5;
 GTK_LAYOUT(layout)->vadjustment->step_increment = 5;

 gdk_color_parse("light blue", &color);
 gdk_color_alloc(gtk_widget_get_colormap(layout), &color);
 gtk_plot_layout_set_background(GTK_PLOT_LAYOUT(layout), color);

 gtk_widget_show(layout);

 active_plot = new_layer(layout);
 gdk_color_parse("light yellow", &color);
 gdk_color_alloc(gtk_widget_get_colormap(active_plot), &color);
 gtk_plot_set_background(GTK_PLOT(active_plot), color);

 gdk_color_parse("white", &color);
 gdk_color_alloc(gtk_widget_get_colormap(layout), &color);
 gtk_plot_legends_set_attributes(GTK_PLOT(active_plot),
                                 NULL, 0,
				 NULL,
                                 &color);
 gtk_plot_set_range(GTK_PLOT(active_plot), 0. ,20., 0., 1.);
 gtk_plot_axis_set_ticks(GTK_PLOT(active_plot), 0, 2, 1);
 gtk_plot_axis_set_ticks(GTK_PLOT(active_plot), 1, .1, .05);
 gtk_plot_axis_labels_set_numbers(GTK_PLOT(active_plot), 2, 0, 0);
 gtk_plot_axis_labels_set_numbers(GTK_PLOT(active_plot), 3, 0, 0);
 gtk_plot_axis_set_visible(GTK_PLOT(active_plot), GTK_PLOT_AXIS_TOP, TRUE);
 gtk_plot_axis_set_visible(GTK_PLOT(active_plot), GTK_PLOT_AXIS_RIGHT, TRUE);
 gtk_plot_set_grids_visible(GTK_PLOT(active_plot), TRUE, TRUE, TRUE, TRUE);
 gtk_plot_canvas_add_plot(GTK_PLOT_CANVAS(canvas), GTK_PLOT(active_plot), .15, .15);
 gtk_plot_axis_hide_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_TOP);
 gtk_plot_axis_hide_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_RIGHT);
 gtk_plot_axis_set_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_LEFT, "Intensity");
 gtk_plot_axis_set_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_BOTTOM, "Time (s)");
 gtk_plot_show_legends_border(GTK_PLOT(active_plot), TRUE, 3);
 gtk_plot_legends_move(GTK_PLOT(active_plot), .60, .10);
 gtk_widget_show(active_plot);

 gtk_widget_show(window1);

 gtk_plot_layout_put_text(GTK_PLOT_LAYOUT(canvas), .45, .05, 0, 
                          "Times-BoldItalic", 20, NULL, NULL,
                          GTK_JUSTIFY_CENTER,
                          "Real Time Demo");

 dataset = gtk_plot_dataset_new(GTK_PLOT(active_plot));
 gtk_plot_add_dataset(GTK_PLOT(active_plot), dataset);

 gdk_color_parse("red", &color);
 gdk_color_alloc(gdk_colormap_get_system(), &color);

 gtk_plot_dataset_set_legend(dataset, "Random pulse");
 gtk_plot_dataset_set_symbol(dataset,
                             GTK_PLOT_SYMBOL_DIAMOND,
                             GTK_PLOT_SYMBOL_OPAQUE,
                             10, 2, color);
 gtk_plot_dataset_set_line_attributes(dataset,
                                      GTK_PLOT_LINE_SOLID,
                                      1, color);

 timer = gtk_timeout_add(1000, update, active_plot);

 gtk_main();

 return(0);
}


