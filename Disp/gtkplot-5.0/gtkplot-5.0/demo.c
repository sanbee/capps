/* plot.c */

#include <gtk/gtk.h>
#include <glib.h>
#include "gtkplot.h"
#include <fcntl.h>
#include "gtkplotcanvas.h"
//#include "gtkplotcanvasplot.h"
//#include "gtkplotdata.h"

#define NPOINTS 11
GtkWidget* canvas;

void quit()
{
    gtk_main_quit();
}

gboolean resize_canvas(GtkWidget* vbox,GdkEvent* event,GtkWidget* vbox_in)
{
     gint w;
     gint h;
     /*
     w = vbox_in->allocation.width;
     h = vbox_in->allocation.height;
     gtk_plot_canvas_set_size(GTK_PLOT_CANVAS(canvas),w,h);
     gtk_plot_canvas_paint(GTK_PLOT_CANVAS(canvas));
     gtk_plot_canvas_refresh(GTK_PLOT_CANVAS(canvas));
     g_message("W = %d H = %d",w,h);
     */
     return FALSE;
}

int main(int argc, char *argv[])
{
     GtkWidget* window;
     GtkWidget* vbox;
     GtkWidget* plot;
     GtkWidget* plot2;
     GtkPlotCanvasChild* child;
     GtkPlotCanvasChild* child2;
     GtkPlotData* dataset;
     GtkPlotData* dataset2;
     GdkColor colour_magenta;
     GdkColor colour_RoyalBlue1;
     GdkColor colour_LightGray;
     GdkColor colour_black;
     gdouble px[NPOINTS];
     gdouble py[NPOINTS];
     gdouble px2[NPOINTS];
     gdouble py2[NPOINTS];
     int error_fd;
     int i;

     error_fd = open("error.log",
                      O_CREAT|O_WRONLY|O_TRUNC, //create file|open for 
writing only|if the file exists, truncate it to zero length
                      0666); //The protection mode for the file
     dup2(error_fd,STDERR_FILENO);

     gtk_init(&argc, &argv);

     //allocate colours
     gdk_color_parse("magenta",&colour_magenta);
     gdk_color_alloc(gdk_colormap_get_system(),&colour_magenta);
     gdk_color_parse("RoyalBlue1",&colour_RoyalBlue1);
     gdk_color_alloc(gdk_colormap_get_system(),&colour_RoyalBlue1);
     gdk_color_parse("LightGray",&colour_LightGray);
     gdk_color_alloc(gdk_colormap_get_system(),&colour_LightGray);
     gdk_color_parse("black",&colour_black);
     gdk_color_alloc(gdk_colormap_get_system(),&colour_black);

     //main window
     window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
     gtk_window_set_title(GTK_WINDOW(window),"Plot");
     gtk_container_border_width(GTK_CONTAINER(window),0);
     g_signal_connect(GTK_OBJECT(window),"destroy",G_CALLBACK(quit),NULL);

     //vbox
     vbox = gtk_vbox_new(TRUE,0);
     gtk_container_add(GTK_CONTAINER(window),vbox);
     g_signal_connect(GTK_OBJECT(window),"configure-event",G_CALLBACK(resize_canvas),vbox);

     //canvas
     canvas = gtk_plot_canvas_new(450, 450, 1.0);//The 1.0 is the scale factor
     gtk_box_pack_start(GTK_BOX(vbox),
                        canvas,
                        FALSE,//expand
                        TRUE,//fill
                        0);  //padding

     //plot 1
     plot = gtk_plot_new(NULL);
     //gtk_plot_set_range(GtkPlot *plot,gdouble xmin,gdouble xmax,gdouble 
ymin,gdouble ymax)
     gtk_plot_set_range(GTK_PLOT(plot),0,10,0,100);
     //gtk_plot_set_ticks(GtkPlot *plot, GtkPlotOrientation orientation, 
gdouble major_step, gint 
nminor)                                                 gint nminor);
     gtk_plot_set_ticks(GTK_PLOT(plot),GTK_PLOT_AXIS_X,2,0);
     gtk_plot_set_ticks(GTK_PLOT(plot),GTK_PLOT_AXIS_Y,10,0);
     //gtk_plot_axis_show_ticks(GtkPlotAxis *axis,gint major_mask,gint 
minor_mask);
     gtk_plot_axis_show_ticks((GTK_PLOT(plot))->left,0,0);
     gtk_plot_axis_show_ticks((GTK_PLOT(plot))->bottom,0,0);
     //gtk_plot_grids_set_visible(GtkPlot *plot,gboolean vmajor,gboolean 
vminor,gboolean hmajor,gboolean hminor);
     gtk_plot_grids_set_visible(GTK_PLOT(plot),1,0,1,0);
     //Botttom X-axis settings
     gtk_plot_axis_set_attributes((GTK_PLOT(plot))->bottom,1.0,&colour_black);
     gtk_plot_major_vgrid_set_attributes(GTK_PLOT(plot),GTK_PLOT_LINE_SOLID,1.0,&colour_LightGray);
     //Top X-axis settings
     gtk_plot_axis_set_visible((GTK_PLOT(plot))->top,0);
     //Left Y-axis settings
     gtk_plot_axis_set_attributes((GTK_PLOT(plot))->left,1.0,&colour_black);
     gtk_plot_major_hgrid_set_attributes(GTK_PLOT(plot),GTK_PLOT_LINE_SOLID,1.0,&colour_LightGray);
     //Right Y-axis settings
     gtk_plot_axis_set_visible((GTK_PLOT(plot))->right,0);
     //Titles
     gtk_plot_axis_hide_title((GTK_PLOT(plot))->right);
     //gtk_plot_axis_set_title(GtkPlotAxis *axis, const gchar *title);
     gtk_plot_axis_set_title((GTK_PLOT(plot))->bottom, "Frequency (MHz)");
     //Add plot to canvas
     child = gtk_plot_canvas_plot_new(GTK_PLOT(plot));
     gtk_plot_canvas_put_child(GTK_PLOT_CANVAS(canvas), child, 0.10, 0.10, 
0.85, 0.85);
     gtk_widget_show(plot);

     /* plot dataset 1*/
     for (i = 0; i < NPOINTS; i++)
     {
       px[i] = i;
       py[i] = i*i;
     }
     dataset = GTK_PLOT_DATA(gtk_plot_data_new());
     gtk_plot_add_data(GTK_PLOT(plot),dataset);
     //Legends
     gtk_plot_data_set_legend(dataset, "DB(|S21|) (L)");
     gtk_plot_data_set_points(dataset, px, py, NULL, NULL, NPOINTS);
     //gtk_plot_data_set_line_attributes(GtkPlotData *data,GtkPlotLineStyle 
style,GdkCapStyle cap_style,GdkJoinStyle join_style,gfloat width,const 
GdkColor *color);
     gtk_plot_data_set_line_attributes(dataset,GTK_PLOT_LINE_SOLID,GDK_CAP_ROUND,GDK_JOIN_MITER,1.0,&colour_RoyalBlue1);
     gtk_plot_data_draw_points(dataset, 1);
     gtk_widget_show(GTK_WIDGET(dataset));

     //plot 2
     plot2 = gtk_plot_new(NULL);
     gtk_plot_set_range(GTK_PLOT(plot2),0,10,0,100);
     gtk_plot_set_ticks(GTK_PLOT(plot2),GTK_PLOT_AXIS_Y,10,0);
     gtk_plot_axis_show_ticks((GTK_PLOT(plot2))->top,0,0);
     gtk_plot_axis_show_ticks((GTK_PLOT(plot2))->right,0,0);
     //Top X-axis settings
     gtk_plot_axis_set_attributes((GTK_PLOT(plot2))->top,1.0,&colour_black);
     gtk_plot_axis_show_ticks((GTK_PLOT(plot2))->top,0,0);
     gtk_plot_axis_show_labels((GTK_PLOT(plot2))->top,0);
     //Bottom X-axis settings
     gtk_plot_axis_set_visible((GTK_PLOT(plot2))->bottom,0);
     //Left Y-axis settings
     gtk_plot_axis_set_visible((GTK_PLOT(plot2))->left,0);
     //Right Y-axis settings
     gtk_plot_axis_set_attributes((GTK_PLOT(plot2))->right,1.0,&colour_black);
     //Titles
     gtk_plot_axis_hide_title((GTK_PLOT(plot2))->right);
     child2 = gtk_plot_canvas_plot_new(GTK_PLOT(plot2));
     gtk_plot_canvas_put_child(GTK_PLOT_CANVAS(canvas), child2, 0.10, 0.10, 
0.85, 0.85);
     gtk_widget_show(plot2);

     //plot 2 dataset
     for (i = 0; i < NPOINTS; i++)
     {
       px2[i] = i;
       py2[i] = 100-i*i;
     }
     dataset2 = GTK_PLOT_DATA(gtk_plot_data_new());
     gtk_plot_add_data(GTK_PLOT(plot2),dataset2);
     gtk_plot_data_set_points(dataset2,px2,py2,NULL,NULL,NPOINTS);
     gtk_plot_data_set_line_attributes(dataset2,GTK_PLOT_LINE_SOLID,GDK_CAP_ROUND,GDK_JOIN_MITER,1.0,&colour_magenta);
     gtk_plot_data_draw_points(dataset2, 1);
     gtk_widget_show(GTK_WIDGET(dataset2));

     gtk_plot_canvas_paint(GTK_PLOT_CANVAS(canvas));
     gtk_plot_canvas_refresh(GTK_PLOT_CANVAS(canvas));
     gtk_widget_show_all(window);

     gtk_main();

     return 0;
}

