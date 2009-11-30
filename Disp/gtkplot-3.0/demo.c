#include <math.h>
#include <stdio.h>
#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include "gtkplot.h"
#include "gtkplotlayout.h"
#include "gtkplotcanvas.h"

GdkPixmap *pixmap=NULL;
GtkWidget **plots;
GtkWidget **buttons;
GtkPlotData *dataset[5];
gint nlayers = 0;


void
quit ()
{
  gtk_main_quit();
}

gdouble function(gdouble x, gboolean *err)
{
 *err = FALSE;

 return(-.5+.3*sin(3.*x)*sin(50.*x));
}

gdouble gaussian(gdouble x, gboolean *err)
{
 gdouble y;
 *err = FALSE;
 y = .65*exp(-.5*pow(x-.5,2)/.02);
 return y;
}


gint
activate_plot(GtkWidget *widget, gpointer data)
{
  GtkWidget **widget_list;
  GtkWidget *active_widget;
  GtkWidget *canvas;
  gint n = 0;

  if(GTK_IS_PLOT_CANVAS(widget)) 
    {
        canvas = GTK_WIDGET(widget);
        widget_list = plots;
        active_widget = GTK_WIDGET(gtk_plot_canvas_get_active_plot(GTK_PLOT_CANVAS(widget)));
    }

  if(GTK_IS_BUTTON(widget)) 
    {
        canvas = GTK_WIDGET(data);
        widget_list = buttons;
        active_widget = widget;
    }

  while(n < nlayers)
    {
      if(widget_list[n] == active_widget){
            gtk_plot_canvas_set_active_plot(GTK_PLOT_CANVAS(canvas),
                                            GTK_PLOT(plots[n]));  
            GTK_BUTTON(buttons[n])->button_down = TRUE;
            GTK_TOGGLE_BUTTON(buttons[n])->active = TRUE;
            gtk_widget_set_state(buttons[n], GTK_STATE_ACTIVE);
      }else{
            GTK_BUTTON(buttons[n])->button_down = FALSE;
            GTK_TOGGLE_BUTTON(buttons[n])->active = FALSE;
            gtk_widget_set_state(buttons[n], GTK_STATE_NORMAL);
      }
      gtk_widget_queue_draw(buttons[n]);
      
      n++;
    }

  return TRUE;
}


GtkWidget *
new_layer(GtkWidget *canvas)
{
 gchar label[10];

 nlayers++;

 buttons = (GtkWidget **)g_realloc(buttons, nlayers * sizeof(GtkWidget *));
 plots = (GtkWidget **)g_realloc(plots, nlayers * sizeof(GtkWidget *));

 sprintf(label, "%d", nlayers);
 
 buttons[nlayers-1] = gtk_toggle_button_new_with_label(label);
/* gtk_button_set_relief(GTK_BUTTON(buttons[nlayers-1]), GTK_RELIEF_NONE);
*/
 gtk_widget_set_usize(buttons[nlayers-1], 20, 20);
 gtk_layout_put(GTK_LAYOUT(canvas), buttons[nlayers-1], (nlayers-1)*20, 0);
 gtk_widget_show(buttons[nlayers-1]);

 gtk_signal_connect(GTK_OBJECT(buttons[nlayers-1]), "toggled",
                    (GtkSignalFunc) activate_plot, canvas);

 plots[nlayers-1] = gtk_plot_new_with_size(NULL, .5, .25);

 gtk_signal_connect(GTK_OBJECT(canvas), "click_on_plot",
                    (GtkSignalFunc) activate_plot, NULL);

 activate_plot(buttons[nlayers-1],canvas);

 return plots[nlayers-1];
}


void
build_example1(GtkWidget *active_plot)
{
 GdkColor color;
 static GtkPlotPoint points[]={{0., 0., 0.2, 0.1, 0.},
                               {.20, .4, 0.2, 0.1, 0.},
                               {.40, .50, 0.2, 0.1, 0.},
                               {.60, .35, 0.2, 0.1, 0.},
                               {.80, .30, 0.2, 0.1, 0.},
                               {1.0, .40, 0.2, 0.1, 0.},
                              };
 static GtkPlotPoint points3[]={{-0., 0., 0.2, 0.1, 0.},
                                {-.20, .4, 0.2, 0.1, 0.},
                                {-.40, .50, 0.2, 0.1, 0.},
                                {-.60, .35, 0.2, 0.1, 0.},
                                {-.80, .30, 0.2, 0.1, 0.},
                                {-1.0, .40, 0.2, 0.1, 0.},
                                };


 dataset[0] = gtk_plot_dataset_new(GTK_PLOT(active_plot));
 gtk_plot_add_dataset(GTK_PLOT(active_plot), dataset[0]);
 gtk_plot_dataset_set_points(dataset[0], points, 6);
 gdk_color_parse("red", &color);
 gdk_color_alloc(gdk_colormap_get_system(), &color); 

 gtk_plot_dataset_set_symbol(dataset[0],
                             GTK_PLOT_SYMBOL_DIAMOND,
			     GTK_PLOT_SYMBOL_OPAQUE,
                             10, 2, color);
 gtk_plot_dataset_set_line_attributes(dataset[0],
                                      GTK_PLOT_LINE_SOLID,
                                      1, color);

 gtk_plot_dataset_set_connector(dataset[0], GTK_PLOT_CONNECT_SPLINE);

 gtk_plot_dataset_show_yerrbars(dataset[0]);
 gtk_plot_dataset_set_legend(dataset[0], "Spline + EY");


 dataset[3] = gtk_plot_dataset_new(GTK_PLOT(active_plot));
 gtk_plot_add_dataset(GTK_PLOT(active_plot), dataset[3]);
 gtk_plot_dataset_set_points(dataset[3], points3, 6);
 gtk_plot_dataset_set_symbol(dataset[3],
                             GTK_PLOT_SYMBOL_SQUARE,
			     GTK_PLOT_SYMBOL_OPAQUE,
                             8, 2, active_plot->style->black);
 gtk_plot_dataset_set_line_attributes(dataset[3],
                                      GTK_PLOT_LINE_SOLID,
                                      4, color);
 gtk_plot_dataset_set_connector(dataset[3], GTK_PLOT_CONNECT_STRAIGHT);
 gtk_plot_dataset_set_xy_attributes(dataset[3], 
                                    GTK_PLOT_LINE_SOLID,
                                    0, active_plot->style->black);
 gtk_plot_dataset_set_legend(dataset[3], "Line + Symbol");


 
 gdk_color_parse("blue", &color);
 gdk_color_alloc(gdk_colormap_get_system(), &color); 

 dataset[1] = gtk_plot_add_function(GTK_PLOT(active_plot), (GtkPlotFunc)function);
 gtk_plot_dataset_set_line_attributes(dataset[1],
                                      GTK_PLOT_LINE_SOLID,
                                      0, color);

 gtk_plot_dataset_set_legend(dataset[1], "Function Plot");
}

void
build_example2(GtkWidget *active_plot)
{
 GdkColor color;
 static GtkPlotPoint points2[]={{.1, .012, 0., 0., .10000},
                               {.2, .067, 0., 0., .10000},
                               {.3, .24, 0., 0., .10000},
                               {.4, .5, 0., 0., .10000},
                               {.5, .65, 0., 0., .10000},
                               {.6, .5, 0., 0., .10000},
                               {.7, .24, 0., 0., .10000},
                               {.8, .067, 0., 0., .10000},
                              };


 dataset[4] = gtk_plot_add_function(GTK_PLOT(active_plot), (GtkPlotFunc)gaussian);
 gdk_color_parse("dark green", &color);
 gdk_color_alloc(gdk_colormap_get_system(), &color); 
 gtk_plot_dataset_set_line_attributes(dataset[4],
                                      GTK_PLOT_LINE_DASHED,
                                      2, color);

 gtk_plot_dataset_set_legend(dataset[4], "Gaussian");


 gdk_color_parse("blue", &color);
 gdk_color_alloc(gdk_colormap_get_system(), &color); 

 dataset[2] = gtk_plot_dataset_new(GTK_PLOT(active_plot));
 gtk_plot_add_dataset(GTK_PLOT(active_plot), dataset[2]);
 gtk_plot_dataset_set_points(dataset[2], points2, 8);
 gtk_plot_dataset_set_symbol(dataset[2],
                             GTK_PLOT_SYMBOL_IMPULSE,
			     GTK_PLOT_SYMBOL_FILLED,
                             10, 5, color);
 gtk_plot_dataset_set_line_attributes(dataset[2],
                                      GTK_PLOT_LINE_SOLID,
                                      5, color);

 gtk_plot_dataset_set_connector(dataset[2], GTK_PLOT_CONNECT_NONE);
 gtk_plot_dataset_set_legend(dataset[2], "Impulses");
}

int main(int argc, char *argv[]){

 GtkWidget *window1;
 GtkWidget *vbox1;
 GtkWidget *scrollw1;
 GtkWidget *layout;
 GtkWidget *active_plot;
 GtkWidget *canvas;
 GdkColor color;
 gint page_width, page_height;
 gfloat scale = 1.;
 
 page_width = GTK_PLOT_LETTER_W * scale;
 page_height = GTK_PLOT_LETTER_H * scale;
 
 gtk_init(&argc,&argv);

 window1=gtk_window_new(GTK_WINDOW_TOPLEVEL);
 gtk_window_set_title(GTK_WINDOW(window1), "GtkPlot Demo");
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
 GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(canvas), GTK_PLOT_CANVAS_DND_FLAGS);
 GTK_PLOT_CANVAS_SET_FLAGS(GTK_PLOT_CANVAS(canvas), GTK_PLOT_CANVAS_ALLOCATE_TITLES);
 layout = canvas;
 gtk_container_add(GTK_CONTAINER(scrollw1),layout);
 gtk_layout_set_size(GTK_LAYOUT(layout), page_width, page_height);
 GTK_LAYOUT(layout)->hadjustment->step_increment = 5;
 GTK_LAYOUT(layout)->vadjustment->step_increment = 5;

/*
 gdk_color_parse("light blue", &color);
 gdk_color_alloc(gtk_widget_get_colormap(layout), &color);
 gtk_plot_layout_set_background(GTK_PLOT_LAYOUT(layout), color);
*/

 gtk_widget_show(layout);

 active_plot = new_layer(canvas);
 gtk_plot_set_range(GTK_PLOT(active_plot), -1. ,1., -1., 1.4);
 gtk_plot_legends_move(GTK_PLOT(active_plot), .500, .05);
 gtk_plot_hide_legends_border(GTK_PLOT(active_plot));
 gtk_plot_axis_hide_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_TOP);
 gtk_plot_axis_show_ticks(GTK_PLOT(active_plot), 3, 15);
 gtk_plot_axis_set_ticks(GTK_PLOT(active_plot), 0, 1., .5);
 gtk_plot_axis_set_ticks(GTK_PLOT(active_plot), 1, 1., .5);
 gtk_plot_axis_set_ticks(GTK_PLOT(active_plot), 2, 1., .5);
 gtk_plot_axis_set_ticks(GTK_PLOT(active_plot), 3, 1., .5);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_TOP_AXIS);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_RIGHT_AXIS);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_X0);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_Y0);
 gtk_plot_canvas_add_plot(GTK_PLOT_CANVAS(canvas), GTK_PLOT(active_plot), .15, .05);
 gtk_widget_show(active_plot);

 build_example1(active_plot);

 active_plot = new_layer(canvas);
 gdk_color_parse("light yellow", &color);
 gdk_color_alloc(gtk_widget_get_colormap(active_plot), &color);
 gtk_plot_set_background(GTK_PLOT(active_plot), color);

 gdk_color_parse("light blue", &color);
 gdk_color_alloc(gtk_widget_get_colormap(layout), &color);
 gtk_plot_legends_set_attributes(GTK_PLOT(active_plot),
                                 NULL, 0,
				 NULL,
                                 &color);
 gtk_plot_set_range(GTK_PLOT(active_plot), 0. ,1., 0., .85);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_V_GRID);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_H_GRID);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_TOP_AXIS);
 GTK_PLOT_SET_FLAGS(GTK_PLOT(active_plot), GTK_PLOT_SHOW_RIGHT_AXIS);
 gtk_plot_canvas_add_plot(GTK_PLOT_CANVAS(canvas), GTK_PLOT(active_plot), .15, .4);
 gtk_plot_axis_hide_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_TOP);
 gtk_plot_axis_hide_title(GTK_PLOT(active_plot), GTK_PLOT_AXIS_RIGHT);
 gtk_plot_show_legends_border(GTK_PLOT(active_plot), TRUE, 3);
 gtk_plot_legends_move(GTK_PLOT(active_plot), .58, .05);
 gtk_widget_show(active_plot);
 build_example2(active_plot);

 gtk_widget_show(window1);

 gtk_plot_layout_put_text(GTK_PLOT_LAYOUT(canvas), .25, .005, 0, 
                          "Times-BoldItalic", 16, NULL, NULL,
                          "DnD titles, legends and plots");

 gtk_plot_layout_export_ps(GTK_PLOT_LAYOUT(canvas), "plotdemo.ps", 0, 0, 
                           GTK_PLOT_LETTER);

 gtk_main();

 return(0);
}


