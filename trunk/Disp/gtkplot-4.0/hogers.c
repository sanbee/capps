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
    gdouble dx;
    gdouble dwidth;

    switch(plot->xscale){
       case GTK_PLOT_SCALE_LOG10:
        if(x <= 0. || plot->xmin <= 0. || plot->xmax <= 0.){
          return 0;
        }else{
          dx = log10(x) - log10(plot->xmin);
          dwidth = log10(plot->xmax)-log10(plot->xmin);
        }
	break;
       case GTK_PLOT_SCALE_LINEAR:
       default:
         dx = x - plot->xmin;
         dwidth = plot->xmax - plot->xmin;
    }

    width = (gdouble)GTK_WIDGET(plot)->allocation.width * plot->width;
    return(roundint(width*dx/dwidth));
}
