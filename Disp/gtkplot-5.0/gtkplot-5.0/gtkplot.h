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

#ifndef __GTK_PLOT_H__
#define __GTK_PLOT_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "gtkplotpc.h"

enum
{
  GTK_PLOT_TRANSPARENT    	 = 1 << 1
};


#define GTK_PLOT(obj)        GTK_CHECK_CAST (obj, gtk_plot_get_type (), GtkPlot)
#define GTK_PLOT_CLASS(klass) GTK_CHECK_CLASS_CAST (klass, gtk_plot_get_type, GtkPlotClass)
#define GTK_IS_PLOT(obj)     GTK_CHECK_TYPE (obj, gtk_plot_get_type ())
#define GTK_PLOT_FLAGS(plot)         (GTK_PLOT(plot)->flags)
#define GTK_PLOT_SET_FLAGS(plot,flag) (GTK_PLOT_FLAGS(plot) |= (flag))
#define GTK_PLOT_UNSET_FLAGS(plot,flag) (GTK_PLOT_FLAGS(plot) &= ~(flag))

#define GTK_PLOT_TRANSPARENT(plot) (GTK_PLOT_FLAGS(plot) & GTK_PLOT_TRANSPARENT)

typedef struct _GtkPlot		GtkPlot;
typedef struct _GtkPlotClass	GtkPlotClass;
typedef struct _GtkPlotText 	GtkPlotText;
typedef struct _GtkPlotLine 	GtkPlotLine;
typedef struct _GtkPlotSymbol 	GtkPlotSymbol;
typedef struct _GtkPlotAxis 	GtkPlotAxis;
typedef struct _GtkPlotData	GtkPlotData;
typedef struct _GtkPlotTicks	GtkPlotTicks;

typedef gdouble (*GtkPlotFunc)	(GtkPlot *plot, GtkPlotData *dataset, gdouble x, gboolean *error);

typedef enum
{
  GTK_PLOT_SCALE_LINEAR	,
  GTK_PLOT_SCALE_LOG10		
} GtkPlotScale;

typedef enum
{
  GTK_PLOT_SYMBOL_NONE		,
  GTK_PLOT_SYMBOL_SQUARE	,
  GTK_PLOT_SYMBOL_CIRCLE	,
  GTK_PLOT_SYMBOL_UP_TRIANGLE	,
  GTK_PLOT_SYMBOL_DOWN_TRIANGLE	,
  GTK_PLOT_SYMBOL_DIAMOND	,
  GTK_PLOT_SYMBOL_PLUS		,
  GTK_PLOT_SYMBOL_CROSS		,
  GTK_PLOT_SYMBOL_STAR		,
  GTK_PLOT_SYMBOL_IMPULSE	,
  GTK_PLOT_SYMBOL_BAR	
} GtkPlotSymbolType;

typedef enum
{
  GTK_PLOT_SYMBOL_EMPTY		,
  GTK_PLOT_SYMBOL_FILLED	,
  GTK_PLOT_SYMBOL_OPAQUE	
} GtkPlotSymbolStyle;

typedef enum
{
  GTK_PLOT_LINE_NONE		,
  GTK_PLOT_LINE_SOLID		,
  GTK_PLOT_LINE_DOTTED		,
  GTK_PLOT_LINE_DASHED		,
  GTK_PLOT_LINE_DOT_DASH	,
  GTK_PLOT_LINE_DOT_DOT_DASH	,
  GTK_PLOT_LINE_DOT_DASH_DASH	
} GtkPlotLineStyle;

typedef enum
{
  GTK_PLOT_CONNECT_NONE		,
  GTK_PLOT_CONNECT_STRAIGHT	,
  GTK_PLOT_CONNECT_SPLINE	,
  GTK_PLOT_CONNECT_HV_STEP	,
  GTK_PLOT_CONNECT_VH_STEP	,
  GTK_PLOT_CONNECT_MIDDLE_STEP	
} GtkPlotConnector;

enum
{
  GTK_PLOT_LABEL_NONE    	= 0,
  GTK_PLOT_LABEL_LEFT    	= 1 << 1,
  GTK_PLOT_LABEL_RIGHT    	= 1 << 2,
  GTK_PLOT_LABEL_TOP    	= 1 << 3,
  GTK_PLOT_LABEL_BOTTOM    	= 1 << 4
};

typedef enum
{
  GTK_PLOT_ERROR_DIV_ZERO,
  GTK_PLOT_ERROR_LOG_NEG
} GtkPlotError;

enum
{
  GTK_PLOT_AXIS_LEFT	,
  GTK_PLOT_AXIS_RIGHT	,
  GTK_PLOT_AXIS_TOP	,
  GTK_PLOT_AXIS_BOTTOM	
};

enum
{
  GTK_PLOT_LABEL_FLOAT	,
  GTK_PLOT_LABEL_EXP	
};

enum
{
  GTK_PLOT_TICKS_NONE		= 0,
  GTK_PLOT_TICKS_LEFT		= 1 << 1,
  GTK_PLOT_TICKS_RIGHT		= 1 << 2,
  GTK_PLOT_TICKS_UP		= 1 << 3,
  GTK_PLOT_TICKS_DOWN		= 1 << 4
};

struct _GtkPlotText
{
  gdouble x, y;
  gint angle; /* 0, 90, 180, 270 */
  GdkColor fg;
  GdkColor bg;
 
  gboolean transparent;

  gchar *font;
  gint height;

  gchar *text;

  gint justification;
};

struct _GtkPlotLine
{
  GtkPlotLineStyle line_style;

  gint line_width;
  GdkColor color;
};

struct _GtkPlotSymbol
{
  GtkPlotSymbolType symbol_type;
  GtkPlotSymbolStyle symbol_style;

  gint size;
  gint line_width;
  GdkColor color;
};

struct _GtkPlotTicks
{
  gint nticks;             /* Number of ticks */

  gdouble step;		   /* ticks step */
  gint *ticks;             /* ticks points */
  gdouble *value;          /* ticks values */

  gboolean set_limits;
  gdouble begin, end; 
};

struct _GtkPlotAxis
{
  gboolean is_visible;

  GtkPlotText title;
  gboolean title_visible;

  GtkOrientation orientation;
  
  GtkPlotScale scale_type;

  GtkPlotLine line;

  gdouble min;
  gdouble max;

  gint major_mask;
  gint minor_mask;
  gint ticks_length;
  gint ticks_width;

  gint labels_offset;

  GtkPlotText label_attr;

  gint label_precision;
  gint label_style;
  gint label_mask;
};

struct _GtkPlotData
{
  gboolean is_function;
  gboolean is_visible;
  gboolean show_legend;

  gchar *name;
  gchar *legend;
 
  GtkPlotSymbol symbol;
  GtkPlotLine line; 
  GtkPlotConnector line_connector;

  GtkPlotLine x_line;
  GtkPlotLine y_line;

  gboolean show_xerrbars;
  gboolean show_yerrbars;
  gint xerrbar_length;
  gint yerrbar_length;

  gint num_points;
  gdouble *x;
  gdouble *y;
  gdouble *dx;
  gdouble *dy;

  GtkPlotFunc function;
  gint x_step;
};

struct _GtkPlot
{
  GtkMisc misc;

  GdkDrawable *drawable;

  guint8 flags;

  GdkColor background;

  gboolean show_vmajor;
  gboolean show_vminor;
  gboolean show_hmajor;
  gboolean show_hminor;

  gboolean show_x0;
  gboolean show_y0;

 /* location and size in percentage of the widget's size */
  gdouble x, y, width, height;

  gdouble xmin, xmax;
  gdouble ymin, ymax;

  GtkPlotScale xscale, yscale;

  GtkPlotAxis bottom; 
  GtkPlotAxis top; 
  GtkPlotAxis left; 
  GtkPlotAxis right; 

  GtkPlotTicks xmajor, xminor;
  GtkPlotTicks ymajor, yminor;

  GtkPlotLine x0_line;
  GtkPlotLine y0_line;

  GtkPlotLine major_vgrid;
  GtkPlotLine minor_vgrid;
  GtkPlotLine major_hgrid;
  GtkPlotLine minor_hgrid;

  gdouble legends_x, legends_y; /* position in % */
  gint legends_width, legends_height; /* absolute size */

  gint legends_line_width;  
  gint legends_border_width;
  gint legends_shadow_width;
  gboolean show_legends;
  gboolean show_legends_border;
  gboolean show_legends_shadow;
  GtkPlotText legends_attr;

  GtkPlotData *active_dataset;

  GList *data_sets;
  GList *text;

  GtkPlotPC *pc;
};

struct _GtkPlotClass
{
  GtkMiscClass parent_class;

  void (* changed) (GtkPlot *plot);

  gint (* moved)   (GtkPlot *plot,
                    gdouble x, gdouble y);

  gint (* resized) (GtkPlot *plot,
                    gdouble width, gdouble height);

  gint (* plot_select_region_pixel) (GtkPlot *plot,
				     gint arg1, 
				     gint arg2, 
				     gint arg3, 
				     gint arg4, 
				     gint arg5, 
				     gint arg6, 
				     gdouble arg7, 
				     gdouble arg8, 
				     gdouble arg9, 
				     gdouble arg10);

  void (* error) (GtkPlot *plot, gint errno);
};

/* Plot */

guint		gtk_plot_get_type		(void);
GtkWidget*	gtk_plot_new			(GdkDrawable *drawable);
GtkWidget*	gtk_plot_new_with_size		(GdkDrawable *drawable,
                                                 gdouble width, gdouble height);
void		gtk_plot_set_drawable		(GtkPlot *plot,
						 GdkDrawable *drawable);
GdkDrawable *	gtk_plot_get_drawable		(GtkPlot *plot);
void		gtk_plot_get_position		(GtkPlot *plot,
						 gdouble *x, gdouble *y);
void		gtk_plot_get_size		(GtkPlot *plot,
						 gdouble *width, 
					  	 gdouble *height);
GtkAllocation 	gtk_plot_get_internal_allocation(GtkPlot *plot);
void		gtk_plot_set_background		(GtkPlot *plot, 
						 GdkColor background);
void 		gtk_plot_paint                  (GtkWidget *widget,
                                                 GdkRectangle *area);

void		gtk_plot_refresh		(GtkPlot *plot,
						 GdkRectangle *area);
void		gtk_plot_move		        (GtkPlot *plot,
						 gdouble x, gdouble y);
void		gtk_plot_resize		        (GtkPlot *plot,
						 gdouble width, gdouble height);
void		gtk_plot_move_resize		(GtkPlot *plot,
						 gdouble x, gdouble y,
						 gdouble width, gdouble height);
void		gtk_plot_get_pixel		(GtkPlot *plot,
                                                 gdouble xx, gdouble yy,
                                                 gint *x, gint *y);
void		gtk_plot_get_point		(GtkPlot *plot,
                                                 gint x, gint y,
                                                 gdouble *xx, gdouble *yy);
void		gtk_plot_set_xrange		(GtkPlot *plot,
						 gdouble xmin, gdouble xmax);
void		gtk_plot_set_yrange		(GtkPlot *plot,
						 gdouble ymin, gdouble ymax);
void		gtk_plot_set_range		(GtkPlot *plot,
						 gdouble xmin, gdouble xmax,
						 gdouble ymin, gdouble ymax);
void		gtk_plot_get_xrange		(GtkPlot *plot,
						 gdouble *xmin, gdouble *xmax);
void		gtk_plot_get_yrange		(GtkPlot *plot,
						 gdouble *ymin, gdouble *ymax);
void 		gtk_plot_set_xscale		(GtkPlot *plot,
						 GtkPlotScale scale_type);
void 		gtk_plot_set_yscale		(GtkPlot *plot,
						 GtkPlotScale scale_type);
GtkPlotScale 	gtk_plot_get_xscale		(GtkPlot *plot);
GtkPlotScale 	gtk_plot_get_yscale		(GtkPlot *plot);
void		gtk_plot_put_text		(GtkPlot *plot,
						 gdouble x, gdouble y, 
                                                 gint angle,
						 gchar *font,	
                                                 gint height,
						 GdkColor *foreground,
						 GdkColor *background,
						 gint justification,
                                                 gchar *text); 
gint            gtk_plot_remove_text            (GtkPlot *plot, char *text);
void		gtk_plot_text_get_size		(GtkPlotText text,
						 gint *width, gint *height);
/* Axis */

GtkPlotAxis *   gtk_plot_get_axis               (GtkPlot *plot, gint axis);
void		gtk_plot_axis_set_visible	(GtkPlot *plot, 
						 gint axis,
                                                 gboolean visible);
gboolean	gtk_plot_axis_get_visible	(GtkPlot *plot, 
						 gint axis);
void		gtk_plot_axis_set_title		(GtkPlot *plot, 
						 gint axis,
						 gchar *title);
void		gtk_plot_axis_show_title	(GtkPlot *plot, 
						 gint axis);
void		gtk_plot_axis_hide_title	(GtkPlot *plot, 
						 gint axis);
void		gtk_plot_axis_move_title	(GtkPlot *plot, 
						 gint axis,
						 gint angle,
						 gdouble x, gdouble y);
void		gtk_plot_axis_justify_title	(GtkPlot *plot, 
						 gint axis,
						 gint justification);
void		gtk_plot_axis_set_attributes 	(GtkPlot *plot,
						 gint axis,
						 gint width,
						 GdkColor color);
void		gtk_plot_axis_set_ticks		(GtkPlot *plot,
						 GtkOrientation orientation,
						 gdouble major_ticks,
						 gdouble minor_ticks);
void		gtk_plot_axis_set_ticks_length	(GtkPlot *plot,
						 gint axis,
						 gint length);
void		gtk_plot_axis_set_ticks_width	(GtkPlot *plot,
						 gint axis,
						 gint width);
void		gtk_plot_axis_show_ticks	(GtkPlot *plot,
						 gint axis,
                                                 gint major_mask,
						 gint minor_mask);
void		gtk_plot_axis_set_ticks_limits	(GtkPlot *plot,
						 GtkOrientation orientation,
						 gdouble begin, gdouble end);
void		gtk_plot_axis_unset_ticks_limits(GtkPlot *plot,
						 GtkOrientation orientation);
void		gtk_plot_axis_show_labels	(GtkPlot *plot, 
						 gint axis,
						 gint labels_mask);
void		gtk_plot_axis_labels_set_attributes	(GtkPlot *plot,
						 	 gint axis,
							 gchar *font,
							 gint height,
							 GdkColor foreground,
							 GdkColor background);
void		gtk_plot_axis_labels_set_numbers	(GtkPlot *plot,
					 		 gint axis,
							 gint style,
							 gint precision); 
/* Grids */
void		gtk_plot_set_x0_visible			(GtkPlot *plot, 
							 gboolean visible);
gboolean 	gtk_plot_get_x0_visible			(GtkPlot *plot);
void 		gtk_plot_set_y0_visible			(GtkPlot *plot, 
							 gboolean visible);
gboolean 	gtk_plot_get_y0_visible			(GtkPlot *plot);
void		gtk_plot_set_grids_visible		(GtkPlot *plot,
                				         gboolean vmajor, 
						 	 gboolean vminor,
                        				 gboolean hmajor,
							 gboolean hminor);
void		gtk_plot_get_grids_visible		(GtkPlot *plot,
                         				 gboolean *vmajor, 
							 gboolean *vminor,
                         				 gboolean *hmajor, 
							 gboolean *hminor);
void		gtk_plot_set_y0line_attributes 	(GtkPlot *plot,
						 GtkPlotLineStyle style,
						 gint width,
						 GdkColor color);
void		gtk_plot_set_x0line_attributes 	(GtkPlot *plot,
						 GtkPlotLineStyle style,
						 gint width,
						 GdkColor color);
void		gtk_plot_set_major_vgrid_attributes 	(GtkPlot *plot,
						 	 GtkPlotLineStyle style,
						 	 gint width,
						 	 GdkColor color);
void		gtk_plot_set_minor_vgrid_attributes 	(GtkPlot *plot,
						 	 GtkPlotLineStyle style,
						 	 gint width,
						 	 GdkColor color);
void		gtk_plot_set_major_hgrid_attributes 	(GtkPlot *plot,
						 	 GtkPlotLineStyle style,
						 	 gint width,
						 	 GdkColor color);
void		gtk_plot_set_minor_hgrid_attributes 	(GtkPlot *plot,
						 	 GtkPlotLineStyle style,
						 	 gint width,
						 	 GdkColor color);
/* Legends */

void 		gtk_plot_show_legends 		(GtkPlot *plot);
void 		gtk_plot_hide_legends 		(GtkPlot *plot);
void 		gtk_plot_show_legends_border 	(GtkPlot *plot,
                                                 gboolean show_shadow,
                                                 gint shadow_width);
void 		gtk_plot_hide_legends_border	(GtkPlot *plot);
void		gtk_plot_legends_move		(GtkPlot *plot,
						 gdouble x, gdouble y);
void		gtk_plot_legends_get_position	(GtkPlot *plot,
						 gdouble *x, gdouble *y);
GtkAllocation	gtk_plot_legends_get_allocation	(GtkPlot *plot);
void		gtk_plot_legends_set_attributes (GtkPlot *plot,
						 gchar *font,
						 gint height,
						 GdkColor *foreground,
						 GdkColor *background);
/* Data Sets */

GtkPlotData *   gtk_plot_dataset_new    	();
void 		gtk_plot_add_dataset		(GtkPlot *plot,
						 GtkPlotData *dataset);
gint 		gtk_plot_remove_dataset		(GtkPlot *plot,
						 GtkPlotData *dataset);
GtkPlotData * 		gtk_plot_add_function	(GtkPlot *plot,
						 GtkPlotFunc function);
/* use gc == NULL for default */
void 		gtk_plot_draw_dataset		(GtkPlot *plot,
					         GdkGC *gc,
						 GtkPlotData *data);
void 		gtk_plot_dataset_set_points	(GtkPlotData *data,
						 gdouble *x, gdouble *y,
						 gdouble *dx, gdouble *dy,
                                                 gint num_points);
void 		gtk_plot_dataset_get_points	(GtkPlotData *data,
						 gdouble *x, gdouble *y,
						 gdouble *dx, gdouble *dy,
                                                 gint *num_points);
void 		gtk_plot_dataset_set_x		(GtkPlotData *data,
						 gdouble *x); 
void 		gtk_plot_dataset_set_y		(GtkPlotData *data,
						 gdouble *y); 
void 		gtk_plot_dataset_set_dx		(GtkPlotData *data,
						 gdouble *dx); 
void 		gtk_plot_dataset_set_dy		(GtkPlotData *data,
						 gdouble *dy); 
gdouble * 	gtk_plot_dataset_get_x		(GtkPlotData *data, 
                                                 gint *num_points);
gdouble * 	gtk_plot_dataset_get_y		(GtkPlotData *data, 
                                                 gint *num_points);
gdouble * 	gtk_plot_dataset_get_dx		(GtkPlotData *data, 
                                                 gint *num_points);
gdouble * 	gtk_plot_dataset_get_dy		(GtkPlotData *data, 
                                                 gint *num_points);
void		gtk_plot_dataset_set_numpoints  (GtkPlotData *data,
                                                 gint num_points);
gint		gtk_plot_dataset_get_numpoints  (GtkPlotData *data);
void		gtk_plot_dataset_set_symbol     (GtkPlotData *data,
                                                 GtkPlotSymbolType type,
                                                 GtkPlotSymbolStyle style,
						 gint size,
						 gint line_width,
						 GdkColor color);
void		gtk_plot_dataset_get_symbol     (GtkPlotData *data,
                                                 gint *type,
                                                 gint *style,
						 gint *size,
						 gint *line_width,
						 GdkColor *color);
void		gtk_plot_dataset_set_connector  (GtkPlotData *data,
						 GtkPlotConnector connector); 
gint		 gtk_plot_dataset_get_connector  (GtkPlotData *data);
void		gtk_plot_dataset_set_line_attributes 	(GtkPlotData *data,
						 	 GtkPlotLineStyle style,
						 	 gint width,
						 	 GdkColor color);
void		gtk_plot_dataset_get_line_attributes 	(GtkPlotData *data,
						 	 gint *style,
						 	 gint *width,
						 	 GdkColor *color);
void		gtk_plot_dataset_set_xy_attributes 	(GtkPlotData *data,
						 	 GtkPlotLineStyle style,
						 	 gint width,
						 	 GdkColor color);
void		gtk_plot_dataset_show_xerrbars  	(GtkPlotData *data);
void		gtk_plot_dataset_show_yerrbars 	 	(GtkPlotData *data);
void		gtk_plot_dataset_hide_xerrbars  	(GtkPlotData *data);
void		gtk_plot_dataset_hide_yerrbars  	(GtkPlotData *data);
void		gtk_plot_dataset_set_legend     	(GtkPlotData *data,
                                                	 gchar *legend);
void 		gtk_plot_dataset_show_legend 		(GtkPlotData *data);
void 		gtk_plot_dataset_hide_legend 		(GtkPlotData *data);
void		gtk_plot_dataset_set_name       	(GtkPlotData *data,
                                                	 gchar *name);
void		gtk_plot_show_dataset			(GtkPlotData *data);
void		gtk_plot_hide_dataset			(GtkPlotData *data);


#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __GTK_PLOT_H__ */
