/* GTK - The GIMP Toolkit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
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

/*
 * Modified by the GTK+ Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GTK+ Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GTK+ at ftp://ftp.gtk.org/pub/gtk/. 
 */

void
gtk_color_selection_hsv_to_rgb (double  h, double  s, double  v,
				double *r, double *g, double *b)
{
  int i;
  double f, w, q, t;

  if (s == 0.0)
    s = 0.000001;

  if (h == -1.0)
    {
      *r = v;
      *g = v;
      *b = v;
    }
  else
    {
      if (h == 360.0)
	h = 0.0;
      h = h / 60.0;
      i = (int) h;
      f = h - i;
      w = v * (1.0 - s);
      q = v * (1.0 - (s * f));
      t = v * (1.0 - (s * (1.0 - f)));

      switch (i)
	{
	case 0:
	  *r = v;
	  *g = t;
	  *b = w;
	  break;
	case 1:
	  *r = q;
	  *g = v;
	  *b = w;
	  break;
	case 2:
	  *r = w;
	  *g = v;
	  *b = t;
	  break;
	case 3:
	  *r = w;
	  *g = q;
	  *b = v;
	  break;
	case 4:
	  *r = t;
	  *g = w;
	  *b = v;
	  break;
	case 5:
	  *r = v;
	  *g = w;
	  *b = q;
	  break;
	}
    }
}


void
gtk_color_selection_rgb_to_hsv (double  r, double  g, double  b,
				double *h, double *s, double *v)
{
  double max, min, delta;

  max = r;
  if (g > max)
    max = g;
  if (b > max)
    max = b;

  min = r;
  if (g < min)
    min = g;
  if (b < min)
    min = b;

  *v = max;

  if (max != 0.0)
    *s = (max - min) / max;
  else
    *s = 0.0;

  if (*s == 0.0)
    *h = -1.0;
  else
    {
      delta = max - min;

      if (r == max)
	*h = (g - b) / delta;
      else if (g == max)
	*h = 2.0 + (b - r) / delta;
      else if (b == max)
	*h = 4.0 + (r - g) / delta;

      *h = *h * 60.0;

      if (*h < 0.0)
	*h = *h + 360;
    }
}
