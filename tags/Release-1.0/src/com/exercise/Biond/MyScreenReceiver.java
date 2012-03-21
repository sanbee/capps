// $Id$
// ******************************************************************
// Copyright (c) 2012 S.Bhatnagar
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
// 
// History:
//       Dark ages version: March, 2012
package com.exercise.Biond;

import android.appwidget.AppWidgetManager;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;
import android.util.Log;

public class MyScreenReceiver extends BroadcastReceiver
{
    //    private static final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
    private static Boolean screenOn = true;
    private static int nScreenVisits=0;

    @Override public void onReceive(Context context, Intent intent)
    {
	nScreenVisits++;
	RemoteViews views_l = new RemoteViews(context.getPackageName(), myApp(context).LAYOUT);
	ComponentName myComponentName = new ComponentName(context, BiondWidgetProvider.class);
	AppWidgetManager manager = AppWidgetManager.getInstance(context);
	manager.updateAppWidget(myComponentName, views_l);

	if (intent.getAction().equals(Intent.ACTION_SCREEN_OFF)) 
	    {
		//		Log.i("####Screen is ", "off");
		screenOn = false;
		//
		// Stop the service that starts the receiver that
		// receivers the ACTION_BATTERY_CHANGED intents.
		//
		context.stopService(new Intent(context, MyBatteryService.class));
	    } 
	else if (intent.getAction().equals(Intent.ACTION_SCREEN_ON)) 
	    {
		//		Log.i("####Screen is ", "on");
		screenOn = true;
		//
		// Start the service that starts the receiver that
		// receivers the ACTION_BATTERY_CHANGED intents.
		//
		context.startService(new Intent(context, MyBatteryService.class));
		myApp(context).batteryServiceIsFresh=true;
	    }
    };
    //
    //-----------------------------------------------------------
    //    
    public BiondApp myApp(Context context)
    {
	return (BiondApp)context.getApplicationContext();
    }
};
