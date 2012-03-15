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

import android.app.Service;
import android.appwidget.AppWidgetManager;
import android.content.BroadcastReceiver;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.TextView;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.app.PendingIntent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.PowerManager;
import android.os.IBinder;
import android.widget.RemoteViews;
import android.text.format.Time;
import android.util.Log;

public class MyBatteryReceiver extends BroadcastReceiver
{
    //
    //-----------------------------------------------------------
    //    
    public void onReceive(Context context, Intent intent)
    {
	//	Log.i(TAG, "##### I received intent: " + action);
	String action = intent.getAction();
	BiondApp myApp=myApp(context);
	if (action.equals(Intent.ACTION_BATTERY_CHANGED))
	    {
		int level  = intent.getIntExtra("level", 0);
		int status = intent.getIntExtra("status", BatteryManager.BATTERY_STATUS_UNKNOWN);
		RemoteViews updateViews = new RemoteViews(context.getPackageName(), 
							  myApp.LAYOUT);
		myApp.displayInfo(context, updateViews, level, status, myApp.batteryServiceIsFresh);
		//		myApp(context).notify(context,level);
	    }
	if (myApp.batteryServiceIsFresh==true)
	    {
		//		Log.i("Bat:Rec:", "##### Fresh notification issued");
		int level  = intent.getIntExtra("level", 0);
		int status = intent.getIntExtra("status", BatteryManager.BATTERY_STATUS_UNKNOWN);
		RemoteViews updateViews = new RemoteViews(context.getPackageName(), 
							  myApp.LAYOUT);
		myApp.displayInfo(context, updateViews, level, status, myApp.batteryServiceIsFresh);
		myApp.batteryServiceIsFresh=false;
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public BiondApp myApp(Context context)
    {
	return (BiondApp)context.getApplicationContext();
    }
}