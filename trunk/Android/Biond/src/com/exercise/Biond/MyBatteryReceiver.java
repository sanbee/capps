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
import android.app.NotificationManager;
import android.app.Notification;
import android.graphics.Color;
import java.lang.Integer;

public class MyBatteryReceiver extends BroadcastReceiver
{
    //
    //-----------------------------------------------------------
    //    
    public void onReceive(Context context, Intent intent)
    {
	//	Log.i(TAG, "##### I received intent: " + action);
	String action = intent.getAction();
	if (action.equals(Intent.ACTION_BATTERY_CHANGED))
	    {
		int level = intent.getIntExtra("level", 0);
		int status = intent.getIntExtra("status", BatteryManager.BATTERY_STATUS_UNKNOWN);
		RemoteViews updateViews = new RemoteViews(context.getPackageName(), 
							  myApp(context).LAYOUT);
		myApp(context).displayInfo(context, updateViews, level, status);
		notify(context,level);

	    	// if ((level != oldbatterylevel) || (status != oldstatus))
	    	//     {
	    	// 	//			Log.i("New level: "," = " + batterylevel + " " + oldbatterylevel);
		// 	String batteryStatus;
	    	// 	oldbatterylevel=level;
	    	// 	oldstatus = status;

	    	// 	// if (level >= 30) normalColor=Color.WHITE;
	    	// 	// else if ((level < 30) && (level >= 20))  normalColor=Color.CYAN;
	    	// 	// else if ((level < 20) && (level >= 5))  normalColor=Color.YELLOW;
	    	// 	// else normalColor=Color.RED;

	    	// 	if (status == BatteryManager.BATTERY_STATUS_CHARGING)          batteryStatus = "Charging"; 
	    	// 	else if (status == BatteryManager.BATTERY_STATUS_DISCHARGING)  batteryStatus = "Dis-charging";
	    	// 	else if (status == BatteryManager.BATTERY_STATUS_NOT_CHARGING) batteryStatus = "Not charging";
	    	// 	else if (status == BatteryManager.BATTERY_STATUS_FULL)         batteryStatus = "Full";
	    	// 	else                                                           batteryStatus = "";
	    	// 	updateAppWidget(context, level, batteryStatus);
	    	//     }
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void notify(Context context, int level)
    {
	//	Log.i("notify", notification.toString());

	String ns = Context.NOTIFICATION_SERVICE;
	NotificationManager mNotificationManager = (NotificationManager) context.getSystemService(ns);

	CharSequence tickerText = Integer.toString(level)+"%";
	long when = System.currentTimeMillis();
	int icon = R.drawable.icon;

	//
	// When tickerText is set to null, notification bar won't
	// scroll when a notifaction is posted.
	//
	Notification notification = new Notification(icon,null/*tickerText*/,when);
	notification.flags |= Notification.FLAG_ONGOING_EVENT;
	notification.flags |= Notification.FLAG_NO_CLEAR;
	// notification.tickerView = new RemoteViews(context.getPackageName(), 
	// 					  myApp(context).LAYOUT);

	CharSequence contentTitle = "Battery Level";

	CharSequence contentText = Integer.toString(level)+"%";
	Intent notificationIntent = new Intent(context, MyBatteryReceiver.class);
	PendingIntent contentIntent = PendingIntent.getBroadcast(context, 0, notificationIntent, 0);

	notification.setLatestEventInfo(context, contentTitle, contentText, contentIntent);
	int HELLO_ID=1;
	mNotificationManager.notify(HELLO_ID, notification);
    }
    //
    //-----------------------------------------------------------
    //    
    public BiondApp myApp(Context context)
    {
	return (BiondApp)context.getApplicationContext();
    }
}