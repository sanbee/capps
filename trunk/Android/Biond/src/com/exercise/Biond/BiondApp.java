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
import android.app.Application;
import android.app.NotificationManager;
import android.app.Notification;
import android.app.PendingIntent;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;
import android.graphics.Color;
import android.util.Log;
import android.os.BatteryManager;
import java.lang.Integer;

public class BiondApp extends Application 
{
    public final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
    public static int blinkDelay=100, blinkColor=Color.GREEN, normalColor=Color.WHITE;
    public static String batteryStatus;
    public static Boolean batteryServiceIsFresh=true;

    private static int oldbatterylevel = 0, oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;
    private final  CharSequence contentTitle = "Battery Level";

    //
    //-----------------------------------------------------------------------------------
    //
    public void globalUpdateAppWidget(Context context, int batteryLevel, 
				      String batteryStatus, RemoteViews updateViews,
				      Boolean writeToScreen, Boolean rollingNotify)
    {
	//Log.i("GlobalUpdate: ", "Level = " + batteryLevel + " Status = " + batteryStatus);

	updateViews.setTextColor(R.id.level, normalColor);

	updateViews.setTextViewText(R.id.level,  batteryLevel + "%");

	updateViews.setTextViewText(R.id.status, batteryStatus);

	updateViews.setProgressBar(R.id.progress_bar,100,batteryLevel,false);

	ComponentName myComponentName = new ComponentName(context, BiondWidgetProvider.class);
	AppWidgetManager manager = AppWidgetManager.getInstance(context);

	//
	// Change the visible displays: the screen and post a notification.
	//
	if (writeToScreen) 
	    {
		manager.updateAppWidget(myComponentName, updateViews);
		notify(context,batteryLevel,rollingNotify);
	    }
    }
    //
    //-----------------------------------------------------------------------------------
    //
    public void displayInfo(Context context, RemoteViews views, int level, int status, Boolean forceDisplay)
    {
	Boolean doit=(level != oldbatterylevel) || (status != oldstatus) || forceDisplay;
	//	Log.i("New level: "," = " + level + " " + oldbatterylevel + doit);
	if (doit)
	    {
		oldbatterylevel=level;
		oldstatus = status;
		
		if (level >= 30)                        normalColor=Color.WHITE;
		else if ((level < 30) && (level >= 20)) normalColor=Color.CYAN;
		else if ((level < 20) && (level >= 5))  normalColor=Color.YELLOW;
		else                                    normalColor=Color.RED;
		
		if (oldstatus == BatteryManager.BATTERY_STATUS_CHARGING)          batteryStatus = "Charging"; 
		else if (oldstatus == BatteryManager.BATTERY_STATUS_DISCHARGING)  batteryStatus = "Dis-charging";
		else if (oldstatus == BatteryManager.BATTERY_STATUS_NOT_CHARGING) batteryStatus = "Not charging";
		else if (oldstatus == BatteryManager.BATTERY_STATUS_FULL)         batteryStatus = "Full";
		else                                                              batteryStatus = "";
	    }
	globalUpdateAppWidget(context, level, batteryStatus, views,doit,forceDisplay);
    }
    //
    //-----------------------------------------------------------
    //    
    public void notify(Context context, int level,Boolean rollingNotify)
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
	Notification notification;
	if (rollingNotify)
	    notification = new Notification(icon,tickerText,when);
	else
	    notification = new Notification(icon,null,when);
	
	notification.flags |= Notification.FLAG_ONGOING_EVENT;
	notification.flags |= Notification.FLAG_NO_CLEAR;
	// notification.tickerView = new RemoteViews(context.getPackageName(), 
	// 					  myApp(context).LAYOUT);

	CharSequence contentText = Integer.toString(level)+"%";
	Intent notificationIntent = new Intent(context, MyBatteryReceiver.class);
	PendingIntent contentIntent = PendingIntent.getBroadcast(context, 0, notificationIntent, 0);

	notification.setLatestEventInfo(context, contentTitle, contentText, contentIntent);
	int HELLO_ID=1;
	mNotificationManager.notify(HELLO_ID, notification);
    }
}

