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

import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Notification;
import android.app.Application;
import android.appwidget.AppWidgetManager;
import android.content.ComponentName;
import android.content.IntentFilter;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;
import android.os.BatteryManager;
import android.graphics.Color;
import java.util.TimerTask;
import android.os.Handler;
import java.lang.Integer;
import android.util.Log;
import java.util.Timer;

public class BiondApp extends Application 
{
    public final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
    public static int blinkDelay=100, blinkColor=Color.GREEN, normalColor=Color.WHITE;
    public static String batteryStatus;
    public static Boolean batteryServiceIsFresh=true, broadcastMode_g=true;
    public static RemoteViews views_g=null;
    public final String ACTION_TOGGLE_BUTTON="toggleButton";
    public final String ACTION_NULL="NULL";

    private static int oldbatterylevel = 0, oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;
    private final  CharSequence contentTitle = "Battery Level";
    private Context thisContext;

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
		//Log.i("writting to screen","rollingNotify = "+rollingNotify);
		manager.updateAppWidget(myComponentName, updateViews);
		notify(context,batteryLevel,rollingNotify);
	    }
    }
    //
    //-----------------------------------------------------------------------------------
    //
    public void displayInfo(Context context, RemoteViews views, int level, int status, 
			    Boolean forceDisplay)
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
		
		if (oldstatus == BatteryManager.BATTERY_STATUS_CHARGING)
		    batteryStatus = "Charging"; 
		else if (oldstatus == BatteryManager.BATTERY_STATUS_DISCHARGING)
		    batteryStatus = "Discharging";
		else if (oldstatus == BatteryManager.BATTERY_STATUS_NOT_CHARGING)
		    batteryStatus = "Not charging";
		else if (oldstatus == BatteryManager.BATTERY_STATUS_FULL)
		    batteryStatus = "Full";
		else
		    batteryStatus = "";
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
	NotificationManager mNotificationManager = 
	    (NotificationManager) context.getSystemService(ns);

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
	Intent notificationIntent = 
	    new Intent(context, MyBatteryReceiver.class);
	PendingIntent contentIntent = 
	    PendingIntent.getBroadcast(context, 0, notificationIntent, 0);

	notification.setLatestEventInfo(context, contentTitle, 
					contentText, contentIntent);
	int HELLO_ID=1;
	mNotificationManager.notify(HELLO_ID, notification);
    }
    //
    //-----------------------------------------------------------
    //    
    public void globalUpdateWidget(Context context, RemoteViews views_l, 
				   Boolean makeNewView)
    {
	//	Log.i("Biond: ", "#####localUpdateWidget called");

	if (makeNewView)
	    views_g = views_l;

	Intent batteryIntent = context.getApplicationContext().registerReceiver
	    (null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));

	int level = batteryIntent.getIntExtra("level", -1);
	int status = batteryIntent.getIntExtra("status",-1);
	//	Log.i("locaUpdate: ", "Level = " + level);
	displayInfo(context, views_l, level, status, false);

	gBlink(context, R.id.level, 
	       blinkColor, normalColor, blinkDelay);
    }   
    //
    //-----------------------------------------------------------
    //    
    public RemoteViews gBuildView(Context context, RemoteViews views_l, Boolean makeNewView,
				  Boolean broadcastMode)
    {
	//Log.i("Biond: ", "gBuildView " + broadcastMode);

    	String modeStr;
    	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);

	if (makeNewView)
	    views_l =  new RemoteViews(context.getPackageName(),
				       LAYOUT);

    	ComponentName thisWidget = new ComponentName(context,
    						     BiondWidgetProvider.class);
    	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
    	if (broadcastMode_g.equals(true))
    	    {
    		gUnregisterForClick(context,   views_l);
    		gRegisterForBroadcast(context, views_l);

    		modeStr="";        views_l.setTextViewText(R.id.blank, modeStr);

    		views_l.setTextColor(R.id.mode_auto,Color.GREEN);
    		modeStr="Auto";    views_l.setTextViewText(R.id.mode_auto, modeStr);

    		views_l.setTextColor(R.id.mode_manual,Color.LTGRAY);
    		modeStr="Manual";  views_l.setTextViewText(R.id.mode_manual, modeStr);
    	    }
    	else
    	    {
    		gUnRegisterForBroadcast(context, views_l);
    		gRegisterForClick(context, views_l,broadcastMode);

    		modeStr="";       views_l.setTextViewText(R.id.blank, modeStr);

    		views_l.setTextColor(R.id.mode_manual,Color.GREEN);
    		modeStr="Manual"; views_l.setTextViewText(R.id.mode_manual, modeStr);

    		views_l.setTextColor(R.id.mode_auto,Color.LTGRAY);
    		modeStr="Auto";   views_l.setTextViewText(R.id.mode_auto, modeStr);
    	    }
	paperPusher.updateAppWidget(thisWidget, views_l);

    	return views_l;
    }
    //
    //-----------------------------------------------------------
    //    
    public void gUnRegisterForBroadcast(Context context, RemoteViews views)
    {
	//	Log.i("Biond: ", "#####unregisteringBroadcast");
	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);

	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		context.stopService(new Intent(context, MyBatteryService.class));
		//
		// The following service is required to always be
		// running if the TextView is to be automatically
		// updated on screen orientation change when in Manual
		// mode.  It is only required for this purpose when in
		// Manual mode.  If this service is stopped in Manual
		// mode (for optimization), then the app has to tapped
		// once on screen orientation change in Manual mode.
		// Not a big deal and half-way reasonable behaviour,
		// in case having this service run all the time
		// becomes an issue.

		// context.stopService(new Intent(context, MyScreenService.class));
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void gRegisterForBroadcast(Context context, RemoteViews views)
    {
	//	Log.i("Biond: ", "#####registeringBroadcast");
	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		context.startService(new Intent(context, MyBatteryService.class));
		context.startService(new Intent(context, MyScreenService.class));
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void gUnregisterForClick(Context context, RemoteViews views)
    {
	//	Log.i("Biond: ", "#####unregisteringOnClick");
	Intent intent = new Intent(context, BiondWidgetProvider.class);
	intent.setAction(ACTION_NULL);
	PendingIntent pendingIntent = 
	    PendingIntent.getBroadcast(context, 0, intent, 0);
	views.setOnClickPendingIntent(R.id.level, pendingIntent);
	views.setOnClickPendingIntent(R.id.status, pendingIntent);
	//	views.setOnClickPendingIntent(R.id.blank, pendingIntent);
	//	views.setOnClickPendingIntent(R.id.mode, pendingIntent);
    }
    //
    //-----------------------------------------------------------
    //    
    public void gRegisterForClick(Context context, RemoteViews views, 
				  Boolean registerOnlyMode)
    {
	//Log.i("Biond: ", "#####registeringOnClick");
	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		if (registerOnlyMode.equals(false))
		{
		    Intent intent = new Intent(context, BiondWidgetProvider.class);
		    intent.setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE);
		    //intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, appWidgetIds);
		    intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, allWidgetIds);
		    PendingIntent pendingIntent = PendingIntent.getBroadcast
			(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

		    views.setOnClickPendingIntent(R.id.level,  pendingIntent);
		    views.setOnClickPendingIntent(R.id.status, pendingIntent);
		    //		    views.setOnClickPendingIntent(R.id.blank,   pendingIntent);
		    //		    views.setOnClickPendingIntent(R.id.mode,   pendingIntent);
		}
		{
		    Intent intent = new Intent(context, BiondWidgetProvider.class);
		    intent.setAction(ACTION_TOGGLE_BUTTON);
		    PendingIntent pendingIntent = PendingIntent.getBroadcast
			(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

		    views.setOnClickPendingIntent(R.id.mode_auto,   pendingIntent);
		    views.setOnClickPendingIntent(R.id.mode_manual,   pendingIntent);
		    //		    views.setOnClickPendingIntent(R.id.button, pendingIntent);
		}
		paperPusher.updateAppWidget(widgetId, views);
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void gBlink(Context context, final int textViewId, int blinkColor, 
		      final int normalColor, int delay)
    {
	//	Log.i("Biond: ", "#####blink called");
	//	views_p.setInt(R.id.level, "setBackgroundColor", android.graphics.Color.WHITE);
	//	views_p.setTextColor(R.id.level,blinkcolor);
	// views_p=null;
	//	views_p = new RemoteViews(context.getPackageName(), myApp(context).LAYOUT);
	views_g.setTextColor(textViewId,blinkColor);
	thisContext = context;
	final Handler handler = new Handler(); 
	Timer t = new Timer(); 
	t.schedule(new TimerTask() 
	    { 
		public void run() 
		{ 
		    handler.post(new Runnable() 
			{ 
			    public void run() 
			    { 
				//views_p.setTextColor(R.id.level,normalcolor);
				views_g.setTextColor(textViewId,normalColor);
				ComponentName thisWidget = 
				    new ComponentName(thisContext,
						      BiondWidgetProvider.class);
				AppWidgetManager appWidgetManager = 
				    AppWidgetManager.getInstance(thisContext);
				int[] allWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);
				for (int widgetId : allWidgetIds) 
				    appWidgetManager.updateAppWidget(widgetId, 
								     views_g);
			    } 
			}
			); 
		} 
	    }, delay); 
    }
}

