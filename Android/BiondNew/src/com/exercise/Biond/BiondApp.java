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
import android.widget.Toast;
import android.os.Message;

public class BiondApp extends Application 
{
    public final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
    public static int blinkDelay=100, blinkColor=Color.GREEN, normalColor=Color.WHITE;
    public static String batteryStatus;
    public static Boolean batteryServiceIsFresh=true, modeAuto_g=true;
    public static RemoteViews views_g=null;
    public final String ACTION_TOGGLE_BUTTON="toggleButton";
    public final String ACTION_NULL="NULL";
    public final String ACTION_CLICK="Click";
    private static int DOUBLE_CLICK_DELAY = 250;
    private final int BIOND_NOTIFICATION_ID=1;
    

    public static int oldbatterylevel = 0, oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;
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
    public Boolean displayInfo(Context context, RemoteViews views, int level, int status, 
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
	return doit;
    }
    //
    //-----------------------------------------------------------
    //    
    public void cancelNotification(Context context)
    {
    	((NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE)).
	    cancel(BIOND_NOTIFICATION_ID);
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

	//	CharSequence contentText = Integer.toString(level)+"%";
	Intent notificationIntent = 
	    new Intent(context, BiondBatteryReceiver.class);
	PendingIntent contentIntent = 
	    PendingIntent.getBroadcast(context, 0, notificationIntent, 0);

	notification.setLatestEventInfo(context, contentTitle, 
					tickerText, contentIntent);
	mNotificationManager.notify(BIOND_NOTIFICATION_ID, notification);
    }
    //
    //-----------------------------------------------------------
    //    
    public void globalUpdateWidget(Context context, RemoteViews views_l, 
				   Boolean makeNewView)
    {
	//	Log.i("Biond: ", "#####globalUpdateWidget called");

	if (makeNewView)
	    views_g = views_l;

	Intent batteryIntent = context.getApplicationContext().registerReceiver
	    (null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));

	int level = batteryIntent.getIntExtra("level", -1),
	    status = batteryIntent.getIntExtra("status",-1),
	    scale = batteryIntent.getIntExtra("scale",-1);
	level *= 100/scale;
	//	if (displayInfo(context, views_l, level, status, false))
	if (displayInfo(context, views_l, level, status, !modeAuto_g))
	    {
		//		Log.i("Blinking","blink...blink...");
		gBlink(context, R.id.level, 
		       blinkColor, normalColor, blinkDelay);
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
	//	installDoubleClick(context, views);//This installs onDoubleClick() for level
	views.setOnClickPendingIntent(R.id.level, pendingIntent);
	views.setOnClickPendingIntent(R.id.status, pendingIntent);
    }
    //
    //-----------------------------------------------------------
    //    
    public void gRegisterForClick(Context context, RemoteViews views)
    {
	//Log.i("Biond: ", "#####registeringOnClick");
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);

	Intent intent = new Intent(context, BiondWidgetProvider.class);
	intent.setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE);
	PendingIntent pendingIntent = PendingIntent.getBroadcast
	    (context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

	//	installDoubleClick(context, views);//This installs onDoubleClick() for level
	views.setOnClickPendingIntent(R.id.level,  pendingIntent);
	views.setOnClickPendingIntent(R.id.status, pendingIntent);

	globalUpdateWidget(context, views, true);
	//	paperPusher.updateAppWidget(thisWidget, views);
    }
    //
    //-----------------------------------------------------------
    //    
    public void gRegisterButtons(Context context, RemoteViews views)
    {
	Intent intent = new Intent(context, BiondWidgetProvider.class);
	intent.setAction(ACTION_TOGGLE_BUTTON);
	PendingIntent pendingIntent = PendingIntent.getBroadcast
	    (context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

	views.setOnClickPendingIntent(R.id.mode_auto,   pendingIntent);
	views.setOnClickPendingIntent(R.id.mode_manual,   pendingIntent);
	//views.setOnClickPendingIntent(R.id.button, pendingIntent);
    }
    //
    //-----------------------------------------------------------
    //    
    public void gBlink(Context context, final int textViewId, int blinkColor, 
		      final int normalColor, int delay)
    {
	//Log.i("Biond: ", "#####blink called");
	//views_p.setInt(R.id.level, "setBackgroundColor", 
	//               android.graphics.Color.WHITE);
	//views_p.setTextColor(R.id.level,blinkcolor);
	//views_p=null;
	//views_p = new RemoteViews(context.getPackageName(), 
	//                          myApp(context).LAYOUT);
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
    //
    //-----------------------------------------------------------
    //    
    public RemoteViews gUpdateButtons(Context context, RemoteViews views_l, 
				     Boolean makeNewView, Boolean modeAuto)
    {
	//Log.i("Biond: ", "gBuildView " + broadcastMode);
    	String modeStr;
    	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);

	if (makeNewView)
	    views_l =  new RemoteViews(context.getPackageName(),
				       LAYOUT);

    	ComponentName thisWidget = new ComponentName(context,
    						     BiondWidgetProvider.class);

    	if (modeAuto_g.equals(true))
    	    {
    		modeStr="";        views_l.setTextViewText(R.id.blank, modeStr);

    		views_l.setTextColor(R.id.mode_auto,Color.GREEN);
    		modeStr="Auto";    views_l.setTextViewText(R.id.mode_auto, modeStr);

    		views_l.setTextColor(R.id.mode_manual,Color.LTGRAY);
    		modeStr="Manual";  views_l.setTextViewText(R.id.mode_manual, modeStr);
    	    }
    	else
    	    {
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
    public void installDoubleClick(Context context, RemoteViews views)
    {
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	AppWidgetManager appWidgetManager = AppWidgetManager.getInstance(context);
	int[] appWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);

	//	Intent intent = new Intent(context, getClass());
	Intent intent = new Intent(context, BiondWidgetProvider.class);
	intent.setAction(ACTION_CLICK);
	PendingIntent pendingIntent = PendingIntent.getBroadcast(context, 0, intent, 0);

	views.setOnClickPendingIntent(R.id.level, pendingIntent);
	views.setOnClickPendingIntent(R.id.status, pendingIntent);

	appWidgetManager.updateAppWidget(appWidgetIds, views);

	context.getSharedPreferences("widget", 0).edit().putInt("clicks", 0).commit();
    }
    //
    //-----------------------------------------------------------
    //    
    // public void onDoubleClick(final Context context, Intent intent)
    // {
    // 	//	if (intent.getAction().equals("Click")) 
    // 	    {
    // 		int clickCount = context.
    // 		    getSharedPreferences("widget", Context.MODE_PRIVATE).
    // 		    getInt("clicks", 0);
    // 		context.getSharedPreferences("widget", Context.MODE_PRIVATE).
    // 		    edit().putInt("clicks", ++clickCount).commit();
	    
    // 		final Handler handler = new Handler() 
    // 		    {
    // 			public void handleMessage(Message msg) 
    // 			{
    // 			    int clickCount = context.
    // 				getSharedPreferences("widget", Context.MODE_PRIVATE).
    // 				getInt("clicks", 0);
			    
    // 			    if (clickCount > 1) 
    // 				{
    // 				    Intent myintent = new Intent(context,MenuActivity.class);
    // 				    myintent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
    // 				    context.startActivity(myintent);
    // 				    //FLAG_ACTIVITY_NEW_TASK
    // 				    Toast.makeText(context, "Whaat?", Toast.LENGTH_SHORT).show();
    // 				}
    // 			    else 
    // 				{
    // 				    gOnSingleClick(context);
    // 				    // Toast.makeText(context, "singleClick", Toast.LENGTH_SHORT).show();
    // 				}
			    
    // 			    context.getSharedPreferences("widget", Context.MODE_PRIVATE).
    // 				edit().putInt("clicks", 0).commit();
    // 			}
    // 		    };
		
    // 		if (clickCount == 1) new Thread() {
    // 			@Override public void run()
    // 			{
    // 			    try 
    // 				{
    // 				    synchronized(this) { wait(DOUBLE_CLICK_DELAY); }
    // 				    handler.sendEmptyMessage(0);
    // 				} 
    // 			    catch(InterruptedException ex) {}
    // 			}
    // 		    }.start();
    // 	    }
	
    // 	//	super.onReceive(context, intent);
    // }
    //
    //-----------------------------------------------------------
    //    
    public void gOnSingleClick(Context context)
    {
	// Looks like it is better to get a new RemoteViews than
	// re-use view_g.  Some docs suggest that without this it
	// might lead to mem. leak.  This way it is also certainly
	// faster.
	//	if (views_g == null) 
	    views_g = new RemoteViews(context.getPackageName(), LAYOUT);
	globalUpdateWidget(context, views_g, true);
	gUpdateButtons(context, views_g, false,!modeAuto_g);
    }
}

