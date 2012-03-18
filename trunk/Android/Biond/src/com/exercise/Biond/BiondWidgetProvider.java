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

import java.lang.Thread;
import java.lang.Object;
import java.util.Timer;
import java.util.TimerTask;
import android.util.Log;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Handler;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.widget.RemoteViews;
import android.text.format.Time;
import android.widget.ProgressBar;
import android.widget.Toast;
import android.graphics.Color;

public class BiondWidgetProvider extends AppWidgetProvider 
{
    private static RemoteViews views_p=null;
    //    private static Boolean broadcastMode_p=true;

    private Toast myToast=null;
    //
    //------------------------------------------------------------------
    //
    @Override public void onReceive(Context context, Intent intent)
    {
	//Log.i("Biond: ", "#####onReceive called");
	String intentAction = intent.getAction();
	String mode;
	BiondApp myapp=myApp(context);
	Boolean modeChanged=false;

    	super.onReceive(context,intent);
	if (intent.getAction().equals(myapp.ACTION_TOGGLE_BUTTON))
	    {
		myapp.broadcastMode_g=!myapp.broadcastMode_g;
		modeChanged=true;
	    }

	//	RemoteViews views_l = buildView(context, myApp(context).broadcastMode_g);
	myapp.views_g = myapp.gBuildView(context, myapp.views_g, 
					 (myapp.views_g == null), 
					 myapp.broadcastMode_g);
    }
    //
    //------------------------------------------------------------------
    //
    @Override public void onUpdate(Context context, AppWidgetManager appWidgetManager,
				   int[] appWidgetIds) 
    {
	//Log.i("onUpdate: ", "########onUpdate called " + myApp(context).broadcastMode_g);
	BiondApp myapp=myApp(context);
	if (myapp.views_g == null)
	    {
		myapp.views_g = new RemoteViews(context.getPackageName(), 
						myapp.LAYOUT);
		views_p = myapp.views_g;
	    }

	//	localUpdateWidget(context, myApp(context).views_g);
	if (myapp.broadcastMode_g.equals(false))
	    myapp.globalUpdateWidget(context,myapp.views_g,false);
	myapp.gRegisterForClick(context, 
				myapp.views_g, 
				myapp.broadcastMode_g);
    }
    //
    //-----------------------------------------------------------
    //    
    public void onDisabled(Context context)
    {
	//	Log.i("OnDisabled: ", "#### SBMon Disabled");
	//	if (views_p != null) views_p = null;
	if (myApp(context).broadcastMode_g)
	    {
		context.stopService(new Intent(context, MyBatteryService.class));
		context.stopService(new Intent(context, MyScreenService.class));
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public BiondApp myApp(Context context)
    {
	return (BiondApp)context.getApplicationContext();
    }
    //
    //-----------------------------------------------------------
    //    
}




    // //
    // //-----------------------------------------------------------
    // //    
    // public void unRegisterForBroadcast(Context context, RemoteViews views)
    // {
    // 	Log.i("Biond: ", "#####unregisteringBroadcast");
    // 	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);

    // 	ComponentName thisWidget = new ComponentName(context,
    // 						     BiondWidgetProvider.class);
    // 	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
    // 	for (int widgetId : allWidgetIds) 
    // 	    {
    // 		context.stopService(new Intent(context, MyBatteryService.class));
    // 		context.stopService(new Intent(context, MyScreenService.class));
    // 	    }
    // }
    // //
    // //-----------------------------------------------------------
    // //    
    // public void registerForBroadcast(Context context, RemoteViews views)
    // {
    // 	Log.i("Biond: ", "#####registeringBroadcast");
    // 	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);
    // 	ComponentName thisWidget = new ComponentName(context,
    // 						     BiondWidgetProvider.class);
    // 	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
    // 	for (int widgetId : allWidgetIds) 
    // 	    {
    // 		//		appWidgetManager.updateAppWidget(widgetId, views);
    // 		context.startService(new Intent(context, MyBatteryService.class));
    // 		context.startService(new Intent(context, MyScreenService.class));
    // 	    }
    // }
    // //
    // //-----------------------------------------------------------
    // //    
    // public void unregisterForClick(Context context, RemoteViews views)
    // {
    // 	Log.i("Biond: ", "#####unregisteringOnClick");
    // 	Intent intent = new Intent(context, BiondWidgetProvider.class);
    // 	intent.setAction(ACTION_NULL);
    // 	PendingIntent pendingIntent = 
    // 	    PendingIntent.getBroadcast(context, 0, intent, 0);
    // 	views.setOnClickPendingIntent(R.id.level, pendingIntent);
    // 	views.setOnClickPendingIntent(R.id.status, pendingIntent);
    // 	//	views.setOnClickPendingIntent(R.id.blank, pendingIntent);
    // 	//	views.setOnClickPendingIntent(R.id.mode, pendingIntent);
    // }
    // //
    // //-----------------------------------------------------------
    // //    
    // public void registerForClick(Context context, RemoteViews views, Boolean registerOnlyMode)
    // {
    // 	Log.i("Biond: ", "#####registeringOnClick");
    // 	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);
    // 	ComponentName thisWidget = new ComponentName(context,
    // 						     BiondWidgetProvider.class);
    // 	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
    // 	for (int widgetId : allWidgetIds) 
    // 	    {
    // 		if (registerOnlyMode.equals(false))
    // 		{
    // 		    Intent intent = new Intent(context, BiondWidgetProvider.class);
    // 		    intent.setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE);
    // 		    //intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, appWidgetIds);
    // 		    intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, allWidgetIds);
    // 		    PendingIntent pendingIntent = PendingIntent.getBroadcast
    // 			(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

    // 		    views.setOnClickPendingIntent(R.id.level,  pendingIntent);
    // 		    views.setOnClickPendingIntent(R.id.status, pendingIntent);
    // 		    //		    views.setOnClickPendingIntent(R.id.blank,   pendingIntent);
    // 		    //		    views.setOnClickPendingIntent(R.id.mode,   pendingIntent);
    // 		}
    // 		{
    // 		    Intent intent = new Intent(context, BiondWidgetProvider.class);
    // 		    intent.setAction(ACTION_TOGGLE_BUTTON);
    // 		    PendingIntent pendingIntent = PendingIntent.getBroadcast
    // 			(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

    // 		    views.setOnClickPendingIntent(R.id.mode_auto,   pendingIntent);
    // 		    views.setOnClickPendingIntent(R.id.mode_manual,   pendingIntent);
    // 		    //		    views.setOnClickPendingIntent(R.id.button, pendingIntent);
    // 		}
    // 		paperPusher.updateAppWidget(widgetId, views);
    // 	    }
    // }


    // //
    // //-----------------------------------------------------------
    // //    
    // public RemoteViews buildView(Context context, Boolean broadcastMode)
    // {
    // 	String modeStr;
    // 	AppWidgetManager paperPusher = AppWidgetManager.getInstance(context);

    // 	RemoteViews views_l =  new RemoteViews(context.getPackageName(),
    // 					       myApp(context).LAYOUT);
    // 	ComponentName thisWidget = new ComponentName(context,
    // 						     BiondWidgetProvider.class);
    // 	int[] allWidgetIds = paperPusher.getAppWidgetIds(thisWidget);
    // 	if (broadcastMode.equals(true))
    // 	    {
    // 		unregisterForClick(context, views_l);
    // 		registerForBroadcast(context, views_l);

    // 		modeStr="";        views_l.setTextViewText(R.id.blank, modeStr);

    // 		views_l.setTextColor(R.id.mode_auto,Color.GREEN);
    // 		modeStr="Auto";    views_l.setTextViewText(R.id.mode_auto, modeStr);

    // 		views_l.setTextColor(R.id.mode_manual,Color.LTGRAY);
    // 		modeStr="Manual";  views_l.setTextViewText(R.id.mode_manual, modeStr);
    // 	    }
    // 	else
    // 	    {
    // 		unRegisterForBroadcast(context,views_l);
    // 		registerForClick(context, views_l,broadcastMode);

    // 		modeStr="";       views_l.setTextViewText(R.id.blank, modeStr);

    // 		views_l.setTextColor(R.id.mode_manual,Color.GREEN);
    // 		modeStr="Manual"; views_l.setTextViewText(R.id.mode_manual, modeStr);

    // 		views_l.setTextColor(R.id.mode_auto,Color.LTGRAY);
    // 		modeStr="Auto";   views_l.setTextViewText(R.id.mode_auto, modeStr);
    // 	    }
    // 	paperPusher.updateAppWidget(thisWidget, views_l);

    // 	return views_l;
    // }
    // //
    // //-----------------------------------------------------------
    // //    
    // public void localUpdateWidget(Context context, RemoteViews views)
    // {
    // 	Log.i("Biond: ", "#####localUpdateWidget called");
    // 	// RemoteViews views_l = new RemoteViews(context.getPackageName(), 
    // 	// 				      R.layout.biondwidget_layout);
    // 	Intent batteryIntent = context.getApplicationContext().registerReceiver
    // 	    (null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));

    // 	int level = batteryIntent.getIntExtra("level", -1);
    // 	int status = batteryIntent.getIntExtra("status",-1);
    // 	//	Log.i("locaUpdate: ", "Level = " + level);
    // 	myApp(context).displayInfo(context, views, level, status, false);

    // 	blink(context, R.id.level, 
    // 	      myApp(context).blinkColor, 
    // 	      myApp(context).normalColor, 
    // 	      myApp(context).blinkDelay);
    // }   
