package com.exercise.Biond;

import android.content.BroadcastReceiver;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.widget.RemoteViews;
import android.app.PendingIntent;
import android.util.Log;
import android.os.BatteryManager;
import android.text.format.Time;
import android.widget.ProgressBar;
import android.os.Bundle;

public class BiondWidgetProvider extends AppWidgetProvider 
{
    //    private static final String TAG = "Biond Provider";
    private static final int LAYOUT=R.layout.biondwidget_layout;//_tablet_xlarge;

    private static int oldbatterylevel = 0;
    private static int oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;

    private static final String ACTION_TOGGLE_BUTTON="toggleButton";
    private static final String ACTION_NULL="NULL";
    private static int nvisits = 0;
    private static String statusStr ="", batteryLevel="";
    private RemoteViews views_p=null;
    private static Boolean broadcastMode=false;

    @Override public void onReceive(Context context, Intent intent)
    {
	String intentAction = intent.getAction();
	//	Log.i("onReceive: ","####called " + intentAction + " " + broadcastMode);
	if (intent.getAction().equals(ACTION_TOGGLE_BUTTON))
	    broadcastMode=!broadcastMode;
	AppWidgetManager gm = AppWidgetManager.getInstance(context);
	RemoteViews views_l =  new RemoteViews(context.getPackageName(),
					       LAYOUT);
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = gm.getAppWidgetIds(thisWidget);
	if (broadcastMode.equals(true))
	    {
		unregisterForClick(context, gm, allWidgetIds, views_l);
		registerForBroadcast(context, gm, allWidgetIds, views_l);
		views_l.setTextViewText(R.id.mode,
					"Auto");
	    }
	else
	    {
		unRegisterForBroadcast(context, gm, allWidgetIds, views_l);
		registerForClick(context, gm, allWidgetIds, views_l);
		views_l.setTextViewText(R.id.mode,
					"Manual");
	    }
	gm.updateAppWidget(thisWidget, views_l);
    	super.onReceive(context,intent);
    }
    // @Override public void onEnabled(Context context)
    // {
    // 	Log.i("OnEnabled: ","####called");
    // }
    @Override public void onUpdate(Context context, AppWidgetManager appWidgetManager,
				   int[] appWidgetIds) 
    {
	//	Log.i(TAG, "########onUpdate called " + broadcastMode);
	nvisits++;
	if (views_p == null)
	    views_p = new RemoteViews(context.getPackageName(), LAYOUT);

	localUpdateWidget(context,views_p);
	registerForClick(context, appWidgetManager, appWidgetIds, views_p);
    }
    //
    //-----------------------------------------------------------
    //    
    public void unRegisterForBroadcast(Context context, AppWidgetManager appWidgetManager,
				       int[] appWidgetIds, RemoteViews views)
    {
	//	Log.i(TAG, "unregisteringBroadcast");
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		context.stopService(new Intent(context, MyBatteryService.class));
		context.stopService(new Intent(context, MyScreenService.class));
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void registerForBroadcast(Context context, AppWidgetManager appWidgetManager,
				     int[] appWidgetIds, RemoteViews views)
    {
	//	Log.i(TAG, "registeringBroadcast");
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		appWidgetManager.updateAppWidget(widgetId, views);
		context.startService(new Intent(context, MyBatteryService.class));
		context.startService(new Intent(context, MyScreenService.class));
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void unregisterForClick(Context context, AppWidgetManager appWidgetManager,
				   int[] appWidgetIds, RemoteViews views)
    {
	//	Log.i(TAG, "unregisteringOnClick");
	Intent intent = new Intent(context, BiondWidgetProvider.class);
	intent.setAction(ACTION_NULL);
	PendingIntent pendingIntent = PendingIntent.getBroadcast
	    (context, 0, intent, 0);
	views.setOnClickPendingIntent(R.id.level, pendingIntent);
	views.setOnClickPendingIntent(R.id.status, pendingIntent);
	views.setOnClickPendingIntent(R.id.mode, pendingIntent);
    }
    //
    //-----------------------------------------------------------
    //    
    public void registerForClick(Context context, AppWidgetManager appWidgetManager,
				 int[] appWidgetIds, RemoteViews views)
    {
	//	Log.i(TAG, "registeringOnClick");
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		{
		    Intent intent = new Intent(context, BiondWidgetProvider.class);
		    intent.setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE);
		    intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, appWidgetIds);
		    PendingIntent pendingIntent = PendingIntent.getBroadcast
			(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

		    views.setOnClickPendingIntent(R.id.level,  pendingIntent);
		    views.setOnClickPendingIntent(R.id.status, pendingIntent);
		    views.setOnClickPendingIntent(R.id.mode, pendingIntent);
		}
		{
		    Intent intent = new Intent(context, BiondWidgetProvider.class);
		    intent.setAction(ACTION_TOGGLE_BUTTON);
		    PendingIntent pendingIntent = PendingIntent.getBroadcast
			(context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT);

		    views.setOnClickPendingIntent(R.id.button, pendingIntent);
		}
		appWidgetManager.updateAppWidget(widgetId, views);
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void localUpdateWidget(Context context, RemoteViews views)
    {
	//	RemoteViews views_l = new RemoteViews(context.getPackageName(), R.layout.biondwidget_layout);
	Intent batteryIntent = context.getApplicationContext().registerReceiver
	    (null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));


	int level = batteryIntent.getIntExtra("level", -1);
	int status = batteryIntent.getIntExtra("status",-1);
	if ((level != oldbatterylevel) || (status != oldstatus))
	    {
		oldbatterylevel = level;
		oldstatus = status;

		double scale = batteryIntent.getIntExtra("scale", -1)/100.0;
		if (scale > 0) level = (int)(level / scale);
		
		if (status == BatteryManager.BATTERY_STATUS_CHARGING)	        statusStr = "Charging"; 
		else if (status == BatteryManager.BATTERY_STATUS_DISCHARGING)   statusStr = "Dis-charging";
		else if (status == BatteryManager.BATTERY_STATUS_NOT_CHARGING)  statusStr = "Not charging";
		else if (status == BatteryManager.BATTERY_STATUS_FULL)          statusStr = "Full";
		//	Log.i("update","...run" + statusStr);
		// Time now = new Time();
		// now.setToNow();
		// String time = now.format("%H:%M:%S");
		
		views.setProgressBar(R.id.progress_bar,100,level,false);
		views.setTextViewText(R.id.level,
				      //			      "Bat. Status:\n" +
				      level + "%");
		
		views.setTextViewText(R.id.status,
				      statusStr);
		// + "\n" +
		// nvisits + "@"+time);
	    }
    }   
    //
    //-----------------------------------------------------------
    //    
    public void onDisabled(Context context)
    {
	if (broadcastMode)
	    {
		context.stopService(new Intent(context, MyBatteryService.class));
		context.stopService(new Intent(context, MyScreenService.class));
	    }
    }
}
