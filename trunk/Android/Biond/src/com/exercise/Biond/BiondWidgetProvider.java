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
    //    private static final String TAG = "Biond Provider";
    // private final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
    private final String ACTION_TOGGLE_BUTTON="toggleButton";
    private final String ACTION_NULL="NULL";

    private static int oldbatterylevel = 0;
    private static int oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;
    //    private static int blinkDelay=200, blinkColor=Color.GREEN, normalColor=Color.WHITE;
    
    private static RemoteViews views_p=null;
    private static Boolean broadcastMode=true;

    private Toast myToast=null;
    //    private String statusStr ="", batteryLevel="";
    private Context thisContext;

    @Override public void onReceive(Context context, Intent intent)
    {
	Boolean modeChanged=false;
	String intentAction = intent.getAction();
	String mode;

	if (intent.getAction().equals(ACTION_TOGGLE_BUTTON))
	    {
		broadcastMode=!broadcastMode;
		modeChanged=true;
	    }
	//	Log.i("onReceive: ","####called " + intentAction + " " + broadcastMode);
	AppWidgetManager gm = AppWidgetManager.getInstance(context);

	RemoteViews views_l =  new RemoteViews(context.getPackageName(),
					       myApp(context).LAYOUT);
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = gm.getAppWidgetIds(thisWidget);
	if (broadcastMode.equals(true))
	    {
		unregisterForClick(context, gm, allWidgetIds, views_l);
		registerForBroadcast(context, gm, allWidgetIds, views_l);
		mode="Auto";
		views_l.setTextColor(R.id.mode_auto,Color.GREEN);
		views_l.setTextViewText(R.id.mode_auto, mode);
		mode="Manual";
		views_l.setTextColor(R.id.mode_manual,Color.LTGRAY);
		views_l.setTextViewText(R.id.mode_manual, mode);
	    }
	else
	    {
		unRegisterForBroadcast(context, gm, allWidgetIds, views_l);
		registerForClick(context, gm, allWidgetIds, views_l,broadcastMode);
		mode="Manual";
		views_l.setTextColor(R.id.mode_manual,Color.GREEN);
		views_l.setTextViewText(R.id.mode_manual, mode);
		mode="Auto";
		views_l.setTextColor(R.id.mode_auto,Color.LTGRAY);
		views_l.setTextViewText(R.id.mode_auto, mode);
	    }
	// if (modeChanged)
	//     {
	// 	if (!myToast.equals(null)) myToast.cancel();
	// 	myToast=Toast.makeText(context, mode,  Toast.LENGTH_SHORT);
		
	// 	myToast.show();
	//     }
	gm.updateAppWidget(thisWidget, views_l);
    	super.onReceive(context,intent);
    }
    // @Override public void onEnabled(Context context)
    // {
    // 	myApp = (BiondApp)context.getApplicationContext();
    //  	Log.i("OnEnabled: ","####called");
    // }
    @Override public void onUpdate(Context context, AppWidgetManager appWidgetManager,
				   int[] appWidgetIds) 
    {
	//	Log.i(TAG, "########onUpdate called " + broadcastMode);
	if (views_p == null)
	    views_p = new RemoteViews(context.getPackageName(), myApp(context).LAYOUT);

	localUpdateWidget(context,views_p);
	registerForClick(context, appWidgetManager, appWidgetIds, views_p,broadcastMode);
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
	//	views.setOnClickPendingIntent(R.id.blank, pendingIntent);
	//	views.setOnClickPendingIntent(R.id.mode, pendingIntent);
    }
    //
    //-----------------------------------------------------------
    //    
    public void registerForClick(Context context, AppWidgetManager appWidgetManager,
				 int[] appWidgetIds, RemoteViews views, Boolean registerOnlyMode)
    {
	//	Log.i(TAG, "registeringOnClick");
	ComponentName thisWidget = new ComponentName(context,
						     BiondWidgetProvider.class);
	int[] allWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);
	for (int widgetId : allWidgetIds) 
	    {
		if (registerOnlyMode.equals(false))
		{
		    Intent intent = new Intent(context, BiondWidgetProvider.class);
		    intent.setAction(AppWidgetManager.ACTION_APPWIDGET_UPDATE);
		    intent.putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, appWidgetIds);
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
	//	myApp(context).displayInfo(context, views, level, status);
	if ((level != oldbatterylevel) || (status != oldstatus))
	    {
		String statusStr="UNKNOWN";
		//		Log.i("update","...run" + statusStr);
		oldbatterylevel = level;
		oldstatus = status;

		double scale = batteryIntent.getIntExtra("scale", -1)/100.0;
		if (scale > 0) level = (int)(level / scale);
		
		if (level >= 30) myApp(context).normalColor=Color.WHITE;
		else if ((level < 30) && (level >= 20))  myApp(context).normalColor=Color.CYAN;
		else if ((level < 20) && (level >= 5))  myApp(context).normalColor=Color.YELLOW;
		else myApp(context).normalColor=Color.RED;
		
		if (status == BatteryManager.BATTERY_STATUS_CHARGING)	        statusStr = "Charging"; 
		else if (status == BatteryManager.BATTERY_STATUS_DISCHARGING)   statusStr = "Dis-charging";
		else if (status == BatteryManager.BATTERY_STATUS_NOT_CHARGING)  statusStr = "Not charging";
		else if (status == BatteryManager.BATTERY_STATUS_FULL)          statusStr = "Full";
		// Time now = new Time();
		// now.setToNow();
		// String time = now.format("%H:%M:%S");
		
		views.setProgressBar(R.id.progress_bar,100,level,false);

		views.setTextViewText(R.id.level, level + "%");

		views.setTextViewText(R.id.status, statusStr);
		// + "\n" +
	    }

	blink(context, R.id.level, 
	      myApp(context).blinkColor, 
	      myApp(context).normalColor, 
	      myApp(context).blinkDelay);
    }   
    //
    //-----------------------------------------------------------
    //    
    public void blink(Context context, final int textViewId, int blinkcolor, final int normalcolor, int delay)
    {
	//	views_p.setInt(R.id.level, "setBackgroundColor", android.graphics.Color.WHITE);
	//	views_p.setTextColor(R.id.level,blinkcolor);
	views_p.setTextColor(textViewId,blinkcolor);
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
				//				views_p.setTextColor(R.id.level,normalcolor);
				views_p.setTextColor(textViewId,normalcolor);
				ComponentName thisWidget = new ComponentName(thisContext,
									     BiondWidgetProvider.class);
				AppWidgetManager appWidgetManager = AppWidgetManager.getInstance(thisContext);
				int[] allWidgetIds = appWidgetManager.getAppWidgetIds(thisWidget);
				for (int widgetId : allWidgetIds) 
				    appWidgetManager.updateAppWidget(widgetId, views_p);
			    } 
			}
			); 
		} 
	    }, delay); 
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
    //
    //-----------------------------------------------------------
    //    
    public BiondApp myApp(Context context)
    {
	return (BiondApp)context.getApplicationContext();
    }

}
