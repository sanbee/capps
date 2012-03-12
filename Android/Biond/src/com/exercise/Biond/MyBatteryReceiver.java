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
    //    private final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
       private static int oldbatterylevel = 0;
       private static int oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;

    // private String batteryStatus ="";
    //    private static final int HELLO_ID = 1;
    //    private static final String TAG = "MyBatteryReceiver";

    //
    //-----------------------------------------------------------
    //    
    public void onReceive(Context context, Intent intent)
    {
	//	Log.i(TAG, "##### I received intent: " + action);

	String action = intent.getAction();
	// int level;

	if (action.equals(Intent.ACTION_BATTERY_CHANGED))
	    {
		int level = intent.getIntExtra("level", 0);
		int status = intent.getIntExtra("status", BatteryManager.BATTERY_STATUS_UNKNOWN);
		// RemoteViews updateViews = new RemoteViews(context.getPackageName(), 
		// 					  myApp(context).LAYOUT);
		// myApp(context).displayInfo(context, updateViews, level, status);



	    	if ((level != oldbatterylevel) || (status != oldstatus))
	    	    {
	    		//			Log.i("New level: "," = " + batterylevel + " " + oldbatterylevel);
			String batteryStatus;
	    		oldbatterylevel=level;
	    		oldstatus = status;

	    		// if (level >= 30) normalColor=Color.WHITE;
	    		// else if ((level < 30) && (level >= 20))  normalColor=Color.CYAN;
	    		// else if ((level < 20) && (level >= 5))  normalColor=Color.YELLOW;
	    		// else normalColor=Color.RED;

	    		if (status == BatteryManager.BATTERY_STATUS_CHARGING)          batteryStatus = "Charging"; 
	    		else if (status == BatteryManager.BATTERY_STATUS_DISCHARGING)  batteryStatus = "Dis-charging";
	    		else if (status == BatteryManager.BATTERY_STATUS_NOT_CHARGING) batteryStatus = "Not charging";
	    		else if (status == BatteryManager.BATTERY_STATUS_FULL)         batteryStatus = "Full";
	    		else                                                           batteryStatus = "";
	    		updateAppWidget(context, level, batteryStatus);
	    	    }
	    }
    }
    //
    //-----------------------------------------------------------
    //    
    public void updateAppWidget(Context context,int batterylevel, String batteryStatus)
    {
    	//	Log.i(TAG, "updateAppWidget czall no. " + nvisits);

    	RemoteViews updateViews = new RemoteViews(context.getPackageName(), 
    						  myApp(context).LAYOUT);
    	// Time now = new Time();
    	// now.setToNow();
    	// String time = now.format("%H:%M:%S");
    	updateViews.setTextViewText(R.id.level,
    				    //				    "Bat. Status:\n" +
    				    batterylevel + "%");

    	updateViews.setTextViewText(R.id.status,
    				    batteryStatus);
    				    // + "\n" +
    				    // nvisits+ "@"+time);
    	updateViews.setProgressBar(R.id.progress_bar,100,batterylevel,false);
    	// updateViews.setTextViewText(R.id.message,
    	// 			    "Updates: " + nvisits);
	
    	//		setupOnClickListener(context);
    	ComponentName myComponentName = new ComponentName(context, BiondWidgetProvider.class);
    	AppWidgetManager manager = AppWidgetManager.getInstance(context);
    	notify(context, batterylevel);
    	manager.updateAppWidget(myComponentName, updateViews);
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

	Notification notification = new Notification(icon,tickerText,when);
	notification.flags |= Notification.FLAG_ONGOING_EVENT;

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