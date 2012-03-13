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