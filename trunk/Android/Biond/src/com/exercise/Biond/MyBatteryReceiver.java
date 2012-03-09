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

public class MyBatteryReceiver extends BroadcastReceiver
{
    private static int oldbatterylevel = 0;
    // private String batteryStatus ="";
    private static int nvisits = 0;
    //    private static final int HELLO_ID = 1;
    //    private static final String TAG = "MyBatteryReceiver";

    public void onReceive(Context context, Intent intent)
    {
	// String ns = Context.NOTIFICATION_SERVICE;
	// NotificationManager mNotificationManager = (NotificationManager) context.getSystemService(ns);

	// CharSequence tickerText = "Hello";
	// long when = System.currentTimeMillis();
	// int icon = R.drawable.icon;
	// Notification notification = new Notification(icon,tickerText,when);
	// Log.i(TAG, notification.toString());
	// CharSequence contentTitle = "My notification";
	// CharSequence contentText = "Hello World!";
	// Intent notificationIntent = new Intent(context, MyBatteryReceiver.class);
	// PendingIntent contentIntent = PendingIntent.getBroadcast(context, 0, notificationIntent, 0);

	// notification.setLatestEventInfo(context, contentTitle, contentText, contentIntent);
	// mNotificationManager.notify(HELLO_ID, notification);






	String action = intent.getAction();
	//	Log.i(TAG, "##### I received intent: " + action);
	String batteryStatus;
	int batterylevel;

	if (action.equals(Intent.ACTION_BATTERY_CHANGED))
	    {
		nvisits++;
		batterylevel = intent.getIntExtra("level", 0);
		if (batterylevel != oldbatterylevel)
		    {
			//			Log.i("New level: "," = " + batterylevel + " " + oldbatterylevel);
			oldbatterylevel=batterylevel;
		
			int status = intent.getIntExtra("status", BatteryManager.BATTERY_STATUS_UNKNOWN);
			String strStatus;
			if (status == BatteryManager.BATTERY_STATUS_CHARGING)          batteryStatus = "Charging"; 
			else if (status == BatteryManager.BATTERY_STATUS_DISCHARGING)  batteryStatus = "Dis-charging";
			else if (status == BatteryManager.BATTERY_STATUS_NOT_CHARGING) batteryStatus = "Not charging";
			else if (status == BatteryManager.BATTERY_STATUS_FULL)         batteryStatus = "Full";
			else                                                          batteryStatus = "";
			updateAppWidget(context,batterylevel, batteryStatus);
		    }
	    }
    }

    public void updateAppWidget(Context context,int batterylevel, String batteryStatus)
    {
	//	Log.i(TAG, "updateAppWidget czall no. " + nvisits);
	RemoteViews updateViews = new RemoteViews(context.getPackageName(), R.layout.biondwidget_layout);
	Time now = new Time();
	now.setToNow();
	String time = now.format("%H:%M:%S");
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
	manager.updateAppWidget(myComponentName, updateViews);
    }


}