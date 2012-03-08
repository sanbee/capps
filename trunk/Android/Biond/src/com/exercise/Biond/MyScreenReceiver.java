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

public class MyScreenReceiver extends BroadcastReceiver
{
    private static Boolean screenOn = true;
    private static int nScreenVisits=0;

    @Override public void onReceive(Context context, Intent intent)
    {
	nScreenVisits++;
	RemoteViews views_l = new RemoteViews(context.getPackageName(), R.layout.biondwidget_layout);
	// views_l.setTextViewText(R.id.screen,
	// 		      "SVisits: " + nScreenVisits);
	ComponentName myComponentName = new ComponentName(context, BiondWidgetProvider.class);
	AppWidgetManager manager = AppWidgetManager.getInstance(context);
	manager.updateAppWidget(myComponentName, views_l);

	if (intent.getAction().equals(Intent.ACTION_SCREEN_OFF)) 
	    {
		//		Log.i("####Screen is ", "off");
		screenOn = false;
		//
		// Stop the service that starts the receiver that
		// receivers the ACTION_BATTERY_CHANGED intents.
		//
		context.stopService(new Intent(context, MyBatteryService.class));
	    } 
	else if (intent.getAction().equals(Intent.ACTION_SCREEN_ON)) 
	    {
		//		Log.i("####Screen is ", "on");
		screenOn = true;
		//
		// Stop the service that starts the receiver that
		// receivers the ACTION_BATTERY_CHANGED intents.
		//
		context.startService(new Intent(context, MyBatteryService.class));
	    }
    };
};
