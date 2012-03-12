package com.exercise.Biond;

import android.graphics.Color;
import android.app.Application;
import android.appwidget.AppWidgetManager;
import android.content.Context;
import android.widget.RemoteViews;
import android.content.ComponentName;
import android.os.BatteryManager;
import android.util.Log;

public class BiondApp extends Application 
{
    public final int LAYOUT=R.layout.biondwidget_layout_relative;//_tablet_xlarge;
    public static int blinkDelay=200, blinkColor=Color.GREEN, normalColor=Color.WHITE;
    public static String batteryStatus;

    private static int oldbatterylevel = 0;
    private static int oldstatus = BatteryManager.BATTERY_STATUS_UNKNOWN;
    //
    //-----------------------------------------------------------------------------------
    //
    public void globalUpdateAppWidget(Context context, int batteryLevel, 
				      String batteryStatus, RemoteViews updateViews,
				      Boolean writeToScreen)
    {
	//Log.i("GlobalUpdate: ", "Level = " + batteryLevel + " Status = " + batteryStatus);

	updateViews.setTextColor(R.id.level, normalColor);

	updateViews.setTextViewText(R.id.level,  batteryLevel + "%");

	updateViews.setTextViewText(R.id.status, batteryStatus);

	updateViews.setProgressBar(R.id.progress_bar,100,batteryLevel,false);

	ComponentName myComponentName = new ComponentName(context, BiondWidgetProvider.class);
	AppWidgetManager manager = AppWidgetManager.getInstance(context);
	//	notify(context, batteryLevel);
	if (writeToScreen) manager.updateAppWidget(myComponentName, updateViews);
    }
    //
    //-----------------------------------------------------------------------------------
    //
    public void displayInfo(Context context, RemoteViews views, int level, int status)
    {
	//Log.i("New level: "," = " + level + " " + oldbatterylevel);
	Boolean doit=(level != oldbatterylevel) || (status != oldstatus);
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
	globalUpdateAppWidget(context, level, batteryStatus, views,doit);
    }
}

