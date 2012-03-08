package com.exercise.Biond;

import android.content.BroadcastReceiver;
import android.content.IntentFilter;
import android.content.Intent;
import android.app.Service;
import android.os.IBinder;
import android.util.Log;

public class MyBatteryService extends Service 
{
    //    private static final String TAG = "MyBatteryReceiver";
    public static BroadcastReceiver myBatteryReceiver=null;

    public void startBatteryReceiver()
    {
	if (myBatteryReceiver == null)
	    myBatteryReceiver=new MyBatteryReceiver();

	IntentFilter batteryFilter = new IntentFilter();
	batteryFilter.addAction(Intent.ACTION_BATTERY_CHANGED);
	registerReceiver(myBatteryReceiver, batteryFilter);
    }

    public void stopBatteryReceiver()
    {
	unregisterReceiver(myBatteryReceiver);
	myBatteryReceiver=null;
    }

    @Override public void onCreate() 
    {
	super.onCreate();
	startBatteryReceiver();
    }
    
    @Override public void onDestroy() 
    {
	//	Log.i(TAG,"onDestroy()");
	super.onDestroy();
	stopBatteryReceiver();
    }

    @Override public IBinder onBind(Intent arg0) 
    {
	return null;
    }
}