package com.exercise.Biond;

import android.content.BroadcastReceiver;
import android.content.IntentFilter;
import android.content.Intent;
import android.app.Service;
import android.os.IBinder;
import android.util.Log;

public class MyScreenService extends Service 
{
    //    private static final String TAG = "MyScreenReceiver";
    public static BroadcastReceiver myScreenStatusReceiver=null;

    public void startScreenReceiver()
    {
	if (myScreenStatusReceiver == null)
	    myScreenStatusReceiver=new MyScreenReceiver();

	IntentFilter screenFilter = new IntentFilter(Intent.ACTION_SCREEN_ON);
	screenFilter.addAction(Intent.ACTION_SCREEN_OFF);
	registerReceiver(myScreenStatusReceiver, screenFilter);
    }

    public void stopScreenReceiver()
    {
	unregisterReceiver(myScreenStatusReceiver);
	myScreenStatusReceiver=null;
    }
    
    @Override public void onCreate() 
    {
	super.onCreate();
	startScreenReceiver();
    }
    
    @Override public void onDestroy() 
    {
	//	Log.i(TAG,"onDestroy()");
	super.onDestroy();
	stopScreenReceiver();
    }
    //
    // This one is requrired -- looks likes a pure virutal
    //
    @Override public IBinder onBind(Intent arg0) 
    {
	return null;
    }
}