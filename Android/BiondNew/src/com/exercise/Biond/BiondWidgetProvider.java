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

import android.view.GestureDetector.OnDoubleTapListener;
import android.app.PendingIntent;
import android.util.Log;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.BroadcastReceiver;
import android.content.IntentFilter;
import android.content.Context;
import android.content.Intent;
import android.widget.RemoteViews;
//import android.widget.Toast;
//import android.graphics.Color;

public class BiondWidgetProvider extends AppWidgetProvider 
{
    private static RemoteViews views_p=null;
    private static BiondBatteryReceiver myBatReceiver_p=null;    
    //    private static Boolean modeAuto_p=true;
    //    private Toast myToast=null;
    //
    //------------------------------------------------------------------
    //
    @Override public void onReceive(Context context, Intent intent)
    {
	BiondApp myapp=myApp(context);
	String mode;
	//	Log.i("Biond: ", "#####onReceive called "+intent.getAction()+" "+myapp.modeAuto_g);
	
	if (intent.getAction().equals(myapp.ACTION_TOGGLE_BUTTON))
	    {
		myapp.modeAuto_g=!myapp.modeAuto_g;
		if (myapp.modeAuto_g.equals(true))
		    {
			context.getApplicationContext().
			    registerReceiver(myBatReceiver_p,
					     new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
			myapp.gUnregisterForClick(context, myapp.views_g);
		    }
		else
		    {
			context.getApplicationContext().unregisterReceiver(myBatReceiver_p);
			myapp.gRegisterForClick(context, myapp.views_g);
		    }
		//		myapp.views_g = new RemoteViews(context.getPackageName(), myapp.LAYOUT);
		myapp.gUpdateButtons(context, myapp.views_g, false, myapp.modeAuto_g);
	    }
	else if (intent.getAction().equals(Intent.ACTION_CONFIGURATION_CHANGED))
	    {
		myapp.globalUpdateWidget(context,myapp.views_g,true);
		myapp.gRegisterForClick(context, myapp.views_g);
		setupWidget(context);
	    }
	else if (intent.getAction().equals(Intent.ACTION_SCREEN_OFF))
	    {
		if (myapp.modeAuto_g.equals(true))
		    context.getApplicationContext().unregisterReceiver(myBatReceiver_p);
	    }
	else if (intent.getAction().equals(Intent.ACTION_SCREEN_ON))
	    {
		if (myapp.modeAuto_g.equals(true))
		    {
			//			myBatReceiver_p = new BiondBatteryReceiver();
			rollingNotify(context);
			context.getApplicationContext().
			    registerReceiver(myBatReceiver_p,
					     new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
		    }
	    }
	else if (intent.getAction().equals(AppWidgetManager.ACTION_APPWIDGET_UPDATE))
	    {
		myapp.gOnSingleClick(context);
	    }
	else if (intent.getAction().equals(AppWidgetManager.ACTION_APPWIDGET_ENABLED))
		 onEnabled(context);
	else if (intent.getAction().equals(AppWidgetManager.ACTION_APPWIDGET_DISABLED))
		 onDisabled(context);
	else if (intent.getAction().equals(AppWidgetManager.ACTION_APPWIDGET_DELETED))
		 onDeleted(context);

	// else if (intent.getAction().equals(myapp.ACTION_CLICK))
	//     myapp.onDoubleClick(context,intent);

    	super.onReceive(context,intent);
    }
    //
    //------------------------------------------------------------------
    //
    public void onEnabled(Context context)
    {
	//	Log.i("onUpdate: ", "########onUpdate called " + myApp(context).modeAuto_g);
	setupWidget(context);
	super.onEnabled(context);
    }
    //
    //------------------------------------------------------------------
    //
    @Override public void onUpdate(Context context, AppWidgetManager appWidgetManager,
				   int[] appWidgetIds) 
    {
	//	Log.i("onUpdate: ", "########onUpdate called " + myApp(context).modeAuto_g);
	setupWidget(context);
    }
    //
    //------------------------------------------------------------------
    //
    public void setupWidget(Context context)
    {
	BiondApp myapp=myApp(context);
	
	context.getApplicationContext().
	    registerReceiver(this, new IntentFilter(Intent.ACTION_SCREEN_OFF));
	
	context.getApplicationContext().
	    registerReceiver(this, new IntentFilter(Intent.ACTION_SCREEN_ON));
	
	context.getApplicationContext().
	    registerReceiver(this, new IntentFilter(Intent.ACTION_CONFIGURATION_CHANGED));
	
	context.getApplicationContext().
	    registerReceiver(this, new IntentFilter(myapp.ACTION_TOGGLE_BUTTON));
	
	if (myapp.views_g == null)
	    {
		myapp.views_g = new RemoteViews(context.getPackageName(), 
						myapp.LAYOUT);
		views_p = myapp.views_g;
	    }
	
	if (myapp.modeAuto_g.equals(true))
	    {
		//		if (myBatReceiver_p==null)
		myBatReceiver_p = new BiondBatteryReceiver();
		context.getApplicationContext().
		    registerReceiver(myBatReceiver_p, 
				     new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
	    }
	
	//
	// Register to recieve events when clicked on the buttons
	//
	myapp.gRegisterButtons(context,myapp.views_g);

	myapp.gUpdateButtons(context, myapp.views_g, 
			     false,
			     !myapp.modeAuto_g);
	myapp.globalUpdateWidget(context,myapp.views_g,true);
		//	localUpdateWidget(context, myApp(context).views_g);
	if (myapp.modeAuto_g.equals(false))
	    myapp.globalUpdateWidget(context,myapp.views_g,false);

	myapp.gRegisterForClick(context, myapp.views_g);//, myapp.modeAuto_g);

	//	myapp.installDoubleClick(context,myapp.views_g);

	//	MenuActivity menu = new MenuActivity();
	
				    // Intent intent = new Intent(context,menu);
				    // context.startActivity(intent);
    }
    //
    //-----------------------------------------------------------
    //    
    public void onDisabled(Context context)
    {
	// Log.i("OnDisabled: ", "#### SBMon Disabled");
	// BiondApp myapp=myApp(context);
	// if (myapp.modeAuto_g.equals(true))
	//     context.getApplicationContext().unregisterReceiver(myBatReceiver_p);
	// super.onDisabled(context);
	super.onDisabled(context);
    }
    //
    //-----------------------------------------------------------
    //    
    public void onDeleted(Context context)
    {
	BiondApp myapp=myApp(context);
	if (myapp.modeAuto_g.equals(true))
	    context.getApplicationContext().unregisterReceiver(myBatReceiver_p);
	myapp.cancelNotification(context);
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
    public void rollingNotify(Context context)
    {
	BiondApp myapp=myApp(context);
	Intent batteryIntent = context.getApplicationContext().registerReceiver
	    (null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
	
	int level = batteryIntent.getIntExtra("level", -1);
	myapp.notify(context, level, (level != myapp.oldbatterylevel));
    }
}
