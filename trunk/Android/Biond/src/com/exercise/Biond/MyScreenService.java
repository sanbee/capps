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