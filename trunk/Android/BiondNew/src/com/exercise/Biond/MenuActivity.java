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

import android.app.ListActivity;
import android.os.Bundle;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.AdapterView;
import java.util.ArrayList;
import android.util.Log;
import android.content.Context;
import android.graphics.Color;
import android.widget.TextView;
import android.view.ViewGroup;

public class MenuActivity extends ListActivity implements AdapterView.OnItemSelectedListener 
{
    Animation anim = null;
    
    /** Called when the activity is first created. */
    @Override public void onCreate(Bundle icicle) 
    {
        super.onCreate(icicle);
	anim = AnimationUtils.loadAnimation( this, R.anim.magnify );
	setContentView(R.layout.menu);
	//        setContentView(R.layout.radiobutton);
	ListView v = getListView();		// set up item selection listener
	v.setOnItemSelectedListener( this );	// see OnItemSelectedListener methods below
	ArrayList<String> items = new ArrayList<String>();
	items.add( "Auto mode" );
	items.add( "Manual mode" );
	items.add( "Notify" );
	items.add( "On Gesture" );
	//	ArrayAdapter itemsAdapter = new ArrayAdapter( this, R.layout.row, items );
	ArrayAdapter itemsAdapter = new ArrayAdapter<String>( this, android.R.layout.simple_list_item_1, items );
	setListAdapter( itemsAdapter );
    }
    
    protected void onListItemClick(ListView l, 
				   View v, 
				   int position,
				   long id) 
    {
	BiondApp myapp=myApp(this);
	v.startAnimation( anim );
	if (position==0)
	    {
		l.getChildAt(0).setBackgroundColor(Color.GREEN);
		l.getChildAt(1).setBackgroundColor(Color.BLACK);
		myapp.modeAuto_g=true;
	    }
	else if (position==1)
	    {
		l.getChildAt(1).setBackgroundColor(Color.GREEN);
		l.getChildAt(0).setBackgroundColor(Color.BLACK);
		myApp(this).modeAuto_g=false;
	    }
	myapp.gUpdateButtons(this, myapp.views_g, 
			     false, myapp.modeAuto_g);
	this.finish();
	//v.getChildAt(position).setTextColor(Color.GREEN);
	//	Log.i("ItemClicked:","Id="+id+" "+position+" "+myApp(this).modeAuto_g);
    }
    
    // --- AdapterView.OnItemSelectedListener methods --- 
    public void onItemSelected(AdapterView parent, 
			       View v, 
			       int position, 
			       long id) 
    {
	v.startAnimation( anim );
    }
    
    public void onNothingSelected(AdapterView parent) {}
    //
    //-----------------------------------------------------------
    //    
    public BiondApp myApp(Context context)
    {
	return (BiondApp)context.getApplicationContext();
    }
}
