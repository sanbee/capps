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

import android.app.Activity;
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
import android.widget.RadioGroup;

public class MenuActivity2 extends Activity
    implements RadioGroup.OnCheckedChangeListener {
    RadioGroup rlist;
    @Override public void onCreate(Bundle icicle)
    {
	super.onCreate(icicle);
	setContentView(R.layout.radiobutton);
	rlist = (RadioGroup)findViewById(R.id.button1);
	rlist.setOnCheckedChangeListener(this);
    }

    public void onCheckedChanged(RadioGroup group, int checkedId)
    {
	if (group==rlist)
	    {
		if (checkedId==R.id.button1)
		    Log.i("Checked: ","button1");
		else if (checkedId==R.id.button2)
		    Log.i("Checked: ","button2");
	    }
    }
};		    