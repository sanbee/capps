package naarad.client.tabhost;
import android.app.Application;
// import com.blahti.example.drag;
// import com.blahti.example.drag.DragController;
// import com.blahti.example.drag.DragLayer;
// import android.os.Handler;
// import niko.dragdrop.view.DragDropView;
//import android.widget.TextView;
//import android.widget.EditText;
//import android.content.SharedPreferences;
//import android.content.Context;
//import android.widget.FrameLayout;
//import android.view.LayoutInflater;
//import android.support.v4.view.ViewPager;
//import android.support.v4.view.ViewCompat;
//import android.view.Gravity;
//import android.view.View.OnLongClickListener;
//import android.view.ViewGroup.LayoutParams;
//import android.view.WindowManager;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import android.widget.Button;
import android.widget.ToggleButton;
import android.util.Log;
import android.os.SystemClock;
import android.os.AsyncTask;
import android.widget.Toast;

import android.widget.RelativeLayout;
import android.widget.ImageView;
import android.graphics.Color;
import android.content.res.Resources;
import android.graphics.drawable.ColorDrawable;
import java.lang.Integer;
import android.view.MotionEvent;
import android.view.View.OnTouchListener;
import android.graphics.drawable.Drawable;
import android.view.GestureDetector.SimpleOnGestureListener;
import android.view.GestureDetector;
import android.app.Activity;

//public class NaaradControlFragment extends Fragment implements View.OnLongClickListener 
public class NaaradControlFragment extends NaaradAbstractFragment implements OnTouchListener
{
    //private EditText textField;
    //private Button sendButton,initButton;
    //    private Handler myHandler;
    //    private DragDropView dragDropView;// = new DragDropView();
    //private LayoutInflater li;
    //    private View.OnLongClickListener onLongPressListener;

    //private DragLayer mDragLayer;             // The ViewGroup that supports drag-drop.
    //private DragController mDragController;
    // private boolean mLongClickStartsDrag = true;    // If true, it takes a long click to start the drag operation.
                                                    // Otherwise, any touch event starts a drag.
    //    private LayoutParams imageParams;

    private static View mView;

    private Socket client;
    private PrintWriter printwriter;
    private ToggleButton lamp0, lamp1, lamp2, currentToggleButton;
    private ToggleButton[] lampArr;
    private ImageView bulb0, bulb1, bulb2;
    private ImageView[] bulbArr;
    
    private String messsage, serverName;
    private int serverPort=1234;
    final private String ALL_WELL="All well";
    private ImageView iv;

    public boolean touchFlag=false;
    private View selected_item=null;
    private int offset_x, offset_y;
    private int topy, leftX, rightX, bottomY;

    private GestureDetector myGestureDetector;
    private View.OnTouchListener myOnTouchListener;
    private boolean touchFlag_p;
    private Activity mActivity0=null;        
    //private MyViewPager mViewPager=null;
    public NaaradApp myApp;

    // public void setViewPager(MyViewPager v)
    // {if (mViewPager == null) mViewPager = v;}

    // public void setActivity(Activity a)
    // {
    // 	if (mActivity0 == null)
    // 	mActivity0 = a;
    // }
    // @Override public void onAttach(Activity activity) 
    // {
    //     super.onAttach(activity);
    //     mActivity0 = activity;
    // }
    //
    //-----------------------------------------------------------------------------------------
    //
    public static final NaaradControlFragment newInstance(String sampleText) 
    {
	NaaradControlFragment f = new NaaradControlFragment();
	
	Bundle b = new Bundle();
	b.putString("bString", sampleText);
	f.setArguments(b);
	
	return f;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public boolean onTouch(View v, MotionEvent event) 
    //@Override public boolean onLongClick(View v, MotionEvent event) 
    {   
    	switch (event.getActionMasked()) 
            {
            case MotionEvent.ACTION_DOWN:
    		//Log.i(null,"Activity onTouch Down");
                touchFlag=true;
                // offset_x = (int) v.getWidth();//event.getX();
                // offset_y = (int) v.getHeight();//event.getY();
    		//System.err.println("Activity Down: "+v.getTop()+" "+event.getX());
                selected_item = v;
    		//                imageParams=v.getLayoutParams();
                break;
            case MotionEvent.ACTION_UP:
    		//Log.i(null,"Activity onTouch Up");
                selected_item=null;
                touchFlag=false;
                break;
            default:
                break;
            }       
    	return false;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public void setBulbBG(View v,boolean on)
    {
	if (on)
	    {
		//((ImageView)(v)).setImageDrawable(mActivity.getResources().getDrawable(R.drawable.ic_launcher_bg));
		//((ImageView)(v)).setImageDrawable(mActivity.getResources().getDrawable(R.drawable.lamp_off));
		((ImageView)(v)).setImageDrawable(mActivity0.getResources().getDrawable(R.drawable.lamp_off));
	    }
	else
	    {
		//((ImageView)(v)).setImageDrawable(mActivity.getResources().getDrawable(R.drawable.ic_launcher));
		//((ImageView)(v)).setImageDrawable(mActivity.getResources().getDrawable(R.drawable.lamp_on));
		((ImageView)(v)).setImageDrawable(mActivity0.getResources().getDrawable(R.drawable.lamp_on));
		//v.setBackgroundColor(Color.YELLOW);
	    }
	v.setBackgroundColor(Color.TRANSPARENT);
	
	
	// if (on) v.setBackgroundColor(Color.TRANSPARENT);
	// else    v.setBackgroundColor(Color.YELLOW);
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public boolean setLampBG(View v)
    {
	boolean on = ((ToggleButton)v).isChecked();
	if (on) 
	    {
		//v.setBackgroundDrawable(mActivity.getResources().getDrawable(R.drawable.lamp_on));
		v.getBackground().setAlpha(255);
	    }
	else    
	    {
		//		v.setBackgroundDrawable(R.drawable.lamp_off);
		v.getBackground().setAlpha(128);
	    }
	return on;
    }
    public void toast (String msg)
    {
	//Toast.makeText (getApplicationContext(), msg, Toast.LENGTH_SHORT).show ();
	Toast.makeText (mActivity0, msg, Toast.LENGTH_SHORT).show ();
    } 
    //
    //-----------------------------------------------------------------------------------------
    //
    public void lampHandler0(View v)
    {
	serverName = getServerName();
	serverPort = getServerPort();
	currentToggleButton = (ToggleButton)(v);
	//Log.i("Ctrl Server: ", serverName+":"+serverPort);
	
	int tag = Integer.parseInt((String)v.getTag());
	boolean on = setLampBG(v);
	setBulbBG(bulbArr[tag],!on);
	messsage="tell "+v.getTag()+" ";
	
	if (on)	messsage += "1";
	else	messsage += "0";
	
	//	Log.i("Cmd: ",messsage);
	
	SendMessage sendMessageTask = new SendMessage();
	sendMessageTask.execute();
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    // @Override public void onSaveInstanceState(Bundle outState)
    // {
    // 	//Log.i("setInstance: ", "Setting 0");
	
    // 	super.onSaveInstanceState(outState);
    // 	ColorDrawable cd=(ColorDrawable)(bulb2.getBackground());
    // 	outState.putInt("bg0", cd.getColor());
    // }
    //
    //-----------------------------------------------------------------------------------------
    //
    public boolean containerOnTouch(View v, MotionEvent event, boolean touchFlag_p) 
    {
	int crashX, crashY, w, h;

	if ((touchFlag_p==true) && (selected_item != null))
	    {
		//System.err.println("Display If  Part ::->"+touchFlag_p);
		switch (event.getActionMasked()) 
		    {
		    // case MotionEvent.ACTION_DOWN :
		    // 	//Log.i(null,"COT Down");
		    // 	// topy=selected_item.getTop();//imageDrop.getTop();
		    // 	// leftX=selected_item.getLeft();
		    // 	// rightX=selected_item.getRight();   
		    // 	// bottomY=selected_item.getBottom();
		    // 	// w=selected_item.getWidth();//selected_item.getLayoutParams().width;
		    // 	// h=selected_item.getLayoutParams().height;
		    // 	// System.err.println("D Display Top-->"+topy);      
		    // 	// System.err.println("D Display Left-->"+leftX);
		    // 	// System.err.println("D Display Right-->"+rightX);
		    // 	// System.err.println("D Display Bottom-->"+bottomY);                
		    // 	// System.err.println("D Display Width-->"+w);                
		    // 	// System.err.println("D Display Height-->"+h);                
		    // 	break;
		    case MotionEvent.ACTION_MOVE:
			int x,y,pX,pY, oX=0, oY=0, iX,iY;
			pX=(int) event.getX();
			pY=(int) event.getY();

			oX=selected_item.getHeight()*2;
			oY=selected_item.getWidth();
			iX=selected_item.getRight();
			iY=selected_item.getTop();
			x=pX - oX + iX;
			y=pY - oY + iY;
			System.err.println("M Display Here X Value-->"+(x)+" "+pX+" "+oX+" "+iX);
			System.err.println("M Display Here Y Value-->"+(y)+" "+pY+" "+oY+" "+iY);

			RelativeLayout.LayoutParams lp = new RelativeLayout.LayoutParams
			    (
			     new ViewGroup.MarginLayoutParams
			     (RelativeLayout.LayoutParams.WRAP_CONTENT,
			      RelativeLayout.LayoutParams.WRAP_CONTENT)
			     );
			lp.setMargins(x, y, 0, 0);                  
			lp.height=30;
			lp.width=30;
			selected_item.setLayoutParams(lp);
			break;  
		    // case MotionEvent.ACTION_UP:
		    // 	Log.i(null,"COT Up");
		    // 	break;
		    default:
			break;
		    }
	    }
	// else
	//     {
	// 	System.err.println("Display Else Part ::->"+touchFlag);
	//     }               
	return true;
    };

    //
    //-----------------------------------------------------------------------------------------
    //
    GestureDetector.SimpleOnGestureListener myGestureListener = new GestureDetector.SimpleOnGestureListener()
	{
	    @Override public boolean onSingleTapUp(MotionEvent e)
	    {
	    	super.onSingleTapUp(e);
		//		Log.i("Gesture: ","TapUp");
		int tag=(Integer)(selected_item.getTag(R.integer.key0));
		//Log.i("Tag: ",Integer.toString(tag));//R.integer.key0);
		boolean on = lampArr[tag].isChecked();
		setBulbBG(selected_item, on);
		lampArr[tag].setChecked(!on);
		lampHandler0(lampArr[tag]);
	    	return false;
	    }

	    @Override public void onLongPress(MotionEvent e)
	    {
		Log.i("Gesture: ","LongPress");
		super.onLongPress(e);
		touchFlag_p=true;
		((ImageView)selected_item).setAlpha(100);
		selected_item.setBackgroundColor(Color.GREEN);
		selected_item.getBackground().setAlpha(100);
		//mViewPager = ((MyViewPager)mActivity0.findViewById(R.id.viewpager));
		//mViewPager.enableSwipe(false);
		myApp.setSwipeState(false);
		//mViewPager.enableSwipe(false);

		//containerOnTouch(selected_item, e, touchFlag_p);
		return;
	    }
	    @Override public boolean onDown(MotionEvent e)
	    {
	    	Log.i("Gesture: ","onDown");
	    	super.onDown(e);  
	    	    // int tag=(Integer)(selected_item.getTag(R.integer.key0));
	    	    // //Log.i("Tag: ",Integer.toString(tag));//R.integer.key0);
	    	    // boolean on = lampArr[tag].isChecked();
	    	    // setBulbBG(selected_item, on);
	    	    // lampArr[tag].setChecked(!on);
	    	    // lampHandler0(lampArr[tag]);
	    	return true;
	    }

	    // @Override public boolean onDoubleTap(MotionEvent e)
	    // {
	    // 	Log.i("Gesture: ","onDoubleTab");
	    // 	super.onDoubleTap(e);  
	    // 	return false;
	    // }
	};
    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	super.onCreateView(inflater, container, savedInstanceState);
	myApp = (NaaradApp) getActivity().getApplication();

	setRetainInstance(true);	
	//getRetainInstance();
	final Resources res = getResources();
	final int k0 = res.getInteger(R.integer.key0);
	final int k1 = res.getInteger(R.integer.key1);
	//
	// This checks mView and recreates if it is null.  Otherwise
	// returns the existing one.
	//
	mActivity0 = getActivity();

	if (recreateView(mView)) return mView;	

	myOnTouchListener = new View.OnTouchListener() 
            {
                public boolean onTouch(View v, MotionEvent event) 
                {
		    boolean ret=true;
		    int act = event.getActionMasked();
		    int tag=(Integer)(v.getTag(R.integer.key0));
		    //Log.i("onTouch: ",Integer.toString(tag));//R.integer.key0);

		    if (act != MotionEvent.ACTION_MOVE)
		    	myGestureDetector.onTouchEvent(event);
		    
		    selected_item = v;

		    switch(act)
			{
			// case MotionEvent.ACTION_DOWN:
			// 	Log.i("onTouch: ","DOWN "+Integer.toString(tag));//R.integer.key0);

			// 	boolean on = lampArr[tag].isChecked();
			// 	setBulbBG(selected_item, on);
			// 	lampArr[tag].setChecked(!on);
			// 	lampHandler0(lampArr[tag]);
			// 	return true;
			case MotionEvent.ACTION_MOVE:
			    if (touchFlag_p)
				ret=containerOnTouch(selected_item, event,true);
			    break;
			case MotionEvent.ACTION_UP:
			    Log.i("onTouch: ","UP "+Integer.toString(tag));//R.integer.key0);
			    touchFlag_p=false;
			    ((ImageView)selected_item).setAlpha(255);
			    selected_item.setBackgroundColor(Color.TRANSPARENT);
			    selected_item=null;
			    //mViewPager = ((MyViewPager)mActivity0.findViewById(R.id.viewpager));
			    //mViewPager.enableSwipe(true);
			    myApp.setSwipeState(true);
			    break;
			};
		    //Log.i("onTouch: ","Touched");//R.integer.key0);

		    //return containerOnTouch(v, event,touchFlag_p);
		    return ret;
		}
	    };
	myGestureDetector = new GestureDetector(mActivity0, myGestureListener);
	
	View.OnClickListener lampHandler = new View.OnClickListener()
	    {
		public void onClick(View v) 
		{
		    lampHandler0(v);
		}
	    };
	
	
	mView = inflater.inflate(R.layout.activity_naarad_control, container, false);
	
	//li=inflater;
	//dragDropView = new DragDropView(mView.getContext());
	
	lampArr = new ToggleButton[3];
	bulbArr = new ImageView[3];
	lampArr[0]  = lamp0 = (ToggleButton) mView.findViewById(R.id.lamp0); // reference to the send button
	lampArr[1]  = lamp1 = (ToggleButton) mView.findViewById(R.id.lamp1); // reference to the send button
	lampArr[2]  = lamp2 = (ToggleButton) mView.findViewById(R.id.lamp2); // reference to the send button
	bulbArr[0]  = bulb0 = (ImageView) mView.findViewById(R.id.iv1); // reference to the send button
	bulbArr[1]  = bulb1 = (ImageView) mView.findViewById(R.id.iv2); // reference to the send button
	bulbArr[2]  = bulb2 = (ImageView) mView.findViewById(R.id.iv3); // reference to the send button
	
	for (int i=0; i<lampArr.length; i++)
	    {
		lampArr[i].setTag(Integer.toString(i)); 
		lampArr[i].setOnClickListener(lampHandler);
		setLampBG(lampArr[i]);
		//lampArr[i].setOnTouchListener(OnTouchToDrag);
	    }
	
	
	View.OnClickListener bulbOnClickListener = new View.OnClickListener()
	    {
		public void onClick(View v)
		{
		    int tag=(Integer)(v.getTag(R.integer.key0));
		    //Log.i("Tag: ",Integer.toString(tag));//R.integer.key0);
		    boolean on = lampArr[tag].isChecked();
		    setBulbBG(v, on);
		    lampArr[tag].setChecked(!on);
		    lampHandler0(lampArr[tag]);
		    // if (mLongClickStartsDrag) 
		    // 	{
		    // 	    // Tell the user that it takes a long click to start dragging.
		    // 	    toast ("Press and hold to drag an image.");
		    // 	}
		};
	    };

	//container.setOnTouchListener(myOnTouchListener);
	// container.setOnTouchListener(new View.OnTouchListener() 
        //     {
        //         public boolean onTouch(View v, MotionEvent event) 
        //         {
	// 	    //		    int tag=(Integer)(v.getTag(R.integer.key0));
	// 	    //		    Log.i("onTouch: ",Integer.toString(tag));//R.integer.key0);
	// 	    Log.i("onTouch: ","Touched");//R.integer.key0);
	// 	    selected_item = v;
	// 	    myGestureDetector.onTouchEvent(event);
	// 	    return false;
	// 	    //return containerOnTouch(v, event,touchFlag);
	// 	}
	//     });
	
	
	for (int i=0; i<bulbArr.length; i++)
	    {
		bulbArr[i].setTag(R.integer.key0,i);//Integer.toString(i)); 	
		bulbArr[i].setTag(R.integer.key1,"0");
		
		//bulbArr[i].setOnClickListener(bulbOnClickListener);


		//bulbArr[i].setOnLongClickListener(onLongPressListener);
		//bulbArr[i].setOnTouchListener(OnTouchToDrag);

		// This installs the onTouch handler, which records
		// the object clicked and return false (transfering
		// control to container's onTouch handler).
		
		//bulbArr[i].setOnTouchListener(this); 
		//bulbArr[i].setOnLongClickListener(this);

		bulbArr[i].setOnTouchListener(myOnTouchListener);
	    }
	
	// if(savedInstanceState != null)
	//     {
	// 	Log.i("BG: ",Integer.toString(savedInstanceState.getInt("bg0")));
	// 	bulb2.setBackgroundColor(savedInstanceState.getInt("bg0"));//		mEditText.setText(savedInstanceState.getString("textKey"));
	//     }
	
	return mView;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    private String mkMessage(String message) 
    {
	Integer n, totalLen;
	n = message.length();
	String msg,lenStr;
	
	lenStr=Integer.toString(n);
	totalLen =lenStr.length() + n + 1;
	lenStr=Integer.toString(totalLen);
	msg = lenStr + " " + message;
	//Log.i("Message: ",msg);
	return msg;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public String sendCommand(String cmd)
    {
	try 
	    {
		if (cmd.length() == 0) return null;
		
		//Log.i("Thread: ",serverName+":"+Integer.toString(serverPort));
		
		client = new Socket(serverName, serverPort);
		printwriter = new PrintWriter(client.getOutputStream(), true);
		
		printwriter.write(mkMessage("open"));
		printwriter.flush();
		SystemClock.sleep(500);
		
		printwriter.write(mkMessage(cmd)); // write the message to output stream
		// printwriter.write((cmd)); // write the message to output stream
		printwriter.flush();
		SystemClock.sleep(500);
		
		printwriter.write(mkMessage("done"));
		printwriter.flush();
		//SystemClock.sleep(500);
		
		printwriter.close();
		client.close(); // closing the connection
		
	    } 
	catch (UnknownHostException e) 
	    {
		String msg = "Unknown host: "+serverName+":"+Integer.toString(serverPort)+"\nCheck settings";
		return msg;
	    } 
	catch (IOException e) 
	    {
		String msg = "Error connecting to "+serverName+":"+Integer.toString(serverPort)+"\nCheck settings";
		return msg;
	    }
	return ALL_WELL;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    private class SendMessage extends AsyncTask<Void, Void, String> 
    {
	//
	//-----------------------------------------------------------------------------------------
	//
	@Override protected String doInBackground(Void... params) 
	    {
		//Log.i("Thread: ",messsage);
		return sendCommand(messsage);
	    }
	@Override protected void onPostExecute(String result) 
	    {
		super.onPostExecute(result);
		if (result != ALL_WELL)		
		    {
			Toast.makeText(mActivity0, result, Toast.LENGTH_SHORT).show();
			boolean on = currentToggleButton.isChecked();
			int tag = Integer.parseInt((String)currentToggleButton.getTag());
			
			currentToggleButton.setChecked(!on);
			setBulbBG(bulbArr[tag],on);
		    }
	    }
    }
}
//
//=====================TEST CODE===================================
//

// RelativeLayout rl = new RelativeLayout(mView.getContext());
// ImageView iv;
// RelativeLayout.LayoutParams params;

// int yellow_iv_id = 123; // Some arbitrary ID value.

// iv = new ImageView(rl.getContext());
// iv.setId(yellow_iv_id);
// iv.setImageResource(R.drawable.ic_launcher);
// //	iv.setBackgroundColor(Color.YELLOW);
// params = new RelativeLayout.LayoutParams(30, 40);
// params.leftMargin = 50;
// params.topMargin = 60;
// iv.setLayoutParams(params);
// rl.addView(iv, params);
// iv.setVisibility(View.VISIBLE);

// iv = new ImageView(rl.getContext());
// iv.setImageResource(R.drawable.ourhouse2);
// //	iv.setBackgroundColor(Color.RED);
// params = new RelativeLayout.LayoutParams(30, 40);
// params.leftMargin = 80;
// params.topMargin = 90;

// // This line defines how params.leftMargin and params.topMargin are interpreted.
// // In this case, "<80,90>" means <80,90> to the right of the yellow ImageView.
// params.addRule(RelativeLayout.RIGHT_OF, yellow_iv_id);

// iv.setVisibility(View.VISIBLE);
// rl.addView(iv, params);
//
//=====================TEST CODE===================================
//
