package naarad.client.tabhost;
//import com.blahti.example.drag;
import com.blahti.example.drag.DragController;
import com.blahti.example.drag.DragLayer;

//import niko.dragdrop.view.DragDropView;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import android.widget.Button;
import android.widget.ToggleButton;
import android.widget.EditText;
import android.util.Log;
import android.os.Handler;
import android.os.SystemClock;
import android.os.AsyncTask;
import android.widget.EditText;
import android.content.SharedPreferences;
import android.content.Context;
import android.widget.Toast;
import java.net.UnknownHostException;
import android.os.Handler;
import android.widget.RelativeLayout;
import android.widget.FrameLayout;
import android.view.LayoutInflater;
import android.widget.ImageView;
import android.graphics.Color;
import android.content.res.Resources;
import android.graphics.drawable.ColorDrawable;
import java.lang.Integer;
import android.view.MotionEvent;
import android.support.v4.view.ViewPager;
import android.support.v4.view.ViewCompat;
import android.view.Gravity;
import android.view.View.OnTouchListener;
import android.view.ViewGroup.LayoutParams;
import android.view.WindowManager;
import android.graphics.drawable.Drawable;

//public class NaaradControlFragment extends Fragment 
public class NaaradControlFragment extends NaaradAbstractFragment implements OnTouchListener
{
    private static View mView;
    private ViewGroup gViewGroup;

    private Socket client;
    private PrintWriter printwriter;
    //private EditText textField;
    //private Button sendButton,initButton;
    private ToggleButton lamp0, lamp1, lamp2, currentToggleButton;
    private ToggleButton[] lampArr;
    private ImageView bulb0, bulb1, bulb2;
    private ImageView[] bulbArr;
    
    private String messsage, serverName;
    private int serverPort=1234;
    private Handler myHandler;
    final private String ALL_WELL="All well";
    //    private DragDropView dragDropView;// = new DragDropView();
    private ImageView iv;
    private LayoutInflater li;
    private View.OnLongClickListener onLongPressListener;
    
    private DragLayer mDragLayer;             // The ViewGroup that supports drag-drop.
    private DragController mDragController;
    private boolean mLongClickStartsDrag = true;    // If true, it takes a long click to start the drag operation.
                                                    // Otherwise, any touch event starts a drag.

    public boolean touchFlag=false;
    private View selected_item=null;
    private int offset_x, offset_y;
    //    private LayoutParams imageParams;
    private int topy, leftX, rightX, bottomY;
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
    {   
	switch (event.getActionMasked()) 
            {
            case MotionEvent.ACTION_DOWN:
		//Log.i(null,"Activity onTouch Down");
                touchFlag=true;
                // offset_x = (int) v.getWidth();//event.getX();
                // offset_y = (int) v.getHeight();//event.getY();
		System.err.println("Activity Down: "+v.getTop()+" "+event.getX());
                selected_item = v;
		//                imageParams=v.getLayoutParams();
                break;
            case MotionEvent.ACTION_UP:
		Log.i(null,"Activity onTouch Up");
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
		//((ImageView)(v)).setImageDrawable(getActivity().getResources().getDrawable(R.drawable.ic_launcher_bg));
		((ImageView)(v)).setImageDrawable(getActivity().getResources().getDrawable(R.drawable.lamp_off));
	    }
	else
	    {
		//((ImageView)(v)).setImageDrawable(getActivity().getResources().getDrawable(R.drawable.ic_launcher));
		((ImageView)(v)).setImageDrawable(getActivity().getResources().getDrawable(R.drawable.lamp_on));
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
		//v.setBackgroundDrawable(getActivity().getResources().getDrawable(R.drawable.lamp_on));
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
	Toast.makeText (getActivity(), msg, Toast.LENGTH_SHORT).show ();
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
    @Override public void onSaveInstanceState(Bundle outState)
    {
	Log.i("setInstance: ", "Setting 0");
	
	super.onSaveInstanceState(outState);
	ColorDrawable cd=(ColorDrawable)(bulb2.getBackground());
	outState.putInt("bg0", cd.getColor());
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public boolean containerOnTouch(View v, MotionEvent event, boolean touchFlag_p) 
    {
	int crashX, crashY, w, h;

	if(touchFlag_p==true)
	    {
		System.err.println("Display If  Part ::->"+touchFlag_p);
		switch (event.getActionMasked()) 
		    {
		    case MotionEvent.ACTION_DOWN :
			Log.i(null,"COT Down");
			// topy=selected_item.getTop();//imageDrop.getTop();
			// leftX=selected_item.getLeft();
			// rightX=selected_item.getRight();   
			// bottomY=selected_item.getBottom();
			// w=selected_item.getWidth();//selected_item.getLayoutParams().width;
			// h=selected_item.getLayoutParams().height;
			// System.err.println("D Display Top-->"+topy);      
			// System.err.println("D Display Left-->"+leftX);
			// System.err.println("D Display Right-->"+rightX);
			// System.err.println("D Display Bottom-->"+bottomY);                
			// System.err.println("D Display Width-->"+w);                
			// System.err.println("D Display Height-->"+h);                
			break;
		    case MotionEvent.ACTION_MOVE:
			int x,y;
			crashX=(int) event.getX();
			crashY=(int) event.getY();

			offset_x=selected_item.getHeight();
			offset_y=selected_item.getWidth();

			x=crashX - offset_x;
			y=crashY - offset_y;
			System.err.println("M Display Here X Value-->"+(x)+" "+crashX+" "+offset_x);
			System.err.println("M Display Here Y Value-->"+(y)+" "+crashY+" "+offset_y);
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
		    case MotionEvent.ACTION_UP:
			Log.i(null,"COT Up");
			break;
		    default:
			break;
		    }
	    }else
	    {
		System.err.println("Display Else Part ::->"+touchFlag);
	    }               
	return true;
    };

    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	super.onCreateView(inflater, container, savedInstanceState);
	setRetainInstance(true);	
	//
	// This checks mView and recreates if it is null.  Otherwise
	// returns the existing one.
	//
	gViewGroup = container;
	if (recreateView(mView)) return mView;	
	
	final Resources res = getResources();
	final int k0 = res.getInteger(R.integer.key0);
	final int k1 = res.getInteger(R.integer.key1);
	
	View.OnClickListener lampHandler = new View.OnClickListener()
	    {
		public void onClick(View v) 
		{
		    lampHandler0(v);
		}
	    };
	
	
	mView = inflater.inflate(R.layout.activity_naarad_control, container, false);
	
	li=inflater;
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
		    if (mLongClickStartsDrag) 
			{
			    // Tell the user that it takes a long click to start dragging.
			    toast ("Press and hold to drag an image.");
			}
		};
	    };

	//container.setOnTouchListener(OnTouchToDrag);
	// container.setOnTouchListener(new View.OnTouchListener() 
        //     {
        //         public boolean onTouch(View v, MotionEvent event) 
        //         {
	// 	    return containerOnTouch(v, event,touchFlag);
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

		//		bulbArr[i].setOnTouchListener(this); 
	    }
	
	if(savedInstanceState != null)
	    {
		Log.i("BG: ",Integer.toString(savedInstanceState.getInt("bg0")));
		bulb2.setBackgroundColor(savedInstanceState.getInt("bg0"));//		mEditText.setText(savedInstanceState.getString("textKey"));
	    }
	
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
			Toast.makeText(getActivity(), result, Toast.LENGTH_SHORT).show();
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
