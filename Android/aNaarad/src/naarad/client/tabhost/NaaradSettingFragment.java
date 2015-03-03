package naarad.client.tabhost;

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
import android.widget.TextView;
import android.widget.CheckBox;
import android.util.Log;
import android.os.Handler;
import android.os.SystemClock;
import android.os.AsyncTask;
import android.widget.EditText;
//import android.widget.CheckedTextView;
import java.lang.Integer;
import android.widget.Toast;
import android.graphics.Color;
import android.view.Gravity;
import android.widget.Toast;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.content.Context;

//public class NaaradSettingFragment extends Fragment 
public class NaaradSettingFragment extends NaaradAbstractFragment
{
    private static View mView;
    // private SharedPreferences prefs; 
    private Socket client;
    private PrintWriter printwriter;
    private EditText serverNameField, serverPortField;
    private Button setButton;
    private CheckBox wakeButton,wifiControl;
    public NaaradApp myApp;
    //private TextView wakeText;
    //    private CheckedTextView ctView;
    private String message;
    private boolean wifiTurnedOnByMe=false;
    private int gWifiControl=0, gWakeLockControl=0;
    private int gServerPort;
    private String gServerName;
    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public void onPause()
    {
	super.onPause();
	boolean giveMsg=false;
	if (myApp.myWakeLock.isHeld()) {myApp.myWakeLock.release();giveMsg=true;}
	if (myApp.myWifiLock.isHeld()) {myApp.myWifiLock.release();giveMsg=true;}
	if (giveMsg) toast("Wake and Wifi locks released.",Gravity.BOTTOM);
	setPreference("serverName",gServerName);
	setPreference("gServerPort",gServerPort);
	setPreference("wifiControl", gWifiControl);
	setPreference("wakeControl", gWakeLockControl);
	if (wifiTurnedOnByMe) 
	    {
		myApp.setWifiState(false);
		toast("Turning off wifi...",Gravity.BOTTOM);
	    }
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public static final NaaradSettingFragment newInstanceNSF(String sampleText) 
    {
	NaaradSettingFragment f = new NaaradSettingFragment();
	
	Bundle b = new Bundle();
	b.putString("bString", sampleText);
	f.setArguments(b);
	
	return f;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public void wifiControlClick(CheckBox v)
    {
	if (v.isChecked())
	    {
		if (!isWifiConnected())
		    {
			wifiTurnedOnByMe=true;
			myApp.setWifiState(true);
			toast("Turning wifi on...",Gravity.BOTTOM);
		    }
		gWifiControl=1;
	    }
	else
	    {
		if (isWifiConnected() && wifiTurnedOnByMe)
		    myApp.setWifiState(false);
		toast("Turning wifi off...",Gravity.BOTTOM);
		gWifiControl=0;
	    }
		
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public void wakeLockClick(CheckBox v)
    {
	if (isWifiConnected())
	    {
		if (((CheckBox)v).isChecked())
		    {
			if (!myApp.myWakeLock.isHeld()) myApp.myWakeLock.acquire();
			if (!myApp.myWifiLock.isHeld()) myApp.myWifiLock.acquire();
			toast("Wake and Wifi locks set.",Gravity.BOTTOM);
			gWakeLockControl=1;
			//System.err.println("Checked");
		    }
		else
		    {
			if (myApp.myWakeLock.isHeld()) myApp.myWakeLock.release();
			if (myApp.myWifiLock.isHeld()) myApp.myWifiLock.release();
			gWakeLockControl=0;
			//System.err.println("UnChecked");
		    }
	    }
	else
	    {
		v.setChecked(false);
		gWakeLockControl=0;
		toast("Wifi not connected.\nWake and Wifi locks not set.",Gravity.BOTTOM);
	    }
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	setRetainInstance(true);	
	myApp = (NaaradApp) getActivity().getApplication();
	myApp.makeLocks();
	// if (mView != null)
	//     {
	// 	// Do not inflate the layout again.
	// 	// The returned View of onCreateView will be added into the fragment.
	// 	// However it is not allowed to be added twice even if the parent is same.
	// 	// So we must remove root View (nView) from the existing parent view group
	// 	// (it will be added back).
	// 	((ViewGroup)mView.getParent()).removeView(mView);
	// 	return mView;
	//     }
	if (mView == null)
	    mView = inflater.inflate(R.layout.activity_naarad_setting,
				     container, false);
	else
	    ((ViewGroup)mView.getParent()).removeView(mView);
	    
	serverNameField  = (EditText) mView.findViewById(R.id.serverName); // reference to the text field
	serverPortField  = (EditText) mView.findViewById(R.id.serverPort); // reference to the text field
	serverPortField.setText(Integer.toString(getDefaultPort()));
	serverNameField.setText(getDefaultServer());

	setButton = (Button)  mView.findViewById(R.id.setButton); // reference to the send button

	wifiControl = (CheckBox) mView.findViewById(R.id.wifiControl);
	wakeButton = (CheckBox) mView.findViewById(R.id.wake);
	wakeButton.setChecked(false);

	if ((gWifiControl = getPreference("wifiControl",0)) == 1)
	    {
		wifiControl.setChecked(true);
		wifiControlClick(wifiControl);
	    }
	wifiControl.setOnClickListener(new View.OnClickListener()
	    {
		public void onClick(View v)
		{
		    wifiControlClick((CheckBox)v);
		}
		    
	    });

	//	if (wakeButton.isChecked()) wakeLockClick(wakeButton);
	if ((gWakeLockControl = getPreference("wakeControl",0)) == 1)
	    {
		wakeButton.setChecked(true);
		wakeLockClick(wakeButton);
	    }

	wakeButton.setOnClickListener(new View.OnClickListener()
	    {
		public void onClick(View v)
		{
		    wakeLockClick((CheckBox)v);
		}
	    });

	// Button press event listener
	setButton.setOnClickListener(new View.OnClickListener() 
	    {
		public void onClick(View v) 
		{
		    gServerPort=getDefaultPort();
		    gServerName = serverNameField.getText().toString(); // get the text message on the text field
		    try
			{
			    gServerPort = Integer.parseInt(serverPortField.getText().toString()); // get the text message on the text field
			    if (gServerPort <= 0)
				throw (new NumberFormatException("Port number < 0"));
			    else
				setButton.setTextColor(Color.GREEN);
			}
		    catch (NumberFormatException nfe)
			{
			    String msg = "Wrong Port number: "+ nfe.getMessage();
			    System.out.println(msg);
			    Toast.makeText(getActivity(), msg, Toast.LENGTH_SHORT).show();
			    setButton.setTextColor(Color.RED);
			}
		    //setServerName(serverName);
		    setPreference("serverName",gServerName);
		    //setServerPort(serverPort);
		    setPreference("serverPort", gServerPort);
		    // setPreference("lamp0X", 100);
		    // setPreference("lamp0Y", 200);
		}
	    });
	return mView;
    }
    // //
    // //-----------------------------------------------------------------------------------------
    // //
    // private class SendMessage extends AsyncTask<Void, Void, Void> 
    // {
    // 	private String mkMessage(String message) 
    // 	{
    // 	    Integer n, totalLen;
    // 	    n = message.length();
    // 	    String msg,lenStr;
	    
    // 	    lenStr=Integer.toString(n);
    // 	    totalLen =lenStr.length() + n + 1;
    // 	    lenStr=Integer.toString(totalLen);
    // 	    msg = lenStr + " " + message;
    // 	    //Log.i("Message: ",msg);
    // 	    return msg;
    // 	}
    // 	//
    // 	//-----------------------------------------------------------------------------------------
    // 	//
    // 	@Override protected Void doInBackground(Void... params) 
    // 	    {
    // 	    try 
    // 		{
    // 		    if (messsage.length() == 0) return null;
    // 		    //client = new Socket("10.0.2.2", 1234); // connect to the server on local machine
    // 		    client = new Socket("raspberrypi", 1234); // connect to the Naarad server
    // 		    printwriter = new PrintWriter(client.getOutputStream(), true);
		    
    // 		    printwriter.write(mkMessage("open"));
    // 		    printwriter.flush();
    // 		    SystemClock.sleep(500);
		    
    // 		    printwriter.write(mkMessage(messsage)); // write the message to output stream
    // 		    // printwriter.write((messsage)); // write the message to output stream
    // 		    printwriter.flush();
    // 		    SystemClock.sleep(500);
		    
    // 		    // Handler handler = new Handler(); 
    // 		    // handler.postDelayed(new Runnable() { 
    // 		    // 	public void run() { 
    // 		    // 	    //			    my_button.setBackgroundResource(R.drawable.defaultcard); 
    // 		    // 	} 
    // 		    //     }, 1000); 
    // 		    printwriter.write(mkMessage("done"));
    // 		    printwriter.flush();
    // 		    //SystemClock.sleep(500);
		    
    // 		    printwriter.close();
    // 		    client.close(); // closing the connection
		    
    // 		} 
    // 	    catch (UnknownHostException e) 
    // 		{
    // 		    e.printStackTrace();
    // 		} 
    // 	    catch (IOException e) 
    // 		{
    // 		    e.printStackTrace();
    // 		}
    // 	    return null;
    // 	    }
    // }
}
