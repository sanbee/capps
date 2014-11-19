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
import android.widget.ImageView;
import android.graphics.Color;
import android.content.res.Resources;
import android.graphics.drawable.ColorDrawable;
import java.lang.Integer;

//public class NaaradControlFragment extends Fragment 
public class NaaradControlFragment extends NaaradAbstractFragment
{
    private static View mView;
    
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
    public void lampHandler0(View v)
    {
	serverName = getServerName();
	serverPort = getServerPort();
	currentToggleButton = (ToggleButton)(v);
	//Log.i("Ctrl Server: ", serverName+":"+serverPort);

	int tag = Integer.parseInt((String)v.getTag());
	boolean on = ((ToggleButton)v).isChecked();
	setBulbBG(bulbArr[tag],!on);
	messsage="tell "+v.getTag()+" ";

	if (on)	messsage += "1";
	else	messsage += "0";

	Log.i("Cmd: ",messsage);

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
    public void setBulbBG(View v,boolean on)
    {
	if (on) v.setBackgroundColor(Color.TRANSPARENT);
	else    v.setBackgroundColor(Color.YELLOW);
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	super.onCreateView(inflater, container, savedInstanceState);
	setRetainInstance(true);	

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
	lampArr = new ToggleButton[3];
	bulbArr = new ImageView[3];
	lampArr[0]  = lamp0 = (ToggleButton) mView.findViewById(R.id.lamp0); // reference to the send button
	lampArr[1]  = lamp1 = (ToggleButton) mView.findViewById(R.id.lamp1); // reference to the send button
	lampArr[2]  = lamp2 = (ToggleButton) mView.findViewById(R.id.lamp2); // reference to the send button
	bulbArr[0] = bulb0 = (ImageView) mView.findViewById(R.id.iv1); // reference to the send button
	bulbArr[1] = bulb1 = (ImageView) mView.findViewById(R.id.iv2); // reference to the send button
	bulbArr[2] = bulb2 = (ImageView) mView.findViewById(R.id.iv3); // reference to the send button

	for (int i=0; i<lampArr.length; i++)
	    {
		lampArr[i].setTag(Integer.toString(i)); 
		lampArr[i].setOnClickListener(lampHandler);
	    }

	View.OnClickListener bulbOnClickListener = new View.OnClickListener()
	    {
		public void onClick(View v)
		{
		    int tag=(Integer)(v.getTag(R.integer.key0));
		    Log.i("Tag: ",Integer.toString(tag));//R.integer.key0);
		    boolean on = lampArr[tag].isChecked();
		    setBulbBG(v, on);
		    lampArr[tag].setChecked(!on);
		    lampHandler0(lampArr[tag]);
		};
	    };

	for (int i=0; i<bulbArr.length; i++)
	    {
		bulbArr[i].setTag(R.integer.key0,i);//Integer.toString(i)); 	
		bulbArr[i].setTag(R.integer.key1,"0");
		
		bulbArr[i].setOnClickListener(bulbOnClickListener);
		
		bulbArr[i].setOnLongClickListener(new View.OnLongClickListener()
		    {
			public boolean onLongClick(View v)
			{
			    Log.i("Test: ","IV0 long clicked");
			    return true;
			};
		    });
	    }

	if(savedInstanceState != null)
	    {
		Log.i("BG: ",Integer.toString(savedInstanceState.getInt("bg0")));
		bulb2.setBackgroundColor(savedInstanceState.getInt("bg0"));//		mEditText.setText(savedInstanceState.getString("textKey"));
	    }

	// bulb2.setOnClickListener(new View.OnClickListener()
	//     {
	// 	public void onClick(View v)
	// 	{
	// 	    boolean on = lamp2.isChecked();
	// 	    setBulbBG(bulb2, on);
	// 	    lamp2.setChecked(!on);
	// 	    lampHandler0(lamp2);
	// 	};
	//     });
	// bulb2.setOnLongClickListener(new View.OnLongClickListener()
	//     {
	// 	public boolean onLongClick(View v)
	// 	{
	// 	    Log.i("Test: ","IV0 long clicked");
	// 	    return true;
	// 	};
	//     });


	// Button press event listener
	// sendButton.setOnClickListener(new View.OnClickListener() 
	//     {
	// 	public void onClick(View v) 
	// 	{
	// 	    messsage = textField.getText().toString(); // get the text message on the text field
	// 	    textField.setText(""); // Reset the text field to blank
	// 	    SendMessage sendMessageTask = new SendMessage();
	// 	    sendMessageTask.execute();
	// 	}
	//     });
	
	// TextView txtSampleText = (TextView) mView
	//     .findViewById(R.id.txtViewSample);
	// txtSampleText.setText(sampleText);
	
	return mView;
    }
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
		Log.i("Thread: ",messsage);
		return sendCommand(messsage);
	    // try 
	    // 	{
	    // 	    if (messsage.length() == 0) return null;

	    // 	    //Log.i("Thread: ",serverName+":"+Integer.toString(serverPort));
		    
	    // 	    client = new Socket(serverName, serverPort);
	    // 	    printwriter = new PrintWriter(client.getOutputStream(), true);
		    
	    // 	    printwriter.write(mkMessage("open"));
	    // 	    printwriter.flush();
	    // 	    SystemClock.sleep(500);
		    
	    // 	    printwriter.write(mkMessage(messsage)); // write the message to output stream
	    // 	    // printwriter.write((messsage)); // write the message to output stream
	    // 	    printwriter.flush();
	    // 	    SystemClock.sleep(500);
		    
	    // 	    printwriter.write(mkMessage("done"));
	    // 	    printwriter.flush();
	    // 	    //SystemClock.sleep(500);
		    
	    // 	    printwriter.close();
	    // 	    client.close(); // closing the connection
		    
	    // 	} 
	    // catch (UnknownHostException e) 
	    // 	{
	    // 	    String msg = "Unknown host: "+serverName+":"+Integer.toString(serverPort)+"\nCheck settings";
	    // 	    return msg;
	    // 	} 
	    // catch (IOException e) 
	    // 	{
	    // 	    String msg = "Error connecting to "+serverName+":"+Integer.toString(serverPort)+"\nCheck settings";
	    // 	    return msg;
	    // 	}
	    // return ALL_WELL;
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
