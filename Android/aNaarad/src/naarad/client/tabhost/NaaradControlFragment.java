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

//public class NaaradControlFragment extends Fragment 
public class NaaradControlFragment extends NaaradAbstractFragment
{
    private static View mView;
    
    private Socket client;
    private PrintWriter printwriter;
    private EditText textField;
    private Button sendButton,initButton;
    private ToggleButton lamp0, lamp1, lamp2, currentToggleButton;
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

	boolean on = ((ToggleButton)v).isChecked();
	messsage="tell "+v.getTag()+" ";
	// Log.i("lamp" + v.getTag() + ": ","clicked");
	if (on)	messsage += "1";
	else	messsage += "0";
	SendMessage sendMessageTask = new SendMessage();
	sendMessageTask.execute();
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	View.OnClickListener lampHandler = new View.OnClickListener()
	    {
		public void onClick(View v) 
		{
		    lampHandler0(v);
		    // if (super.getServerName() != null)
		    // 	Log.i("Ctrl Server: ", super.getServerName());
		    // boolean on = ((ToggleButton)v).isChecked();
		    // messsage="tell "+v.getTag()+" ";
		    // // Log.i("lamp" + v.getTag() + ": ","clicked");
		    // if (on)	messsage += "1";
		    // else	messsage += "0";
		    // SendMessage sendMessageTask = new SendMessage();
		    // sendMessageTask.execute();
		}
	    };
	
	
	mView = inflater.inflate(R.layout.activity_naarad_control,
				 container, false);
	String sampleText = getArguments().getString("bString");
	textField  = (EditText) mView.findViewById(R.id.editText1); // reference to the text field
	sendButton = (Button)  mView.findViewById(R.id.sendButton); // reference to the send button
	initButton = (Button)  mView.findViewById(R.id.initButton); // reference to the send button
	lamp0      = (ToggleButton) mView.findViewById(R.id.lamp0); // reference to the send button
	lamp1      = (ToggleButton) mView.findViewById(R.id.lamp1); // reference to the send button
	lamp2      = (ToggleButton) mView.findViewById(R.id.lamp2); // reference to the send button

	lamp0.setTag("0"); lamp0.setOnClickListener(lampHandler);
	lamp1.setTag("1"); lamp1.setOnClickListener(lampHandler);
	lamp2.setTag("2"); lamp2.setOnClickListener(lampHandler);

	// Button press event listener
	sendButton.setOnClickListener(new View.OnClickListener() 
	    {
		public void onClick(View v) 
		{
		    messsage = textField.getText().toString(); // get the text message on the text field
		    textField.setText(""); // Reset the text field to blank
		    SendMessage sendMessageTask = new SendMessage();
		    sendMessageTask.execute();
		}
	    });
	
	// TextView txtSampleText = (TextView) mView
	//     .findViewById(R.id.txtViewSample);
	// txtSampleText.setText(sampleText);
	
	return mView;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    private class SendMessage extends AsyncTask<Void, Void, String> 
    {
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
	@Override protected String doInBackground(Void... params) 
	    {
	    try 
		{
		    if (messsage.length() == 0) return null;

		    //Log.i("Thread: ",serverName+":"+Integer.toString(serverPort));
		    
		    client = new Socket(serverName, serverPort);

		    //		    client = new Socket(serverName, serverPort);
		    //		    client = new Socket("10.0.2.2", 1234); // connect to the server on local machine
		    //client = new Socket("raspberrypi", 1234); // connect to the Naarad server
		    printwriter = new PrintWriter(client.getOutputStream(), true);
		    
		    printwriter.write(mkMessage("open"));
		    printwriter.flush();
		    SystemClock.sleep(500);
		    
		    printwriter.write(mkMessage(messsage)); // write the message to output stream
		    // printwriter.write((messsage)); // write the message to output stream
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
	@Override protected void onPostExecute(String result) 
	    {
		super.onPostExecute(result);
		if (result != ALL_WELL)		
		    {
			Toast.makeText(getActivity(), result, Toast.LENGTH_SHORT).show();
			currentToggleButton.setChecked(!currentToggleButton.isChecked());
		    }
	    }
    }
}
