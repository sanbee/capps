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
import java.lang.Integer;
import android.widget.Toast;
import android.graphics.Color;

//public class NaaradSettingFragment extends Fragment 
public class NaaradSettingFragment extends NaaradAbstractFragment
{
    private static View mView;
    // private SharedPreferences prefs; 
    private Socket client;
    private PrintWriter printwriter;
    private EditText serverNameField, serverPortField;
    private Button setButton;
    private String message;
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
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	setRetainInstance(true);	
	if (mView != null)
	    {
		// Do not inflate the layout again.
		// The returned View of onCreateView will be added into the fragment.
		// However it is not allowed to be added twice even if the parent is same.
		// So we must remove root View (nView) from the existing parent view group
		// (it will be added back).
		((ViewGroup)mView.getParent()).removeView(mView);
		return mView;
	    }

	mView = inflater.inflate(R.layout.activity_naarad_setting,
				 container, false);
	serverNameField  = (EditText) mView.findViewById(R.id.serverName); // reference to the text field
	serverPortField  = (EditText) mView.findViewById(R.id.serverPort); // reference to the text field
	setButton = (Button)  mView.findViewById(R.id.setButton); // reference to the send button
	serverPortField.setText(Integer.toString(getDefaultPort()));
	serverNameField.setText(getDefaultServer());

	// Button press event listener
	setButton.setOnClickListener(new View.OnClickListener() 
	    {
		public void onClick(View v) 
		{
		    String serverName;
		    int serverPort=getDefaultPort();
		    serverName = serverNameField.getText().toString(); // get the text message on the text field
		    try
			{
			    serverPort = Integer.parseInt(serverPortField.getText().toString()); // get the text message on the text field
			    if (serverPort <= 0)
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
		    setPreference("serverName",serverName);
		    //setServerPort(serverPort);
		    setPreference("serverPort", serverPort);
		    setPreference("lamp0X", 100);
		    setPreference("lamp0Y", 200);
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
