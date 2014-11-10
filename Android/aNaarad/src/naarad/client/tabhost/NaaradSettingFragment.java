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

public class NaaradSettingFragment extends Fragment 
{
    private static View mView;
    
    private Socket client;
    private PrintWriter printwriter;
    private EditText textField;
    private Button connectButton;
    private String messsage;
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
	mView = inflater.inflate(R.layout.activity_naarad_setting,
				 container, false);
	String sampleText = getArguments().getString("bString");
	textField  = (EditText) mView.findViewById(R.id.editText1); // reference to the text field
	connectButton = (Button)  mView.findViewById(R.id.connectButton); // reference to the send button

	// Button press event listener
	connectButton.setOnClickListener(new View.OnClickListener() 
	    {
		public void onClick(View v) 
		{
		    messsage = textField.getText().toString(); // get the text message on the text field
		    textField.setText(""); // Reset the text field to blank
		    Log.i("Msg: ",messsage);
		}
	    });
	return mView;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    private class SendMessage extends AsyncTask<Void, Void, Void> 
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
	@Override protected Void doInBackground(Void... params) 
	    {
	    try 
		{
		    if (messsage.length() == 0) return null;
		    //client = new Socket("10.0.2.2", 1234); // connect to the server on local machine
		    client = new Socket("raspberrypi", 1234); // connect to the Naarad server
		    printwriter = new PrintWriter(client.getOutputStream(), true);
		    
		    printwriter.write(mkMessage("open"));
		    printwriter.flush();
		    SystemClock.sleep(500);
		    
		    printwriter.write(mkMessage(messsage)); // write the message to output stream
		    // printwriter.write((messsage)); // write the message to output stream
		    printwriter.flush();
		    SystemClock.sleep(500);
		    
		    // Handler handler = new Handler(); 
		    // handler.postDelayed(new Runnable() { 
		    // 	public void run() { 
		    // 	    //			    my_button.setBackgroundResource(R.drawable.defaultcard); 
		    // 	} 
		    //     }, 1000); 
		    printwriter.write(mkMessage("done"));
		    printwriter.flush();
		    //SystemClock.sleep(500);
		    
		    printwriter.close();
		    client.close(); // closing the connection
		    
		} 
	    catch (UnknownHostException e) 
		{
		    e.printStackTrace();
		} 
	    catch (IOException e) 
		{
		    e.printStackTrace();
		}
	    return null;
	    }
    }
}
