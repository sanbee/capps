package com.lakj.comspace.simpleclient;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

import android.app.Activity;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ToggleButton;
import android.widget.EditText;
import android.util.Log;
import android.os.Handler;
import android.os.SystemClock;
import android.util.Log;
/**
 * This is a simple Android mobile client
 * This application read any string massage typed on the text field and 
 * send it to the server when the Send button is pressed
 * Author by Lak J Comspace
 *
 */

public class NaaradClientActivity extends Activity {
    
    private Socket client;
    private PrintWriter printwriter;
    private EditText textField;
    private Button sendButton,initButton;
    private ToggleButton lamp0, lamp1, lamp2;
    private String messsage;
    
    @Override
	protected void onCreate(Bundle savedInstanceState) {
	super.onCreate(savedInstanceState);
	setContentView(R.layout.activity_naarad_client);
	
	View.OnClickListener lampHandler = new View.OnClickListener()
	{
	    public void onClick(View v) {
		boolean on = ((ToggleButton)v).isChecked();
		messsage="tell "+v.getTag()+" ";
		// Log.i("lamp" + v.getTag() + ": ","clicked");
		if (on)	messsage += "1";
		else	messsage += "0";
		SendMessage sendMessageTask = new SendMessage();
		sendMessageTask.execute();
	    }
	};

	textField = (EditText) findViewById(R.id.editText1); // reference to the text field
	sendButton = (Button) findViewById(R.id.sendButton); // reference to the send button
	initButton = (Button) findViewById(R.id.initButton); // reference to the send button
	lamp0 = (ToggleButton) findViewById(R.id.lamp0); // reference to the send button
	lamp1 = (ToggleButton) findViewById(R.id.lamp1); // reference to the send button
	lamp2 = (ToggleButton) findViewById(R.id.lamp2); // reference to the send button

	lamp0.setTag("0"); lamp0.setOnClickListener(lampHandler);
	lamp1.setTag("1"); lamp1.setOnClickListener(lampHandler);
	lamp2.setTag("2"); lamp2.setOnClickListener(lampHandler);

	// Button press event listener
	sendButton.setOnClickListener(new View.OnClickListener() {
		
		public void onClick(View v) {
		    messsage = textField.getText().toString(); // get the text message on the text field
		    textField.setText(""); // Reset the text field to blank
		    SendMessage sendMessageTask = new SendMessage();
		    sendMessageTask.execute();
		}
	    });
	// lamp0.setOnClickListener(new View.OnClickListener() {
		
	// 	public void onClick(View v) {
	// 	    boolean on = lamp0.isChecked();
	// 	    Log.i("lamp" + v.getTag() + ": ","clicked");
	// 	    if (on)
	// 		messsage = "tell 0 1";
	// 	    else
	// 		messsage = "tell 0 0";
	// 	    SendMessage sendMessageTask = new SendMessage();
	// 	    sendMessageTask.execute();
	// 	}
	//     });
	// lamp1.setOnClickListener(new View.OnClickListener() {
		
	// 	public void onClick(View v) {
	// 	    boolean on = lamp1.isChecked();
	// 	    if (v.getId() == R.id.lamp1) Log.i("lamp1: ","clicked");
	// 	    if (on)
	// 		messsage = "tell 1 1";
	// 	    else
	// 		messsage = "tell 1 0";
	// 	    SendMessage sendMessageTask = new SendMessage();
	// 	    sendMessageTask.execute();
	// 	}
	//     });

	// lamp2.setOnClickListener(new View.OnClickListener() {
		
	// 	public void onClick(View v) {
	// 	    boolean on = lamp2.isChecked();
	// 	    if (v.getId() == R.id.lamp2) Log.i("lamp2: ","clicked");
	// 	    if (on)
	// 		messsage = "tell 2 1";
	// 	    else
	// 		messsage = "tell 2 0";
	// 	    SendMessage sendMessageTask = new SendMessage();
	// 	    sendMessageTask.execute();
	// 	}
	//     });
    }
    
    private class SendMessage extends AsyncTask<Void, Void, Void> {
	
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
	
	@Override
	    protected Void doInBackground(Void... params) {
	    try {
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
		
	    } catch (UnknownHostException e) {
		e.printStackTrace();
	    } catch (IOException e) {
		e.printStackTrace();
	    }
	    return null;
	}
	
    }
    
    @Override
	public boolean onCreateOptionsMenu(Menu menu) {
	// Inflate the menu; this adds items to the action bar if it is present.
	getMenuInflater().inflate(R.menu.slimple_text_client, menu);
	return true;
    }
    
}
