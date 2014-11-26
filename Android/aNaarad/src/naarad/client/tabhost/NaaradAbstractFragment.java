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

public abstract class NaaradAbstractFragment extends Fragment 
{
    private SharedPreferences prefs; 
    
    // client = new Socket("10.0.2.2", 1234); // connect to the server on local machine
    // client = new Socket("raspberrypi", 1234); // connect to the Naarad server
    final public int getDefaultPort() {return 1234;}
    final public String getDefaultServer() {return "10.0.2.2";}

    public boolean recreateView(View v)
    {
	if (v != null)
	    {
		// Do not inflate the layout again.
		// The returned View of onCreateView will be added into the fragment.
		// However it is not allowed to be added twice even if the parent is same.
		// So we must remove root View (nView) from the existing parent view group
		// (it will be added back).
		((ViewGroup)v.getParent()).removeView(v);
		return true;
	    }
	else return false;
    }

    public void setServerName(String name)
    {
	prefs = getActivity().getSharedPreferences("nSettings", Context.MODE_PRIVATE); 
	SharedPreferences.Editor editor = prefs.edit();
	editor.putString("serverName", name);  //or you can use putInt, putBoolean ... 
	editor.commit();
    }
    public String getServerName()    
    {
	prefs = getActivity().getSharedPreferences("nSettings", Context.MODE_PRIVATE);
	return prefs.getString("serverName",getDefaultServer());
    }


    public void setServerPort(int port)
    {
	prefs = getActivity().getSharedPreferences("nSettings", Context.MODE_PRIVATE); 
	SharedPreferences.Editor editor = prefs.edit();
	editor.putInt("serverPort", port);  //or you can use putInt, putBoolean ... 
	editor.commit();
    }
    public int getServerPort()    
    {
	prefs = getActivity().getSharedPreferences("nSettings", Context.MODE_PRIVATE);
	return prefs.getInt("serverPort",getDefaultPort());
    }

    
}
