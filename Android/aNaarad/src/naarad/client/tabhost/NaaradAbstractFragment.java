package naarad.client.tabhost;

import android.content.SharedPreferences;
import android.support.v4.app.Fragment;
import android.content.Context;
import android.view.ViewGroup;
import android.app.Activity;
import android.view.View;
import android.util.Log;


public abstract class NaaradAbstractFragment extends Fragment 
{
    private SharedPreferences prefs; 
    private Activity mActivity=null;        
    // client = new Socket("10.0.2.2", 1234); // connect to the server on local machine
    // client = new Socket("raspberrypi", 1234); // connect to the Naarad server
    final public int getDefaultPort() {return 1234;}
    //final public String getDefaultServer() {return "10.0.2.2";}
    final public String getDefaultServer() {return "raspberrypi";}

    //
    // Hold a reference to the activity and use it in place of
    // getActivity().  This is required since getActivity() may return
    // NULL when the view is created from existing activity (either
    // via swipping or re-start of the app).
    //
    @Override public void onAttach(Activity activity) 
    {
        super.onAttach(activity);
	if (mActivity==null)
        mActivity = activity;
    }

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
	prefs = mActivity.getSharedPreferences("nSettings", Context.MODE_PRIVATE); 
	SharedPreferences.Editor editor = prefs.edit();
	editor.putString("serverName", name);  //or you can use putInt, putBoolean ... 
	editor.commit();
    }
    public String getServerName()    
    {
	prefs = mActivity.getSharedPreferences("nSettings", Context.MODE_PRIVATE);
	return prefs.getString("serverName",getDefaultServer());
    }


    public void setServerPort(int port)
    {
	prefs = mActivity.getSharedPreferences("nSettings", Context.MODE_PRIVATE); 
	SharedPreferences.Editor editor = prefs.edit();
	editor.putInt("serverPort", port);  //or you can use putInt, putBoolean ... 
	editor.commit();
    }
    public int getServerPort()    
    {
	prefs = mActivity.getSharedPreferences("nSettings", Context.MODE_PRIVATE);
	return prefs.getInt("serverPort",getDefaultPort());
    }

    //
    //-----------------------------------------------------------------------------------------
    //
    public String mkMessage(String message) 
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
    
}
