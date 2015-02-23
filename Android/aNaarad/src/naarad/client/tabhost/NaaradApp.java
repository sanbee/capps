package naarad.client.tabhost;
//import android.net.wifi.WifiManager.WifiLock;
import android.os.PowerManager.WakeLock;
import android.util.DisplayMetrics;
import android.app.Application;
import android.os.PowerManager;
import android.net.wifi.WifiManager;
import android.net.wifi.WifiManager.WifiLock;

public class NaaradApp extends Application 
{
    public boolean swipeEnabled=true;
    public float densityDpi;
    public WakeLock myWakeLock=null;
    public WifiLock myWifiLock=null;

    public boolean getSwipeState()
    {
	return swipeEnabled;
    }

    public void setSwipeState(boolean b)
    {
	swipeEnabled = b;
    }

    @Override public void onCreate() 
    {
        super.onCreate();
	//DisplayMetrics dm = getResources().getDisplayMetrics(); 
	//densityDpi = dm.densityDpi;
	densityDpi = getResources().getDisplayMetrics().density;

	swipeEnabled=true;

	PowerManager wakeLockPM = (PowerManager) getSystemService(POWER_SERVICE);
	WifiManager wifiLockPM = (WifiManager) getSystemService(WIFI_SERVICE);
	if (myWakeLock == null)
	    {
		myWakeLock = wakeLockPM.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MyWakelockTag");
		myWifiLock = wifiLockPM.createWifiLock(WifiManager.WIFI_MODE_FULL, "MyWifiLockTag");

		System.err.println("Making wakelock");
	    }
    }
    public int dpToPixel(int dp)
    {
	return (int)(dp*densityDpi);
    }
}
