package naarad.client.tabhost;
import android.app.Application;
import android.util.DisplayMetrics;

public class NaaradApp extends Application 
{
    public boolean swipeEnabled=true;
    public float densityDpi;

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
    }
}
