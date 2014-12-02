package naarad.client.tabhost;
import android.app.Application;

public class NaaradApp extends Application 
{
    private boolean swipeEnabled=true;

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
	swipeEnabled=true;
    }
}
