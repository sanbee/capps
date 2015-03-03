package naarad.client.tabhost;

import android.util.Log;
import java.util.ArrayList;
import java.util.List;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
// import android.support.v4.view.ViewPager;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.support.v4.view.ViewPager.OnPageChangeListener;
import android.widget.TabHost;
import android.widget.TextView;
import android.widget.TabHost.OnTabChangeListener;
import android.text.Spanned;
import android.text.Html;
import android.graphics.Color;
//import java.lang.Integer;
/**
 * ---------------------------------------------------------------------------------------Freely inspired by: 
 * --------------- http://thepseudocoder.wordpress.com/2011/10/13/android-tabs-viewpager-swipe-able-tabs-ftw/
 */

public class MainActivity extends FragmentActivity implements
						       OnTabChangeListener, OnPageChangeListener,
						       NaaradPlotFragment.nPlotDataArrivalListener 
{
    
    MyPageAdapter pageAdapter;
    private MyViewPager mViewPager=null;
    private TabHost mTabHost;
    private NaaradControlFragment gControlFragment;
    private TextView b1;
    //private int defTextColor1;

    public void onDataArrival(String json)
    {
	//NaaradControlFragment frag = (NaaradControlFragment)pageAdapter.getItem(0);
	// NaaradControlFragment frag = (NaaradControlFragment)
	//     getSupportFragmentManager()
	//     .findFragmentById(R.id.viewpager);

	// if (frag==null)
	//     System.err.println("$@#%@# - it's NULL!");
	// else
	//     //gControlFragment.setSensorValues(json);
	//     frag.setSensorValues(json);

	b1 = (TextView) gControlFragment.setSensorValues(json);

	//
	// Give a visual feedback for the arrival of a RF packet.
	// Blink a dot in the title of the sensor data tab.  
	//
	// This is done by (1) creating a Handler attached to the
	// looper in the UI thread, (2) making a Runnable object with
	// the code to change the text of the tab label, (3) setting
	// the Runnable with the label text, posting it and again
	// setting up the Runnable with a "." in the label and posting
	// it with a delay of 200 ms.  These posts via the Handler are
	// queued and executed in the UI thread.
	//
	class mRunnable implements Runnable
	{
	    Spanned theSText=null;
	    String theText=null;
	    int thisColor;
	    
	    public void setText(String thisText,
				int color) 
	    {
		theText=thisText;
		thisColor=color;
	    }
	    public void setText(Spanned thisText,
				int color
				) 
	    {
		theSText=thisText;
		thisColor=color;
		//iThisColor=iColor;
	    }
	    		
	    public void run()
	    {
		TextView label = (TextView) mTabHost.getTabWidget().getChildAt(1).findViewById(android.R.id.title); 
		b1.setTextColor(thisColor);
		// //b1.getBackground().setAlpha(thisAlpha);
		// if (thisAlpha == 0)
		//     {
		// 	//b1.setBackgroundColor(Color.GREEN);
		// 	//b1.setBackgroundResource(R.drawable.speech_bubble_green);
		// 	// String text=b1.getText().toString();
		// 	// theSText = Html.fromHtml("<p><b>"+text+"<sup>.</sup></b></p>");
		// 	// b1.setText(theSText); theSText=null;
		// 	b1.setTextColor(Color.GREEN);
		//     }
		// else
		//     {
		// 	//b1.setBackgroundColor(Color.RED);
		// 	//b1.setBackgroundResource(R.drawable.speech_bubble_l_orange);
		
		// 	// String text=b1.getText().toString();
		// 	// theSText = Html.fromHtml("<p><b>"+text+"</b></p>");
		// 	// b1.setText(theSText); theSText=null;
		// 	b1.setTextColor(Color.WHITE);
		//     }

		if (theSText != null) label.setText(theSText);
		else                  label.setText(theText);
	    }
	}
	final Handler hUpdate = new Handler(Looper.getMainLooper());
	final mRunnable rUpdate = new mRunnable();

	rUpdate.setText("Sensors .",Color.GREEN); hUpdate.post(rUpdate);
	rUpdate.setText("Sensors  ",Color.WHITE); hUpdate.postDelayed(rUpdate, 250);
	//Spanned tmp = Html.fromHtml("<p>Sensors<b> .</b></p>");
	//rUpdate.setText(tmp,"");	hUpdate.post(rUpdate);
	//	tmp = Html.fromHtml("<p>Sensors<b>  </b></p>");
	//rUpdate.setText(tmp,"");	hUpdate.postDelayed(rUpdate, 250);


	// Thread tUpdate = new Thread() 
	//     {
	// 	public void run() 
	// 	{
	// 	    // TextView label = (TextView) mTabHost.getTabWidget().getChildAt(1).findViewById(android.R.id.title); 
	// 	    // int defColor = label.getCurrentTextColor();
		    
	// 	    // //String hexColor = String.format("#%06X", (0xFFFFFF & defColor));
	// 	    // String hexColor = Integer.toHexString(defColor);

	// 	    //Spanned tmp = Html.fromHtml("<p>"+"Sensors<b>."+"</b><font size =\"50\" color=\"green\"></font></p>");
	// 	    //rUpdate.setText("Sensors","green");
	// 	    //rUpdate.setText("Sensors.",Color.parseColor("green"));
	// 	    //rUpdate.setText(tmp,"");
	// 	    rUpdate.setText("Sensors.","");
	// 	    hUpdate.post(rUpdate);

	// 	    //SystemClock.sleep(200);
		    
	// 	    //tmp = Html.fromHtml("<p><b>"+"Sensors."+"</b><font size =\"50\" color=\"#0066FF\"></font></p>");
	// 	    //tmp = Html.fromHtml("<p>"+"Sensors<b> "+"</b><font size =\"50\" color=\"white\"></font></p>");
		    
	// 	    //rUpdate.setText("Sensors","white");
	// 	    //rUpdate.setText("Sensors",defTextColor1);
	// 	    //rUpdate.setText("Sensors ",Color.parseColor("white"));
	// 	    //rUpdate.setText(tmp,"");
	// 	    rUpdate.setText("Sensors ","");
		    
	// 	    //hUpdate.post(rUpdate);
	// 	    hUpdate.postDelayed(rUpdate,200);
	// 	}
	//     };
	//tUpdate.start();
    }

    // @Override public void onPause()
    // {
    // 	super.onPause();
    // 	System.err.println("MainActivity::onPause()");
    // }

    @Override protected void onResume()
    {
	super.onResume();
	if (this.mViewPager == null) this.mViewPager = (MyViewPager) findViewById(R.id.viewpager);
    }
    @Override protected void onCreate(Bundle savedInstanceState) 
    {
	super.onCreate(savedInstanceState);
	setContentView(R.layout.activity_main);
	
	if (this.mViewPager == null)
	    this.mViewPager = (MyViewPager) findViewById(R.id.viewpager);
	
	// Tab Initialization
	initialiseTabHost();
	
	// Fragments and ViewPager Initialization
	List<Fragment> fragments = getFragments();
	pageAdapter = new MyPageAdapter(getSupportFragmentManager(), fragments);
	this.mViewPager.setAdapter(pageAdapter);
	this.mViewPager.setOnPageChangeListener(MainActivity.this);

	// ((NaaradControlFragment)fragments.get(0)).setViewPager(this.mViewPager);
	// ((NaaradControlFragment)fragments.get(0)).setActivity(this);
    }
    
    // Method to add a TabHost
    private static void AddTab(MainActivity activity, TabHost tabHost,
			       TabHost.TabSpec tabSpec) 
    {
	tabSpec.setContent(new MyTabFactory(activity));
	tabHost.addTab(tabSpec);
    }
    
    // Manages the Tab changes, synchronizing it with Pages
    public void onTabChanged(String tag) 
    {
	int pos = this.mTabHost.getCurrentTab();
	this.mViewPager.setCurrentItem(pos);
	// TextView label = (TextView) mTabHost.getTabWidget().getChildAt(1).findViewById(android.R.id.title); 
	// defTextColor1 = label.getCurrentTextColor();
    }
    
    // Manages the Page changes, synchronizing it with Tabs
    @Override public void onPageScrolled(int arg0, float arg1, int arg2) 
    {
	int pos = this.mViewPager.getCurrentItem();
	this.mTabHost.setCurrentTab(pos);

	// Log.i("Main: ","Tab no. "+Integer.toString(pos));

	// fragmentTransaction.setCustomAnimations(animEnter, animExit, animPopEnter, animPopExit);
	// fragmentTransaction.add(android.R.id.content, fragment,"MyStringIdentifierTag");
	// fragmentTransaction.addToBackStack(null);
	// fragmentTransaction.commit();
    }
    
    
    private List<Fragment> getFragments() 
    {
	List<Fragment> fList = new ArrayList<Fragment>();
	
	// TODO Put here your Fragments
	//MySampleFragment f1 = MySampleFragment.newInstance("Sample Fragment 1");
	NaaradControlFragment f1 = NaaradControlFragment.newInstance("Naarad Control Fragment");
        // Capture the article fragment from the activity layout
        gControlFragment = f1;
	
	//MySampleFragment f2 = MySampleFragment.newInstance("Sample Fragment 2");
	NaaradPlotFragment    f2 = NaaradPlotFragment.newInstanceNPF("Naarad Plot Fragment");

	
	//MySampleFragment f3 = MySampleFragment.newInstance("Sample Fragment 3");
	NaaradSettingFragment f3 = NaaradSettingFragment.newInstanceNSF("Naarad Setting Fragment");
	
	
	fList.add(f1);
	fList.add(f2);
	fList.add(f3);
	
	return fList;
    }
    
    // Tabs Creation
    private void initialiseTabHost() 
    {
	mTabHost = (TabHost) findViewById(android.R.id.tabhost);
	mTabHost.setup();
	
	// TODO Put here your Tabs
	MainActivity.AddTab(this, this.mTabHost,
			    this.mTabHost.newTabSpec("Controls").setIndicator("Controls"));
	MainActivity.AddTab(this, this.mTabHost,
			    this.mTabHost.newTabSpec("Sensors  ").setIndicator("Sensors  "));
	MainActivity.AddTab(this, this.mTabHost,
			    this.mTabHost.newTabSpec("Settings").setIndicator("Settings"));
	mTabHost.getTabWidget().getChildAt(0).getLayoutParams().height =40;
	mTabHost.getTabWidget().getChildAt(1).getLayoutParams().height =40;
	mTabHost.getTabWidget().getChildAt(2).getLayoutParams().height =40;
	mTabHost.setOnTabChangedListener(this);
    }
    //
    // MainActivity is an abstract class and the following methods
    // needs an implementation.
    //
    @Override public void onPageScrollStateChanged(int arg0) {}
    @Override public void onPageSelected(int arg0) {}
}
