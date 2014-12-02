package naarad.client.tabhost;

import android.util.Log;
import java.util.ArrayList;
import java.util.List;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
// import android.support.v4.view.ViewPager;
import android.support.v4.view.ViewPager.OnPageChangeListener;
import android.widget.TabHost;
import android.widget.TabHost.OnTabChangeListener;
//import java.lang.Integer;

/**
 * ---------------------------------------------------------------------------------------Freely inspired by: 
 * --------------- http://thepseudocoder.wordpress.com/2011/10/13/android-tabs-viewpager-swipe-able-tabs-ftw/
 */

public class MainActivity extends FragmentActivity implements
						       OnTabChangeListener, OnPageChangeListener 
{
    
    MyPageAdapter pageAdapter;
    private MyViewPager mViewPager=null;
    private TabHost mTabHost;
    
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
    }
    
    @Override public void onPageScrollStateChanged(int arg0) 
    {
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
    
    @Override public void onPageSelected(int arg0) 
    {
    }
    
    private List<Fragment> getFragments() 
    {
	List<Fragment> fList = new ArrayList<Fragment>();
	
	// TODO Put here your Fragments
	//MySampleFragment f1 = MySampleFragment.newInstance("Sample Fragment 1");
	NaaradControlFragment f1 = NaaradControlFragment.newInstance("Naarad Control Fragment");
	
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
			    this.mTabHost.newTabSpec("Sensors").setIndicator("Sensors"));
	MainActivity.AddTab(this, this.mTabHost,
			    this.mTabHost.newTabSpec("Settings").setIndicator("Settings"));
	mTabHost.getTabWidget().getChildAt(0).getLayoutParams().height =40;
	mTabHost.getTabWidget().getChildAt(1).getLayoutParams().height =40;
	mTabHost.getTabWidget().getChildAt(2).getLayoutParams().height =40;
	mTabHost.setOnTabChangedListener(this);
    }
}
