package naarad.client.tabhost;

import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.widget.TextView;
import android.view.ViewGroup;
import android.view.View;
import android.os.Bundle;

import android.widget.ToggleButton;
import android.widget.EditText;
import android.os.SystemClock;
import android.widget.Button;
import android.os.AsyncTask;
import android.os.Handler;
import android.util.Log;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Random;

import org.achartengine.ChartFactory;
import org.achartengine.GraphicalView;
import org.achartengine.chart.PointStyle;
import org.achartengine.util.MathHelper;

import org.achartengine.model.SeriesSelection;
import org.achartengine.model.XYMultipleSeriesDataset;
import org.achartengine.model.XYSeries;
import org.achartengine.renderer.XYMultipleSeriesRenderer;
import org.achartengine.renderer.XYSeriesRenderer;
import org.achartengine.tools.PanListener;
import org.achartengine.tools.ZoomEvent;
import org.achartengine.tools.ZoomListener;
import android.graphics.Color;
import android.widget.LinearLayout;
import android.content.Context;
import android.view.ViewGroup.LayoutParams;
import android.os.Build.VERSION;
import java.util.Date;

// import java.util.Calendar;
// import java.util.TimeZone;

class DynamicDataSource implements Runnable 
{
    int i=0;
    private boolean keepRunning = true;
    
    // @Override 
    public void run() 
    {
	i=0;
	if (keepRunning == true)
	    {
		keepRunning = false;
		Log.i("Run: ","stopping");
	    }
	else
	    {
		keepRunning=true;
		Log.i("Run: ","starting");
	    }
	while(keepRunning)
	    {
		Log.i("Run: ",Integer.toString(i++));
		SystemClock.sleep(1000);
	    }
    }
    public void stopThread() 
    {
	keepRunning = false;
    }
}

public class NaaradPlotFragment extends Fragment 
{
    private XYMultipleSeriesDataset mDataset = new XYMultipleSeriesDataset();
    private XYMultipleSeriesRenderer mRenderer = new XYMultipleSeriesRenderer();
    //    private XYSeriesRenderer mCurrentRenderer;
    private String mDateFormat;
    private Button mNewSeries;
    private Button mAdd;
    private GraphicalView mChartView, mTimeChartView;
    private int index = 0;
    protected Update mUpdateTask0, mUpdateTask1;


    private int apiLevel;
    private static View mView;
    
    private Socket client;
    private PrintWriter printwriter;
    private EditText textField;
    private ToggleButton plotButton;
    private String messsage;
    private DynamicDataSource dataSource;
    private Thread myThread;

    private XYSeries series0, series1;
    private XYSeriesRenderer renderer0,renderer1;

    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
	apiLevel=android.os.Build.VERSION.SDK_INT;
	Log.i("API Level: ",android.os.Build.VERSION.RELEASE+" "+Integer.toString(apiLevel));
	setHasOptionsMenu(true);
	mView = inflater.inflate(R.layout.activity_naarad_plot, container, false);
	//
	//--------------------------------------------------------------------------
	//    
	// setContentView(R.layout.xy_chart);
	initMultiRenderer(mRenderer);
	makeSeries(mRenderer, mDataset, Color.GREEN);
	makeSeries(mRenderer, mDataset, Color.BLUE);
	//
	//--------------------------------------------------------------------------
	//    
	plotButton = (ToggleButton)  mView.findViewById(R.id.plotButton); // reference to the send button
	//	plotButton.setChecked(true);

	// Button press event listener
	//plotButton.setOnClickListener(new View.OnClickListener() 
	View.OnClickListener plotButtonHandler = new View.OnClickListener() 
	    {
		public void onClick(View v) 
		{
		    boolean on = ((ToggleButton)v).isChecked();
		    if (on) 
		    	{
			    makeChart(mRenderer,mDataset);
			    startAllCharts();
		    	}
		    else 
		    	{
			    stopAllCharts();
		    	}
		}
	    };
	plotButton.setOnClickListener(plotButtonHandler);
	//
	//--------------------------------------------------------------------------
	//    

	return mView;
    }
    @Override public void onResume() 
    {
        // // kick off the data generating thread:
        // myThread = new Thread(dataSource);
        // myThread.start();

	//mUpdateTask0.execute();
        super.onResume();
    }
    
    @Override public void onPause() 
    {
	if (mUpdateTask0 != null) mUpdateTask0.cancel(true);
	if (mUpdateTask1 != null) mUpdateTask1.cancel(true);

	super.onPause();
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    public static final NaaradPlotFragment newInstanceNPF(String sampleText) 
    {
	NaaradPlotFragment f = new NaaradPlotFragment();
	
	Bundle b = new Bundle();
	b.putString("bString", sampleText);
	f.setArguments(b);
	
	return f;
    }
    
    private void makeSeries(XYMultipleSeriesRenderer multiRenderer,
			   XYMultipleSeriesDataset multiDataset,
			   int color)
    {
	String seriesTitle = "Series " + (multiDataset.getSeriesCount() + 1);
	XYSeries series0 = new XYSeries(seriesTitle);
	multiDataset.addSeries(series0);
	XYSeriesRenderer renderer0 = new XYSeriesRenderer();
	renderer0.setPointStyle(PointStyle.CIRCLE);
	renderer0.setFillPoints(true);
	renderer0.setLineWidth(2f);
	//renderer0.setColor(Color.GREEN);
	renderer0.setColor(color);
	multiRenderer.addSeriesRenderer(renderer0);
    }
    private void initMultiRenderer(XYMultipleSeriesRenderer multiRenderer)
    {
	mRenderer.setApplyBackgroundColor(true);
	mRenderer.setBackgroundColor(Color.argb(100, 50, 50, 50));
	mRenderer.setAxisTitleTextSize(10);
	mRenderer.setChartTitleTextSize(10);
	mRenderer.setLabelsTextSize(10);
	mRenderer.setLegendTextSize(10);
	mRenderer.setMargins(new int[] { 2, 20, 5, 2 }); //Top, Left, Bottom, Right
	mRenderer.setZoomButtonsVisible(false);
	mRenderer.setPointSize(2);
	mRenderer.setXTitle("");
	mRenderer.setYTitle("Temperature");
	mRenderer.setShowGrid(true);
	mRenderer.setShowLegend(false);
	//mRenderer.setShowLegend(true);
	mRenderer.setXLabels(10); // No. of xtics
	mRenderer.setClickEnabled(false);
	mRenderer.setPanEnabled(false);
    }
    public void makeChart(XYMultipleSeriesRenderer multiRenderer,
			  XYMultipleSeriesDataset multiDataset)
    {
	mTimeChartView = ChartFactory.getTimeChartView(getActivity(), mDataset, mRenderer,"hh:mm:ss\ndd/MM");	// "%tT"
	
	LinearLayout layout = (LinearLayout) mView.findViewById(R.id.chart);
	LayoutParams lp = new LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT);
	layout.addView(mTimeChartView,0, lp);
    }
    public void startAllCharts()
    {
	if (mUpdateTask0 != null) mUpdateTask0.cancel(true);
	if (mUpdateTask1 != null) mUpdateTask1.cancel(true);
	mUpdateTask0 = new Update();
	mUpdateTask1 = new Update();
	
	mUpdateTask0.stopRecording(false);
	if (apiLevel <= 9)
	    mUpdateTask0.execute(mDataset.getSeriesAt(0));
	else
	    mUpdateTask0.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR,mDataset.getSeriesAt(0));

	mUpdateTask1.stopRecording(false);
	if (apiLevel <= 9)
	    mUpdateTask1.execute(mDataset.getSeriesAt(1));
	else
	    mUpdateTask1.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR,mDataset.getSeriesAt(1));
    }
    public void stopAllCharts()
    {
	if (mUpdateTask0 != null) 
	    {
		mUpdateTask0.stopRecording(true);
		mUpdateTask0.cancel(true);
	    }
	if (mUpdateTask1 != null) 
	    {
		mUpdateTask1.stopRecording(true);
		mUpdateTask1.cancel(true);
	    }
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    //
    //-----------------------------------------------------------------------------------------
    //
    private int generateRandomNum(int n) 
    {
	Random randomGenerator = new Random();
	int randomVal = randomGenerator.nextInt(n);
	return randomVal;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    protected class Update extends AsyncTask<XYSeries, XYSeries, String> 
    {
	protected double x0,y0, xMax, xMin, yMax, yMin;
	protected XYSeries thisSeries;
	protected String seriesTitle;
	protected Boolean stopRecording=false;
	
	private void stopRecording(Boolean status)
	{
	    stopRecording = status;
	}

	@Override protected String doInBackground(XYSeries... params) 
	    {
		int i = 0;
		thisSeries = params[0];
		seriesTitle = thisSeries.getTitle();
		Log.i("onExec: ",seriesTitle+Integer.toString(thisSeries.getItemCount()));
		while (true)
		    {
			if (isCancelled()) break;
			try 
			    {
				Thread.sleep(1*1000);
				//x0 = new Date().getTime();
				x0 =  new Date().getTime();
				y0 = generateRandomNum(100);
				publishProgress(thisSeries);
				i++;
			    } 
			catch (Exception e) 
			    {
			    }
		    }
		String str="done";
		return str;
	    }

	
	// -- gets called just before thread begins
	@Override protected void onPreExecute() 
	    {
		super.onPreExecute();
	    }
	//
	//--------------------------------------------------------------------------
	//    
	public void addNewData(XYSeries thisSeries, double x,double y)
	{
	    double xMax,xMin, dX=0, dT=60000.0*10;
	    int n;
	    //mRenderer.setXAxisMax(x+360000.0);	    
	    // xMax = thisSeries.getMaxX();
	    // xMin = thisSeries.getMinX();
	    // if (x > xMax) xMax = x;
	    // if (x < xMin)) xMin = x;

	    n=thisSeries.getItemCount();
	    if (n>0)
	    	{
	    	    xMax = thisSeries.getX(n-1);
	    	    xMin = thisSeries.getX(0);
	    	    dX=xMax-xMin;
	    	     // Log.i("xrange0",Double.toString(xMax)+" "+Double.toString(xMin)+" "+Double.toString(dX)+" "+Integer.toString(n));
	    	    if (dX > dT) thisSeries.remove(0);
	    	}
	    if ((y != MathHelper.NULL_VALUE) && (y > yMax)) yMax = y;
	    if ((y != MathHelper.NULL_VALUE) && (y < yMin)) yMin = y;
	    thisSeries.add(x, y);

	    // if (dX > dT)
	    // 	{
	    // 	    xMax = mCurrentSeries.getX(n-1);
	    // 	    xMin = mCurrentSeries.getX(0);
	    // 	    dX=xMax-xMin;
	    // 	    Log.i("xrange1",Double.toString(xMax)+" "+Double.toString(xMin)+" "+Double.toString(dX)+" "+Integer.toString(n));
	    // 	}
	}
	//
	//--------------------------------------------------------------------------
	//    
	//	@Override protected void onProgressUpdate(Integer... values) 
	@Override protected void onProgressUpdate(XYSeries... values) 
	    {
		super.onProgressUpdate(values);
		
		//mCurrentSeries.add(x, y);
		//addNewData(mCurrentSeries,x0,y0);
		//		Log.i("onProg: ",seriesTitle+Integer.toString(values[0].getItemCount()));
		addNewData(values[0],x0,y0);
		
		if ((mTimeChartView != null) && (stopRecording==false))
		    {
			mTimeChartView.zoomReset();
			// mRenderer.setYAxisMax(yMax);
			// mRenderer.setYAxisMin(yMin);
			mRenderer.setYAxisMax(500.0);
			mRenderer.setYAxisMin(-500.0);
			mTimeChartView.repaint();
		    }
		// Bitmap bitmap = mChartView.toBitmap();
		// try {
		// 	File file = new File(Environment.getExternalStorageDirectory(),
		// 			"test" + index++ + ".png");
		// 	FileOutputStream output = new FileOutputStream(file);
		// 	bitmap.compress(CompressFormat.PNG, 100, output);
		
		// } catch (Exception e) {
		// 	e.printStackTrace();
		// }
	    }
	
	// -- called if the cancel button is pressed
	@Override protected void onCancelled() 
	    {
		int i=0;
		if (stopRecording)
		    {
			if (seriesTitle != null)
			    {
				Log.i("Log: ","Cancelled "+seriesTitle);
				x0 = new Date().getTime();
				y0 = (double)(MathHelper.NULL_VALUE);
				addNewData(thisSeries, x0,y0);
			    }
		    }
		super.onCancelled();
	    }
	
	@Override protected void onPostExecute(String result) 
	    {
		super.onPostExecute(result);
	    }
    }
    
    
}
