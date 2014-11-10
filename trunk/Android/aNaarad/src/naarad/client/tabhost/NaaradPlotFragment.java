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
    private XYSeries mCurrentSeries;
    private XYSeriesRenderer mCurrentRenderer;
    private String mDateFormat;
    private Button mNewSeries;
    private Button mAdd;
    private GraphicalView mChartView, mTimeChartView;
    private int index = 0;
    // Calendar cal = Calendar.getInstance(TimeZone.getTimeZone("MST"));
    // long offset = cal.get(Calendar.ZONE_OFFSET) + cal.get(Calendar.DST_OFFSET);
    // long time = cal.getTimeInMillis();

    //static double x = System.currentTimeMillis()+new SimpleTimeZone().getRawOffset();
    // Date d=new Date();
    //double x =  d.getTime() + d.get(Calendar.ZONE_OFFSET) + d.get(Calendar.DST_OFFSET);
    //double x =  new Date().getTime() - offset;
    // static double x = time;
    //static double x = 0;
    static double x = new Date().getTime();
    static double y = 0;
    protected Update mUpdateTask;


    private static View mView;
    
    private Socket client;
    private PrintWriter printwriter;
    private EditText textField;
    private ToggleButton plotButton;
    private String messsage;
    private DynamicDataSource dataSource;
    private Thread myThread;

    @Override public void onResume() 
    {
        // // kick off the data generating thread:
        // myThread = new Thread(dataSource);
        // myThread.start();

	//mUpdateTask.execute();
        super.onResume();
    }
    
    @Override public void onPause() 
    {
	mUpdateTask.cancel(true);
	// x=0;
	// y=0;
        // dataSource.stopThread();
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
    
    //
    //-----------------------------------------------------------------------------------------
    //
    @Override public View onCreateView(LayoutInflater inflater, ViewGroup container,
				       Bundle savedInstanceState) 
    {
        // dataSource = new DynamicDataSource();
	//	myThread = new Thread(dataSource);
	mView = inflater.inflate(R.layout.activity_naarad_plot,
				 container, false);
	String sampleText = getArguments().getString("bString");
	// textField  = (EditText) mView.findViewById(R.id.editText1); // reference to the text field
	plotButton = (ToggleButton)  mView.findViewById(R.id.plotButton); // reference to the send button
	// Button press event listener
	// plotButton.setOnClickListener(new View.OnClickListener() 
	//     {
	// 	public void onClick(View v) 
	// 	{
	// 	    messsage = textField.getText().toString(); // get the text message on the text field
	// 	    textField.setText(""); // Reset the text field to blank
	// 	    Log.i("Msg: ",messsage);
		    
	// 	    boolean on = ((ToggleButton)v).isChecked();
	// 	    if (on) 
	// 	    	{
	// 	    	    myThread = new Thread(dataSource);
	// 	    	    myThread.start();
	// 	    	}
	// 	    else 
	// 	    	{
	// 	    	    dataSource.stopThread();
	// 	    	}
			
	// 	    // myThread = new Thread(dataSource);
	// 	    // myThread.start();
	// 	}
	//     });

	//
	//--------------------------------------------------------------------------
	//    
	// setContentView(R.layout.xy_chart);
	
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
	mRenderer.setXLabels(10); // No. of xtics
	mRenderer.setClickEnabled(false);
	mRenderer.setPanEnabled(false);
	//mRenderer.setXAxisMax(x+360000.0);
	
	String seriesTitle = "Series " + (mDataset.getSeriesCount() + 1);
	XYSeries series = new XYSeries(seriesTitle);
	mDataset.addSeries(series);
	mCurrentSeries = series;
	XYSeriesRenderer renderer = new XYSeriesRenderer();
	mRenderer.addSeriesRenderer(renderer);
	renderer.setPointStyle(PointStyle.CIRCLE);
	renderer.setFillPoints(true);
	renderer.setLineWidth(2f);
	mCurrentRenderer = renderer;

	LinearLayout layout = (LinearLayout) mView.findViewById(R.id.chart);
	//mChartView = ChartFactory.getLineChartView(getActivity(), mDataset, mRenderer);
	mTimeChartView = ChartFactory.getTimeChartView(getActivity(), mDataset, mRenderer,"hh:mm:ss\ndd/MM");	// "%tT"

	LayoutParams lp = new LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT);
	//layout.addView(mChartView, lp);
	layout.addView(mTimeChartView,0, lp);
	//mChartView.repaint();
	mTimeChartView.repaint();

	mUpdateTask = new Update();
	mUpdateTask.execute();
	//
	//--------------------------------------------------------------------------
	//    

	return mView;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    private int generateRandomNum() 
    {
	Random randomGenerator = new Random();
	int randomInt = randomGenerator.nextInt(100);
	return randomInt;
    }
    //
    //-----------------------------------------------------------------------------------------
    //
    protected class Update extends AsyncTask<Context, Integer, String> 
    {
	@Override protected String doInBackground(Context... params) 
	    {
		int i = 0;
		while (true)
		    {
			if (isCancelled()) break;
			try 
			    {
				Thread.sleep(1000);
				x = new Date().getTime();

				//x = x + 5000;
				y = generateRandomNum();
				
				publishProgress(i);
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
	public void addNewData(double x,double y)
	{
	    double xMax,xMin, dX=0, dT=360000.0;
	    int n;
	    //mRenderer.setXAxisMax(x+360000.0);	    
	    xMax = mCurrentSeries.getMaxX();
	    xMin = mCurrentSeries.getMinX();
	    n=mCurrentSeries.getItemCount();
	    if (n>0)
		{
		    xMax = mCurrentSeries.getX(n-1);
		    xMin = mCurrentSeries.getX(0);
		    dX=xMax-xMin;
		    // Log.i("xrange0",Double.toString(xMax)+" "+Double.toString(xMin)+" "+Double.toString(dX)+" "+Integer.toString(n));
		    if (dX > dT) mCurrentSeries.remove(0);
		}
	    mCurrentSeries.add(x, y);
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
	@Override protected void onProgressUpdate(Integer... values) 
	    {
		super.onProgressUpdate(values);
		
		//mCurrentSeries.add(x, y);
		addNewData(x,y);
		
		// if (mChartView != null) 
		//     {
		// 	mChartView.repaint();
		//     }
		if (mTimeChartView != null) 
		    {
			mTimeChartView.zoomReset();
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
		super.onCancelled();
	    }
	
	@Override protected void onPostExecute(String result) 
	    {
		super.onPostExecute(result);
	    }
    }
    
    
}
