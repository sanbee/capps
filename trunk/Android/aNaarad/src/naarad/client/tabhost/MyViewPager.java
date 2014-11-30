package naarad.client.tabhost;
import android.content.Context;
import android.view.MotionEvent;
import android.util.AttributeSet;
import android.support.v4.view.ViewPager;

public class MyViewPager extends ViewPager {

    private boolean enabled;

    public MyViewPager(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.enabled = true;
    }

    public MyViewPager(Context context) {
        super(context);
        this.enabled = false;
    }

    public void enableSwipe(boolean enable) {
	this.enabled=enable;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (this.enabled) {
            return super.onTouchEvent(event);
        }

        return false;
    }

    @Override
    public boolean onInterceptTouchEvent(MotionEvent event) {
        if (this.enabled) {
            return super.onInterceptTouchEvent(event);
        }

        return false;
    }

    public void setPagingEnabled(boolean enabled) {
        this.enabled = enabled;
    }
}