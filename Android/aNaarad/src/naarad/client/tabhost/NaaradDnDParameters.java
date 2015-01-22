package naarad.client.tabhost;
import android.view.View;

public class NaaradDnDParameters
{
    // The item which was clicked.  
    public View selected_item; 

    // The location at which longPress event happened.
    public int longPress_x, longPress_y;

    // Set in GestureDetector on LongPress and used in OnTouchListener
    // ACTION_MOVE to call containerOnTouch() to do the actual move of
    // the selected_item.
    public boolean touchFlag_p=false; 
}