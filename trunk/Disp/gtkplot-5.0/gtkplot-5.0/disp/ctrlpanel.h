#ifndef FD_ctrlpanel_h_
#define FD_ctrlpanel_h_
/* Header file generated with fdesign. */

/**** Callback routines ****/

void XMax_cb(FL_OBJECT *ob, long data);
void XMin_cb(FL_OBJECT *ob, long data);
void YMax_cb(FL_OBJECT *ob, long data);
void YMin_cb(FL_OBJECT *ob, long data);
void ResetSliders();
void Quit_cb(FL_OBJECT *ob, long data);
void Reset_cb(FL_OBJECT *ob, long data);
void FreezX_cb(FL_OBJECT *ob, long data);
void FreezY_cb(FL_OBJECT *ob, long data);

/**** Forms and Objects ****/

typedef struct {
	FL_FORM *ctrlpanel;
	void *vdata;
	long ldata;
} FD_ctrlpanel;

extern FD_ctrlpanel * create_form_ctrlpanel(void);

#endif /* FD_ctrlpanel_h_ */
