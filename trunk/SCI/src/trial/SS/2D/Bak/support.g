include 'images.g'

function imgtoasc(val FileName)
{
    local Box, Mask, Shape,Arr,AsciiImage;

    im:=image(FileName);
    Shape:=im.shape();
    NX:=Shape[1]; NY:=Shape[2];
    Box:=im.box(blc=[1,1,1],trc=[NX,NY,1]);
    im.getregion(Arr,Mask,Box);

    AsciiImage:=spaste(FileName,".ascii");
    fd := open(AsciiImage);

    fprintf(fd,"%f %f\n", NX,NY);
    write(fd);

    for(i in 1:NX)
	for(j in 1:NY)
	{
	    fprintf(fd,"%f\n",Arr[i,j]);
	    write(fd);
	}
}
