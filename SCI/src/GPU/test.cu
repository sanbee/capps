#include <stdio.h>

void cpuflip(int *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
{
#if 1
      int cx=nx/2;
      int cy=ny/2;
 int i, j;

      for (i=0; i<cx; i++)
        for (j=0; j< cy; j++)
          {
            int tmp;
            tmp=buf[i+j*ny];
            buf[i+j*ny] = buf[cx+i + (cy+j)*ny];
            buf[cx+i + (cy+j)*ny] = tmp;
          }
      for ( i=cx; i < nx; i++)
        for (j=0; j < cy; j++)
          {
            int tmp;
            tmp=buf[i-cx +(j+cy)*ny];
            buf[i-cx +(j+cy)*ny] = buf[i + j*ny];
            buf[i + j*ny] = tmp;
          }
#endif
     

}
//buf, 4,4,2,2
__global__ void kernel_flip(int *buf, const int nx, const int ny, const int tileWidthX, const int tileWidthY)
    {
#if 1
//working kernel
      // calculate thread id
      unsigned int i = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int j = tileWidthY*blockIdx.y + threadIdx.y ;
      
      int cx=nx/2, cy=ny/2;
      int tmp=0; 

      if (i < cx  && j <cy) 
        {
          //printf("i=%d, j=%d, tmp=%d, buf[%d]=%d\n", i,j, tmp, (cx+i + (cy+j)*ny), buf[cx+i + (cy+j)*ny]);
          tmp=buf[i+j*ny];
          buf[i+j*ny] = buf[cx+i + (cy+j)*ny];
          buf[cx+i + (cy+j)*ny] = tmp;
        }
      else if (j < cy)
        {
          //printf("i=%d, j=%d, cx=%d cy=%d nx=%d ny=%d\n",i,j,cx,cy,nx,ny);
          //printf("i=%d, j=%d, buf[%d]=%d buf_s[%d]=%d\n", i,j, (i-cx + (cy+j)*ny),buf[i-cx+(cy+j)*ny],(i + j*ny), buf[i + j*ny]);
          tmp=buf[i-cx +(j+cy)*ny];
          buf[i-cx +(j+cy)*ny] = buf[i + j*ny];
          buf[i + j*ny] = tmp;
        }
#endif

#if 0
      unsigned int i = tileWidthX*blockIdx.x + threadIdx.x ;
      unsigned int j = tileWidthY*blockIdx.y + threadIdx.y ;
      int cx=nx/2, cy=ny/2;
      int tmp=0;

#endif








    }


int main()
{
   int nx=2048;
   int ny=2048;
   int tileWidthX =32;
   int tileWidthY =32;
   int *buf_cpu;
   int *buf_orig;
   int *buf_gpu;
   int i;
   int size = (nx*ny*sizeof(int));
   buf_cpu = (int*)malloc(size);
   buf_orig = (int*)malloc(size);
   buf_gpu = (int*)malloc(size);

   for (i=0;i<nx*ny;i++)
   {
       buf_cpu[i] = buf_orig[i]=i;
   }

  
   //CPU Call 
   cpuflip(buf_cpu,nx, ny,tileWidthX, tileWidthY);

   //GPU call
   int *d_buf;
   cudaMalloc((void**)&d_buf,size );
   cudaMemcpy(d_buf, buf_orig, size, cudaMemcpyHostToDevice);
   dim3 dimGrid ( (nx/32) , (ny/32) ,1 ) ;
   dim3 dimBlock( 32, 32, 1 ) ;
   //printf(" Kernel Parameters::#blks=%d, #tpB=%d\n", (nx*ny/2), nx/2 );
   kernel_flip<<<dimGrid,dimBlock>>>(d_buf, nx,ny,tileWidthX, tileWidthY);
   cudaMemcpy(buf_gpu, d_buf, size, cudaMemcpyDeviceToHost);


//   for (i=0;i<nx*ny;i++)
   //{
       //printf("buf_orig[%d]=%d buf_cpu[%d]=%d buf_gpu[%d]=%d\n",i,buf_orig[i],i,buf_cpu[i],i,buf_gpu[i]);
   //}

}

