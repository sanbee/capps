/* $Id: mach.c,v 1.5 1998/02/17 05:43:15 sanjay Exp $ */
/* Courtesy Rajiv Singh */
#include <stdio.h>
#include <mach.h>

void GetMachineSpec(MachineSpecType *mac)
{
  struct { char x; char y;        } a1;
  struct { char x; short y;       } a2;
  struct { char x; int   y;       } a3;
  struct { char x; long y;        } a4;
  struct { char x; float y;       } a5;
  struct { char x; double y;      } a6;
  /* struct { char x; char* y;       } a7; */

  mac->type_size[Char]       = sizeof(char);
  mac->type_size[Short]      = sizeof(short);
  mac->type_size[Int]        = sizeof(int);
  mac->type_size[Long]       = sizeof(long);
  mac->type_size[Float]      = sizeof(float);
  mac->type_size[Double]     = sizeof(double);

# define align(a) (int)((char*)&(a.y) - (char*)&(a.x))
  mac->type_align[Char]       = align(a1);
  mac->type_align[Short]      = align(a2);
  mac->type_align[Int]        = align(a3);
  mac->type_align[Long]       = align(a4);
  mac->type_align[Float]      = align(a5);
  mac->type_align[Double]     = align(a6);

  mac->twos_comp = ((unsigned short)(-1) == 65535U);
  { unsigned short big_end=1;
    mac->big_endian= *(unsigned char *)&big_end == 0;
  }
  { int i, j;
    float f[3] = { 1.2345, -1.2345, -1.2345e-42 };
    unsigned char big_rep[3][4] =
    { {0x3f, 0x9e, 0x04, 0x19},
      {0xbf, 0x9e, 0x04, 0x19},
      {0x80, 0x00, 0x03, 0x71}
    };
    unsigned char *f_rep = (unsigned char*)f;
    for (i=0; i < 3; i++)
    { if (mac->big_endian)
      { for (j=0; j < 4; j++) if (big_rep[i][j] != f_rep[i*4+j]) break; }
      else
      { for (j=0; j < 4 ; j++) if (big_rep[i][j] != f_rep[i*4+3-j]) break; }
      if (j != 4) break;
      /* for (j=0; j < 4; j++) printf("%02x ", *(f_rep+i*4+j)); */
    }
    if (i == 0)      mac->float_ieee = 0;
    else if (i == 3) mac->float_ieee = 1;
    else             mac->float_ieee = 2;
  }
}

void swap_bytes(unsigned short *p, int n)
{ while (n-- > 0) p[n] = p[n]/256 + (p[n] % 256)*256; }

void swap_short(unsigned *p, int n)
{ while (n-- > 0) p[n] = p[n]/65536 + (p[n] % 65536)*65536; }

void swap_long(void *p, int n)
{ swap_short((unsigned *)p, n); swap_bytes((unsigned short *)p, 2*n); }

void swap_d(double *p, int n)
{ float temp, *v;
  swap_long(p, 2*n) ;
  while(n-- > 0)
  { v = (float *)p ;
    temp = *v; *v = v[1] ; v[1] = temp ;
    p++ ;
  }
} 
  
