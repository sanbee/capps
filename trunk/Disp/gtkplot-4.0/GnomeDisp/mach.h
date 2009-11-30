#if !defined(MACH_H)
#define MACH_H

#ifdef	__cplusplus
extern "C" {
#endif

typedef enum
{ Char, Short, Int, Long, Float, Double, BaseTypes } BaseDataType;
  /*
char *DataTypeName[] =
{"char", "short", "int", "long", "float", "double" };
*/

typedef struct
{ int type_size[BaseTypes], type_align[BaseTypes];
  int twos_comp, float_ieee, big_endian;
} MachineSpecType;
/*
MachineSpecType MachineSpec;
*/
void GetMachineSpec(MachineSpecType *);
void swap_bytes(unsigned short *, int);
void swap_short(unsigned *, int);
void swap_long(void *, int);
void swap_d(double *, int);
#ifdef	__cplusplus
}
#endif

#endif
