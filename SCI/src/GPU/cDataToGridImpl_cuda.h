  template <class T>
  __global__ void kernel_cuDataToGridImpl_p(T* gridStore, Int* gridShape, VBStore* vbs,
				     Matrix<Double>* sumwt, 
				     const Bool dopsf ,
				     const Int* polMap_ptr, const Int *chanMap_ptr,
				     const Double *uvwScale_ptr, const Double *offset_ptr,
				     const Double *dphase_ptr, Int XThGrid=0, Int YThGrid=0
				     );
