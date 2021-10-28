#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>


__device__ inline void MyAtomicAdd(float* address, float value) {

	int oldval, newval, readback;
	oldval = __float_as_int(*address);
	newval = __float_as_int(__int_as_float(oldval) + value);
	while ((readback = atomicCAS((int*)address, oldval, newval)) != oldval) {
		oldval = readback;
		newval = __float_as_int(__int_as_float(oldval) + value);
	}

}