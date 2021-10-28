#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

__global__ void kernelFunc(float* dst, const float* src) {
	//dst -> global
	float p = src[threadIdx.x]; // p goes in a register
	float heap[10]; // global -> local ; thread
	__shared__ float partial_sum[1024];//shared memory
}