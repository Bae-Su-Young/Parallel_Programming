#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

const int GRIDSIZE = (32*1024);
const int BLOCKSIZE = 1024;
const int TOTALSIZE = GRIDSIZE * BLOCKSIZE;
const int TargetSIze = 2 * BLOCKSIZE;

__global__ void kernel(unsigned* pData, unsigned* pAnswer, unsigned target) {
	//each thread loads multiple element from global to shared memory
	register unsigned tid = threadIdx.x;
	if (tid == 0) {
		*pAnswer = TOTALSIZE;
	}
	__syncthreads();

	register unsigned start = tid * GRIDSIZE;
	register unsigned end = (tid + 1) * GRIDSIZE;
	register unsigned index;

	for (index = start; index < end; index++) {
		register unsigned value = pData[index];
		if (value == target) {
			atomicMin(pAnswer, index);
		}
	}
}

//Better memory access
__global__ void kernel(unsigned* pData, unsigned* pAnswer, unsigned target) {
	//each thread loads multiple element from global to shared memory
	register unsigned tid = threadIdx.x;
	register unsigned i;
	if (tid == 0) {
		*pAnswer = TOTALSIZE;
	}
	__syncthreads();

	for (i = 0; i < GRIDSIZE; i++) {
		register unsigned index = tid + i * BLOCKSIZE;
		register unsigned value = pData[index];
		if (value == target) {
			atomicMin(pAnswer, index);
		}
	}
}

__global__ void kernel_binSearch(unsigned* pData, unsigned* pAnswer, unsigned target) {
	register unsigned tid = threadIdx.x;
	register unsigned first = tid * GRIDSIZE;
	register unsigned last = (tid + 1) * GRIDSIZE;

	while (first < last) {
		register unsigned mid = (first + last) / 2;
		if (target == pData[mid]) {
			atomicMin(pAnswer, mid);
			last = first;
		}
		else if (target < pData[mid]) {
			last = mid - 1;
		}
		else {
			first = mid + 1;
		}
	}
}

//early cutoff BinSearh
__global__ void kernel_binSearch(unsigned* pData, unsigned* pAnswer, unsigned target) {
	register unsigned tid = threadIdx.x;
	register unsigned first = tid * GRIDSIZE;
	register unsigned last = (tid + 1) * GRIDSIZE;

	if (pData[first] <= target && target <= pData[last - 1]) {
		while (first < last) {
			register unsigned mid = (first + last) / 2;
			if (target == pData[mid]) {
				atomicMin(pAnswer, mid);
				last = first;
			}
			else if (target < pData[mid]) {
				last = mid - 1;
			}
			else {
				first = mid + 1;
			}
		}
	}
}