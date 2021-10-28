#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GRIDSIZE 2
#define BLOCKSIZE 1024
#define TOTALSIZE (BLOCKSIZE*GRIDSIZE)

const unsigned TargetSize = 2 * BLOCKSIZE;

__device__ void makeIncreasing(unsigned* sData, int i, int j) {
	if (sData[i] > sData[j]) {
		unsigned t = sData[i];
		sData[i] = sData[j];
		sData[j] = t;
	}
}

__device__ void makeDecreasing(unsigned* sData, int i, int j) {
	if (sData[i] < sData[j]) {
		unsigned t = sData[i];
		sData[i] = sData[j];
		sData[j] = t;
	}
}

__global__ void kernel(unsigned* pData) {
	__shared__ unsigned sData[TargetSize];
	register unsigned i1 = threadIdx.x;
	register unsigned i2 = i1 + TargetSize/2;

	//load the data
	sData[i1] = pData[i1];
	sData[i2] = pData[i2];
	__syncthreads();

	//main loop
	for (unsigned k = 2; k <= TargetSize; k *= 2) {
		for (unsigned j = k / 2; j > 0; j /= 2) {
			unsigned ij = i1 ^ j;
			if ((ij) > i1) { //thread끼리의 간섭을 피하기 위해서(=bubble sort)
				if ((i1 & k) == 0) {
					makeIncreasing(sData, i1, ij);
				}
				else {
					makeDecreasing(sData, i1, ij);
				}
			}
			__syncthreads();
			ij = i2 ^ j;
			if ((ij) > i2) { //thread끼리의 간섭을 피하기 위해서
				if ((i2 & k) == 0) {
					makeIncreasing(sData, i2, ij);
				}
				else {
					makeDecreasing(sData, i2, ij);
				}
			}
			__syncthreads();
		}
	}

	pData[i1] = sData[i1];
	pData[i2] = sData[i2];
}