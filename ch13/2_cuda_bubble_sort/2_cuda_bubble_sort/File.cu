#include <iostream>
#include <algorithm>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define GRIDSIZE 2

#define BLOCKSIZE 1024
#define TOTALSIZE (BLOCKSIZE*GRIDSIZE)

const unsigned TargetSize = 2 * BLOCKSIZE;

__global__ void kernel(unsigned* pData) {
	__shared__ unsigned sData[TargetSize];
	register unsigned tid = threadIdx.x;
	register unsigned data0, data1;

	sData[tid] = pData[tid];
	sData[tid + TargetSize / 2] = pData[tid + TargetSize / 2];

	for (register unsigned i = 0; i < TargetSize / 2; i++) {

		__syncthreads();
		//stage 1
		data0 = sData[2 * tid];
		data1 = sData[2 * tid + 1];
		if (data0 > data1) {
			sData[2 * tid] = data1;
			sData[2 * tid + 1] = data0;
		}
		__syncthreads();
		if (tid != 0) {
			//stage2
			data0 = sData[2 * tid - 1];
			data1 = sData[2 * tid];
			if (data0 > data1) {
				sData[2 * tid - 1] = data1;
				sData[2 * tid] = data0;
			}
		}
	}
	__syncthreads();
	pData[tid] = sData[tid];
	pData[tid + TargetSize / 2] = sData[tid + TargetSize / 2];
}

void genData(unsigned* ptr, unsigned size) {
	while (size--) {
		*ptr++ = (unsigned)(rand() % 10000);
	}
}

bool isSortedData(unsigned* ptr, unsigned size) {
	unsigned prev = *ptr++;
	size--;
	while (size--) {
		unsigned cur = *ptr++;
		if (prev > cur) return false;
		prev = cur;
	}
	return true;
}
int main(void) {

	printf("\n===========HOST VERSION=========\n");

	unsigned* pData;
	long long start, end, freq;
	pData = (unsigned*)malloc(TargetSize * sizeof(unsigned));

	genData(pData, TargetSize);

	//check
	printf("sorting %d data\n", TargetSize);
	printf("%u %u %u %u %u %u\n",
		pData[0], pData[1], pData[2], pData[3], pData[4], pData[5]);
	printf("is sorted? -- %s\n", isSortedData(pData, TargetSize) ? "yes" : "no");
	
	//perform the action
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	QueryPerformanceCounter((LARGE_INTEGER*)(&start));
	std::sort(pData, pData + TargetSize);
	QueryPerformanceCounter((LARGE_INTEGER*)(&end));

	//print result
	printf("elapsed time = %f msec\n", (double)(end - start) * 1000.0 / (double)(freq));

	//check 
	printf("%u %u %u %u %u %u\n",
		pData[0], pData[1], pData[2], pData[3], pData[4], pData[5]);
	printf("is sorted? -- %s\n", isSortedData(pData, TargetSize) ? "yes" : "no");

	//--------------------------------------------------------------------------------------------------------
	//CUDA VERSION
	printf("\n===========CUDA BUBLLE SORT VERSION=========\n");
	long long s_c, e_c,f_c;
	unsigned* pSourceDev = NULL;
	unsigned* pData_c;
	pData_c = (unsigned*)malloc(TargetSize * sizeof(unsigned));

	genData(pData_c, TargetSize);

	//check
	printf("sorting %d data\n", TargetSize);
	printf("%u %u %u %u %u %u\n",
		pData_c[0], pData_c[1], pData_c[2], pData_c[3], pData_c[4], pData_c[5]);
	printf("is sorted? -- %s\n", isSortedData(pData_c, TargetSize) ? "yes" : "no");

	
	cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(unsigned));
	QueryPerformanceFrequency((LARGE_INTEGER*)(&f_c));
	QueryPerformanceCounter((LARGE_INTEGER*)(&s_c));
	cudaMemcpy(pSourceDev, pData_c, TOTALSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);
	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	kernel << <dimGrid, dimBlock >> > (pSourceDev);
	cudaMemcpy(pData_c, pSourceDev, TOTALSIZE * sizeof(unsigned), cudaMemcpyDeviceToHost);
	QueryPerformanceCounter((LARGE_INTEGER*)(&e_c));

	//print result
	printf("elapsed time = %f msec\n", (double)(e_c - s_c) * 1000.0 / (double)(f_c));

	//check 
	printf("%u %u %u %u %u %u\n",
		pData_c[0], pData_c[1], pData_c[2], pData_c[3], pData_c[4], pData_c[5]);
	printf("is sorted? -- %s\n", isSortedData(pData_c, TargetSize) ? "yes" : "no");



}