#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <windows.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

#define GRIDSIZE (8*1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE * BLOCKSIZE)

__global__ void adj_diff_naive(float* result, float* input) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > 0) {
		float x_i = input[i];
		float x_i_minus_1 = input[i-1];
		result[i] = x_i - x_i_minus_1;
	}
}

//shared memory version kernel
__global__ void adj_diff_shared(float* result, float* input) {
	__shared__ float s_data[BLOCKSIZE];
	register unsigned int tx = threadIdx.x;
	register unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_data[tx] = input[i];
	__syncthreads();//모든 데이터가 shared memory에 복사되었음을 확신할 수 있다.
	if (tx > 0) {
		result[i] = s_data[tx] - s_data[tx - 1];

	}
	else if (i > 0) {
		result[i] = s_data[tx] - input[i - 1];
	}
}

//Dyamic allocate Shared memory 
__global__ void adj_diff_shared_d(float* result, float* input) {
	extern __shared__ float s_data[];
	register unsigned int tx = threadIdx.x;
	register unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

	s_data[tx] = input[i];
	__syncthreads();
	if (tx > 0) {
		result[i] = s_data[tx] - s_data[tx - 1];
	}
	else if (i > 0) {
		result[i] = s_data[tx] - input[i - 1];
	}
}


void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}
void getDiff(float* dst, const float* src, unsigned int size) {
	for (register int i = 1; i < size; ++i) {
		dst[i] = src[i] - src[i - 1];
	}
}

int main(void) {
	float* pSource = NULL;
	float* pResult1 = NULL; float* pResult2 = NULL; float* pResult3 = NULL;
	int i;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));

	//malloc
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult1 = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult2 = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult3 = (float*)malloc(TOTALSIZE * sizeof(float));

	//device-side
	float* pSourceDev = NULL;
	float* pResultDev1 = NULL; float* pResultDev2 = NULL; float* pResultDev3 = NULL;

	cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float));
	cudaMalloc((void**)&pResultDev1, TOTALSIZE * sizeof(float));
	cudaMalloc((void**)&pResultDev2, TOTALSIZE * sizeof(float));
	cudaMalloc((void**)&pResultDev3, TOTALSIZE * sizeof(float));

	//memcpy
	cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);

	//launch the kernel
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	adj_diff_naive <<<dimGrid, dimBlock >>> (pResultDev1, pSourceDev);

	//copy tfrom device to host
	cudaMemcpy(pResult1, pResultDev1, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	pResult1[0] = 0.0F;
	getDiff(pResult1, pSource, TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("[naive] elpased time=%f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));

	//==========================================================================================
	adj_diff_shared << <dimGrid, dimBlock >> > (pResultDev2, pSourceDev);

	//copy tfrom device to host
	cudaMemcpy(pResult2, pResultDev2, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	pResult2[0] = 0.0F;
	getDiff(pResult2, pSource, TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("[shared] elpased time=%f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));

	//==========================================================================================
	adj_diff_shared_d << <dimGrid, dimBlock,BLOCKSIZE * sizeof(float) >> > (pResultDev3, pSourceDev);

	//copy tfrom device to host
	cudaMemcpy(pResult3, pResultDev3, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	pResult3[0] = 0.0F;
	getDiff(pResult3, pSource, TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("[Dynamic shared] elpased time=%f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));




	//print sampe casess
	//i = 1;
	//printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	//i = TOTALSIZE - 1;
	//printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	//i = TOTALSIZE / 2;
	//printf("i=%2d: %f = %f - %f\n", i, pResult[i], pSource[i], pSource[i - 1]);
	//
	//free
	free(pSource);
	free(pResult1);

	free(pResult2);
	free(pResult3);

}
