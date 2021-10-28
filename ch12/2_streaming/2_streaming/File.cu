#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <iostream>
#include <string>
#include "./streaming-config.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct ElapTime {
	long long Start;
	long long End;
}ElapTime;

void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

void kernel(float* dst, float* src, int size) {
	while (size--) {
		*dst = 0.0F;
		for (register int j = 0; j < REPEAT; j++) {
			*dst += *src; //src를 REPEAT번만큼 더한 값이 저장
		}
		dst++;
		src++;
	}
}

__global__ void kernel_c(float* dst, float* src) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	dst[i] = 0.0F;
	//global memory access
	for (register int j = 0; j < REPEAT; j++) {
		dst[i] += src[i];
	}
}
void printSample(std::string version,float* result, float* src) {
	std::cout << version << " version sample case RESULT\n";
	int i = 0;
	printf("i=%8d: %f = %f * %d\n", i, result[i], src[i], REPEAT);
	i = TOTALSIZE - 1;
	printf("i=%8d: %f = %f * %d\n", i, result[i], src[i], REPEAT);
	i = TOTALSIZE / 2;
	printf("i=%8d: %f = %f * %d\n", i, result[i], src[i], REPEAT);
}
int main(void) {
	float* pSource = NULL;
	float* pResult = NULL;
	ElapTime hTime;
	long long freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));

	//malloc memories on the host-side
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));

	//generate source data
	genData(pSource, TOTALSIZE);

	//Host version
	cudaEvent_t start_h, stop_h;
	cudaEventCreate(&start_h);
	cudaEventCreate(&stop_h);
	cudaEventRecord(start_h, 0);
	QueryPerformanceCounter((LARGE_INTEGER*)(&hTime.Start));
	kernel(pResult, pSource, TOTALSIZE);
	float time_h;
	cudaEventRecord(stop_h, 0);
	cudaEventSynchronize(stop_h);
	cudaEventElapsedTime(&time_h, start_h, stop_h);

	QueryPerformanceCounter((LARGE_INTEGER*)(&hTime.End));
	printSample("HOST", pResult, pSource);
	printf("cudaEvent_t Elapsed Time= %f msec\n", time_h);
	printf("elpased time= %f usec\n\n", (double)(hTime.End - hTime.Start) * 1000000.0 / (double)(freq));
	cudaEventDestroy(start_h);
	cudaEventDestroy(stop_h);
	//CUDA version
	ElapTime cTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* pSourceDev = NULL;
	float* pResultDev = NULL;
	cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float));
	cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float));

	cudaEventRecord(start, 0);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cTime.Start));
	cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float), cudaMemcpyHostToDevice);
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	kernel_c << <dimGrid, dimBlock >> > (pResultDev, pSourceDev);
	cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost);
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cTime.End));


	printSample("CUDA", pResult, pSource);
	printf("cudaEvent_t Elapsed Time= %f msec\n", time);
	printf("elpased time= %f usec\n\n", (double)(cTime.End - cTime.Start) * 1000000.0 / (double)(freq));
	
}