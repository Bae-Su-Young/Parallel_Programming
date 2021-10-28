#include <stdio.h>
#include <stdlib.h>
#include <windows.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define GRIDSIZE 1024
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

#define NUMHIST 16 

void genData(unsigned int* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (unsigned int)(rand() % (NUMHIST - 1));
	}
}

__global__ void kernel(unsigned int* hist, unsigned int* img, unsigned int size) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pixelVal = img[i];
	atomicAdd(&(hist[pixelVal]), 1);

}
//sharedm memory version
__global__ void kernel_sharedMem(unsigned int* hist, unsigned int* img, unsigned int size) {
	__shared__ int histShared[NUMHIST];
	if (threadIdx.x < NUMHIST) {		//처음 16스레드는 shared memory initialize
		histShared[threadIdx.x] = 0;
	}
	__syncthreads();
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int pixelVal = img[i];
	atomicAdd(&(histShared[pixelVal]), 1);
	__syncthreads();

	if (threadIdx.x < NUMHIST) {
		atomicAdd(&(hist[threadIdx.x]), histShared[threadIdx.x]); //global memory에 총합으로 더하는 과정
	}
}

int main(void) {
	unsigned int* pImage = NULL;
	unsigned int* pHistogram = NULL;
	int i;

	//prepare timer
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//malloc
	pImage = (unsigned int*)malloc(TOTALSIZE * sizeof(unsigned int));
	pHistogram = (unsigned int*)malloc(NUMHIST * sizeof(unsigned int));
	for (i = 0; i < NUMHIST; i++) {
		pHistogram[i] = 0;
	}

	//generate src data
	genData(pImage, TOTALSIZE);

	//CUAD: allocate device memory
	unsigned int* pImageDev;
	unsigned int* pHistogramDev;
	cudaMalloc((void**)&pImageDev, TOTALSIZE * sizeof(unsigned int));
	cudaMalloc((void**)&pHistogramDev, NUMHIST * sizeof(unsigned int));
	cudaMemset(pHistogramDev, 0, NUMHIST * sizeof(unsigned int));
	
	//CUDA: copy from host to side
	cudaMemcpy(pImageDev, pImage, TOTALSIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

	//start timer
	cudaEventRecord(start, 0);

	//kernel(pHistogram, pImage, TOTALSIZE);
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	//kernel << <dimGrid, dimBlock >> > (pHistogramDev, pImageDev, TOTALSIZE);
	kernel_sharedMem << <dimGrid, dimBlock >> > (pHistogramDev, pImageDev, TOTALSIZE);
	//end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("elapsed time=%f msec\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//CUDA: copy from device to host
	cudaMemcpy(pHistogram, pHistogramDev, NUMHIST * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//rpitn histogram
	long total = 0L;
	for (i = 0; i < NUMHIST; i++) {
		printf("%2d: %10d\n", i, pHistogram[i]);
		total += pHistogram[i];
	}
	printf("total: %10ld (should be %ld)\n", total, TOTALSIZE);
}