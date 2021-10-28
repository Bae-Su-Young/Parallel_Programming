#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define GRIDSIZE (32*1024)
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

__global__ void kernel(unsigned long long int* pCount) {
	(*pCount) = (*pCount) + 1;
}

//atomic version
__global__ void kernel_atomic(unsigned long long int* pCount) {
	atomicAdd(pCount, 1ULL);
}

__global__ void kernel_atomic_sharedMem(unsigned long long int* pCount) {
	__shared__ int nCountShared;		
	if (threadIdx.x == 0) {
		nCountShared = 0;
	}

	__syncthreads();
	atomicAdd(&nCountShared, 1);		//atomic add 연산
	__syncthreads();

	if (threadIdx.x == 0) {				//shared memory에 있던 값을
		atomicAdd(pCount, nCountShared);//바로 global memory에 저장
	}

}
int main(void) {
	unsigned long long int aCount[1];

	//prepare timer
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//cuda: alocate device memory
	unsigned long long int* pCountDev = NULL;
	cudaMalloc((void**)&pCountDev, sizeof(unsigned long long int));
	cudaMemset(pCountDev, 0, sizeof(unsigned long long int));

	//start timer
	cudaEventRecord(start, 0);

	//launch kernel
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	//kernel << <dimGrid, dimBlock >> > (pCountDev);
	//kernel_atomic << <dimGrid, dimBlock >> > (pCountDev);
	kernel_atomic_sharedMem << <dimGrid, dimBlock >> > (pCountDev);

	//copy from device to host
	cudaMemcpy(aCount, pCountDev, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

	printf("total number of threads=%d\n", TOTALSIZE);
	printf("count = % llu\n", aCount[0]);

	//end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("elpased time= %f msec\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//cudaFree
	cudaFree(pCountDev);


}