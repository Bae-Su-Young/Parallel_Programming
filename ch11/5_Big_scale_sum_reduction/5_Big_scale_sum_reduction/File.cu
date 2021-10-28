#include <iostream>
#include <stdio.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GRIDSIZE 1
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genData(unsigned* ptr, unsigned size) {
	while (size--) {
		*ptr += (unsigned)(rand() % 100);
	}
}

void kernel(unsigned* pData, unsigned* pAnswer, unsigned size) {
	register unsigned answer = 0;
	while (size--) {
		answer += *pData;
	}
	*pAnswer = answer;
}

//device - atomic operation
//__global__ void kernel(unsigned* pData, unsigned* pAnswer) {
//	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
//	register unsigned fValue = pData[i];
//	atomicAdd(pAnswer, fValue);
//}

//shared memory - atomic operation
__global__ void kernel_a(unsigned* pData, unsigned* pAnswer) {
	__shared__ unsigned answerShared;
	if (threadIdx.x == 0) { answerShared = 0.0F; }
	__syncthreads();
	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	register unsigned value = pData[i];//register <- global memory
	atomicAdd(&answerShared, value);//update the shared variable
	__syncthreads();

	if (threadIdx.x == 0) {
		atomicAdd(pAnswer, answerShared);//update the global variable
	}
}

////shared memory - reduction version
__global__ void kernel_s(unsigned* pdata, unsigned* pAnswer) {
	__shared__ unsigned datashared[BLOCKSIZE];

	//each thread loads one element from global to shared memory
	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	register unsigned tid = threadIdx.x;
	datashared[tid] = pdata[i];
	__syncthreads();

	//do reduction in the shared memory
	//s=stride
	for (register unsigned s = 1; s < BLOCKSIZE; s *= 2) {
		if (tid % (2 * s) == 0) {
			datashared[tid] += datashared[tid + s];
		}
		__syncthreads();
	}

	//add the partial sum to the global answer
	if (tid == 0) {
		atomicAdd(pAnswer, datashared[0]);
	}
}
int main(void) {
	unsigned* pData = NULL;
	unsigned* pAnswer = NULL;
	unsigned answer = 0;

	pData = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));
	pAnswer = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	genData(pData, TOTALSIZE);

	//start the timer
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	kernel(pData, &answer, TOTALSIZE);

	//end the timer
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::nanoseconds du = end - start;

	printf("Host Version\n%lld nano-seconds\nanswer=%lld\n", du, answer);
	
	/*CUDA version*/
	unsigned* pDataDev;
	unsigned* pAnwerDev;

	cudaMalloc((void**)&pDataDev, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev, 4 * sizeof(unsigned));
	cudaMemset(pAnwerDev, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);

	std::chrono::system_clock::time_point start_a = std::chrono::system_clock::now();
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	kernel_a << <dimGrid, dimBlock >> > (pDataDev, pAnwerDev);
	std::chrono::system_clock::time_point end_a = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_a = end_a - start_a;
	//==============================================================================================
	unsigned* pDataDev_s;
	unsigned* pAnwerDev_s;
	pAnswer = NULL;

	cudaMalloc((void**)&pDataDev_s, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev_s, 4 * sizeof(unsigned));
	cudaMemset(pAnwerDev, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev_s, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);

	std::chrono::system_clock::time_point start_s = std::chrono::system_clock::now();
	kernel_s << <dimGrid, dimBlock >> > (pDataDev_s, pAnwerDev_s);
	std::chrono::system_clock::time_point end_s = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_s = end_s - start_s;

	cudaMemcpy(pAnswer, pAnwerDev_s, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	printf("\nCUDA atomic Version\n%lld nano-seconds\nanswer=%lld\n", du_a, pAnswer);
	printf("\nCUDA Shared atomic Version\n%lld nano-seconds\nanswer=%lld\n", du_s, pAnswer);
	//free(pData);

}