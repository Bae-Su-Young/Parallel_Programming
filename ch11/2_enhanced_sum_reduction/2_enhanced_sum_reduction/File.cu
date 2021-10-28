#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
using namespace chrono;

#define GRIDSIZE 1
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genData(unsigned* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}
__global__ void kernel(unsigned* pData, unsigned* pAnswer) {
	__shared__ unsigned dataShared[2 * BLOCKSIZE];
	unsigned tx = threadIdx.x;
	dataShared[tx] = pData[tx];
	dataShared[BLOCKSIZE + tx] = pData[BLOCKSIZE + tx];

	//do reduction in the shared memory
	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		if (tx % stride == 0) {
			dataShared[2 * tx] += dataShared[2 * tx + stride];
		}
	}

	//final sychronize
	__syncthreads();
	if (tx == 0) {
		pAnswer[tx] = dataShared[tx];
	}
}
void kernel(unsigned* pData, unsigned* pAnswer, unsigned size) {
	while (size--) {
		unsigned value = *pData++;
		*pAnswer = *pAnswer + value;
	}
}
int main(void) {
	unsigned* pData = NULL;
	unsigned answer = 0;

	//malloc memories on the host-side
	pData = (unsigned*)malloc(2 * BLOCKSIZE * sizeof(unsigned));

	//generate source data
	genData(pData, 2 * BLOCKSIZE);

	//HOST: start the timer
	system_clock::time_point h_start = system_clock::now();
	kernel(pData, &answer, 2 * BLOCKSIZE);

	//end the timer
	system_clock::time_point h_end = system_clock::now();
	nanoseconds h_du = h_end - h_start;
	printf("%lld nano-secodns\n", h_du);
	printf("answer= %lld\n", answer);

	//CUDA: allocate device memory
	unsigned* pDataDev;
	unsigned* pAnswerDev;
	cudaMalloc((void**)&pDataDev, 2 * BLOCKSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnswerDev, 2 * BLOCKSIZE * sizeof(unsigned));
	cudaMemset(pAnswerDev, 0, 1 * sizeof(unsigned));

	//cuda: copy from host to device
	cudaMemcpy(pDataDev, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);

	//start timer
	system_clock::time_point start = system_clock::now();

	//laucnh the kernel
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	kernel << <dimGrid, dimBlock >> > (pDataDev, pAnswerDev);

	//end timer
	system_clock::time_point end = system_clock::now();

	nanoseconds du = end - start;
	printf("%lld nano-seconds\n", du);
	printf("answer= %lld \n", answer);
}