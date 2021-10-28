#include <chrono>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define GRIDSIZE 8*1024
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)
#define HALF (TOTALSIZE/2)

#if defined(NDEBUG)
//code for release mode
#define CUDA_CHECK(x) (x)
#else
//code for debug mode
#define CUDA_CHECK(x) do{\
	 (x);\
	cudaError_t e = cudaGetLastError();\
	 if (cudaSuccess != e) {\
		 printf("cuda failure %s at %s:%d\n",\
			cudaGetErrorString(e),\
			 __FILE__, __LINE__);\
			 exit(1);\
	 }\
	} while (0)
#endif

__global__ void evenodd(float* result, float* input) {
	register unsigned int tx = threadIdx.x;
	register unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (threadIdx.x % 2 == 0) {
		result[gx / 2] = input[gx];
	}
	else {
		result[HALF + gx / 2] = input[gx];
	}
}
__global__ void evenodd2(float* result, float* input) {
	register unsigned int tx = threadIdx.x;
	register unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

	if (gx < HALF) {
		result[gx] = input[gx * 2];//짝수번째 input을 저장
	}
	else {
		result[gx] = input[(gx - HALF) * 2 + 1];//홀수번째 저장
	}
}

void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}
int main() {
	float* pSource = NULL;
	float* pResult = NULL;
	float* pResult2 = NULL;

	int i;

	//malloc memories on the host side
	pSource = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult = (float*)malloc(TOTALSIZE * sizeof(float));
	pResult2 = (float*)malloc(TOTALSIZE * sizeof(float));

	//generate source data
	genData(pSource, TOTALSIZE);

	//device variable
	float* pSourceDev = NULL;
	float* pResultDev = NULL;
	float* pResultDev2 = NULL;

	CUDA_CHECK(cudaMalloc((void**)&pSourceDev, TOTALSIZE * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&pResultDev, TOTALSIZE * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&pResultDev2, TOTALSIZE * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(pSourceDev, pSource, TOTALSIZE * sizeof(float),cudaMemcpyHostToDevice));

	//evenodd
	//start timer
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	//launch kernel
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);

	evenodd << <dimGrid, dimBlock >> > (pResultDev, pSourceDev);

	//end timer
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::nanoseconds du = end - start;
	printf("[kernel 1] %lld nano-seconds\n", du);

	//evenodd2
	std::chrono::system_clock::time_point start2 = std::chrono::system_clock::now();

	//launch kernel2
	evenodd2 << <dimGrid, dimBlock >> > (pResultDev2, pSourceDev);

	//end timer
	std::chrono::system_clock::time_point end2 = std::chrono::system_clock::now();
	std::chrono::nanoseconds du2 = end2 - start2;
	printf("[kernel 2] %lld nano-seconds\n", du2);


	CUDA_CHECK(cudaMemcpy(pResult, pResultDev, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(pResult2, pResultDev2, TOTALSIZE * sizeof(float), cudaMemcpyDeviceToHost));

	//print
	printf("\n[Kernel 1 Result]\n");
	printf("Source: [0]=%f,[1]=%f, [2]=%f, [3]=%f\n", pSource[0], pSource[1], pSource[2], pSource[3]);
	printf("Result: [0]=%f,[1]=%f, [HALF]=%f, [HALF+1]=%f\n", pResult[0], pResult[1], pResult[HALF], pResult[HALF+1]);

	printf("\n[Kernel 2 Result]\n");
	printf("Source: [0]=%f,[1]=%f, [2]=%f, [3]=%f\n", pSource[0], pSource[1], pSource[2], pSource[3]);
	printf("Result: [0]=%f,[1]=%f, [HALF]=%f, [HALF+1]=%f\n", pResult2[0], pResult2[1], pResult2[HALF], pResult2[HALF + 1]);

}


