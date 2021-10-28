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

//shared memory - reduction version
__global__ void kernel_s(unsigned* pdata, unsigned* pAnswer) {
	__shared__ unsigned datashared[BLOCKSIZE];

	//each thread loads one element from global to shared memory
	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	register unsigned tid = threadIdx.x;
	datashared[tid] = pdata[i];//copy to shared memory
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
//Shared + atomic + reversed tournament
__global__ void kernel_r(unsigned* pData, unsigned* pAnswer) {
	__shared__ unsigned dataShared[BLOCKSIZE];

	//each thread loads one element from global to shared memory
	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	register unsigned tid = threadIdx.x;
	dataShared[tid] = pData[i];//하나의 스레드가 하나의 데이터만 읽어옴
	__syncthreads();

	//do reduction in the reduction memory
	//시작하자마 절반의 스레드가 Idle에 빠지는 문제가 있다.
	for (register unsigned s = BLOCKSIZE / 2; s > 0; s >>= 1) {
		if (tid < s) {
			dataShared[tid] += dataShared[tid + s];
		}
		__syncthreads();
	}

	//add the partial sum to the global answer
	if (tid == 0) {
		atomicAdd(pAnswer, dataShared[0]);
	}
}

//resolve the idle threads
__global__ void kernel_rr(unsigned* pData, unsigned* pAnswer) {
	__shared__ unsigned dataShared[BLOCKSIZE];

	//each thread loads one element from global to shared memory
	register unsigned i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	register unsigned tid = threadIdx.x;
	dataShared[tid] = pData[i]+pData[i+blockDim.x];//모든 스레드가 처음부터 두 개의 데이터를 읽어와서 한번의 더하기 연산을 하도록 한다.
	__syncthreads();

	//do reduction in the reduction memory
	//시작하자마 절반의 스레드가 Idle에 빠지는 문제가 있다.
	for (register unsigned s = BLOCKSIZE / 2; s > 0; s >>= 1) {
		if (tid < s) {
			dataShared[tid] += dataShared[tid + s];
		}
		__syncthreads();
	}

	//add the partial sum to the global answer
	if (tid == 0) {
		atomicAdd(pAnswer, dataShared[0]);
	}
}
__device__ void warpReduce(volatile unsigned* dataShared, unsigned tid) {
	dataShared[tid] += dataShared[tid + 32];
	dataShared[tid] += dataShared[tid + 16];
	dataShared[tid] += dataShared[tid + 8];
	dataShared[tid] += dataShared[tid + 4];
	dataShared[tid] += dataShared[tid + 2];
	dataShared[tid] += dataShared[tid + 1];
}
__global__ void kernel_lastwarp(unsigned* pData, unsigned* pAnswer) {
	__shared__ unsigned dataShared[BLOCKSIZE];

	register unsigned tid = threadIdx.x;
	register unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	dataShared[tid] = pData[i] + pData[i + blockDim.x];
	__syncthreads();

	for (register unsigned s = BLOCKSIZE / 2; s > 32; s >>= 1) {
		if (tid < s) {
			dataShared[tid] += dataShared[tid + s];
		}
		__syncthreads();
	}
	//when only one warp is actived
	if (tid < 32) {
		warpReduce(dataShared, tid);

		if (tid == 0) {
			atomicAdd(pAnswer, dataShared[0]);
		}
	}
}
__global__ void kernel_complete(unsigned* pData, unsigned* pAnswer) {
	__shared__ unsigned dataShared[BLOCKSIZE];

	register unsigned tid = threadIdx.x;
	register unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	dataShared[tid] = pData[i] + pData[i + blockDim.x];
	__syncthreads();

	if (tid < 512) {
		dataShared[tid] += dataShared[tid + 512];
		__syncthreads();
		if (tid < 256) {
			dataShared[tid] += dataShared[tid + 256];
			__syncthreads();
			if (tid < 128) {
				dataShared[tid] += dataShared[tid + 128];
				__syncthreads();
				if (tid < 64) {
					dataShared[tid] += dataShared[tid + 64];
					__syncthreads();
					if (tid < 32) {
						warpReduce(dataShared, tid);
						if (tid == 0) {
							atomicAdd(pAnswer, dataShared[0]);
						}
					}
				}
			}
		}
	}
}
int main(void) {
	unsigned* pData = NULL;
	unsigned* pAnswer = NULL;
	unsigned answer = 0;

	pData = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));
	pAnswer = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	genData(pData, TOTALSIZE);

	//==============================================================================================
	// Host version

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	kernel(pData, &answer, TOTALSIZE);
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::nanoseconds du = end - start;

	
	//==============================================================================================
	//CUDA version
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

	cudaMemcpy(pAnswer, pAnwerDev, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);


	//==============================================================================================
	//atomic+shared (tournament)
	unsigned* pDataDev_s;
	unsigned* pAnwerDev_s;
	unsigned* pAnswer_s = NULL;
	pAnswer_s = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	cudaMalloc((void**)&pDataDev_s, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev_s, 4 * sizeof(unsigned));
	
	cudaMemset(pAnwerDev, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev_s, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);
	

	std::chrono::system_clock::time_point start_s = std::chrono::system_clock::now();
	kernel_s << <dimGrid, dimBlock >> > (pDataDev_s, pAnwerDev_s);
	std::chrono::system_clock::time_point end_s = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_s = end_s - start_s;

	cudaMemcpy(pAnswer_s, pAnwerDev_s, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	//==============================================================================================
	//atomic+shared (reversed-tournament)
	unsigned* pDataDev_r;
	unsigned* pAnwerDev_r;
	unsigned* pAnswer_r = NULL;
	pAnswer_r = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	cudaMalloc((void**)&pDataDev_r, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev_r, 4 * sizeof(unsigned));
	cudaMemset(pAnwerDev_r, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev_r, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);
	

	std::chrono::system_clock::time_point start_r = std::chrono::system_clock::now();
	kernel_r << <dimGrid, dimBlock >> > (pDataDev_r, pAnwerDev_r);
	std::chrono::system_clock::time_point end_r = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_r = end_r - start_r;
	cudaMemcpy(pAnswer_r, pAnwerDev_r, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	//==============================================================================================
	//atomic+shared (reversed-tournament) + Idle Threads
	unsigned* pDataDev_rr;
	unsigned* pAnwerDev_rr;
	unsigned* pAnswer_rr = NULL;
	pAnswer_rr = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	cudaMalloc((void**)&pDataDev_rr, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev_rr, 4 * sizeof(unsigned));
	cudaMemset(pAnwerDev_rr, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev_rr, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);

	dim3 dimGrid_rr(GRIDSIZE/2, 1, 1);            //Gride의 수도 절반으로 줄여줘야한다.
	std::chrono::system_clock::time_point start_rr = std::chrono::system_clock::now();
	kernel_rr << <dimGrid_rr, dimBlock >> > (pDataDev_rr, pAnwerDev_rr);
	std::chrono::system_clock::time_point end_rr = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_rr = end_rr - start_rr;

	cudaMemcpy(pAnswer_rr, pAnwerDev_rr, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);
	//==============================================================================================
	//atomic+shared (reversed-tournament) + Idle Threads + last warp 
	unsigned* pDataDev_w;
	unsigned* pAnwerDev_w;
	unsigned* pAnswer_w = NULL;
	pAnswer_w = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	cudaMalloc((void**)&pDataDev_w, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev_w, 4 * sizeof(unsigned));
	cudaMemset(pAnwerDev_w, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev_w, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);

	dim3 dimGrid_w(GRIDSIZE / 2, 1, 1);            //Gride의 수도 절반으로 줄여줘야한다.
	std::chrono::system_clock::time_point start_w = std::chrono::system_clock::now();
	kernel_rr << <dimGrid_w, dimBlock >> > (pDataDev_w, pAnwerDev_w);
	std::chrono::system_clock::time_point end_w = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_w = end_w - start_w;

	cudaMemcpy(pAnswer_w, pAnwerDev_w, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);
	//==============================================================================================
	//atomic+shared (reversed-tournament) + Idle Threads + Complete Unrolling
	unsigned* pDataDev_c;
	unsigned* pAnwerDev_c;
	unsigned* pAnswer_c = NULL;
	pAnswer_c = (unsigned*)malloc(TOTALSIZE * sizeof(unsigned));

	cudaMalloc((void**)&pDataDev_c, TOTALSIZE * sizeof(unsigned));
	cudaMalloc((void**)&pAnwerDev_c, 4 * sizeof(unsigned));
	cudaMemset(pAnwerDev_c, 0, 4 * sizeof(unsigned));

	cudaMemcpy(pDataDev_c, pData, 2 * BLOCKSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);

	dim3 dimGrid_c(GRIDSIZE / 2, 1, 1);            //Gride의 수도 절반으로 줄여줘야한다.
	std::chrono::system_clock::time_point start_c = std::chrono::system_clock::now();
	kernel_rr << <dimGrid_w, dimBlock >> > (pDataDev_c, pAnwerDev_c);
	std::chrono::system_clock::time_point end_c = std::chrono::system_clock::now();
	std::chrono::nanoseconds du_c = end_c - start_c;

	cudaMemcpy(pAnswer_c, pAnwerDev_c, 1 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	//==============================================================================================
	//Print Result
	printf("Host Version\n%lld nano-seconds\nanswer=%lld\n", du, answer);
	printf("\nCUDA atomic Version\n%lld nano-seconds\nanswer=%lld\n", du_a, pAnswer);
	printf("\nCUDA Shared atomic Version\n%lld nano-seconds\nanswer=%lld\n", du_s, pAnswer_s);
	printf("\nCUDA Shared atomic Reversed Version\n%lld nano-seconds\nanswer=%lld\n", du_r, pAnswer_r);
	printf("\nCUDA Reversed Version + Idle Thread\n%lld nano-seconds\nanswer=%lld\n", du_rr, pAnswer_rr);
	printf("\nUrolling the last Warp \n%lld nano-seconds\nanswer=%lld\n", du_w, pAnswer_w);
	printf("\nComplete Unrolling \n%lld nano-seconds\nanswer=%lld\n", du_c, pAnswer_c);
	//free(pData);

}