#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

__global__ void adj_diff_naive(int* g_result, int* g_input) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i > 0) {
		int x_i = g_input[i];//global memory read   -> result[i], result[i+1]을 구할 때 2번이나 read됨. 한번만 해도 되는데 -> 같은 값을 모든 스레드가 2번씩 읽게 된다.(Bad Case) 
		//해결법: global memory에서 읽어서 shared memory에 쓴다.

		int x_i_minus_1 = g_input[i - 1];//global memory read
		
		g_result[i] = x_i - x_i_minus_1;//global memory write
	}
}

//Shared Memory를 이용하는 Kernel
__global__ void adj_diff(int* g_result, int* g_input) {
	int tx = threadIdx.x;
	__shared__ int s_data[BLOCK_SIZE];
	unsigned int i = blockDim.x * blockIdx + tx;
	s_data[tx] = g_input[i];
	__syncthreads();
	if (tx > 0) {
		g_result[i] = s_data[tx] - s_data[tx - 1];
	}
	else if (i > 0) {
		g_result[i] = s_data[tx] - g_input[i - 1];
		// 다른 블록의 thread값이 필요한 경우 불가피하게 글로벌 메모리에 접근
	}
}