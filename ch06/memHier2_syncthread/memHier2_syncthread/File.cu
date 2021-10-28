#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

__global__ void adj_diff_naive(int* g_result, int* g_input) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i > 0) {
		int x_i = g_input[i];//global memory read   -> result[i], result[i+1]�� ���� �� 2���̳� read��. �ѹ��� �ص� �Ǵµ� -> ���� ���� ��� �����尡 2���� �а� �ȴ�.(Bad Case) 
		//�ذ��: global memory���� �о shared memory�� ����.

		int x_i_minus_1 = g_input[i - 1];//global memory read
		
		g_result[i] = x_i - x_i_minus_1;//global memory write
	}
}

//Shared Memory�� �̿��ϴ� Kernel
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
		// �ٸ� ����� thread���� �ʿ��� ��� �Ұ����ϰ� �۷ι� �޸𸮿� ����
	}
}