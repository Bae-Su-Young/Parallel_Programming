#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include  <cstdio> 
#include <stdlib.h> //for rand(),malloc, free()
#include <windows.h> //for QueryPerformanceCouter()


#define _DEBUG //for debug mode
//#define NDEBUG //for release mode

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

const int WIDTH = 1024;//matrix size
const int TILE_WIDTH = 32;
const int GRID_WIDTH = (WIDTH / TILE_WIDTH);

//shared memory version
__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int width) {
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

	int by = blockIdx.y; int bx = blockIdx.x;
	int ty = threadIdx.y; int tx = threadIdx.x;

	int gy = by * TILE_WIDTH + ty;
	int gx = bx * TILE_WIDTH + tx;

	float sum = 0.0f;
	for (register int m = 0; m < width / TILE_WIDTH; m++) {
		s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
		s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * width + gx];

		__syncthreads();
		//use the sharead memory blocks to get the partial sum
		for (register int k = 0; k < TILE_WIDTH; k++) {
			sum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();// shared memory를 다 사용후에 다시 한번더 sync why?) for문을 정지시킴과 동시에 다음 타일을 읽을 준비를 하기 때문

	}
	g_C[gy * width + gx] = sum;
}

//global memory version
__global__ void matmul(float* c, const float* a, const float* b, const int width) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0F;

	for (register int k = 0; k < width; k++) {
		float lhs = a[y * width + k];
		float rhs = b[k * width + x];
		sum += lhs * rhs;

	}
	c[y * width + x] = sum;
}

void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

int main() {
	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;

	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));

	//malloc memories on the host-side
	pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));

	//generate source data
	genData(pA, WIDTH * WIDTH);
	genData(pB, WIDTH * WIDTH);

	//CUDA: allocate device memory
	float* pAdev = NULL;
	float* pBdev = NULL;
	float* pCdev = NULL;

	CUDA_CHECK(cudaMalloc((void**)&pAdev, WIDTH * WIDTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&pBdev, WIDTH * WIDTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&pCdev, WIDTH * WIDTH * sizeof(float)));

	//copy host -> device
	CUDA_CHECK(cudaMemcpy(pAdev, pA, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(pBdev, pB, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	//CUDA_CHECK(cudaMemcpy(pCdev, pC, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice));

	//start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));

	//launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul << <dimGrid, dimBlock >> > (pCdev, pAdev, pBdev, WIDTH);

	//end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	CUDA_CHECK(cudaPeekAtLastError());
	printf("elapsed time=%f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));

	//copyt deviece -> host
	CUDA_CHECK(cudaMemcpy(pC, pCdev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

	//free device memory
	CUDA_CHECK(cudaFree(pAdev));
	CUDA_CHECK(cudaFree(pBdev));
	CUDA_CHECK(cudaFree(pCdev));

	//print
	int i, j;
	i = 0; j = 0;
	printf("c[%4d][%4d]=%f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH / 2; j = WIDTH / 2;
	printf("c[%4d][%4d]=%f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH - 1; j = WIDTH - 1;
	printf("c[%4d][%4d]=%f\n", i, j, pC[i * WIDTH + j]);

	//done
	return 0;

}


