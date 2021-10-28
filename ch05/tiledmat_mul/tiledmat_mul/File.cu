#include <stdio.h>
#include <stdlib.h> // rand(), malloc(), free()
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int WIDTH = 1024;
const int TILE_WIDTH = 32;
const int GRID_WIDTH = (WIDTH / TILE_WIDTH);



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
	while (size--)
	{
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

int main(void) {
	float a[WIDTH][WIDTH];
	float b[WIDTH][WIDTH];
	float c[WIDTH][WIDTH] = { 0 };

	//generate source data
	genData(&(a[0][0]), WIDTH * WIDTH);
	genData(&(b[0][0]), WIDTH * WIDTH);

	//divice-side data
	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;

	//allocate device memory
	cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(float));

	//copy from host to device
	cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	//CUDA: launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b, WIDTH);
	//CUDA_CHECK(cudaPeekAtLastError());

	//copy from device to host
	cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

	//free device memory
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	//print the result
//#if defined(YOU_really_need_it)
//	for (int y = 0; y < WIDTH; y++) {
//		for (int x = 0; x < WIDTH; x++) {
//			printf("%5d", c[y][x]);
//		}
//		printf("\n");
//	}
//#endif

	//print the part of result
	int i, j;
	i = 0; j = 0;
	printf("c[%4d][%4d]=%f\n", i, j, c[i * WIDTH + j]);
	i = WIDTH / 2; j = WIDTH / 2;
	printf("c[%4d][%4d]=%f\n", i, j, c[i * WIDTH + j]);
	i = WIDTH - 1; j = WIDTH - 1;
	printf("c[%4d][%4d]=%f\n", i, j, c[i * WIDTH + j]);


	//done
	return 0;
}