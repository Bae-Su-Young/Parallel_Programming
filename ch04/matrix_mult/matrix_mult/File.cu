#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

__global__ void mulKernel(int* c, const int* a, const int* b, const int WIDTH) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = y * WIDTH + x;

	int sum = 0;
	for (int k = 0; k < WIDTH; k++) {
		sum += a[y*WIDTH+k] * b[k*WIDTH+x];

	}
	c[i] = sum;

}

int main() {
	//host-side
	const int WIDTH = 5;
	int a[WIDTH][WIDTH];
	int b[WIDTH][WIDTH];
	int c[WIDTH][WIDTH] = { 0 };

	//make a,b matrices
	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < WIDTH; y++) {
			a[x][y] = x * 10 + y;
			b[x][y] = (x * 10 + y) * 100;
		}
	}

	//device-side
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	//allocate device memory
	cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int));

	// copy from host to device
	cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

	//launch the kernel
	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(WIDTH, WIDTH, 1);
	mulKernel << <1, dimBlock >> > (dev_c, dev_a, dev_b, WIDTH);

	cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int),cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < WIDTH; y++) {

			printf("%5d", c[x][y]);
		}
		printf("\n");
	}
	return 0;

}