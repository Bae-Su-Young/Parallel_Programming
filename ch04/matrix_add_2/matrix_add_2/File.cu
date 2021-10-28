#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>


//kernel for GPU
__global__ void addKernel(int* c, const int* a, const int* b) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = y * (blockDim.x) + x;
	c[i] = a[i] + b[i];
}

//CPU
void add(const int x, const int y, const int WIDTH, int* c, const int* a, const int* b) {
	int i = y * (WIDTH)+x;//cuda에서는 2차원 배열에 접근할 수 있는 방법이 없기 때문에 이렇게 접근
	c[i] = a[i] + b[i];
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
	cudaMalloc((void**)&dev_a, WIDTH * WIDTH*sizeof(int));
	cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int));
	cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int));

	// copy from host to device
	cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

	//launch a kernel
	dim3 dimBlock(WIDTH, WIDTH, 1);
	dim3 dimGrid(1, 1, 1);
	addKernel << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b);

	//copy from device to host
	cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

	//Free
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);


	//calculate
	//for (int x = 0; x < WIDTH; x++) {
	//	for (int y = 0; y < WIDTH; y++) {
	//		add(x, y, WIDTH, (int*)(c), (int*)(a), (int*)(b));
	//	}
	//}

	//print
	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < WIDTH; y++) {

			printf("%5d", c[x][y]);
		}
		printf("\n");
	}
	return 0;
}