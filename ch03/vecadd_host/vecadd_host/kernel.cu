// PP_vectadd_host.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
//#include <iostream>



//c,a,b 배열은 device에 선언되어 있어야 한다.
__global__ void addKernel(int* c, const int* a, const int* b) {
	int i = threadIdx.x;//어느 core가 실행중인지가 (몇번쨰 core가) x에 담김
	c[i] = a[i] + b[i];
}

int main(void) {
	const int SIZE = 5;
	const int a[SIZE] = { 1,2,3,4,5 };
	const int b[SIZE] = { 10,20,30,40,50 };
	int c[SIZE] = { 0 };

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;

	//allocate device memory
	cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, SIZE * sizeof(int));
	cudaMalloc((void**)&dev_c, SIZE * sizeof(int));

	//copy from host to device
	cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	//lauch a kernel on the GPU with one thread for each element
	addKernel << <1, SIZE >> > (dev_c, dev_a, dev_b);


	//copy from device to host
	cudaMemcpy(c, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	//free device memory
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	//print the result
	printf("{%d, %d, %d,%d,%d}+{%d, %d, %d,%d,%d}""={%d, %d, %d,%d,%d}\n",
		a[0], a[1], a[2], a[3], a[4],
		b[0], b[1], b[2], b[3], b[4],
		c[0], c[1], c[2], c[3], c[4]);
	return 0;
}
