#include <iostream>
#include "./cpu-config.h"
#include "./cuda-config.cuh"
using namespace std;


//#define GRIDSIZE 2
//#define BLOCKSIZE 1024
//#define TOTALSIZE (BLOCKSIZE*GRIDSIZE)
//
//const unsigned TargetSize = 2 * BLOCKSIZE;

int main(void) {
	unsigned* pData;
	long long start, end, freq;
	pData = (unsigned*)malloc(TargetSize * sizeof(unsigned));
	

	genData(pData, TargetSize);

	//check
	printf("sorting %d data\n", TargetSize);
	printf("%u %u %u %u %u %u\n",
		pData[0], pData[1], pData[2], pData[3], pData[4], pData[5]);
	printf("is sorted? -- %s\n", isSortedData(pData, TargetSize) ? "yes" : "no");

	//perform
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	QueryPerformanceCounter((LARGE_INTEGER*)(&start));
	bit_sort_iter(pData, TargetSize);
	QueryPerformanceCounter((LARGE_INTEGER*)(&end));
	
	//check
	printf("elapsed time = %f msec\n", (double)(end - start) * 1000.0 / (double)(freq));
	printf("sorting %d data\n", TargetSize);
	printf("%u %u %u %u %u %u\n",
		pData[0], pData[1], pData[2], pData[3], pData[4], pData[5]);
	printf("is sorted? -- %s\n", isSortedData(pData, TargetSize) ? "yes" : "no");


	//CUDA
	long long s_c, e_c,f_c;
	unsigned* pData_c;
	pData_c = (unsigned*)malloc(TargetSize * sizeof(unsigned));
	unsigned* pData_cDev=NULL;

	genData(pData_c, TargetSize);

	printf("\n==== CUDA ====\n");
	printf("sorting %d data\n", TargetSize);
	printf("%u %u %u %u %u %u\n",
		pData_c[0], pData_c[1], pData_c[2], pData_c[3], pData_c[4], pData_c[5]);
	printf("is sorted? -- %s\n", isSortedData(pData_c, TargetSize) ? "yes" : "no");


	cudaMalloc((void**)&pData_cDev, TOTALSIZE * sizeof(unsigned));
	QueryPerformanceFrequency((LARGE_INTEGER*)(&f_c));
	
	cudaMemcpy(pData_cDev, pData_c, TOTALSIZE * sizeof(unsigned), cudaMemcpyHostToDevice);
	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	QueryPerformanceCounter((LARGE_INTEGER*)(&s_c));
	kernel << <dimGrid, dimBlock >> > (pData_cDev);
	cudaMemcpy(pData_c, pData_cDev, TOTALSIZE * sizeof(unsigned), cudaMemcpyDeviceToHost);
	QueryPerformanceCounter((LARGE_INTEGER*)(&e_c));

	//print result
	printf("elapsed time = %f msec\n", (double)(e_c - s_c) * 1000.0 / (double)(f_c));

	//check 
	printf("%u %u %u %u %u %u\n",
		pData_c[0], pData_c[1], pData_c[2], pData_c[3], pData_c[4], pData_c[5]);
	printf("is sorted? -- %s\n", isSortedData(pData_c, TargetSize) ? "yes" : "no");

}