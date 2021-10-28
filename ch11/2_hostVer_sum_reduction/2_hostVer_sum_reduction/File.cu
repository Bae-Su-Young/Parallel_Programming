#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
using namespace chrono;

#define GRIDSIZE 1
#define BLOCKSIZE 1024
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genData(unsigned* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

void kernel(unsigned* pData, unsigned* pAnswer, unsigned size) {
	while (size--) {
		unsigned value = *pData++;
		*pAnswer = *pAnswer + value;
	}
}
int main(void) {
	unsigned* pData = NULL;
	unsigned answer = 0;

	//malloc memories on the host-side
	pData = (unsigned*)malloc(2 * BLOCKSIZE * sizeof(unsigned));

	//generate source data
	genData(pData, 2 * BLOCKSIZE);

	//HOST: start the timer
	system_clock::time_point h_start = system_clock::now();
	kernel(pData, &answer, 2 * BLOCKSIZE);

	//end the timer
	system_clock::time_point h_end = system_clock::now();
	nanoseconds h_du = h_end - h_start;
	printf("%lld nano-secodns\n", h_du);
	printf("answer= %lld \n", answer);
}