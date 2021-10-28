#include <stdio.h>
#include <stdlib.h>
#include <windows.h> 
#include "./hist-config.h"

void genData(unsigned int* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (unsigned int)(rand() % (NUMHIST - 1));
	}
}

void kernel(unsigned int* hist, unsigned int* img, unsigned int size) {
	while (size--) {
		unsigned int pixelVal = *img++;
		hist[pixelVal] = hist[pixelVal] + 1;

	}
}
int main(void) {
	unsigned int* pImage = NULL;
	unsigned int* pHistogram = NULL;
	int i;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));

	//malloc
	pImage = (unsigned int*)malloc(TOTALSIZE * sizeof(unsigned int));
	pHistogram = (unsigned int*)malloc(NUMHIST * sizeof(unsigned int));
	for (i = 0; i < NUMHIST; i++) {
		pHistogram[i] = 0;
	}

	//generate src data
	genData(pImage, TOTALSIZE);

	//star the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	kernel(pHistogram, pImage, TOTALSIZE);

	//end timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("elapsed time=%f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));

	//rpitn histogram
	long total = 0L;
	for (i = 0; i < NUMHIST; i++) {
		printf("%2d: %10d\n", i, pHistogram[i]);
		total += pHistogram[i];
	}
	printf("total: %10ld (should be %ld)\n", total, TOTALSIZE);
}