#include <iostream>
#include <algorithm>
#include <windows.h>
using namespace std;

#define TargetSize 2048

void genData(unsigned* ptr, unsigned size) {
	while (size--) {
		*ptr++ = (unsigned)(rand() % 10000);
	}
}

bool isSortedData(unsigned* ptr, unsigned size) {
	unsigned prev = *ptr++;
	size--;
	while (size--) {
		unsigned cur = *ptr++;
		if (prev > cur) return false;
		prev = cur;
	}
	return true;
}
int main(void) {
	unsigned* pData;
	long long start,end,freq;

	pData = (unsigned*)malloc(TargetSize * sizeof(unsigned));

	genData(pData, TargetSize);

	//check
	printf("sorting %d data\n", TargetSize);
	printf("%u %u %u %u %u %u\n",
		pData[0], pData[1], pData[2], pData[3], pData[4], pData[5]);
	printf("is sorted? -- %s\n", isSortedData(pData, TargetSize) ? "yes" : "no");

	//perform the action
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	QueryPerformanceCounter((LARGE_INTEGER*)(&start));
	std::sort(pData, pData + TargetSize);
	QueryPerformanceCounter((LARGE_INTEGER*)(&end));

	//print result
	printf("elapsed time = %f msec\n", (double)(end - start) * 1000.0 / (double)(freq));

	//check 
	printf("%u %u %u %u %u %u\n",
		pData[0], pData[1], pData[2], pData[3], pData[4], pData[5]);
	printf("is sorted? -- %s\n", isSortedData(pData, TargetSize) ? "yes" : "no");


}
