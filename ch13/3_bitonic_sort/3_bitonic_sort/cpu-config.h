#pragma once
#include <windows.h>

const int INCREASING = 1;
const int DECREASING = 0;

void compare(unsigned* pData, int i, int j, int dir) {
	if (dir == (pData[i] > pData[j])) {
		unsigned t;
		t = pData[i];
		pData[i] = pData[j];
		pData[j] = t;
	}
}

void bitonicMerge(unsigned* pData, int lo, int cnt, int dir) {
	if (cnt > 1) {
		int k = cnt / 2;
		for (int i = lo; i < lo + k; i++) {
			compare(pData, i, i + k, dir);
		}
		bitonicMerge(pData, lo, k, dir);
		bitonicMerge(pData, lo + k, k, dir);

	}
}

void recBitonicSort(unsigned* pData, int lo, int cnt, int dir) {
	if (cnt > 1) {
		int k = cnt / 2;
		recBitonicSort(pData, lo, k, INCREASING);
		recBitonicSort(pData, lo + k, k, DECREASING);
		bitonicMerge(pData, lo, cnt, dir);
	}
}

void bit_sort(unsigned* pData, unsigned size) {
	recBitonicSort(pData, 0, size, INCREASING);
}
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
void exchange(unsigned* pData, int i, int j) {
	unsigned t;
	t = pData[i];
	pData[i] = pData[j];
	pData[j] = t;
}
void bit_sort_iter(unsigned* pData, unsigned size) {
	int i, j, k;
	for (k = 2; k <= size; k *= 2) {
		for (j = k / 2; j > 0; j /= 2) {
			for (i = 0; i < size; i++) {
				int ij = i ^ j;
				if ((ij) > i) {
					if ((i & k) == 0) {
						if (pData[i] > pData[ij]) exchange(pData, i, ij);
					}
					else {
						if (pData[i] < pData[ij]) exchange(pData, i, ij);
					}
				}
			}
		}
	}
}