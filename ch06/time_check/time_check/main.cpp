#include <stdio.h>
#include <time.h>

/*
int main(void) {
	time_t t;
	char buf[256];
	time(&t);
	printf("%lld\n", t);
	ctime_s(buf, sizeof(buf), &t);
	printf(buf);
}
*/

/*
int main(void) {
	time_t t;
	struct tm ptm;
	int hour, minute, second;
	time(&t);
	localtime_s(&ptm,&t);
	hour = ptm.tm_hour;
	minute = ptm.tm_min;
	second = ptm.tm_sec;
	printf("%02d:%02d:%02d\n", hour, minute, second);
}


#include <windows.h>
int main(void) {
	time_t t;
	struct tm ptm;
	int hour, minute, second;
	while (1) {
		time(&t);
		localtime_s(&ptm, &t);
		hour = ptm.tm_hour;
		minute = ptm.tm_min;
		second = ptm.tm_sec;
		printf("%02d:%02d:%02d\n", hour, minute, second);
		fflush(stdout);
		Sleep(1000);
	}
}


#include <windows.h>
#include <stdio.h>

int main(void) {
	DWORD start, end;
	start = GetTickCount();
	Sleep(30);
	end = GetTickCount();
	printf("elapsed=%d msec, start= %d, end= %d\n", end - start, start, end);
}


#include <windows.h>
#include <winbase.h>

void main() {
	LARGE_INTEGER start, end, f;
	QueryPerformanceFrequency(&f);
	QueryPerformanceCounter(&start);
	Sleep(3000);
	QueryPerformanceCounter(&end);
	__int64 ms_interval = (end.QuadPart - start.QuadPart) / (f.QuadPart / 1000);
	__int64 micro_interval= (end.QuadPart - start.QuadPart) / (f.QuadPart / 1000000);
	printf("millisecond: %d, microsecond: %d\n", (int)ms_interval, (int)micro_interval);
}
*/

#include <chrono>
#include <stdio.h>
#include <windows.h>

//using namespace std;
//using namespace chrono;

int main() {
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	Sleep(2000);
	std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::nanoseconds du = end - start;
	printf("%lld nano-seconds\n", du);
	return 0;
}