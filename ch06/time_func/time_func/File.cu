#include <stdio.h>
#include <time.h>

int main(void) {
	time_t t;
	time(&t);
	printf("%ld\n", t);
	printf(ctime(&t));
}